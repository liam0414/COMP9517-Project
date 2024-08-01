import numpy as np
import os
import gc

from ultralytics import YOLO
from .base import YOLOSemantic
from .. import NUM_CLASSES, METAINFO
from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .helpers import SHAPE, CV2_SHAPE, ORG_SHAPE, CV2_ORG_SHAPE, \
    get_corresponding_prediction, read_img, read_batch, replace_border_black_regions

PREDICT_PATH = 'predictions/'
MODEL_PATH = 'trained/weights/best.pt'


class EnsembleYOLO:
    """
    YOLO-based Ensemble learning model.
    """
    __slots__ = [
        "_yolo"     ,
        "_other"    ,
        "_path"     ,
        "transform" ,
    ]

    def __init__(
        self,
        yolo      : YOLO               | YOLOSemantic, # YOLO model
        other     : torch.nn.Module    | None = None , # Second model
        transform : transforms.Compose | None = None , # tensor transformation
        path      : str                | None = None , # path to yolo model
    ):
        assert not yolo is None
        self._yolo = yolo
        self._other = other
        self._path = path + PREDICT_PATH
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]) if transform is None else transform

    def transform_arr(self, arr: Iterable) -> torch.Tensor:
        """
        Transform any iterable type to tensor
        """
        ret: List[torch.Tensor] = [self.transform(x) for x in arr]
        return torch.stack(ret)

    def _labels_iou(
        self,
        test_labels: Iterable[str] , # all testing labeled mask
        num_classes: int           , # number of classes
        batch_size : int           , # batch size for speed improvement
        resize     : bool          , # whether image is to be resized for prediction
        verbose    : bool          , # verbose printing
    )             -> List[float]   : # list of IoU scores corresponding to each class
        """
        Get IoU score for every label (as a list of scores)
        """
        batch_inputs = np.array_split(
            test_labels, range(batch_size, len(test_labels), batch_size)
        )
        num_batches = len(batch_inputs)
        total_intersections = np.zeros((num_classes, num_batches))
        total_unions = np.zeros((num_classes, num_batches))

        # for each batch
        for idx in range(num_batches):
            batch = batch_inputs[idx]
            if verbose and idx % 10 == 0:
                print(f'Batch: {idx}/{num_batches}...')

            # ground truths and predictions of the entire batch
            ground_truths = [None] * len(batch)
            predictions   = [None] * len(batch)
            for i, test_label in enumerate(batch):
                ground_truths[i] = read_img(test_label, resize=resize)
                predictions[i] = get_corresponding_prediction(test_label, self._path)
            ground_truths = np.array(ground_truths)
            predictions   = np.array(predictions  )

            # create binary mask for each label
            for label in range(num_classes):
                gt_mask = (ground_truths == label)
                pred_mask = (predictions == label)

                # calculate intersection and union
                intersection = np.sum(gt_mask & pred_mask)
                union = np.sum(gt_mask | pred_mask)
                total_intersections[label][idx] += intersection
                total_unions[label][idx] += union
                del gt_mask
                del pred_mask
                gc.collect()

            del ground_truths
            del predictions
            gc.collect()

        # calculate total IoU of each class
        intersect_by_label = np.sum(total_intersections, axis=1)
        union_by_label = np.sum(total_unions, axis=1)
        ious = np.divide(intersect_by_label, union_by_label, where=union_by_label!=0)
        ious[union_by_label == 0] = np.nan
        return ious

    def _display_iou_score(self, ious: List[float]):
        classes = palette = METAINFO['classes']
        merged_classes = []
        merged_ious = []

        for idx, class_name in enumerate(classes):
            if class_name == "pole":
                # Merge pole IoU into other-object
                other_object_idx = classes.index("other-object")
                ious[other_object_idx] += ious[idx]
            elif class_name == "asphalt":
                # Merge asphalt IoU into other-terrain
                other_terrain_idx = classes.index("other-terrain")
                ious[other_terrain_idx] += ious[idx]
            elif class_name not in ["vehicle", "pole", "asphalt", "unlabelled"]:
                merged_classes.append(class_name)
                merged_ious.append(ious[idx])

        class_iou_pairs = list(zip(merged_classes, merged_ious))
        sorted_class_iou_pairs = sorted(class_iou_pairs, key=lambda x: x[0])
        print("\nIoU score\n" + "="*26)
        for class_name, iou in sorted_class_iou_pairs:
            tabs = '\t\t ' if len(class_name) < 10 else '\t '
            if len(class_name) >= 16:
                tabs = ' '
            print(f"{class_name}{tabs}: {iou * 100: .2f}")
        miou = np.nanmean(merged_ious)
        print(f"Mean IoU: {miou * 100: .2f}")

    def display_iou_score(
        self,
        test_labels : Iterable[str] ,
        batch_size  : int  = 9      ,
        resize      : bool = False  ,
        verbose     : bool = False  ,
    ):
        if not os.path.exists(self._path):
            os.makedirs(self._path)
        ious = self._labels_iou(
            test_labels, NUM_CLASSES, batch_size=batch_size, resize=resize, verbose=verbose
        )
        self._display_iou_score(ious)

    def _interpret_predictions(
        self,
        results     : Iterable[np.ndarray] ,
        img_shape   : Tuple[int, int]      ,
        resize      : bool = False         ,
        torch       : bool = False         ,
        border_fill : bool = False         ,
    )            -> List[np.ndarray]     :
        predictions = [None] * len(results)
        for i, result in enumerate(results):
            if not torch:
                prediction = np.zeros(img_shape, dtype=np.uint8)
                masks  = result.masks.data.cpu().numpy()
                labels = result.boxes.cls.cpu().numpy()
                for label, mask in zip(labels, masks):
                    prediction[mask != 0] = label
                del masks
                del labels
                gc.collect()
            else:
                prediction = result.cpu().detach().numpy()
            if border_fill:
                replace_border_black_regions(prediction)
            predictions[i] = prediction
        del results
        gc.collect()
        return predictions

    def _other_predict(self, images: Iterable[np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            tensors = self.transform_arr(images)
            # print(tensors)
            with autocast('cuda'):
                results = self._other(tensors)
            predictions = tensors.argmax(results, dim=1)
        ret = np.array(predictions)
        del predictions
        gc.collect()
        return ret

    def predict(
        self,
        inputs      : Iterable[np.ndarray] ,
        conf        : float      = 0.2     ,
        batch_size  : int        = 12      ,
        ret         : bool       = False   ,
        border_fill : bool = False         ,
    )             -> np.ndarray | None     :
        if not os.path.exists(self._path):
            os.makedirs(self._path)
        if batch_size > len(inputs):
            batch_size = len(inputs)

        # get unet prediction and apply immediately here --> save memory
        batch_inputs = np.array_split(inputs, range(batch_size, len(inputs), batch_size))
        resize = not self._other is None
        img_shape = SHAPE if resize else ORG_SHAPE
        ret = [None] * len(batch_inputs) if ret else None

        for i, batch in enumerate(batch_inputs):
            images = list(batch) if self._other is None else read_batch(batch)

            # prediction
            yolo_results = self._yolo(
                images, verbose=False, device=0, max_det=100, conf=conf, retina_masks=True
            )
            predictions = self._interpret_predictions(
                yolo_results, img_shape, resize=resize, border_fill=border_fill
            )

            # apply unlabeled mask to other model
            if not self._other is None:
                unet_results = self._other_predict(images)
                unet_preds   = self._interpret_predictions(
                    unet_results, img_shape, resize=resize, torch=True, border_fill=border_fill
                )
                unlabeled_mask = predictions == 0
                predictions[unlabeled_mask] = unet_preds[unlabeled_mask]

                del unet_preds
                del unet_results
                del unlabeled_mask
                gc.collect()

            if not ret:
                for path, prediction in zip(batch, predictions):
                    cv2.imwrite(self._path + path.split('/')[-1], prediction)
                del predictions
            else:
                ret[i] = predictions

            # manually free memory
            del images
            del yolo_results
            gc.collect()

        # return only if required
        if ret:
            return np.concatenate(np.array(ret))


    def predict_and_display_score(
        self,
        inputs      : Iterable[np.ndarray]     ,
        num_classes : int        = NUM_CLASSES ,
        conf        : float      = 0.2         ,
        batch_size  : int        = 9           ,
        ret         : bool       = False       ,
        border_fill : bool       = False       ,
    ):
        if not os.path.exists(self._path):
            os.makedirs(self._path)
        if batch_size > len(inputs):
            batch_size = len(inputs)

        # get unet prediction and apply immediately here --> save memory
        batch_inputs = np.array_split(inputs, range(batch_size, len(inputs), batch_size))
        resize = not self._other is None
        img_shape = SHAPE if resize else ORG_SHAPE

        # intersection and union
        total_intersections = np.zeros((num_classes, len(batch_inputs)))
        total_unions = np.zeros((num_classes, len(batch_inputs)))

        for idx, batch in enumerate(batch_inputs):
            images = list(batch) if self._other is None else read_batch(batch)

            # prediction
            yolo_results = self._yolo(
                images, verbose=False, device=0, max_det=100, conf=conf, retina_masks=True
            )
            predictions = self._interpret_predictions(
                yolo_results, img_shape, resize=resize, border_fill=border_fill
            )

            # apply unlabeled mask to other model
            if not self._other is None:
                unet_results = self._other_predict(images)
                unet_preds   = self._interpret_predictions(
                    unet_results, img_shape, resize=resize, torch=True, border_fill=border_fill
                )
                unlabeled_mask = predictions == 0
                predictions[unlabeled_mask] = unet_preds[unlabeled_mask]

                del unet_preds
                del unet_results
                del unlabeled_mask
                gc.collect()

            for label_path, prediction in zip(batch, predictions):
                # ground truths and predictions of the entire batch
                ground_truth = read_img(label_path.replace('image', 'indexLabel'), resize=resize)

                # create binary mask for each label
                for label in range(num_classes):
                    gt_mask = (ground_truth == label)
                    pred_mask = (prediction == label)

                    # calculate intersection and union
                    intersection = np.sum(gt_mask & pred_mask)
                    union = np.sum(gt_mask | pred_mask)
                    total_intersections[label][idx] += intersection
                    total_unions[label][idx] += union
                    del gt_mask
                    del pred_mask
                    gc.collect()

                del ground_truth
                gc.collect()

            # manually free memory
            del predictions
            del images
            del yolo_results
            gc.collect()

        # calculate total IoU of each class
        intersect_by_label = np.sum(total_intersections, axis=1)
        union_by_label = np.sum(total_unions, axis=1)
        ious = np.divide(intersect_by_label, union_by_label, where=union_by_label!=0)
        ious[union_by_label == 0] = np.nan
        self._display_iou_score(ious)


def get_yolo_model(path: str, model_path: str = MODEL_PATH) -> EnsembleYOLO:
    return EnsembleYOLO(YOLO(path + model_path), path=path)


# exporting models
cuda_avail = torch.cuda.is_available()
if cuda_avail:
    torch.cuda.set_device(0)
    print("GPU available")
else:
    print("No GPU found")
print(f'Device count  :  {torch.cuda.device_count()}')
print(f'Device in-use : \'{torch.cuda.current_device()}\'')
