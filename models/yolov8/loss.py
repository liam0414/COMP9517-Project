import torch
import torch.nn.functional as F

from ultralytics.utils.tal import make_anchors
from ultralytics.utils.loss import VarifocalLoss, v8DetectionLoss


class YOLOLoss(v8DetectionLoss):
    """
    Criterion class for computing training losses. The following class is the modified version
    used for YOLO semantic segmentation. To make it work, please directly add the following
    code to the library hard-code at `Lib/site-packages/ultralytics/utils/loss.py`.
    """

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the YOLOLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask
        self.varifocal_loss = VarifocalLoss().to(self.device)

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        # batch size, number of masks, mask height, mask width
        batch_size, _, mask_h, mask_w = proto.shape
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError("Segment dataset incorrectly formatted or not a segment dataset") from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)

        # Varifocal loss
        target_labels = target_bboxes.unsqueeze(-1).expand(-1, -1, self.nc)
        one_hot = torch.zeros(target_labels.size(), device=self.device)
        one_hot.scatter_(-1, target_labels, 1)
        loss[1] = self.varifocal_loss(pred_scores, target_scores, one_hot) / target_scores_sum

        # BCE, but automatically calculated as MCE
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()

        loss[0] *= self.hyp.dfl  # BBox weight      = 1.75
        loss[1] *= self.hyp.box  # Varifocal weight = 7.75
        loss[2] *= self.hyp.cls  # MCE weight       = 0.45
        loss[3] *= self.hyp.dfl  # DFL weight       = 1.75

        return loss.sum() * batch_size, loss.detach()
