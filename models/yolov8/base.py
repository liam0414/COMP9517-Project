# NOTE: all changes made were directly on ultralytics library, and the following code only highlights
# all important changes that were made there.
#
# This includes: loss function change, and layer architectural changes. That said, all changes were
# made for train ONLY, and prediction process is not tampered with.
#
# Therefore, as long as the `.pt` model file can be loaded, prediction results should remain the same.


import numpy as np
import yaml
from copy import deepcopy
from pathlib import Path
from typing import Union, List
from .. import NUM_CLASSES, METAINFO, ORG_HEIGHT, ORG_WIDTH, id_labels
from .loss import YOLOLoss

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from ultralytics import YOLO
from ultralytics.nn import parse_model
from ultralytics.nn.modules import Conv, Segment, Concat, C2f, SPPF
from ultralytics.nn.modules.conv import autopad
from ultralytics.utils.torch_utils import initialize_weights
from ultralytics.models import yolo
from ultralytics.engine.results import Results
from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    RANK,
    callbacks,
    checks,
    yaml_load,
)


class BaseModel(nn.Module):
    """
    Base YOLO Model, closely following the conventions of YOLO codebase for compatibility,
    with modifications only where needed.
    Note that actual changes were made directly to the library of ultralytics.
    """
    def forward(
        self,
        x: torch.Tensor | dict,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of the model on a single scale. Wrapper for `_forward_once` method.
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(
        self,
        x         : torch.Tensor       , # input tensor to the model
        augment   : bool = False       , # augment image during prediction
        embed     : list | None = None , # list of feature vectors
    )            -> torch.Tensor:        # The last output of the model
        """
        Perform a forward pass through the network.
        """
        return self._predict_augment(x) if augment else self._predict_once(x, embed)

    def _predict_once(
        self,
        x     : torch.Tensor,
        embed : list | None = None
    )        -> torch.Tensor:
        """
        Perform a forward pass through the network.
        """
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)  # run
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        return self._predict_once(x)

    def load(
        self,
        weights: dict | torch.nn.Module, # pre-trained weights to be loaded
    ):
        """
        Load the weights into the model.
        """
        model = weights["model"] if isinstance(weights, dict) else weights
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load

    def loss(self, batch: dict, preds: torch.Tensor | list[torch.Tensor] | None = None):
        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)


class YOLOSegmentation(BaseModel):
    """
    YOLO Semantic Segmentation Base.
    """
    def __init__(
        self,
        cfg={},
        n_channels=3,
        names=id_labels
    ):
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super().__init__()

        # Define
        self.n_channels = n_channels
        self.n_classes  = len(id_labels)
        self.names      = names
        self.inplace    = cfg.get("inplace", True)
        self.criterion  = YOLOLoss
        if cfg == {}:
            depth, width, max_channels = 0.67, 0.75, 768
            layers = [
                # backbone
                Conv(3  , 32 , 3, 2, 1),
                Conv(32 , 64 , 3, 2, 1),
                C2f (64 , 64 , 1, True, 1, 0.5),
                Conv(64 , 128, 3, 2, 1),
                C2f (128, 128, 2, True, 1, 0.5),
                Conv(128, 256, 3, 2, 1),
                C2f (256, 256, 2, True, 1, 0.5),
                Conv(256, 512, 3, 2, 1),
                C2f (512, 512, 1, True, 1, 0.5),
                SPPF(512, 512, 1),

                # head
                nn.Upsample(scale_factor=2.0, mode='nearest'),
                Concat(),
                C2f (768, 256, 1, False, 1, 0.5),
                nn.Upsample(scale_factor=2.0, mode='nearest'),
                Concat(),
                C2f (384, 128, 1, False, 1, 0.5),
                Conv(128, 128, 3, 2, 1),
                Concat(),
                C2f (384, 256, 1, False, 1, 0.5),
                Conv(256, 256, 3, 2, 1),
                Concat(),
                C2f (768, 512, 1, False, 1, 0.5),
                Segment(self.n_classes, 32, 128, [128, 256, 512]),
            ]

            f = -1 # lookback
            for i, layer in enumerate(layers):
                layer = layers[i]
                # seq = nn.Sequential(layer)
                t = str(layer)[8:-2].replace("__main__.", "")
                layer.np = sum(x.numel() for x in layer.parameters())
                layer.i, layer.f, layer.type = i, f, t
                layers[i] = layer

            self.model = nn.Sequential(*layers)
        else:
            self.model = parse_model(deepcopy(cfg), n_channels=n_channels, verbose=False)
        self.end2end = getattr(self.model[-1], "end2end", False)
        

        # Build strides
        m = self.model[-1]
        s = 256  # 2x min stride
        m.inplace = self.inplace

        def _forward(x):
            """
            Performs a forward pass through the model"""
            if self.end2end:
                return self.forward(x)["one2many"]
            return self.forward(x)[0] if isinstance(m, (Segment)) else self.forward(x)

        m.stride = torch.tensor(
            [s / x.shape[-2] for x in _forward(torch.zeros(1, self.n_channels, s, s))]
        )  # forward
        self.stride = m.stride
        m.bias_init()  # only run once
        # Init weights, biases
        initialize_weights(self)

    def _predict_augment(self, x):
        """
        Perform augmentations on input image x and return augmented inference and train outputs.
        """
        if getattr(self, "end2end", False):
            return self._predict_once(x)
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torn_channels.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torn_channels.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y


class YOLOTrainer(BaseTrainer):
    """
    YOLO Trainer, used for training.
    """
    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        shuffle = not (getattr(dataset, "rect", False) and shuffle)
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(self.args.imgsz * 0.5, self.args.imgsz * 1.5 + self.stride)
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_model(self, cfg=None, weights=None, verbose=False):
        """Return a YOLO detection model."""
        model = YOLOSegmentation()
        if weights:
            model.load(weights)
        return model

    def progress_string(self):
        """
        Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size.
        """
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )


class YOLOModel(nn.Module):
    """
    YOLO generic model modified.
    """
    def __init__(
        self,
        model     : nn.Module    ,
        predictor : nn.Module    ,
        trainer   : nn.Module    ,
        validator : nn.Module    ,
        task      : str  = None  ,
        verbose   : bool = False ,
    ):
        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = predictor
        self.trainer   = trainer
        self.validator = validator
        self.model     = model
        self.ckpt      = None  # if loaded from *.pt
        self.cfg       = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}    # overrides for trainer object
        self.metrics   = None  # validation/training metrics
        self.session   = None  # HUB session
        self.task      = task  # task type

    def save(self, filename: Union[str, Path] = "saved_model.pt", use_dill=True) -> None:
        from copy import deepcopy
        from datetime import datetime
        from ultralytics import __version__

        updates = {
            "model": deepcopy(self.model).half() if isinstance(self.model, nn.Module) else self.model,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }
        torch.save({**self.ckpt, **updates}, filename, use_dill=use_dill)

    def embed(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs,
    ) -> list:
        if not kwargs.get("embed"):
            kwargs["embed"] = [len(self.model.model) - 2]
        return self.predict(source, stream, **kwargs)

    def predict(
        self,
        source    : Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream    : bool = False,
        **kwargs,
    ) -> List[Results]:
        source = ASSETS
        is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ultralytics")) and any(
            x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
        )

        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict"}  # method defaults
        args = {**self.overrides, **custom, **kwargs}  # highest priority args on the right
        prompts = args.pop("prompts", None)  # for SAM-type models

        self.predictor.args = get_cfg(self.predictor.args, args)
        if "project" in args or "name" in args:
            self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, "set_prompts"):  # for SAM-type models
            self.predictor.set_prompts(prompts)
        return self.predictor.predict_cli(source=source) \
            if is_cli else self.predictor(source=source, stream=stream)

    def val(
        self,
        **kwargs,
    ):
        custom = {"rect": True}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = self.validator(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def benchmark(
        self,
        **kwargs,
    ):
        from ultralytics.utils.benchmarks import benchmark

        custom = {"verbose": False}  # method defaults
        args = {**DEFAULT_CFG_DICT, **self.model.args, **custom, **kwargs, "mode": "benchmark"}
        return benchmark(
            model=self,
            data=kwargs.get("data"),  # if no 'data' argument passed set data=None for default datasets
            imgsz=args["imgsz"],
            half=args["half"],
            int8=args["int8"],
            device=args["device"],
            verbose=kwargs.get("verbose"),
        )

    def export(
        self,
        **kwargs,
    ) -> str:
        from .exporter import Exporter

        custom = {
            "imgsz": self.model.args["imgsz"],
            "batch": 1,
            "data": None,
            "device": None,  # reset to avoid multi-GPU errors
            "verbose": False,
        }  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}  # highest priority args on the right
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def train(
        self,
        **kwargs,
    ):
        if hasattr(self.session, "model") and self.session.model.id: # Ultralytics HUB session with loaded model
            kwargs = self.session.train_args  # overwrite kwargs

        checks.check_pip_update_available()
        overrides = yaml_load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
        custom = {
            # NOTE: handle the case when 'cfg' includes 'data'.
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task],
            "task": self.task,
        }  # method defaults
        args = {**overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right
        if args.get("resume"):
            args["resume"] = self.ckpt_path

        self.trainer = YOLOTrainer(overrides=args, _callbacks=self.callbacks)
        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train()
        # Update model and cfg after training
        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, _ = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, "metrics", None)  # TODO: no metrics returned by DDP
        return self.metrics

    def tune(
        self,
        use_ray=False,
        iterations=10,
        *args,
        **kwargs,
    ):
        if use_ray:
            from ultralytics.utils.tuner import run_ray_tune
            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)
        else:
            from ultralytics.engine.tuner import Tuner
            custom = {}  # method defaults
            args = {**self.overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right
            return Tuner(args=args, _callbacks=self.callbacks)(model=self, iterations=iterations)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

    @property
    def transforms(self):
        return self.model.transforms if hasattr(self.model, "transforms") else None

    @staticmethod
    def _reset_ckpt_args(args: dict) -> dict:
        include = {"imgsz", "data", "task", "single_cls"}
        return {k: v for k, v in args.items() if k in include}


class YOLOSemantic(YOLOModel):
    """
    YOLO Semantic Segmentation model.
    """
    def __init__(self):
        super().__init__(
            model     = YOLOSegmentation,
            trainer   = YOLOTrainer,
            validator = yolo.segment.SegmentationValidator,
            predictor = yolo.segment.SegmentationPredictor,
            task      = "segment",
            verbose   = False
        )