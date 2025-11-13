#!/usr/bin/env python3
"""
Lightweight wrapper for YOLO segmentation to produce a binary foreground mask.
- Works with numpy RGB images (H, W, 3). Returns a boolean mask (H, W) where True=foreground.
- Optionally filter by class ids (COCO indices) via target_classes.

Usage:
    from ultralytics import YOLO
    from yolo_segmentation import YoloSegmentation
    model = YOLO('yolov8n-seg.pt')
    model.to('cpu')  # or 'cuda:0'
    seg = YoloSegmentation(model=model, target_classes=None, conf=0.25)
    mask = seg(rgb_image)  # bool HxW or None if no masks
"""
from __future__ import annotations
import numpy as np
import cv2
from typing import Iterable, Optional, Any, cast
from ultralytics import YOLO  # type: ignore

class YoloSegmentation:
    def __init__(self,
                 model: YOLO,
                 device: Optional[str] = None,
                 target_classes: Optional[Iterable[int]] = None,
                 conf: float = 0.25):
        """Create a segmentation mask provider from an existing YOLO model instance.

        Args:
            model: An initialized ultralytics.YOLO model (e.g., YOLO('yolov8n-seg.pt')).
            device: Optional device string. If provided, the model will be moved via model.to(device).
            target_classes: Optional set/iterable of COCO class ids to keep. If None, keep all.
            conf: Confidence threshold for inference.
        """
        self.device = device
        self.target_classes = set(target_classes) if target_classes is not None else None
        self.conf = conf
        self._model = model
        if device is not None:
            # Move the provided model to the requested device if specified
            self._model.to(device)

    def __call__(self, rgb_image: np.ndarray) -> Optional[np.ndarray]:
        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        res = self._model(bgr, conf=self.conf, verbose=False)[0]
        if res.masks is None:
            return None
        H, W = rgb_image.shape[:2]
        mask = np.zeros((H, W), dtype=bool)
        cls_ids = res.boxes.cls.detach().cpu().numpy().astype(int)
        n = len(res.masks.data)
        idxs = range(n) if self.target_classes is None else [i for i, c in enumerate(cls_ids[:n]) if c in self.target_classes]
        for i in idxs:
            m = res.masks.data[i].detach().cpu().numpy().astype(bool)
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            mask |= m
        return mask if mask.any() else None


if __name__ == "__main__":
    MODEL = 'yolov8n-seg.pt'
    DEVICE = 'cuda:0'            # 'cpu' or 'cuda:0'
    CLASSES = {39}            # e.g., {39, 41} or None
    CONF = 0.25
    COLOR_SIZE = (640, 480)   # e.g., (960, 540) for L515 color
    FPS = 30

    # Build YOLO model first, then pass the instance into YoloSegmentation
    yolo_model = YOLO(MODEL)
    yolo_model.to(DEVICE)
    seg = YoloSegmentation(model=yolo_model, target_classes=CLASSES, conf=CONF)

    def visualize_mask(bgr: np.ndarray, mask: Optional[np.ndarray], alpha: float = 0.5) -> np.ndarray:
        if mask is None:
            return bgr
        m = mask.astype(bool)
        o = bgr.copy()
        o[m] = (0, 255, 0)
        return cv2.addWeighted(o, alpha, bgr, 1.0 - alpha, 0)

    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_SIZE[0], COLOR_SIZE[1], rs.format.rgb8, FPS)
    pipeline.start(config)
    print("[INFO] Press 'q' to quit.")
    while True:
        frames = pipeline.wait_for_frames()
        cf = frames.get_color_frame()
        if not cf:
            continue
        color = np.asanyarray(cf.get_data())  # RGB
        mask = seg(color)
        bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        vis = visualize_mask(bgr, mask, alpha=0.5)
        cv2.imshow('YoloSeg (RealSense)', vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    pipeline.stop()
    cv2.destroyAllWindows()
