#!/usr/bin/env python3
"""
Lightweight wrapper for YOLO detection to produce a binary foreground mask from bounding boxes.
- Works with numpy RGB images (H, W, 3). Returns a boolean mask (H, W) where True=inside any bbox.
- Optionally filter by class ids (COCO indices) via target_classes.

Usage:
    from ultralytics import YOLO
    from yolo_detection import YoloDetection
    model = YOLO('yolov8n.pt')  # detection weights, but seg weights also output boxes
    model.to('cpu')  # or 'cuda:0'
    det = YoloDetection(model=model, target_classes=None, conf=0.25)
    mask = det(rgb_image)  # bool HxW or None if no boxes

Notes:
- You can also use segmentation weights (e.g., 'yolov8n-seg.pt'); the results still contain boxes.
- This class only uses bounding boxes to form a rectangular mask.
"""
from __future__ import annotations
import numpy as np
import cv2
from typing import Iterable, Optional
from ultralytics import YOLO  # type: ignore

class YoloDetection:
    def __init__(self,
                 model: YOLO,
                 device: Optional[str] = None,
                 target_classes: Optional[Iterable[int]] = None,
                 conf: float = 0.25):
        """Create a bbox-mask provider from an existing YOLO model instance.

        Args:
            model: An initialized ultralytics.YOLO model (e.g., YOLO('yolov8n.pt')).
            device: Optional device string. If provided, the model will be moved via model.to(device).
            target_classes: Optional set/iterable of COCO class ids to keep. If None, keep all.
            conf: Confidence threshold for inference.
        """
        self.device = device
        self.target_classes = set(target_classes) if target_classes is not None else None
        self.conf = conf
        self._model = model
        if device is not None:
            self._model.to(device)

    def __call__(self, rgb_image: np.ndarray) -> Optional[np.ndarray]:
        """Run YOLO detection and convert detected boxes into a boolean mask.

        Args:
            rgb_image: uint8 array of shape (H, W, 3) in RGB order.
        Returns:
            (H, W) boolean mask (True inside any bbox) or None if no boxes kept.
        """
        H, W = rgb_image.shape[:2]
        if H == 0 or W == 0:
            return None
        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        res = self._model(bgr, conf=self.conf, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None

        # boxes.xyxy: (N, 4), boxes.cls: (N,)
        boxes_xyxy = res.boxes.xyxy.detach().cpu().numpy()
        cls_ids = res.boxes.cls.detach().cpu().numpy().astype(int)

        # filter by classes if provided
        idxs = range(len(boxes_xyxy)) if self.target_classes is None \
               else [i for i, c in enumerate(cls_ids) if c in self.target_classes]
        if not idxs:
            return None

        mask = np.zeros((H, W), dtype=bool)
        for i in idxs:
            x1, y1, x2, y2 = boxes_xyxy[i]
            # clip to image bounds and cast to int
            xi1 = int(max(0, min(W - 1, np.floor(x1))))
            yi1 = int(max(0, min(H - 1, np.floor(y1))))
            xi2 = int(max(0, min(W,     np.ceil(x2))))
            yi2 = int(max(0, min(H,     np.ceil(y2))))
            if xi2 > xi1 and yi2 > yi1:
                mask[yi1:yi2, xi1:xi2] = True
        return mask if mask.any() else None


if __name__ == "__main__":
    MODEL = 'yolov8n-seg.pt'  # seg weights also provide boxes
    DEVICE = 'cuda:0'         # 'cpu' or 'cuda:0'
    CLASSES = {39}            # e.g., {0} for person
    CONF = 0.25

    # Build YOLO model first, then pass the instance into YoloDetection
    yolo_model = YOLO(MODEL)
    yolo_model.to(DEVICE)
    det = YoloDetection(model=yolo_model, target_classes=CLASSES, conf=CONF)

    # Small RealSense demo overlay
    import pyrealsense2 as rs  # type: ignore
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipeline.start(config)
    print("[INFO] Press 'q' to quit.")
    while True:
        frames = pipeline.wait_for_frames()
        cf = frames.get_color_frame()
        if not cf:
            continue
        color = np.asanyarray(cf.get_data())  # RGB
        mask = det(color)
        bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        if mask is not None:
            o = bgr.copy()
            o[mask] = (0, 255, 0)
            vis = cv2.addWeighted(o, 0.5, bgr, 0.5, 0)
        else:
            vis = bgr
        cv2.imshow('YoloDet BBox Mask (RealSense)', vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    pipeline.stop()
    cv2.destroyAllWindows()
