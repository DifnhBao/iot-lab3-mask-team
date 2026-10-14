"""Cloud detector. Only trained three-class weights are accepted."""
from pathlib import Path
from time import perf_counter_ns

import numpy as np

LABELS = {0: "with_mask", 1: "without_mask", 2: "mask_incorrect"}


class MaskDetector:
    def __init__(self, weights: str, imgsz: int = 640, confidence: float = 0.4):
        from ultralytics import YOLO

        self.model = YOLO(str(Path(weights)))
        names = {int(key): value for key, value in self.model.names.items()}
        if names != LABELS:
            raise ValueError("Model must have class IDs 0/1/2: with_mask, without_mask, mask_incorrect")
        self.imgsz = imgsz
        self.confidence = confidence
        # Baseline CPU inference; no dependency on the laptop's GPU.
        self.model.predict(np.zeros((480, 640, 3), dtype=np.uint8),
                           imgsz=imgsz, device="cpu", verbose=False)

    def predict(self, image_bgr: np.ndarray) -> dict:
        start = perf_counter_ns()
        result = self.model.predict(
            image_bgr,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=0.5,
            agnostic_nms=True,
            max_det=100,
            device="cpu",
            verbose=False,
        )[0]
        pipeline_ms = (perf_counter_ns() - start) / 1_000_000
        detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append({
                "class_id": class_id,
                "label": LABELS[class_id],
                "confidence": float(box.conf.item()),
                "bbox_xyxy": [float(value) for value in box.xyxy[0].tolist()],
            })
        counts = {name: 0 for name in LABELS.values()}
        for detection in detections:
            counts[detection["label"]] += 1
        return {"detections": detections, "counts": counts,
                "timing_ms": {**result.speed, "model_pipeline": pipeline_ms}}
