import ast
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort


LABELS = {
    0: "with_mask",
    1: "without_mask",
    2: "mask_incorrect",
}


class MaskDetector:
    def __init__(self, weights, confidence=0.4):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(weights),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.confidence = confidence
        self.input = self.session.get_inputs()[0]

        shape = self.input.shape
        if (
            len(shape) != 4
            or shape[0] != 1
            or shape[1] != 3
            or not isinstance(shape[2], int)
            or not isinstance(shape[3], int)
        ):
            raise ValueError("Expected static input [1, 3, height, width]")

        self.input_h, self.input_w = shape[2], shape[3]

        metadata = self.session.get_modelmeta().custom_metadata_map
        if "names" not in metadata:
            raise ValueError("Model metadata is missing class names")

        names = ast.literal_eval(metadata["names"])
        names = {int(k): v for k, v in names.items()}
        if names != LABELS:
            raise ValueError(f"Unexpected class mapping: {names}")

        # Warm up before reporting the model as ready.
        self.predict(np.zeros((480, 640, 3), dtype=np.uint8))

    def predict(self, image_bgr):
        start = perf_counter()
        height, width = image_bgr.shape[:2]

        scale = min(self.input_w / width, self.input_h / height)
        new_w = round(width * scale)
        new_h = round(height * scale)

        resized = cv2.resize(image_bgr, (new_w, new_h))
        left = (self.input_w - new_w) // 2
        top = (self.input_h - new_h) // 2

        canvas = np.full(
            (self.input_h, self.input_w, 3), 114, dtype=np.uint8
        )
        canvas[top:top + new_h, left:left + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(
            rgb.transpose(2, 0, 1)[None], dtype=np.float32
        )
        tensor /= 255.0
        prepared = perf_counter()

        raw = self.session.run(None, {self.input.name: tensor})[0]
        inferred = perf_counter()

        if raw.ndim != 3 or raw.shape[0] != 1:
            raise ValueError(f"Unexpected output shape: {raw.shape}")

        pred = raw[0]
        channels = 4 + len(LABELS)

        if pred.shape[0] == channels:
            pred = pred.T
        elif pred.shape[1] != channels:
            raise ValueError(f"Unexpected output shape: {raw.shape}")

        class_ids = pred[:, 4:].argmax(axis=1)
        scores = pred[:, 4:].max(axis=1)
        valid = np.isfinite(pred).all(axis=1)
        valid &= scores >= self.confidence

        pred = pred[valid]
        class_ids = class_ids[valid]
        scores = scores[valid]

        detections = []

        if len(pred):
            centers = pred[:, :2]
            sizes = pred[:, 2:4]
            boxes = np.concatenate(
                (centers - sizes / 2, centers + sizes / 2), axis=1
            )

            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)

            positive = (boxes[:, 2] > boxes[:, 0])
            positive &= (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[positive]
            scores = scores[positive]
            class_ids = class_ids[positive]

            if len(boxes):
                xywh = boxes.copy()
                xywh[:, 2:] -= xywh[:, :2]

                # Class-agnostic NMS, matching the current lab approach.
                kept = cv2.dnn.NMSBoxes(
                    xywh.tolist(),
                    scores.tolist(),
                    self.confidence,
                    0.5,
                )
                kept = np.asarray(kept, dtype=int).reshape(-1)[:100]

                for index in kept:
                    class_id = int(class_ids[index])
                    detections.append({
                        "class_id": class_id,
                        "label": LABELS[class_id],
                        "confidence": float(scores[index]),
                        "bbox_xyxy": boxes[index].tolist(),
                    })

        counts = {name: 0 for name in LABELS.values()}
        for detection in detections:
            counts[detection["label"]] += 1

        end = perf_counter()
        return {
            "detections": detections,
            "counts": counts,
            "timing_ms": {
                "preprocess": (prepared - start) * 1000,
                "inference": (inferred - prepared) * 1000,
                "postprocess": (end - inferred) * 1000,
                "model_pipeline": (end - start) * 1000,
            },
        }