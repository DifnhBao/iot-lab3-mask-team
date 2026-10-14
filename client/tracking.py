"""Local optical flow; Cloud detections remain the only source of class labels."""
import queue
import threading
from dataclasses import dataclass
from time import perf_counter_ns

import cv2
import numpy as np

LABELS = {"with_mask", "without_mask", "mask_incorrect"}


class FlowTracker:
    """Track small grayscale frames, returning boxes in original image coordinates."""
    def __init__(self):
        self.previous = None
        self.boxes = []

    @staticmethod
    def features(gray, bbox):
        x1, y1, x2, y2 = np.rint(bbox).astype(int)
        height, width = gray.shape
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        mask = np.zeros_like(gray)
        mask[y1:y2, x1:x2] = 255
        points = cv2.goodFeaturesToTrack(gray, maxCorners=40, qualityLevel=0.02,
                                        minDistance=4, mask=mask, blockSize=3)
        return points if points is not None else np.empty((0, 1, 2), np.float32)

    def seed(self, frame, detections):
        self.previous = frame
        self.boxes = []
        sy = frame.gray.shape[0] / frame.image_shape[0]
        sx = frame.gray.shape[1] / frame.image_shape[1]
        for detection in detections:
            bbox = np.array(detection["bbox_xyxy"], dtype=np.float32, copy=True)
            confidence = float(detection["confidence"])
            if (detection["label"] not in LABELS or bbox.shape != (4,) or
                    not np.isfinite(bbox).all() or not np.isfinite(confidence) or
                    not 0 <= confidence <= 1):
                raise ValueError("Invalid detection")
            bbox *= [sx, sy, sx, sy]
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, frame.gray.shape[1] - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, frame.gray.shape[0] - 1)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            self.boxes.append({"bbox": bbox, "label": detection["label"],
                               "confidence": confidence,
                               "points": self.features(frame.gray, bbox)})

    def advance(self, frame):
        previous = self.previous
        if previous is None:
            self.previous = frame
            return
        if frame.session_id != previous.session_id or frame.image_shape != previous.image_shape:
            self.boxes = []
            self.previous = frame
            return
        if frame.frame_id <= previous.frame_id:
            return
        # Missing history is not a license to extrapolate a face across a large gap.
        if frame.frame_id != previous.frame_id + 1:
            self.boxes = []
            self.previous = frame
            return
        survivors = []
        for box in self.boxes:
            points = box["points"]
            if len(points) < 6:
                continue
            moved, status, error = cv2.calcOpticalFlowPyrLK(
                previous.gray, frame.gray, points, None, winSize=(21, 21), maxLevel=3)
            if moved is None:
                continue
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                frame.gray, previous.gray, moved, None, winSize=(21, 21), maxLevel=3)
            if back is None:
                continue
            valid = ((status.ravel() == 1) & (back_status.ravel() == 1) &
                     (error.ravel() < 25) &
                     (np.linalg.norm(back - points, axis=2).ravel() < 1.5))
            if valid.sum() < max(6, int(len(points) * 0.4)):
                continue
            old = points[valid].reshape(-1, 2)
            new = moved[valid].reshape(-1, 2)
            transform, inliers = cv2.estimateAffinePartial2D(
                old, new, method=cv2.RANSAC, ransacReprojThreshold=2)
            if transform is None or inliers.sum() < max(6, len(old) * 0.5):
                continue
            scale = float(np.hypot(transform[0, 0], transform[1, 0]))
            if not 0.8 <= scale <= 1.25:
                continue
            bbox = box["bbox"]
            center = (bbox[:2] + bbox[2:]) / 2
            center = transform[:, :2] @ center + transform[:, 2]
            half_size = (bbox[2:] - bbox[:2]) * scale / 2
            bbox = np.concatenate((center - half_size, center + half_size)).astype(np.float32)
            # A box leaving the image or losing its texture must disappear.
            height, width = frame.gray.shape
            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] >= width or bbox[3] >= height:
                continue
            box = {**box, "bbox": bbox, "points": self.features(frame.gray, bbox)}
            if len(box["points"]) >= 6:
                survivors.append(box)
        self.boxes = survivors
        self.previous = frame

    def detections(self):
        if self.previous is None:
            return []
        sy = self.previous.image_shape[0] / self.previous.gray.shape[0]
        sx = self.previous.image_shape[1] / self.previous.gray.shape[1]
        return [{"label": box["label"], "confidence": box["confidence"],
                 "bbox_xyxy": (box["bbox"] * [sx, sy, sx, sy]).tolist()}
                for box in self.boxes]


@dataclass(frozen=True)
class DisplayFrame:
    frame: object
    detections: list
    result_key: object
    prediction_ns: int


class TrackingWorker:
    """Replay delayed detections on their source history, outside the Tk thread."""
    def __init__(self, camera, config, events):
        self.camera = camera
        self.events = events
        self.max_age_ns = max(0.1, float(config.get("tracking_max_age_seconds", 1.0))) * 1e9
        self.results = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.thread = None
        self._lock = threading.Lock()
        self._generation = 0
        self._latest = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def clear(self):
        with self._lock:
            self._generation += 1
            self._latest = None

    def get_latest(self):
        with self._lock:
            return self._latest

    def submit(self, event):
        with self._lock:
            generation = self._generation
        try:
            self.results.put_nowait((generation, event))
        except queue.Full:
            try:
                _, old = self.results.get_nowait()
                self._complete(old, False, "SUPERSEDED")
            except queue.Empty:
                pass
            self.results.put_nowait((generation, event))

    def _complete(self, event, accepted, reason=""):
        # Same bounded queue as transport; GUI owns CSV and all Tk widgets.
        try:
            self.events.put_nowait({"type": "tracking_result", "event": event,
                                    "accepted": accepted, "reason": reason})
        except queue.Full:
            pass

    def _run(self):
        flow = FlowTracker()
        generation = -1
        result_key = None
        prediction_ns = 0
        last_frame_key = None
        while not self.stop_event.is_set():
            with self._lock:
                current_generation = self._generation
            if current_generation != generation:
                flow = FlowTracker()
                generation = current_generation
                result_key = None
                prediction_ns = 0
                last_frame_key = None
            frame = self.camera.get_latest(max_age_seconds=0.5)
            pending = None
            while True:
                try:
                    item_generation, event = self.results.get_nowait()
                except queue.Empty:
                    break
                if item_generation != generation:
                    self._complete(event, False, "RESET")
                    continue
                if pending is not None:
                    self._complete(pending, False, "SUPERSEDED")
                pending = event
            if frame is None:
                flow = FlowTracker()
                result_key = None
                prediction_ns = 0
                with self._lock:
                    self._latest = None
                if pending:
                    self._complete(pending, False, "CAMERA_UNAVAILABLE")
                self.stop_event.wait(0.01)
                continue
            key = (frame.session_id, frame.frame_id)
            now = perf_counter_ns()
            if flow.previous is not None and (flow.previous.session_id != frame.session_id or
                                               flow.previous.image_shape != frame.image.shape[:2]):
                flow = FlowTracker()
                prediction_ns = 0
                result_key = None
            if prediction_ns and now - prediction_ns > self.max_age_ns:
                flow = FlowTracker()
                prediction_ns = 0
                result_key = None
            if key == last_frame_key and pending is None:
                # Also refresh the expiry state while the camera pauses.
                with self._lock:
                    if generation == self._generation and self._latest:
                        self._latest = DisplayFrame(frame, flow.detections(), result_key, prediction_ns)
                self.stop_event.wait(0.005)
                continue
            history = self.camera.history.through(frame)
            accepted = False
            reason = ""
            if pending is not None:
                response = pending["response"]
                anchor = next((item for item in history
                               if item.session_id == response["session_id"] and
                               item.frame_id == response["frame_id"]), None)
                if anchor is None or now - pending["received_ns"] > self.max_age_ns:
                    reason = "STALE_RESULT"
                elif result_key and response["session_id"] == result_key[0] and response["frame_id"] <= result_key[1]:
                    reason = "OUT_OF_ORDER"
                else:
                    try:
                        flow.seed(anchor, response["detections"])
                        prediction_ns = pending["received_ns"]
                        result_key = (response["session_id"], response["frame_id"])
                        accepted = True
                    except (ValueError, KeyError, TypeError, cv2.error):
                        flow = FlowTracker()
                        result_key = None
                        prediction_ns = 0
                        reason = "INVALID_DETECTIONS"
            if flow.previous is not None:
                try:
                    for item in history:
                        flow.advance(item)
                except cv2.error:
                    flow = FlowTracker()
                    result_key = None
                    prediction_ns = 0
            if flow.previous is not None and (flow.previous.session_id, flow.previous.frame_id) != key:
                flow = FlowTracker()
                result_key = None
                prediction_ns = 0
            # Replay may take time: never publish expired labels after it finishes.
            if prediction_ns and perf_counter_ns() - prediction_ns > self.max_age_ns:
                flow = FlowTracker()
                result_key = None
                prediction_ns = 0
                accepted = False
                reason = "STALE_RESULT"
            with self._lock:
                if generation == self._generation and not self.stop_event.is_set():
                    self._latest = DisplayFrame(frame, flow.detections(), result_key, prediction_ns)
                else:
                    accepted = False
                    reason = "RESET"
            last_frame_key = key
            if pending:
                self._complete(pending, accepted, reason)

    def stop(self):
        self.stop_event.set()
        self.clear()
