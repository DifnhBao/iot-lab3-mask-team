"""One continuously drained source, shared by webcam and ESP32 MJPEG."""
import threading
import sys
import uuid
from collections import deque
from dataclasses import dataclass
from time import perf_counter_ns

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    frame_id: int
    session_id: str
    received_ns: int
    image: np.ndarray


@dataclass(frozen=True)
class GrayFrame:
    frame_id: int
    session_id: str
    received_ns: int
    image_shape: tuple
    gray: np.ndarray


class FrameHistory:
    """Bounded, reduced grayscale history for aligning delayed detections."""
    def __init__(self, seconds=1.5, max_frames=90, width=320):
        self.seconds = max(0.1, float(seconds))
        self.width = max(80, int(width))
        self._frames = deque(maxlen=max(2, int(max_frames)))
        self._lock = threading.Lock()

    def put(self, frame):
        height, width = frame.image.shape[:2]
        scale = min(1.0, self.width / width)
        image = cv2.resize(frame.image, (round(width * scale), max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA) if scale < 1 else frame.image
        item = GrayFrame(frame.frame_id, frame.session_id, frame.received_ns,
                         frame.image.shape[:2], cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        with self._lock:
            if self._frames and (self._frames[-1].session_id != frame.session_id or
                                 self._frames[-1].image_shape != item.image_shape):
                self._frames.clear()
            self._frames.append(item)
            cutoff = frame.received_ns - self.seconds * 1e9
            while self._frames and self._frames[0].received_ns < cutoff:
                self._frames.popleft()

    def through(self, frame):
        with self._lock:
            return [item for item in self._frames
                    if item.session_id == frame.session_id and item.frame_id <= frame.frame_id]


class LatestFrame:
    """One slot: old frames are overwritten rather than queued."""
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def put(self, frame: Frame):
        with self._lock:
            self._frame = frame

    def get(self):
        with self._lock:
            return self._frame


class CameraSource:
    def __init__(self, config):
        if config["type"] not in {"webcam", "esp32_mjpeg"}:
            raise ValueError("camera.type must be webcam or esp32_mjpeg")
        if config["type"] == "esp32_mjpeg" and not config.get("stream_url"):
            raise ValueError("ESP32 source requires stream_url")
        self.config = config
        self.latest = LatestFrame()
        self.history = FrameHistory(config.get("history_seconds", 1.5),
                                    config.get("history_max_frames", 90),
                                    config.get("tracking_width", 320))
        self.status = "Camera stopped"
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        # Windows' default MSMF backend can open a device but fail to read it.
        # Try DirectShow first, then switch backend after an open/read failure.
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if sys.platform == "win32" else [cv2.CAP_ANY]
        backend_index = 0
        while not self.stop_event.is_set():
            session_id = uuid.uuid4().hex
            if self.config["type"] == "webcam":
                backend = backends[backend_index % len(backends)]
                capture = cv2.VideoCapture(int(self.config.get("index", 0)), backend)
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            else:
                capture = cv2.VideoCapture(
                    self.config["stream_url"], cv2.CAP_FFMPEG,
                    [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                     cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
            try:
                if not capture.isOpened():
                    self.status = "Cannot connect to camera; retrying"
                else:
                    frame_id = 0
                    while not self.stop_event.is_set():
                        success, image = capture.read()
                        if not success:
                            self.status = "Camera disconnected; retrying"
                            break
                        frame = Frame(frame_id, session_id, perf_counter_ns(), image)
                        self.history.put(frame)
                        self.latest.put(frame)
                        self.status = "Camera connected"
                        frame_id += 1
            except Exception as error:
                self.status = f"Camera error: {error}"
            finally:
                capture.release()
            if not self.stop_event.is_set():
                if self.config["type"] == "webcam":
                    backend_index += 1
                self.stop_event.wait(2)
        self.status = "Camera stopped"

    def get_latest(self, max_age_seconds=2):
        frame = self.latest.get()
        if frame is None or perf_counter_ns() - frame.received_ns > max_age_seconds * 1e9:
            return None
        return frame

    def stop(self):
        self.stop_event.set()

