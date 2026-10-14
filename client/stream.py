"""Continuous frame transport. One inference request in flight per client."""
import csv
import json
import queue
import threading
from pathlib import Path
from time import perf_counter_ns

import cv2
from websockets.sync.client import connect


CSV_FIELDS = ["session_id", "frame_id", "source", "status", "inference_ms",
              "frame_turnaround_ms", "gateway_to_result_ms", "gateway_to_gui_ms", "error"]


def duration_ms(start_ns, end_ns):
    return (end_ns - start_ns) / 1_000_000


class CsvLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not self.path.exists() or self.path.stat().st_size == 0
        self.file = self.path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDS)
        if needs_header:
            self.writer.writeheader()

    def write(self, row):
        self.writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
        self.file.flush()

    def close(self):
        self.file.close()


class StreamWorker:
    def __init__(self, camera, config, events):
        self.camera = camera
        self.config = config
        self.events = events
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def emit(self, event):
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            self.events.put_nowait(event)

    def _run(self):
        fps = max(0.1, min(float(self.config.get("target_fps", 5)), 30))
        last_key = None
        while not self.stop_event.is_set():
            try:
                with connect(self.config["cloud_url"], open_timeout=5, max_size=2 * 1024 * 1024) as socket:
                    self.emit({"type": "status", "message": "Cloud connected"})
                    while not self.stop_event.is_set():
                        frame = self.camera.get_latest()
                        if frame is None or (frame.session_id, frame.frame_id) == last_key:
                            self.stop_event.wait(0.02)
                            continue
                        last_key = (frame.session_id, frame.frame_id)
                        sent_image = frame.image.copy()
                        ok, jpeg = cv2.imencode(".jpg", sent_image,
                                               [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if not ok:
                            continue
                        header = {"frame_id": frame.frame_id, "session_id": frame.session_id,
                                  "source": self.camera.config["type"]}
                        start_ns = perf_counter_ns()
                        try:
                            socket.send(json.dumps(header))
                            socket.send(jpeg.tobytes())
                            deadline_ns = perf_counter_ns() + 15_000_000_000
                            while not self.stop_event.is_set():
                                try:
                                    message = socket.recv(timeout=0.25)
                                    result = json.loads(message)
                                    break
                                except TimeoutError:
                                    if perf_counter_ns() >= deadline_ns:
                                        raise TimeoutError("Inference response timed out after 15 seconds")
                            if self.stop_event.is_set():
                                return
                            end_ns = perf_counter_ns()
                        except Exception as error:
                            self.emit({"type": "failure", "row": {**header, "status": "error",
                                      "frame_turnaround_ms": duration_ms(start_ns, perf_counter_ns()),
                                      "error": str(error)}})
                            raise
                        if result.get("type") != "result":
                            self.emit({"type": "failure", "row": {**header, "status": "error",
                                      "frame_turnaround_ms": duration_ms(start_ns, end_ns),
                                      "error": result.get("code", "INVALID_RESPONSE")}})
                            raise RuntimeError(result.get("code", "INVALID_RESPONSE"))
                        if any(result.get(key) != value for key, value in header.items()):
                            raise RuntimeError("Response frame ID/source does not match input")
                        if result.get("image") != {"width": sent_image.shape[1], "height": sent_image.shape[0]}:
                            raise RuntimeError("Response image dimensions do not match input")
                        row = {**header, "status": "ok", "error": "",
                               "inference_ms": result["timing_ms"].get("inference", ""),
                               "frame_turnaround_ms": duration_ms(start_ns, end_ns),
                               "gateway_to_result_ms": duration_ms(frame.received_ns, end_ns)}
                        self.emit({"type": "result", "image": sent_image, "response": result,
                                   "received_ns": frame.received_ns, "row": row})
                        remaining = 1 / fps - duration_ms(start_ns, perf_counter_ns()) / 1000
                        self.stop_event.wait(max(0, remaining))
            except Exception as error:
                if not self.stop_event.is_set():
                    self.emit({"type": "status", "message": f"Cloud unavailable: {error}"})
                    self.stop_event.wait(3)

    def stop(self):
        self.stop_event.set()
