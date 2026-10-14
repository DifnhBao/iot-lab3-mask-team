"""Hidden Tk integration: live camera continues through slow/missing inference."""
import threading
import time
import tkinter as tk
import unittest
import uuid
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from client.app import Application
from client.camera import CameraSource, Frame
from client.stream import duration_ms


class SyntheticCamera(CameraSource):
    def start(self):
        self.thread = threading.Thread(target=self._feed, daemon=True)
        self.thread.start()

    def _feed(self):
        session = uuid.uuid4().hex
        texture = np.random.default_rng(42).integers(30, 255, (80, 80, 3), np.uint8)
        index = 0
        next_frame = time.monotonic()
        while not self.stop_event.is_set():
            image = np.zeros((480, 640, 3), np.uint8)
            shift = index % 60
            for x in (70, 250, 430):
                image[160:240, x + shift:x + shift + 80] = texture
            frame = Frame(index, session, time.perf_counter_ns(), image)
            self.history.put(frame)
            self.latest.put(frame)
            self.status = "Camera connected"
            index += 1
            next_frame += 1 / 30
            self.stop_event.wait(max(0, next_frame - time.monotonic()))


class SlowDetectorWorker:
    def __init__(self, camera, config, events):
        self.camera, self.events = camera, events
        self.stop_event = threading.Event()
        self.pause = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        self.events.put({"type": "status", "message": "Cloud connected"})
        while not self.stop_event.is_set():
            frame = self.camera.get_latest()
            if frame is None or self.pause.is_set():
                self.stop_event.wait(0.01)
                continue
            start = time.perf_counter_ns()
            if self.stop_event.wait(0.12):
                break
            detections = []
            for x, label in zip((70, 250, 430), ("with_mask", "without_mask", "mask_incorrect")):
                left = x + frame.frame_id % 60
                detections.append({"label": label, "confidence": 0.9,
                                   "bbox_xyxy": [left, 160, left + 80, 240]})
            self.events.put({"type": "result", "image": frame.image.copy(),
                             "received_ns": frame.received_ns,
                             "response": {"session_id": frame.session_id, "frame_id": frame.frame_id,
                                          "detections": detections},
                             "row": {"session_id": frame.session_id, "frame_id": frame.frame_id,
                                     "status": "ok", "inference_ms": 90,
                                     "frame_turnaround_ms": duration_ms(start, time.perf_counter_ns())}})
            self.stop_event.wait(0.08)

    def stop(self):
        self.stop_event.set()


class UnavailableWorker(SlowDetectorWorker):
    def _run(self):
        self.events.put({"type": "status", "message": "Cloud unavailable: test offline"})
        self.stop_event.wait()


class LiveUITests(unittest.TestCase):
    def make_app(self, worker=SlowDetectorWorker):
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        root.withdraw()
        rows = []
        logger = SimpleNamespace(path="memory", write=lambda row: rows.append(dict(row)), close=lambda: None)
        with patch("client.app.CsvLog", return_value=logger):
            app = Application(root, {"camera": {"type": "webcam", "tracking_width": 160}, "cloud_url": "unused",
                                     "display_fps": 30, "tracking_max_age_seconds": 0.6})
        def cleanup():
            objects = [app.camera, app.worker, app.tracker]
            app.close()
            for obj in objects:
                if obj and obj.thread:
                    obj.thread.join(timeout=2)
        self.addCleanup(cleanup)
        with patch("client.app.CameraSource", SyntheticCamera), patch("client.app.StreamWorker", worker):
            app.start()
        return root, app, rows

    @staticmethod
    def pump(root, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            root.update()
            time.sleep(0.003)

    def test_live_video_three_labels_expiry_and_stop_restart(self):
        root, app, rows = self.make_app()
        self.pump(root, 2.0)
        self.assertTrue(app.result_view.cget("image"))
        self.assertGreater(len(app.display_times), 2 * len(app.detection_times))
        self.assertTrue(any(row["status"] == "ok" and row["gateway_to_gui_ms"] > 120 for row in rows))
        self.assertIn("Đeo đúng: 1", app.status.get())
        self.assertIn("Không đeo: 1", app.status.get())
        self.assertIn("Đeo sai: 1", app.status.get())
        self.assertIsNotNone(app.last_row)
        for field, variable in app.latency_values.items():
            self.assertEqual(variable.get(), f"{float(app.last_row[field]):.0f} ms")
        app.worker.pause.set()
        self.pump(root, 0.9)
        self.assertTrue(app.result_view.cget("image"))
        self.assertIn("đang chờ kết quả mới", app.status.get())
        self.assertIsNone(app.last_row)
        self.assertTrue(all(value.get() == "— ms" for value in app.latency_values.values()))
        objects = [app.camera, app.worker, app.tracker]
        app.stop()
        self.pump(root, 0.4)
        self.assertFalse(app.preview.cget("image"))
        self.assertFalse(app.result_view.cget("image"))
        self.assertTrue(all(not obj.thread.is_alive() for obj in objects))
        self.assertEqual(str(app.start_button.cget("state")), "normal")
        with patch("client.app.CameraSource", SyntheticCamera), patch("client.app.StreamWorker", SlowDetectorWorker):
            app.start()
        self.pump(root, 0.4)
        self.assertTrue(app.result_view.cget("image"))

    def test_offline_server_does_not_stop_live_video(self):
        root, app, rows = self.make_app(UnavailableWorker)
        self.pump(root, 0.5)
        self.assertTrue(app.result_view.cget("image"))
        self.assertGreater(len(app.display_times), 5)
        self.assertIn("Cloud unavailable", app.status.get())
        self.assertEqual(rows, [])

    def test_image_pixels_update_without_rebinding_visible_image(self):
        root, app, _ = self.make_app(UnavailableWorker)
        image = np.full((80, 80, 3), (0, 0, 255), np.uint8)
        app.show_image(app.preview, image)
        photo = app.preview.photo
        self.assertEqual(root.tk.call(str(photo), "get", 10, 10), (255, 0, 0))
        with patch.object(app.preview, "configure", wraps=app.preview.configure) as configure:
            app.show_image(app.preview, np.full_like(image, (0, 255, 0)))
            self.assertIs(app.preview.photo, photo)
            configure.assert_not_called()
        self.assertEqual(root.tk.call(str(photo), "get", 10, 10), (0, 255, 0))
        app.preview.configure(image="")
        app.show_image(app.preview, image)
        attached = tuple(map(str, root.tk.splitlist(app.preview.cget("image"))))
        self.assertEqual(attached, (str(photo),))

    def test_short_camera_gap_keeps_pixels_then_clears_on_long_gap(self):
        root, app, _ = self.make_app(UnavailableWorker)
        self.pump(root, 0.3)
        self.assertIsNotNone(app.last_camera_frame)
        with patch.object(app.camera, "get_latest", return_value=None):
            self.pump(root, 0.1)
            self.assertTrue(app.preview.cget("image"))
            self.assertTrue(app.result_view.cget("image"))
            self.assertIn("Camera tạm dừng", app.status.get())
            self.assertIsNone(app.last_row)
            app.last_camera_frame = replace(
                app.last_camera_frame,
                received_ns=time.perf_counter_ns() - 3_000_000_000,
            )
            self.pump(root, 0.1)
            self.assertFalse(app.preview.cget("image"))
            self.assertFalse(app.result_view.cget("image"))
        self.pump(root, 0.2)
        self.assertTrue(app.preview.cget("image"))
        self.assertTrue(app.result_view.cget("image"))


if __name__ == "__main__":
    unittest.main()
