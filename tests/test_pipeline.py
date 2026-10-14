"""Protocol/source tests; synthetic detector exists only inside this test file."""
import io
import json
import queue
import socket
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import uvicorn
from fastapi.testclient import TestClient
from PIL import Image

from client.camera import CameraSource, Frame, LatestFrame
from client.stream import StreamWorker, duration_ms
from server.app import create_app, decode_frame
from scripts.mock_mjpeg import StreamHandler, ThreadingHTTPServer
from scripts.prepare_dataset import convert_box


def jpeg(width=80, height=60):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (240, 10, 20)).save(buffer, "JPEG")
    return buffer.getvalue()


class TestDetector:
    def predict(self, image):
        return {"detections": [{"class_id": 2, "label": "mask_incorrect",
                                "confidence": 0.9, "bbox_xyxy": [2, 3, 20, 25]}],
                "counts": {"with_mask": 0, "without_mask": 0, "mask_incorrect": 1},
                "timing_ms": {"inference": 1.0}}


class PipelineTests(unittest.TestCase):
    def test_rgb_image_is_decoded_to_bgr_with_original_size(self):
        image = decode_frame(jpeg())
        self.assertEqual(image.shape, (60, 80, 3))
        self.assertGreater(int(image[0, 0, 2]), int(image[0, 0, 0]))

    def test_bad_and_oversize_image_rejected(self):
        for payload in (b"", b"not an image", b"x" * (5 * 1024 * 1024 + 1)):
            with self.assertRaises((ValueError, OSError)):
                decode_frame(payload)

    def test_missing_model_is_not_reported_ready(self):
        with patch("server.app.Path.is_file", return_value=False), TestClient(create_app()) as client:
            self.assertEqual(client.get("/health").status_code, 503)
            with client.websocket_connect("/ws/predict") as stream:
                result = stream.receive_json()
                self.assertEqual(result["type"], "error")
                self.assertNotIn("detections", result)

    def test_result_keeps_input_id_dimensions_and_class(self):
        header = {"frame_id": 19, "session_id": "test-session", "source": "esp32_mjpeg"}
        with TestClient(create_app(TestDetector())) as client:
            with client.websocket_connect("/ws/predict") as stream:
                stream.send_json(header)
                stream.send_bytes(jpeg())
                result = stream.receive_json()
                self.assertEqual(result["frame_id"], 19)
                self.assertEqual(result["session_id"], "test-session")
                self.assertEqual(result["image"], {"width": 80, "height": 60})
                self.assertEqual(result["counts"]["mask_incorrect"], 1)
                self.assertGreaterEqual(result["timing_ms"]["server_processing"], 0)

    def test_bad_image_does_not_break_following_frame(self):
        with TestClient(create_app(TestDetector())) as client:
            with client.websocket_connect("/ws/predict") as stream:
                for index, payload in enumerate((b"broken", jpeg())):
                    stream.send_json({"frame_id": index, "session_id": "x", "source": "webcam"})
                    stream.send_bytes(payload)
                    result = stream.receive_json()
                    self.assertEqual(result["type"], "error" if index == 0 else "result")

    def test_invalid_header_rejected(self):
        with TestClient(create_app(TestDetector())) as client:
            with client.websocket_connect("/ws/predict") as stream:
                stream.send_text('{"frame_id": -1}')
                self.assertEqual(stream.receive_json()["code"], "INVALID_HEADER")

    def test_latest_frame_overwrites_and_stale_frame_is_not_sent(self):
        store = LatestFrame()
        for index in range(100):
            store.put(Frame(index, "a", time.perf_counter_ns(), np.zeros((2, 2, 3), np.uint8)))
        self.assertEqual(store.get().frame_id, 99)
        camera = CameraSource({"type": "webcam", "index": 0})
        camera.latest.put(Frame(1, "old", time.perf_counter_ns() - 3_000_000_000,
                                np.zeros((2, 2, 3), np.uint8)))
        self.assertIsNone(camera.get_latest())

    def test_duration_uses_nanoseconds(self):
        self.assertEqual(duration_ms(20_000_000, 120_000_000), 100)

    def test_dataset_class_mapping_and_box_conversion(self):
        values = convert_box("mask_weared_incorrect", 100, 50, 300, 250, 640, 480).split()
        self.assertEqual(values[0], "2")
        self.assertAlmostEqual(float(values[1]), .3125)
        self.assertAlmostEqual(float(values[4]), 200 / 480, places=6)
        with self.assertRaises(ValueError):
            convert_box("unknown", 1, 1, 5, 5, 10, 10)
        with self.assertRaises(ValueError):
            convert_box("with_mask", 5, 1, 2, 5, 10, 10)

    def test_esp_adapter_reads_continuous_mock_mjpeg(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), StreamHandler)
        server.daemon_threads = True
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        camera = CameraSource({"type": "esp32_mjpeg",
                               "stream_url": f"http://127.0.0.1:{server.server_port}/stream"})
        camera.start()
        try:
            deadline = time.monotonic() + 8
            frame = None
            while time.monotonic() < deadline:
                frame = camera.get_latest()
                if frame and frame.frame_id >= 3:
                    break
                time.sleep(0.05)
            self.assertIsNotNone(frame, camera.status)
            self.assertGreaterEqual(frame.frame_id, 3)
            self.assertEqual(frame.image.shape, (480, 640, 3))
        finally:
            camera.stop()
            camera.thread.join(timeout=6)
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)
        self.assertFalse(camera.thread.is_alive())

    def test_actual_websocket_client_and_server_roundtrip(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(create_app(TestDetector()), log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        worker = None
        try:
            deadline = time.monotonic() + 5
            while not server.started and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(server.started)
            camera = CameraSource({"type": "webcam", "index": 0})
            camera.latest.put(Frame(9, "loopback", time.perf_counter_ns(),
                                    np.zeros((60, 80, 3), np.uint8)))
            events = queue.Queue(maxsize=100)
            worker = StreamWorker(camera, {"cloud_url": f"ws://127.0.0.1:{port}/ws/predict"}, events)
            worker.start()
            while True:
                event = events.get(timeout=5)
                if event["type"] == "result":
                    break
                if event["type"] == "failure":
                    self.fail(str(event))
            self.assertEqual(event["response"]["frame_id"], 9)
            self.assertEqual(event["row"]["status"], "ok")
            self.assertGreaterEqual(event["row"]["gateway_to_result_ms"], event["row"]["frame_turnaround_ms"])
        finally:
            if worker:
                worker.stop()
                worker.thread.join(timeout=5)
            server.should_exit = True
            thread.join(timeout=5)
            listener.close()


if __name__ == "__main__":
    unittest.main()
