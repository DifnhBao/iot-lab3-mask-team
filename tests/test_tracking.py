"""Alignment, expiry and real optical-flow regression tests; no real camera."""
import queue
import time
import unittest
from unittest.mock import patch

import numpy as np

from client.camera import CameraSource, Frame, FrameHistory
from client.tracking import FlowTracker, TrackingWorker


def moving_frame(index, session="test", timestamp=None, blank=False):
    image = np.zeros((160, 240, 3), np.uint8)
    if not blank:
        texture = np.random.default_rng(42).integers(30, 255, (60, 60, 3), np.uint8)
        left = 50 + index * 3
        image[45:105, left:left + 60] = texture
    return Frame(index, session, timestamp or time.perf_counter_ns(), image)


def detection(label="with_mask"):
    return {"label": label, "confidence": 0.92, "bbox_xyxy": [50, 45, 110, 105]}


def result_event(frame, detections=None):
    return {"type": "result", "received_ns": frame.received_ns, "image": frame.image.copy(),
            "response": {"session_id": frame.session_id, "frame_id": frame.frame_id,
                         "detections": [detection()] if detections is None else detections},
            "row": {"status": "ok", "frame_id": frame.frame_id, "session_id": frame.session_id,
                    "inference_ms": 50, "frame_turnaround_ms": 60}}


class TrackingTests(unittest.TestCase):
    def test_history_bounded_by_age_count_and_session(self):
        history = FrameHistory(seconds=0.1, max_frames=3, width=120)
        frames = [moving_frame(i, timestamp=1_000_000_000 + i * 30_000_000) for i in range(6)]
        for frame in frames:
            history.put(frame)
        items = history.through(frames[-1])
        self.assertEqual([item.frame_id for item in items], [3, 4, 5])
        self.assertEqual(items[-1].gray.shape, (80, 120))
        changed = moving_frame(0, session="reconnected", timestamp=1_500_000_000)
        history.put(changed)
        self.assertEqual(history.through(frames[-1]), [])
        self.assertEqual(len(history.through(changed)), 1)

    def test_optical_flow_follows_motion_and_preserves_class(self):
        history = FrameHistory(width=240)
        frames = [moving_frame(i) for i in range(7)]
        for frame in frames:
            history.put(frame)
        items = history.through(frames[-1])
        tracker = FlowTracker()
        tracker.seed(items[0], [detection("mask_incorrect")])
        for item in items[1:]:
            tracker.advance(item)
        boxes = tracker.detections()
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["label"], "mask_incorrect")
        self.assertEqual(boxes[0]["confidence"], 0.92)
        np.testing.assert_allclose(boxes[0]["bbox_xyxy"], [68, 45, 128, 105], atol=2)

    def test_lost_texture_removes_box(self):
        history = FrameHistory(width=240)
        first, blank = moving_frame(0), moving_frame(1, blank=True)
        history.put(first)
        history.put(blank)
        items = history.through(blank)
        tracker = FlowTracker()
        tracker.seed(items[0], [detection()])
        tracker.advance(items[1])
        self.assertEqual(tracker.detections(), [])

    def test_missing_intermediate_frame_does_not_extrapolate(self):
        history = FrameHistory(width=240)
        first, last = moving_frame(0), moving_frame(3)
        history.put(first)
        history.put(last)
        items = history.through(last)
        tracker = FlowTracker()
        tracker.seed(items[0], [detection()])
        tracker.advance(items[1])
        self.assertEqual(tracker.detections(), [])

    @staticmethod
    def wait_until(predicate, timeout=2):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.005)
        raise AssertionError("Tracking worker did not publish in time")

    def make_worker(self, age=5, clock=None):
        camera = CameraSource({"type": "webcam", "tracking_width": 240})
        # Source-age rejection is tested in test_pipeline. These tests isolate
        # tracking expiry from OS scheduling and from a stopped synthetic camera.
        camera.get_latest = lambda max_age_seconds=2: camera.latest.get()
        events = queue.Queue(maxsize=100)
        worker = TrackingWorker(camera, {"tracking_max_age_seconds": age}, events)
        if clock is not None:
            timer = patch("client.tracking.perf_counter_ns", side_effect=lambda: clock[0])
            timer.start()
            self.addCleanup(timer.stop)
        def cleanup():
            worker.stop()
            if worker.thread:
                worker.thread.join(timeout=2)
        self.addCleanup(cleanup)
        return camera, events, worker

    def test_delayed_result_replayed_to_current_frame_then_expires(self):
        clock = [time.perf_counter_ns()]
        camera, events, worker = self.make_worker(age=0.4, clock=clock)
        frames = [moving_frame(i, timestamp=clock[0] + i * 10_000_000) for i in range(6)]
        clock[0] += 60_000_000
        for frame in frames:
            camera.history.put(frame)
            camera.latest.put(frame)
        worker.submit(result_event(frames[0]))
        worker.start()
        completion = events.get(timeout=2)
        self.assertTrue(completion["accepted"])
        packet = worker.get_latest()
        self.assertEqual(packet.frame.frame_id, 5)
        self.assertEqual(packet.result_key, ("test", 0))
        np.testing.assert_allclose(packet.detections[0]["bbox_xyxy"], [65, 45, 125, 105], atol=2)
        clock[0] += 500_000_000
        self.wait_until(lambda: worker.get_latest() and worker.get_latest().result_key is None)
        self.assertEqual(worker.get_latest().detections, [])

    def test_old_session_and_evicted_source_are_rejected(self):
        camera, events, worker = self.make_worker()
        old = moving_frame(0, session="old")
        new = moving_frame(0, session="new")
        camera.history.put(new)
        camera.latest.put(new)
        worker.submit(result_event(old))
        worker.start()
        completion = events.get(timeout=2)
        self.assertFalse(completion["accepted"])
        self.assertEqual(completion["reason"], "STALE_RESULT")
        self.assertEqual(worker.get_latest().detections, [])

    def test_clear_removes_labels_and_new_camera_session_resets(self):
        camera, events, worker = self.make_worker()
        first = moving_frame(0)
        camera.history.put(first)
        camera.latest.put(first)
        worker.submit(result_event(first))
        worker.start()
        self.assertTrue(events.get(timeout=2)["accepted"])
        worker.clear()
        self.wait_until(lambda: worker.get_latest() and worker.get_latest().result_key is None)
        worker.submit(result_event(first))
        self.assertTrue(events.get(timeout=2)["accepted"])
        changed = moving_frame(0, session="new")
        camera.history.put(changed)
        camera.latest.put(changed)
        packet = self.wait_until(lambda: worker.get_latest() if worker.get_latest() and
                                 worker.get_latest().frame.session_id == "new" else None)
        self.assertIsNone(packet.result_key)
        self.assertEqual(packet.detections, [])

    def test_empty_detection_clears_previous_box(self):
        camera, events, worker = self.make_worker()
        first = moving_frame(0)
        camera.history.put(first)
        camera.latest.put(first)
        worker.submit(result_event(first))
        worker.start()
        self.assertTrue(events.get(timeout=2)["accepted"])
        second = moving_frame(1)
        camera.history.put(second)
        camera.latest.put(second)
        worker.submit(result_event(second, []))
        self.assertTrue(events.get(timeout=2)["accepted"])
        self.assertEqual(worker.get_latest().detections, [])


if __name__ == "__main__":
    unittest.main()
