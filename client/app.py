"""Live video independent of Cloud detection, with local optical flow."""
import argparse
import json
import queue
import tkinter as tk
from collections import deque
from pathlib import Path
from time import perf_counter_ns
from tkinter import ttk
from urllib.parse import urlparse

import cv2
from PIL import Image, ImageTk

from client.camera import CameraSource
from client.stream import CsvLog, StreamWorker, duration_ms
from client.tracking import TrackingWorker

LABEL_TEXT = {"with_mask": "Deo dung", "without_mask": "Khong deo", "mask_incorrect": "Deo sai"}
COLORS = {"with_mask": (0, 190, 0), "without_mask": (0, 0, 230), "mask_incorrect": (0, 190, 230)}


class Application:
    def __init__(self, root, config):
        self.root, self.config = root, config
        # Small optical-flow images need a small OpenCV thread pool.
        cv2.setNumThreads(max(1, int(config.get("opencv_threads", 1))))
        self.camera = self.worker = self.tracker = None
        self.events = queue.Queue(maxsize=100)
        self.logger = CsvLog(config.get("csv_path", "artifacts/session.csv"))
        self.running = False
        self.closed = False
        self._tick_after = self._stop_after = None
        self.display_interval_ms = round(1000 / max(1, min(60, config.get("display_fps", 30))))
        self.max_age_ns = max(0.1, float(config.get("tracking_max_age_seconds", 1.0))) * 1e9
        self.preview_key = self.result_key = None
        self.display_times = deque(maxlen=120)
        self.detection_times = deque(maxlen=120)
        self.last_row = None
        self.last_measurement_ns = 0
        self.last_camera_frame = None
        self.connection_status = "Chưa kết nối"
        root.title("Lab3 - Khẩu trang / Cloud")
        root.protocol("WM_DELETE_WINDOW", self.close)
        toolbar = ttk.Frame(root, padding=8)
        toolbar.pack(fill="x")
        self.start_button = ttk.Button(toolbar, text="Bắt đầu", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(toolbar, text="Dừng", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        ttk.Label(toolbar, text=f"Nguồn: {config['camera']['type']} | {config['cloud_url']}").pack(side="left")
        panels = ttk.Frame(root, padding=8)
        panels.pack(fill="both", expand=True)
        preview_panel = ttk.LabelFrame(panels, text="Camera trực tiếp — ảnh đầu vào")
        preview_panel.pack(side="left", padx=4)
        result_panel = ttk.LabelFrame(panels, text="Kết quả server — tracking box trên laptop")
        result_panel.pack(side="left", padx=4)
        self.preview = ttk.Label(preview_panel)
        self.preview.pack()
        self.result_view = ttk.Label(result_panel, text="Bấm Bắt đầu để mở camera", width=65)
        self.result_view.pack()
        self.status = tk.StringVar(value="Video và nhận dạng chạy độc lập; nhãn được cập nhật từ server.")
        self.metrics = tk.StringVar(value=f"CSV: {self.logger.path}")
        ttk.Label(root, textvariable=self.status, padding=8).pack(fill="x")
        timing_panel = ttk.LabelFrame(root, text="3. Thời gian xử lý và độ trễ", padding=8)
        timing_panel.pack(fill="x", padx=12, pady=(0, 8))
        local_server = urlparse(config["cloud_url"]).hostname in {"localhost", "127.0.0.1", "::1"}
        inference_place = "server local" if local_server else "Cloud"
        captions = {
            "inference_ms": f"(a) Thời gian suy luận trên {inference_place}:",
            "frame_turnaround_ms": "(b) Thời gian phản hồi local ↔ Cloud:" if not local_server
                                   else "(b) Thời gian phản hồi client ↔ server local:",
            "gateway_to_gui_ms": "(c) Độ trễ toàn trình*:",
        }
        self.latency_values = {}
        for index, (field, caption) in enumerate(captions.items()):
            ttk.Label(timing_panel, text=caption, font=("Segoe UI", 12)).grid(
                row=index, column=0, sticky="w", pady=2)
            value = tk.StringVar(value="— ms")
            self.latency_values[field] = value
            ttk.Label(timing_panel, textvariable=value, width=14, anchor="w",
                      font=("Consolas", 13, "bold")).grid(
                row=index, column=1, sticky="w", padx=(16, 0), pady=2)
        ttk.Label(timing_panel,
                  text="* Đo từ laptop nhận frame đến cập nhật GUI; chưa gồm đoạn camera → laptop.").grid(
                      row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(timing_panel, textvariable=self.metrics).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(timing_panel, text="Số đo của frame đã hiển thị gần nhất; giữ giữa các lần nhận kết quả.").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self.tick()

    def start(self):
        if self.running or self.camera is not None:
            return
        self.running = True
        self.preview_key = self.result_key = None
        self.display_times.clear()
        self.detection_times.clear()
        self.last_row = None
        self.last_measurement_ns = 0
        self.last_camera_frame = None
        self.set_latency_values(None)
        self.connection_status = "Đang kết nối server"
        self.camera = CameraSource(self.config["camera"])
        self.tracker = TrackingWorker(self.camera, self.config, self.events)
        self.worker = StreamWorker(self.camera, self.config, self.events)
        self.camera.start()
        self.tracker.start()
        self.worker.start()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

    def stop(self):
        self.running = False
        for obj in (self.worker, self.tracker, self.camera):
            if obj:
                obj.stop()
        self.stop_button.configure(state="disabled")
        self.status.set("Đang dừng worker/camera...")
        self.preview.configure(image="", text="Đã dừng camera")
        self.result_view.configure(image="", text="Đã dừng / kết quả cũ đã bỏ")
        self.last_row = None
        self.last_measurement_ns = 0
        self.last_camera_frame = None
        self.set_latency_values(None)
        self.metrics.set(f"CSV: {self.logger.path}")
        self._stop_after = self.root.after(100, self.finish_stop)

    def finish_stop(self):
        if self.closed:
            return
        threads = [obj.thread for obj in (self.camera, self.worker, self.tracker) if obj and obj.thread]
        if any(thread.is_alive() for thread in threads):
            self._stop_after = self.root.after(100, self.finish_stop)
            return
        self.camera = self.worker = self.tracker = None
        while not self.events.empty():
            self.events.get_nowait()
        self.start_button.configure(state="normal")
        self.status.set("Đã dừng. Có thể bắt đầu lại.")

    def show_image(self, widget, image):
        height, width = image.shape[:2]
        scale = min(1.0, 560 / width, 420 / height)
        if scale < 1:
            image = cv2.resize(image, (round(width * scale), round(height * scale)),
                               interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        picture = Image.fromarray(rgb)
        photo = getattr(widget, "photo", None)
        if photo is not None and (photo.width(), photo.height()) == picture.size:
            photo.paste(picture)
        else:
            # Attach the replacement before releasing the displayed image.
            photo = ImageTk.PhotoImage(picture, master=widget)
            widget.configure(image=photo, text="")
            widget.photo = photo
        # Reconfiguring a visible ttk image each frame schedules extra redraws.
        # Reattach only after a placeholder (stop/camera unavailable) hid it.
        attached = tuple(map(str, widget.tk.splitlist(widget.cget("image"))))
        if attached != (str(widget.photo),):
            widget.configure(image=widget.photo, text="")

    @staticmethod
    def measured_fps(times, now):
        recent = [t for t in times if now - t < 2e9]
        if len(recent) < 2:
            return 0.0
        return (len(recent) - 1) * 1e9 / max(1, now - recent[0])

    def set_latency_values(self, row):
        for field, variable in self.latency_values.items():
            value = row.get(field) if row else None
            text = f"{float(value):.0f} ms" if value is not None and value != "" else "— ms"
            if variable.get() != text:
                variable.set(text)

    def tick(self):
        if self.closed:
            return
        tick_start = perf_counter_ns()
        completions = []
        for _ in range(100):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if not self.running:
                if event["type"] == "failure":
                    self.logger.write(event["row"])
                continue
            if event["type"] == "status":
                self.connection_status = event["message"]
                if event["message"].startswith("Cloud unavailable"):
                    self.tracker.clear()
                    self.last_row = None
            elif event["type"] == "failure":
                self.logger.write(event["row"])
                self.tracker.clear()
                self.last_row = None
                self.connection_status = event["row"]["error"]
            elif event["type"] == "result":
                self.detection_times.append(perf_counter_ns())
                self.tracker.submit(event)
            elif event["type"] == "tracking_result":
                completions.append(event)

        displayed_prediction = None
        if self.running and self.camera:
            frame = self.camera.get_latest(max_age_seconds=0.5)
            packet = self.tracker.get_latest()
            now = perf_counter_ns()
            if frame is None:
                previous = self.last_camera_frame
                if previous is not None and now - previous.received_ns <= 2_000_000_000:
                    # A short read gap must not repeatedly collapse both panels.
                    # Show the last raw frame, with no stale prediction boxes.
                    if self.result_key is not None:
                        self.show_image(self.result_view, previous.image)
                        self.result_key = None
                    self.status.set(f"Camera tạm dừng; ảnh cuối đang được giữ | {self.connection_status}")
                else:
                    if self.preview.cget("image"):
                        self.preview.configure(image="", text="Camera chưa có frame mới")
                    if self.result_view.cget("image"):
                        self.result_view.configure(image="", text="Đang chờ camera")
                    self.preview_key = self.result_key = None
                    self.last_row = None
                    self.status.set(f"{self.camera.status} | {self.connection_status}")
            else:
                self.last_camera_frame = frame
                key = (frame.session_id, frame.frame_id)
                if key != self.preview_key:
                    self.show_image(self.preview, frame.image)
                    self.preview_key = key
                # Pixels and boxes belong to the same frame. Fall back to raw live
                # video if tracking is more than 100 ms behind the camera.
                usable = (packet is not None and packet.frame.session_id == frame.session_id and
                          packet.frame.image.shape == frame.image.shape and
                          # The tracker snapshot can be newer than the camera
                          # snapshot read just above. Its pixels still match its boxes.
                          abs(frame.received_ns - packet.frame.received_ns) <= 100_000_000)
                if (usable and self.result_key is not None and
                        self.result_key[0] == frame.session_id and packet.frame.frame_id < self.result_key[1]):
                    usable = False
                if usable:
                    display_frame = packet.frame
                    fresh = packet.prediction_ns and now - packet.prediction_ns <= self.max_age_ns
                    detections = packet.detections if fresh else []
                    displayed_prediction = packet.result_key if fresh else None
                else:
                    display_frame = frame
                    detections = []
                display_key = (display_frame.session_id, display_frame.frame_id, displayed_prediction)
                if display_key != self.result_key:
                    image = display_frame.image.copy()
                    for detection in detections:
                        x1, y1, x2, y2 = map(int, detection["bbox_xyxy"])
                        label = detection["label"]
                        color = COLORS[label]
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                        text = f"{LABEL_TEXT[label]} {detection['confidence']:.0%}"
                        cv2.putText(image, text, (x1, max(18, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    self.show_image(self.result_view, image)
                    if self.result_key is None or display_key[:2] != self.result_key[:2]:
                        self.display_times.append(perf_counter_ns())
                    self.result_key = display_key
                if displayed_prediction is not None:
                    counts = {label: sum(d["label"] == label for d in detections) for label in LABEL_TEXT}
                    age_ms = duration_ms(packet.prediction_ns, now)
                    self.status.set(f"{self.connection_status} | Đeo đúng: {counts['with_mask']} | "
                                    f"Không đeo: {counts['without_mask']} | Đeo sai: {counts['mask_incorrect']} | "
                                    f"Tuổi kết quả: {age_ms:.0f} ms")
                else:
                    self.status.set(f"{self.connection_status} | Video trực tiếp; đang chờ kết quả mới")

        # GUI timing ends only when an aligned live frame is submitted to Tk.
        for completion in completions:
            event = completion["event"]
            row = dict(event["row"])
            response = event["response"]
            key = (response["session_id"], response["frame_id"])
            if completion["accepted"] and displayed_prediction == key:
                row["gateway_to_gui_ms"] = duration_ms(event["received_ns"], perf_counter_ns())
                self.last_row = row
                self.last_measurement_ns = event["received_ns"]
            else:
                row["status"] = "stale"
                row["error"] = completion["reason"] or "NOT_DISPLAYED"
                row["gateway_to_gui_ms"] = ""
            self.logger.write(row)
        if self.running:
            now = perf_counter_ns()
            # A temporary tracking/display miss is not a new latency sample.
            # Clear only when the last measured source frame really expires.
            if self.last_row and now - self.last_measurement_ns > self.max_age_ns:
                self.last_row = None
            fps_text = (f"Hiển thị: {self.measured_fps(self.display_times, now):.1f} FPS | "
                        f"Nhận dạng: {self.measured_fps(self.detection_times, now):.1f} FPS")
            self.set_latency_values(self.last_row)
            if self.last_row:
                fps_text += f" | Frame đo: {self.last_row['frame_id']}"
            self.metrics.set(f"{fps_text} | CSV: {self.logger.path}")
        elapsed = duration_ms(tick_start, perf_counter_ns())
        self._tick_after = self.root.after(max(1, self.display_interval_ms - round(elapsed)), self.tick)

    def close(self):
        self.closed = True
        self.running = False
        for callback in (self._tick_after, self._stop_after):
            if callback:
                self.root.after_cancel(callback)
        for obj in (self.worker, self.tracker, self.camera):
            if obj:
                obj.stop()
        self.logger.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/webcam.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = tk.Tk()
    Application(root, config)
    root.mainloop()


if __name__ == "__main__":
    main()
