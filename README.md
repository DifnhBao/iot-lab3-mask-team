# Lab3 — Nhận dạng khẩu trang qua Cloud

**Đồng đội chạy thử:** đọc [docs/TEAM_RUN.md](docs/TEAM_RUN.md). Client Render hiện dùng `configs/webcam_render.json`; chạy client không cần cài PyTorch hoặc train lại.

**Kế hoạch chính:** [KE_HOACH_LAB3.md](KE_HOACH_LAB3.md). Ba nhãn: with_mask / without_mask / mask_incorrect. Model: YOLOv8n detection, huấn luyện trên dữ liệu khẩu trang.

```text
Webcam hoặc ESP32-CAM MJPEG
    → laptop gateway lấy frame mới liên tục
    → WebSocket → Cloud YOLOv8n
    → boxes/nhãn/counts/timing → giao diện laptop và CSV
```

## Trạng thái hiện tại

Đã có code nguồn camera, adapter MJPEG, WebSocket server/client, GUI, CSV logging, script chuẩn bị dữ liệu/train và Docker template.

Dataset đã chuẩn bị tại máy phát triển: 848 ảnh không trùng, 4.021 box; 12 box vượt biên 1 pixel đã được cắt về mép ảnh có log. Dataset không nằm trong repo bàn giao. Hiện có `models/best.pt` và `models/best.onnx`; server local mặc định nạp `models/best.pt`, hoặc file chỉ định bởi `MASK_MODEL_PATH`. Bản ONNX trong `deploy_render/` đã được triển khai lên Render, client dùng `configs/webcam_render.json`. Config `webcam.json` dành cho server localhost. Server báo ready=false/HTTP503 khi thiếu model hoặc nạp lỗi. Detector giả chỉ dùng trong tests.

Webcam → Render đã có kết quả nhận dạng ba trạng thái trong các lượt thử của chủ repo. Webcam laptop hiện gặp vấn đề nên bản này được gửi để đồng đội kiểm tra trên máy khác; chưa xác nhận ESP32-CAM thật. GUI tách ba dòng thời gian (a), (b), (c), có chú thích phạm vi đo. Không coi FPS video/tracking là FPS suy luận Cloud.

Đã kiểm tra 44 test, gồm giao thức/dataset, tracking và GUI Tkinter với camera/nhận diện giả lập. Smoke model thật trên ảnh có sẵn đã chạy; chưa kiểm chứng độ mượt bằng webcam hoặc ESP32-CAM thật trong lần cập nhật này. Chi tiết: [kiểm tra video/tracking](docs/SMOOTH_VIDEO_TEST.md).

## 1. Môi trường

Môi trường `.venv` hiện có thư viện client và server, gồm Ultralytics/PyTorch. Nếu dùng môi trường mới, cài dependencies tương ứng.

Nếu tạo môi trường riêng, dùng một bản Python phù hợp rồi cài:

```powershell
Set-Location 'D:\PTIT\N4\HK1\IoT\Lab3'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-client.txt -r requirements-test.txt
```

Các lệnh tiếp theo dùng `python` của môi trường đã có dependencies. Với venv thay `python` bằng `.\.venv\Scripts\python.exe`.

## 2. Chạy khung server và giao diện

Terminal 1, tại thư mục Lab3:

```powershell
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --workers 1 --ws-max-size 6000000
```

Terminal 2:

```powershell
python -m client.app --config configs/webcam.json
```

Bấm **Bắt đầu** để mở webcam và kết nối server. Thiếu model sẽ có thông báo MODEL_FILE_MISSING. Chỉ source camera mở khi người dùng bấm Bắt đầu.

Giao diện có camera gốc và video trực tiếp kèm nhận dạng/tracking. Video hiển thị độc lập với tốc độ Cloud; config đặt `display_fps=30`, `target_fps=5`. Đây là mức mục tiêu, tốc độ thực phụ thuộc camera, máy và mạng. Model, độ phân giải suy luận và chất lượng JPEG vẫn giữ nguyên.

Camera giữ frame màu mới nhất cùng bộ đệm ảnh xám thu nhỏ (mặc định rộng tối đa 320 px, tối đa 1,5 giây/90 frame). Khi Cloud trả kết quả, tracking worker tìm đúng frame nguồn bằng session/frame ID rồi chạy optical flow qua các frame trung gian đến hiện tại. Nhãn/confidence chỉ lấy từ Cloud; tracker cập nhật vị trí khung bao. Các tác vụ này chạy ngoài luồng Tkinter.

Giao diện vẽ bbox lên đúng ảnh mà tracker đã xử lý. Nếu tracking chậm hơn camera trên 100 ms, giao diện chuyển sang video mới không có bbox để tránh chặn video hoặc vẽ sai tọa độ. Khung bao bị bỏ khi mất dấu, thiếu frame trung gian, đổi session/kích thước, mất kết nối hoặc kết quả cũ quá `tracking_max_age_seconds` (mặc định 1 giây tính từ lúc nhận ảnh ở gateway). Cloud nhận diện lại định kỳ để cập nhật nhãn và sửa tracking. Tracking không cải thiện độ chính xác phân loại của model.

Thanh thông tin hiển thị FPS video thực tế, FPS nhận kết quả nhận diện, tuổi kết quả và ba số đo thời gian. Số đếm là số bbox đang hiển thị, có thể giảm nếu tracker mất dấu. Trong CSV, `status=stale` là kết quả nhận được nhưng không hiển thị (quá cũ, bị thay thế hoặc không còn frame nguồn); `gateway_to_gui_ms` để trống trong các trường hợp này. Với kết quả `ok`, mốc GUI là lần cập nhật video đã căn chỉnh dùng kết quả đó, không phải phép đo vật lý màn hình.

Các tùy chọn bổ sung: `opencv_threads` ở gốc config (mặc định 1); `history_seconds`, `history_max_frames`, `tracking_width` trong `camera`. Dừng xóa cả preview, bbox và metrics cũ; bắt đầu lại tạo session mới.

## 3. Dataset và huấn luyện

Nguồn: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

Tải và giải nén để có:

```text
datasets/raw/images/*.png (hoặc jpg)
datasets/raw/annotations/*.xml
```

Script chuẩn bị nhận Pascal VOC XML, chuyển tên mask_weared_incorrect thành class 2, loại exact duplicates rồi chia train/val/test khoảng 70/15/15:

Annotation vượt biên tối đa 1 pixel được cắt về mép ảnh và ghi từng thay đổi vào `summary.json`; XML gốc giữ nguyên. Box sai lớn hơn, đảo cạnh hoặc rỗng vẫn bị chặn và báo tên file/object.

```powershell
python scripts/prepare_dataset.py --raw datasets/raw --output datasets/mask
```

Output: datasets/mask/data.yaml, images/labels theo split và summary.json. Xem counts/ảnh mẫu, xác nhận bboxes phù hợp quy ước tọa độ nguồn và đủ ba lớp mỗi split trước train. Script không thay thế review nhãn hoặc bảo đảm public dataset tách theo danh tính người.

Train chính trên GPU notebook để giảm tải laptop 8 GB. Cài Ultralytics tại môi trường train, upload dataset và sửa path trong data.yaml cho đúng notebook. Lệnh baseline:

```text
python scripts/train.py --data datasets/mask/data.yaml --epochs 50 --batch 8
```

Nếu chỉ kiểm tra train thử trên CPU:

```text
python scripts/train.py --data datasets/mask/data.yaml --epochs 3 --batch 2 --device cpu
```

Copy best.pt thực tế của lượt train vào models/best.pt. Model phải có names đúng ID 0/1/2. Không dùng yolov8n.pt COCO thay cho weights khẩu trang.

Đánh giá cơ bản bằng thư viện:

```text
yolo detect val model=models/best.pt data=datasets/mask/data.yaml split=test imgsz=640
```

Lưu kết quả từng lớp và vài ảnh dự đoán đúng/sai cho báo cáo. Không cần làm nghiên cứu so sánh model.

## 4. Chạy model thật và Cloud

Server có weights rồi mới cài dependencies model tại môi trường server:

```powershell
python -m pip install -r requirements-server.txt
```

Restart server; http://127.0.0.1:8000/health phải ready=true. Nếu load lỗi, xem console server; nếu names sai, sửa dataset/train chứ không đổi chữ hiển thị để che sai mapping.

Sau local pass, triển khai Docker theo deploy/README.md. Docker template dùng CPU PyTorch, port 7860. Chưa build/deploy template này ở bước hiện tại.

Đổi cloud_url trong config thành `wss://<hostname-app-thuc-te>/ws/predict`. Kiểm tra health/version/hash và chạy webcam lại. Laptop không cần Ultralytics/PyTorch khi chỉ làm client.

## 5. Thử ESP32 source khi chưa có mạch

Terminal nguồn thử:

```powershell
python scripts/mock_mjpeg.py
```

Terminal giao diện:

```powershell
python -m client.app --config configs/mock_esp32.json
```

Stream này hiển thị chữ di chuyển để thử nguồn camera và reconnect; không phải ảnh người để đánh giá model. Chưa có model thì chỉ preview và báo thiếu weights.

Khi có ESP32-CAM: nạp CameraWebServer phù hợp đúng board/pins, cấu hình Wi-Fi, kiểm tra stream trong LAN; thay stream_url trong configs/esp32.json bằng URL thật rồi chạy:

```powershell
python -m client.app --config configs/esp32.json
```

Laptop là gateway; ESP không trực tiếp chạy model hoặc kết nối Cloud trong bản lab này. CameraWebServer tham khảo chính thức: https://github.com/espressif/arduino-esp32/tree/master/libraries/ESP32/examples/Camera/CameraWebServer

## 6. Test và số đo

```powershell
python -m unittest discover -s tests -v
```

Tests dùng localhost và không mở webcam thật. Nếu môi trường sandbox chặn loopback, chạy trong terminal local thông thường.

CSV lưu tại artifacts/ theo config:

- inference_ms: tốc độ inference do model trả.
- frame_turnaround_ms: gateway bắt đầu gửi frame tới nhận đủ response.
- gateway_to_result_ms: nhận frame tại gateway tới nhận result.
- gateway_to_gui_ms: nhận frame tại gateway tới cập nhật ảnh kết quả trong ứng dụng.
- status/error: giữ lỗi thay vì coi là dự đoán thành công.

Gateway timings dùng cùng đồng hồ monotonic. Với ESP, chúng không bao gồm đầy đủ sensor tới gateway; không gọi là sensor-to-display chính xác. GUI timing là mốc cập nhật trong ứng dụng, không phải phép đo vật lý màn hình.

## Checklist demo đủ cho bài lab

- [ ] Model train đủ ba lớp, có kết quả test.
- [ ] Cloud health ready=true, đúng model.
- [ ] Camera nhận dạng liên tục, nhiều mặt, cảnh đủ ba trạng thái.
- [ ] ESP32-CAM là nguồn thật trong demo cuối khi lấy được mạch.
- [ ] Dừng/mở lại, mất mạng và thiếu model có trạng thái rõ.
- [ ] CSV khoảng 100 kết quả và số lỗi, thống kê mean/p95.
- [ ] Báo cáo ngắn, video và hướng dẫn chạy.

Phân tích cũ ở docs/archive chỉ để tham khảo. Phạm vi thực thi là kế hoạch gọn và checklist trên.
