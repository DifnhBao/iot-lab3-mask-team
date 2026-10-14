# Chạy thử Lab 3 trên máy đồng đội

## 1. Clone repository private

Chủ repo cần mời tài khoản GitHub của đồng đội vào Settings → Collaborators. Sau khi chấp nhận lời mời, chạy:

```powershell
git clone https://github.com/DifnhBao/iot-lab3-mask-team.git
cd iot-lab3-mask-team
```

Hoặc tải ZIP từ GitHub sau khi đăng nhập và được cấp quyền.

## 2. Chạy webcam với Cloud (khuyến nghị thử trước)

Dùng Python 3.11–3.13. Trong thư mục gốc repository:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-client.txt
.\.venv\Scripts\python.exe -m client.app --config configs/webcam_render.json
```

Không bắt buộc kích hoạt venv hoặc cài PyTorch để chạy client Cloud. Đóng các ứng dụng đang dùng webcam và chỉ mở một client.

Mở `https://iot-lab3-mask-cloud.onrender.com/health`, đợi `ready=true` trước khi bấm Bắt đầu. Dịch vụ miễn phí có thể cần khởi động lại sau thời gian không dùng. Không cần chạy Uvicorn local.

Config hiện dùng webcam index 0; nếu máy có nhiều camera, đổi index trong `configs/webcam_render.json` thành 1 và thử lại. CSV sẽ tự được tạo tại `artifacts/webcam_render.csv`.

## 3. Kiểm thử cần báo lại

- Camera có hình và hai khung không bị đen/nhấp nháy.
- Box theo mặt; thử không đeo, đeo đúng, để lộ mũi, dưới cằm và nhiều người.
- (a) suy luận Cloud, (b) phản hồi local ↔ Cloud, (c) laptop nhận frame → GUI hiển thị.
- Video hiển thị FPS và nhận dạng FPS là hai chỉ số khác nhau; tracking box chạy trên laptop, nhãn từ server.
- Nếu lỗi, gửi nội dung terminal, ảnh GUI và CSV cho nhóm; không công khai ảnh người dùng.

## 4. Server PyTorch local (tùy chọn)

Repository có `models/best.pt`; không phải train lại.

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-server.txt "ultralytics==8.4.175"
.\.venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --workers 1 --ws-max-size 6000000
```

Terminal thứ hai, tại thư mục gốc:

```powershell
.\.venv\Scripts\python.exe -m client.app --config configs/webcam.json
```

## 5. Cloud ONNX và ESP32-CAM

`deploy_render/` chứa server ONNX, requirements và `models/best.onnx`; trên Render đặt Root Directory là `deploy_render`.

ESP32-CAM phải có firmware phát MJPEG và cùng mạng với laptop. Điền URL stream thật và URL Cloud vào `configs/esp32.json` trước khi chạy `python -m client.app --config configs/esp32.json`. Phần cứng thật chưa được xác nhận hoạt động trong lần bàn giao này.

## 6. Phạm vi bản bàn giao

Gồm mã nguồn hiện tại, cấu hình, notebook, tài liệu và weights chính. Không gồm `.venv`, dataset, ảnh thu thập, CSV phiên chạy hoặc khóa truy cập. Dataset có thể tải lại theo tài liệu nếu cần train; không cần dataset để chạy thử nhận dạng.

Các tài liệu kế hoạch cũ có thể ghi trạng thái trước đây; hướng dẫn này dành cho chạy thử bản hiện tại. Mô hình Render phải được test nhiều người và số đo trên chính máy đồng đội.
