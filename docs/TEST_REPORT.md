# Kiểm tra khung phần mềm

## Bổ sung kiểm tra chuyển dataset

- 4 test mới trong `tests/test_prepare_dataset.py`: pass trong venv của dự án.
- Xác nhận dữ liệu raw có 853 annotation, 12 box xmax=401 vượt ảnh width=400, không có mismatch kích thước.
- Chuyển dataset thực thành công: 848 ảnh không trùng, 4.021 box; 594 train, 127 val, 127 test.
- Audit toàn bộ output: class ID hợp lệ, bbox có diện tích dương và nằm trong ảnh (tolerance làm tròn 1e-6); hash ảnh không trùng xuyên split.
- 12 chỉnh sửa chỉ diễn ra lúc chuyển đổi; raw XML vẫn xmax=401. Chi tiết ở datasets/mask/summary.json.
- Chưa huấn luyện model và chưa đánh giá chất lượng nhận dạng.

Ngày: 10/10/2026. Phạm vi: khung nguồn camera và WebSocket; chưa đánh giá model thật.

## Kết quả thực tế

- `python -m unittest discover -s tests -v`: **11/11 pass**.
- Python syntax và JSON config: pass.
- Khởi tạo GUI ở chế độ ẩn, không mở webcam: pass.
- Localhost networking được kiểm tra ngoài sandbox do sandbox làm socket loopback bị chặn. Không gửi dữ liệu tới dịch vụ ngoài, không mở camera thật.

## Các trường hợp đã kiểm tra

1. RGB input được decode BGR, kích thước giữ đúng.
2. Ảnh hỏng/rỗng/vượt giới hạn bị từ chối.
3. Thiếu model: health503, không trả detection giả.
4. Frame ID, session, source và dimensions giữ đúng qua protocol.
5. Ảnh lỗi không phá request hợp lệ tiếp theo.
6. Header sai bị từ chối.
7. Slot frame giữ frame mới nhất, frame cũ không gửi tiếp.
8. Timing đổi nanosecond sang millisecond đúng.
9. Client WebSocket thật nối server localhost và nhận response.
10. Adapter ESP32 MJPEG đọc được stream HTTP giả lập liên tục và dừng được.
11. Mapping lớp đeo sai và chuyển bbox sang YOLO đúng ví dụ.

Detector giả chỉ nằm trong tests/test_pipeline.py để test protocol. Stream giả chỉ dùng để kiểm tra đọc nguồn, không dùng đánh giá nhận dạng người.

## Môi trường đã dùng

Python3.13.9; FastAPI0.135.1; Uvicorn0.41.0; websockets16.0; OpenCV5.0.0.93; Pillow12.3.0; NumPy2.4.2; httpx0.28.1. Không cài PyTorch/Ultralytics trong bước này.

## Chưa kiểm tra

- Dataset Kaggle thực và chất lượng nhãn.
- Train, best.pt và chất lượng dự đoán ba trạng thái.
- Webcam thật và thao tác GUI nhận dạng.
- ESP32-CAM thật.
- Docker build, Cloud deploy và tốc độ trên mạng thật.

Kết quả này chứng minh khung kết nối đã hoạt động ở môi trường thử, không chứng minh hệ thống nhận dạng khẩu trang đã hoàn thành.
