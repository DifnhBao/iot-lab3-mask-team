# Lab3 — Quyết định và kế hoạch cuối

Đây là bài lab môn học. Mục tiêu: một hệ thống chạy được, đúng yêu cầu, có demo và số đo cơ bản. Tài liệu phân tích dài giữ trong docs/archive, không phải danh sách bắt buộc phải làm.

## Quyết định đã chốt

- Nhận dạng khẩu trang theo ba nhãn: 0 with_mask, 1 without_mask, 2 mask_incorrect.
- YOLOv8n detection huấn luyện cho ba nhãn; file dùng thực tế là models/best.pt.
- Nhận dạng nhiều khuôn mặt trong frame; ba là số lớp, không giới hạn ba người.
- Camera hoạt động liên tục; laptop lấy frame mới và gửi liên tục qua WebSocket tới Cloud. Không cần bấm chụp từng ảnh.
- Webcam dùng phát triển. ESP32-CAM phát MJPEG trong LAN; laptop làm gateway. Chuyển nguồn bằng configs/esp32.json.
- Train chính bằng GPU notebook; suy luận trên Cloud; laptop xử lý camera/giao diện. Không phụ thuộc GPU Radeon của laptop.
- FastAPI server, giao diện Python/OpenCV/Tkinter. Bắt đầu target_fps = 5 và CPU Cloud; đo tốc độ thực, không cam kết 30 FPS.
- Dataset Kaggle trong slide làm nguồn chính; kiểm tra và chuyển annotation sang YOLO. Bổ sung ảnh nếu lớp đeo sai thiếu hoặc demo yếu.
- Không bắt buộc so sánh nhiều model, tracking, nhận diện danh tính, database, Telegram, benchmark nhiều nền tảng hoặc nghiên cứu thuật toán.

## Sáu bước thực hiện

| Bước | Output cần có | Kiểm tra đủ cho lab | Trạng thái |
|---|---|---|---|
| 1. Khung phần mềm | Camera adapters, WebSocket server/client, GUI, cấu hình | Giao thức/ID, stream giả lập, lỗi thiếu model | Đã dựng; 11 test tự động pass, GUI smoke pass; chưa test webcam thật |
| 2. Dataset | images/labels train-val-test, data.yaml | Xem bbox/nhãn mẫu; đếm đủ ba lớp; không exact duplicate xuyên split | Đã chuyển: 594 train / 127 val / 127 test; audit 4.021 box pass; cần xem nhãn mẫu |
| 3. Train model | best.pt, kết quả test theo lớp | Nhận đúng ba trạng thái trên ảnh test; ghi precision/recall/mAP thực tế | Có script; chưa train |
| 4. Nhận dạng local rồi Cloud | Giao diện có boxes/counts/timing, URL Cloud | Webcam liên tục, nhiều mặt, lỗi mạng và đúng model | Chưa có weights/Cloud |
| 5. ESP32-CAM | Firmware camera, stream URL, config nguồn | Stream thật, thay config, chạy ổn định vài phút | Chưa có mạch |
| 6. Bài nộp | README, báo cáo ngắn, video, CSV | Đối chiếu slide; số đo thật; hướng dẫn chạy | README có; báo cáo/video/số liệu chưa có |

## Demo và số đo tối thiểu

1. Camera nhận dạng liên tục ít nhất 5 phút.
2. Thử từng trạng thái và cảnh nhiều người; có cảnh xuất hiện đủ ba loại nhãn.
3. Kết quả gồm khung bao, nhãn, confidence và số mặt mỗi lớp; không gọi confidence là accuracy.
4. Ghi khoảng 100 lượt thành công cùng số lỗi: inference trên Cloud, khứ hồi frame, gateway tới cập nhật GUI. Báo mean và p95.
5. Với ESP, số đo bắt đầu ở gateway không được gọi là đầy đủ sensor tới màn hình. Giải thích giới hạn trong báo cáo; không giả lập upload một chiều bằng khứ hồi/2.
6. Không áp các ngưỡng mAP/FPS hoặc 43 test từ bản kế hoạch dài như thang điểm bắt buộc. Báo kết quả thật và cải thiện lỗi ảnh hưởng demo.

## Hướng dẫn sử dụng

Đọc README.md để chạy code. Tài liệu trong docs/archive giữ để tham khảo nếu gặp vấn đề, không mở rộng phạm vi triển khai chỉ vì có nội dung ở đó.
