# Video trực tiếp và tracking — kiểm tra ngày 13/10/2026

## Thay đổi

- Camera đọc liên tục; chỉ giữ một frame màu mới nhất và lịch sử ảnh xám thu nhỏ có giới hạn.
- Nhận diện WebSocket giữ một request đang chờ, mặc định tối đa 5 FPS. Model, JPEG quality 80 và kích thước suy luận giữ nguyên.
- Tracking Lucas–Kanade chạy trong thread riêng. Kết quả server được gắn vào frame nguồn đúng session/ID, rồi theo dõi qua lịch sử tới frame hiện tại.
- Video đặt mục tiêu 30 FPS, không đợi server. Ảnh và bbox thuộc cùng một frame; nếu tracker chậm trên 100 ms, hiển thị camera mới không có bbox.
- Nhãn/confidence chỉ đến từ model. Bbox mất dấu hoặc kết quả quá 1 giây tính từ frame nguồn bị bỏ. Dừng/reconnect xóa trạng thái cũ.
- Thanh thông tin tách FPS hiển thị, FPS nhận kết quả và tuổi kết quả. CSV giữ schema cũ; kết quả không được hiển thị có status=stale và không ghi gateway_to_gui_ms.
- Render dùng resize OpenCV và tái sử dụng ảnh Tk khi kích thước giữ nguyên; OpenCV mặc định một thread cho tác vụ client.

## Xác minh

42 test của dataset/protocol/tracking đã pass trong một lượt discover đầy đủ trước khi thêm 2 test GUI. Sau tối ưu render, chạy lại cả 8 test tracking và 2 test GUI: 10/10 pass. Tổng cộng 44 test riêng biệt đã được kiểm tra.

Tracking: dịch chuyển ảnh có texture, giữ đúng nhãn/confidence, căn chỉnh phản hồi trễ về frame hiện tại, mất texture, thiếu frame trung gian, giới hạn lịch sử, TTL, reset và đổi session. Kiểm tra TTL dùng đồng hồ điều khiển để tránh phụ thuộc lịch chạy thread của hệ điều hành.

GUI: Tkinter thật ở chế độ ẩn, camera giả lập 640×480, ba đối tượng/ba lớp, phản hồi giả lập trễ 120 ms và khoảng 5 lượt/giây. Kiểm tra video cập nhật nhanh hơn nhận diện, ghi timing sau hiển thị, kết quả hết hạn, server offline, dừng/xóa hai khung ảnh và bắt đầu lại.

Một lần đo ngắn 3 giây sau tối ưu:

| Chỉ số | Kết quả |
|---|---:|
| Nguồn camera giả lập trung bình | 30,2 FPS |
| Cập nhật ảnh ở GUI, cửa sổ khoảng 2 giây cuối | 24,8 FPS |
| Nhận phản hồi nhận diện, cửa sổ khoảng 2 giây cuối | 4,5 FPS |
| Lượt đã ghi log | 13 |
| Lượt hiển thị / bị bỏ | 13 / 0 |

Đây là số đo cập nhật trong ứng dụng với GUI ẩn và nguồn giả lập, không phải benchmark webcam thật hoặc phép đo màn hình vật lý. Tốc độ phụ thuộc tải máy/camera/mạng; không bảo đảm 30 FPS.

Smoke model thật: nạp models/best.pt và nhận diện 3 ảnh đầu tiên trong datasets/mask/images/test. Model trả tổng 13 detection; sau dịch ảnh ngang 2 pixel, tracker giữ được 12 bbox. Bbox không theo dõi được bị bỏ, không giữ một vị trí giả. Đây là kiểm tra tích hợp, không phải đánh giá precision/recall hoặc chứng minh chất lượng ba lớp trên webcam.

## Cách thử thực tế

Tại thư mục Lab3, mở hai terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --workers 1 --ws-max-size 6000000
```

```powershell
.\.venv\Scripts\python.exe -m client.app --config configs/webcam.json
```

Bấm Bắt đầu. Quan sát FPS hiển thị so với camera, bbox khi di chuyển đầu, nhãn khi đeo/tháo khẩu trang, cảnh nhiều người, mất mạng và Dừng/Bắt đầu lại. Ba config mẫu vẫn trỏ localhost; cần đổi cloud_url khi triển khai Cloud thật.

Các tùy chọn trong config:

- display_fps: mục tiêu hiển thị, mặc định 30.
- target_fps: mức gửi nhận diện, hiện giữ 5.
- tracking_max_age_seconds: tuổi tối đa của kết quả tính từ frame nguồn, mặc định 1,0 giây.
- opencv_threads: mặc định 1.
- camera.tracking_width: chiều rộng ảnh tracking tối đa, mặc định 320; không thay đổi ảnh gửi model.
- camera.history_seconds / camera.history_max_frames: lịch sử tối đa 1,5 giây / 90 ảnh xám.

Nếu bbox hay biến mất với mặt nhỏ/thiếu texture hoặc mạng chậm, cần quan sát tuổi kết quả và mất dấu trước khi điều chỉnh. Tăng thời hạn giữ kết quả có thể giảm mất nhãn nhưng tăng thời gian tồn tại của nhãn cũ.
