# Kế hoạch Lab3 đã chốt

> Cập nhật: nhận dạng liên tục, số người thay đổi theo khung hình; phát triển bằng webcam và tích hợp ESP32-CAM cho demo cuối qua laptop gateway. Phần camera/streaming/transport trong kế hoạch đầu được thay bằng [đặc tả tích hợp mới](docs/05_STREAMING_VA_ESP32_CAM.md). Hiện mới có tài liệu, chưa có code/test streaming hoặc phần cứng đã chạy.

**Đề tài:** Nhận diện đồng thời tình trạng đeo khẩu trang của nhiều người từ webcam bằng mô hình học sâu nhẹ triển khai trên Cloud.

**Ba nhãn:** 0 `with_mask`, 1 `without_mask`, 2 `mask_incorrect`.

**Hướng:** object detection; mỗi mặt có khung bao và nhãn riêng; demo camera thật có ba người thuộc ba lớp trong cùng ảnh.

**Công nghệ:** YOLOv8n + Python/FastAPI trên Cloud; OpenCV/Tkinter trên laptop. File model của bài là best.pt sau huấn luyện; inference chạy ở Cloud.

## Tài liệu sử dụng

1. [Đặc tả đề tài, kiến trúc, API và cách đo](docs/01_DAC_TA_DE_TAI.md).
2. [15 công việc: cách làm, output, lệnh và điều kiện hoàn thành](docs/02_CONG_VIEC_VA_CACH_LAM.md).
3. [43 test: input, expected, evidence và nghiệm thu](docs/03_TEST_VA_NGHIEM_THU.md).
4. [Bảng theo dõi tiến độ thật](docs/04_THEO_DOI_TIEN_DO.md).
5. [Streaming và tích hợp ESP32-CAM, cấu hình chuyển nguồn và test phần cứng](docs/05_STREAMING_VA_ESP32_CAM.md).

Hướng dẫn brainstorm cũ còn trong `HUONG_DAN_LAB3.md`. Các lựa chọn đề tài/cách nhận dạng trong hướng dẫn cũ đã được thay bằng quyết định ở bộ tài liệu này.

## Điều đã thực hiện

- Đọc slide Lab2 và Lab3; chốt đề tài theo lựa chọn người học.
- Lập đặc tả, kế hoạch triển khai, đầu ra, test plan và bảng tiến độ.
- Kiểm tra read-only: máy có Python3.13.9 và lệnh Docker. Chưa xác minh Docker daemon, GPU/RAM hay webcam.

## Điều chưa thực hiện

Chưa viết hệ thống server/client/train, chưa cài môi trường dự án, tải/thu dữ liệu, huấn luyện model, triển khai Cloud, chạy 43 test hoặc thu số liệu. Các đường dẫn output và lệnh script trong kế hoạch là quy hoạch triển khai, không phải bằng chứng đã hoàn thành.

## Mốc thực hiện

W00 đặc tả đã lập. Tiếp theo W01 môi trường, W02/W03 dữ liệu, W04/W05 nhãn và chia tập, W06/W07 train, W08 test model, W09/W10 API và GUI, W11 Cloud, W12/W13 nghiệm thu và đo, W14 bài nộp.

Chỉ gọi hệ thống hoàn chỉnh khi có artifacts thực tế và nghiệm thu đạt theo checklist. Ngưỡng mAP/recall, demo và độ trễ trong spec là mục tiêu đề xuất của dự án, không phải yêu cầu định lượng do giảng viên cung cấp.
