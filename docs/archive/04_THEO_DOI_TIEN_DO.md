# Theo dõi tiến độ — Lab3 khẩu trang

> Phạm vi mới: streaming liên tục, nhiều người, thay nguồn webcam bằng ESP32-CAM qua gateway với ít cấu hình. Đặc tả tích hợp đã có ở `05_STREAMING_VA_ESP32_CAM.md`; code adapter/firmware/WebSocket và tests S/H chưa triển khai/chạy. W00 có phần cập nhật thiết kế, W01–W14 vẫn chưa hoàn thành.

Ngày khởi tạo: 10/10/2026. Bảng này ghi trạng thái thực tế, không đánh dấu hoàn thành chỉ vì đã có hướng dẫn.

Trạng thái công việc: TODO / IN_PROGRESS / DONE / BLOCKED.

| ID | Công việc | Phụ thuộc | Trạng thái | Bằng chứng hiện có / cần có |
|---|---|---|---|---|
| W00 | Chốt đặc tả | Lựa chọn đề tài | DONE | docs/01_DAC_TA_DE_TAI.md; đề tài khẩu trang đã chọn |
| W01 | Môi trường | W00 | TODO | Chỉ xác minh Python3.13.9 và Docker command; chưa smoke dependencies/camera |
| W02 | Dataset nguồn | W01 | TODO | Chưa tải hoặc audit dataset |
| W03 | Thu ảnh webcam | W01,W02 | TODO | Chưa thu ảnh và metadata |
| W04 | Nhãn YOLO | W02,W03 | TODO | Chưa có converter, labels hoặc YAML thực |
| W05 | Chia tập/audit | W04 | TODO | Chưa có split manifest và audit |
| W06 | Train thử | W05 | TODO | Chưa có train smoke |
| W07 | Train chính | W06 | TODO | Chưa có weights của bài |
| W08 | Test/đóng gói model | W07 | TODO | Chưa có metrics/hash model |
| W09 | API local | W08 | TODO | Chưa có server source |
| W10 | GUI local | W09 | TODO | Chưa có client source |
| W11 | Deploy Cloud | W09,W10 | TODO | Chưa có tài khoản/endpoint được cấu hình |
| W12 | Test đầu-cuối | W11 | TODO | Chưa chạy webcam/Cloud tests |
| W13 | Đo độ trễ | W12 | TODO | Chưa có raw CSV/số đo thật |
| W14 | Báo cáo/demo/tái chạy | W08,W12,W13 | TODO | Chưa có báo cáo/video cuối |

## Sổ test

Toàn bộ T01–T43 hiện **NOT_RUN**. Một số lệnh read-only kiểm tra công cụ đã chạy nhưng chưa đáp ứng đầy đủ test môi trường. Khi thực thi, tạo kết quả theo template trong `03_TEST_VA_NGHIEM_THU.md`.

| Nhóm | IDs | Trạng thái ban đầu |
|---|---|---|
| Môi trường/dataset | T01–T13 | NOT_RUN |
| Model | T14–T18 | NOT_RUN |
| API | T19–T25 | NOT_RUN |
| GUI | T26–T30 | NOT_RUN |
| Cloud/nhận dạng | T31–T37 | NOT_RUN |
| Timing/bài nộp | T38–T43 | NOT_RUN |

## Điểm cần có trong quá trình thực hiện

- Tài khoản hoặc server Cloud có quyền triển khai; kiểm tra tài nguyên/chi phí trước dùng.
- Dataset thực tế và điều kiện sử dụng; chưa xác minh bằng tải files.
- Thông tin GPU/RAM hoặc chọn train GPU notebook nếu laptop chưa phù hợp.
- Người tham gia/ảnh thật cho demo cuối ba lớp cùng khung hình.
- Hướng dẫn bổ sung của giảng viên về độ trễ một chiều và định dạng bài nộp nếu có.

Các điểm này chưa chặn việc lập kế hoạch. Không cần hỏi lại xác nhận đề tài: đã chọn khẩu trang.

## Công việc tiếp theo

W01: chuẩn bị môi trường dự án, chạy smoke dependencies và kiểm tra webcam; sau đó W02 kiểm kê dataset nguồn. Chưa install dependencies, tải dataset hoặc mở camera trong lượt lập kế hoạch này.
