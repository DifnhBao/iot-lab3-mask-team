# Bổ sung trường hợp tay che mặt

Chưa chạy train vì chưa có ảnh mới được gán nhãn. Giữ nguyên `models/best.pt` cho tới khi model mới được đánh giá tốt hơn.

## 1. Thu và gán nhãn

Đóng giao diện nhận dạng để nhường webcam. Tại thư mục Lab3:

```powershell
.\.venv\Scripts\python.exe scripts/collect_hard_cases.py --split train
```

- SPACE: đóng băng ảnh hiện tại.
- 0: đeo khẩu trang đúng; 1: không có khẩu trang, kể cả dùng tay che mũi/miệng; 2: có khẩu trang nhưng đeo sai.
- R: vẽ box bao quanh toàn bộ khuôn mặt, giống quy ước dataset cũ; ENTER xác nhận box, C hủy. Chọn nhãn trước khi bấm R.
- Gán nhãn **tất cả khuôn mặt nhìn thấy** trong ảnh. U bỏ box cuối. Nếu nhiều người ở nền khó gán nhãn, chọn góc quay khác.
- S: lưu ảnh và nhãn. N: bỏ ảnh để quay lại camera. Q: thoát.

Thu cả tay che miệng, tay che mũi và miệng, không che mặt, khẩu trang đúng và sai. Thay đổi tư thế, ánh sáng, khoảng cách và người. Bắt đầu khoảng 150 ảnh train đa dạng, trong đó khoảng 70–100 ảnh tay che mặt; đây là mục tiêu thu thử, không bảo đảm chất lượng.

Thu thêm khoảng 30 ảnh val và 30 ảnh test trong **buổi quay khác hoặc người khác**. Mỗi tập đều có tay che mặt và cả ba trạng thái khẩu trang. Không chia các frame liên tiếp của một video sang các tập khác nhau.

```powershell
.\.venv\Scripts\python.exe scripts/collect_hard_cases.py --split val
.\.venv\Scripts\python.exe scripts/collect_hard_cases.py --split test
```

## 2. Trộn dữ liệu và train tiếp

```powershell
.\.venv\Scripts\python.exe scripts/merge_hard_cases.py
.\.venv\Scripts\python.exe scripts/train.py --data datasets/mask_hard_cases/data.yaml --weights models/best.pt --epochs 30 --batch 2 --device cpu --name mask_hard_cases
```

Script tạo dataset mới, giữ dữ liệu cũ, kiểm tra nhãn và chặn ảnh trùng byte. Cần xem lại ảnh/box thủ công; kiểm tra tự động không phát hiện mọi ảnh gần trùng. Máy hiện không có CUDA; CPU có thể mất nhiều thời gian. Có thể dùng dataset mới và script trên notebook GPU, sửa `path` trong YAML cho môi trường đó và đổi `--device cpu` thành `--device 0`.

Train tiếp là một lượt huấn luyện mới khởi tạo bằng weights cũ, không phải resume trạng thái optimizer cũ.

## 3. So sánh trước khi thay model

Dùng cùng tập test mới và cùng ngưỡng inference 0.4 để so sánh:

```powershell
.\.venv\Scripts\yolo.exe detect val model=models/best.pt data=datasets/hard_cases/data.yaml split=test conf=0.4 agnostic_nms=True plots=True project=artifacts/evaluation name=old_hard_cases
.\.venv\Scripts\yolo.exe detect val model=artifacts/training/mask_hard_cases/weights/best.pt data=datasets/hard_cases/data.yaml split=test conf=0.4 agnostic_nms=True plots=True project=artifacts/evaluation name=new_hard_cases
.\.venv\Scripts\yolo.exe detect val model=artifacts/training/mask_hard_cases/weights/best.pt data=datasets/mask/data.yaml split=test conf=0.4 agnostic_nms=True plots=True project=artifacts/evaluation name=new_original_test
```

Nếu train tự tạo tên có hậu tố do thư mục đã tồn tại, dùng đúng đường dẫn weights mà train báo. Xem ảnh dự đoán, confusion matrix, precision/recall từng lớp; đếm riêng tay che mặt bị báo `with_mask`. Test dữ liệu cũ để phát hiện khả năng nhận khẩu trang thật bị giảm. Không chọn ngưỡng hoặc sửa model dựa lặp lại vào tập test; dùng val để điều chỉnh.

Chỉ sau khi đạt yêu cầu mới sao lưu model cũ, thay `models/best.pt` bằng model mới và khởi động lại server.
