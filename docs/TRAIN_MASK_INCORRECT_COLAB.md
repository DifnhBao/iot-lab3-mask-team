# Lượt 2: bổ sung đeo sai, giữ khả năng phân biệt tay che mặt

## Đã chuẩn bị

Model baseline: `models/best_mafa_hand.pt` đang dùng sau lượt 1. Dataset nền: `datasets/mask_mafa`, có 2.275 box đeo đúng, 883 không đeo, 94 đeo sai trong train. Lượt 1 chỉ thêm 2 box đeo sai. Notebook mới: `notebooks/Mask_Incorrect_Finetune_Colab.ipynb`.

Không train trên laptop. Chỉ chuẩn bị dữ liệu và kiểm tra trên laptop; train trên GPU Colab. Chưa train lượt 2 vì ảnh mới cần bạn duyệt và gán nhãn.

## 1. Duyệt MAFA nhãn đeo sai

Có 702 ảnh nguồn train và 167 ảnh nguồn test nhãn 2 chưa dùng ở vòng 1, trước kiểm tra box. Báo cáo chi tiết: `artifacts/mask_incorrect_analysis.json`. Đọc box/nhãn thủ công là bắt buộc; hình học hợp lệ không đồng nghĩa nhãn đúng.

```powershell
Set-Location 'D:\PTIT\N4\HK1\IoT\Lab3'
.\.venv\Scripts\python.exe scripts/prepare_mafa.py review --target-class 2 --source train --exclude-dataset datasets/mask_mafa
```

- K giữ, X loại, Q thoát. Quyết định lưu riêng vào `datasets/mafa/review_incorrect.json`; lựa chọn tay che mặt cũ vẫn giữ nguyên.
- Box nhãn 2 màu cam. Nhãn hiển thị vẫn là mã nguồn: 0 không đeo, 1 đeo đúng, 2 đeo sai, 3 vật che mặt.
- Bắt đầu giữ 250–400 ảnh train đa dạng. Đây là mục tiêu thu thử, không bảo đảm chất lượng. Không cần duyệt hết.
- Giữ khi có khẩu trang thực sự đeo sai, mọi khuôn mặt nhìn thấy đều có box đúng và mọi nhãn đều phù hợp quy ước dưới đây.
- Loại ảnh thiếu box, box ở tóc/trán thay vì khuôn mặt, ảnh quá mơ hồ, nhãn sai. Công cụ không cho bấm K với box vượt biên lớn. Đừng giữ ảnh lỗi chỉ để đạt số lượng.
- Với mặt nhãn 3 trong ảnh, chỉ giữ khi thực sự không có khẩu trang; class 3 sẽ được đổi thành without_mask. Nếu có khẩu trang sau vật che hoặc không xác định được thì loại.

Quy ước bài lab:

| Tình huống | Nhãn YOLO |
|---|---|
| Khẩu trang che đúng mũi và miệng | 0 with_mask |
| Không có khẩu trang, kể cả tay che mặt | 1 without_mask |
| Khẩu trang lộ mũi, lộ miệng hoặc kéo xuống cằm | 2 mask_incorrect |

Nhận diện khẩu trang kéo xuống cằm là quy ước của bài lab. Nguồn MAFA có thể coi một số trường hợp mũi/miệng đều hở là không đeo: nếu thấy mặt nguồn nhãn 0 nhưng có khẩu trang dưới cằm, hãy loại ảnh khỏi lượt nhập tự động, hoặc gán nhãn lại thủ công ở bộ riêng. Không nhập nhãn mâu thuẫn vào train. Bổ sung webcam bước 2 giúp học đúng trường hợp này.

Duyệt riêng nguồn test, giữ khoảng 40–60 ảnh phù hợp, chưa dùng train/val:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_mafa.py review --target-class 2 --source test --exclude-dataset datasets/mask_mafa
```

## 2. Thu ảnh webcam phù hợp cảnh thực tế

Đóng giao diện client để nhường webcam, không cần tự bật hoặc tắt server cho công cụ này. Không dùng screenshot giao diện làm dữ liệu chính; công cụ lưu ảnh camera và box trực tiếp.

```powershell
.\.venv\Scripts\python.exe scripts/collect_hard_cases.py --output datasets/webcam_incorrect --split train --class-id 2
```

- SPACE giữ frame; 2 chọn đeo sai; R vẽ box toàn bộ khuôn mặt, ENTER xác nhận; S lưu.
- 0 chọn đeo đúng; 1 chọn không đeo/tay che mặt. Chọn nhãn trước khi vẽ box. U bỏ box cuối; N bỏ frame; Q thoát.
- Box gồm toàn bộ khuôn mặt và vùng cằm, không chỉ vùng khẩu trang. Nhãn tất cả khuôn mặt trong ảnh, tránh người nền khó gán nhãn.
- Khoảng 80 ảnh train: 50 đeo sai, 15 đeo đúng, 15 không đeo/tay che. Trong 50 đeo sai, cần đủ lộ mũi, lộ miệng và kéo xuống cằm; dùng khẩu trang đen đang gây lỗi.
- Thay đổi khoảng cách, góc đầu, ánh sáng và nếu có thể thêm người. Không lưu hàng loạt frame gần như giống nhau.
- Trường hợp khẩu trang che mắt làm mất box là occlusion cực mạnh; không dùng làm trọng tâm lượt đầu. Ưu tiên ba kiểu đeo sai thông thường.

Trong buổi quay khác hoặc với người khác, thu val khoảng 20 ảnh (10 đeo sai, 5 đeo đúng, 5 không đeo/tay che), test khoảng 30 ảnh (15 đeo sai, 5 đeo đúng, 10 không đeo/tay che):

```powershell
.\.venv\Scripts\python.exe scripts/collect_hard_cases.py --output datasets/webcam_incorrect --split val --class-id 2
.\.venv\Scripts\python.exe scripts/collect_hard_cases.py --output datasets/webcam_incorrect --split test --class-id 2
```

Không chia frame liền nhau của cùng cảnh sang train/val/test. Nếu chỉ có một người, ít nhất tách buổi quay và điều kiện để giảm ảnh gần trùng. Test cuối chưa dùng để điều chỉnh model.

## 3. Chuyển và trộn dữ liệu mới

Chạy sau khi duyệt MAFA và thu webcam:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_mafa.py convert --target-class 2 --exclude-dataset datasets/mask_mafa --skip-invalid-images
.\.venv\Scripts\python.exe scripts/merge_hard_cases.py --base datasets/mask_mafa --extra datasets/mafa_incorrect --output datasets/mask_mafa_with_incorrect
.\.venv\Scripts\python.exe scripts/merge_hard_cases.py --base datasets/mask_mafa_with_incorrect --extra datasets/webcam_incorrect --output datasets/mask_mafa_v2
```

Giữ mọi split cũ. MAFA mới không nhập lại ảnh đã có ở vòng 1; split MAFA train mới được chia 85/15 train/val, nguồn test giữ riêng. Không làm lại việc chia ngẫu nhiên toàn bộ dataset cũ vì có thể chuyển test cũ vào train. Công cụ chặn ảnh trùng byte; ảnh gần trùng vẫn phải xem thủ công.

`conversion_report.json` ghi ảnh bị bỏ vì box không hợp lệ. Không tự cắt các box vượt biên lớn. Output tồn tại thì dùng tên mới và cập nhật tên ở các lệnh sau.

## 4. Đóng gói cho Colab

```powershell
.\.venv\Scripts\python.exe scripts/pack_mafa_colab.py --training-dataset mask_mafa_v2 --baseline models/best_mafa_hand.pt --extra-dataset mafa_incorrect --extra-dataset webcam_incorrect --output artifacts/mask_incorrect_colab_bundle.zip
```

Archive dùng alias `datasets/mask_mafa` cho dataset train v2 để notebook giữ đường dẫn portable. `models/best.pt` bên trong archive là bản sao baseline `best_mafa_hand.pt`, không thay file model cũ trên laptop. Metadata ghi nguồn và hash.

## 5. Chạy Colab

- Tạo MyDrive/lab3_mafa_v2, upload `mask_incorrect_colab_bundle.zip` vào đó.
- Upload `notebooks/Mask_Incorrect_Finetune_Colab.ipynb` lên Colab.
- Runtime → Change runtime type → GPU. Chạy cell theo thứ tự: cài thư viện, GPU, Drive, giải nén, sửa YAML, kiểm tra nhãn.
- Notebook khởi tạo từ model vòng 1 để giữ khả năng tay che mặt; train mới 30 epoch, batch 8, AdamW, lr0=0.0001, imgsz 640, patience 10. Đây là cấu hình khởi đầu để thử, không phải giá trị tối ưu đã đo. Nếu hết VRAM giảm batch 4 hoặc 2.
- Checkpoints lưu vào Drive. Không resume=True khi bắt đầu lượt 2 với dữ liệu mới; chỉ resume lượt 2 bị ngắt và còn checkpoint hợp lệ.

## 6. Đánh giá đồng thời bốn bộ

Notebook so sánh baseline vòng 1 với model mới trên val của:

1. MAFA đeo sai mới.
2. Webcam mới.
3. MAFA tay che mặt vòng 1.
4. Dataset cũ.

Không nhìn mAP tổng đơn lẻ. Xem precision/recall từng lớp trong log, confusion matrix, ảnh dự đoán tại conf=0.4. Bộ chỉ có ít lớp không chứng minh chất lượng cả ba trạng thái.

Đếm theo mặt đeo sai: nhận đúng mask_incorrect, nhầm without_mask, nhầm with_mask, mất box. Đếm riêng mặt tay che bị báo with_mask. Mục tiêu là giảm nhầm đeo sai thành không đeo và giữ khả năng nhận tay che, không chỉ làm box biến mất. Chọn model/ngưỡng bằng val.

Cell ảnh dự đoán mặc định xem webcam. Đổi PRED_DATASET thành 'mafa_incorrect' hoặc 'mafa_hand', chạy lại cell đó để xem hai bộ kia. Test cuối đặt CONFIRM_FINAL_TEST=True sau khi chốt cấu hình. Test tay che vòng 1 đã xem trước là bộ hồi quy; test webcam buổi mới cung cấp phép kiểm tra bổ sung chưa dùng train.x

## 7. Đưa candidate về ứng dụng

Export từ notebook, lưu file mới thành `models/best_mask_incorrect_v2.pt`. Dừng server bằng tay, rồi tại PowerShell:

```powershell
$env:MASK_MODEL_PATH = 'models/best_mask_incorrect_v2.pt'
.\.venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --workers 1 --ws-max-size 6000000
```

Mở client như trước; kiểm tra lộ mũi, xuống cằm, đeo đúng, không đeo và tay che. Giữ các weights cũ để quay lại nếu candidate không đạt. Không đổi chữ hiển thị without_mask thành đeo sai để che lỗi model.

## Việc cần làm ngay

Bạn bắt đầu bước 1 (duyệt MAFA nhãn 2). Sau đó thu webcam bước 2. Chưa thể train có ý nghĩa trước khi ảnh được duyệt/gán nhãn. Mọi bước tự động đã chuẩn bị; notebook chưa được chạy trên Colab thực tế ở lượt này.
