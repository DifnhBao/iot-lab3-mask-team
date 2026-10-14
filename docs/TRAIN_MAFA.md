# Train tiếp với MAFA và AnnotationMAFA

Đã tải hai CSV vào `datasets/mafa/annotations`. Đã tạo `scripts/prepare_mafa.py` để duyệt ảnh và chuyển nhãn, chưa tải ảnh MAFA và chưa chạy train. Giữ model hiện tại để làm baseline.

## 1. Nguồn dữ liệu

- Nhãn đã sửa: https://github.com/ElenaRyumina/AnnotationMAFA
- Nguồn dẫn tới ảnh MAFA từ repo nghiên cứu FMLD: https://github.com/borutb-fri/FMLD
- Google Drive được repo FMLD dẫn: https://drive.google.com/drive/folders/1nbtM1n0--iZ3VVbNGhocxbnBGhMau_OG
- Bản mirror Kaggle được repo FMLD dẫn: https://www.kaggle.com/datasets/rahulmangalampalli/mafa-data

Repo AnnotationMAFA chứa nhãn, không chứa toàn bộ ảnh. Tải cả ảnh train và test, giải nén dưới `datasets/mafa/raw`. Script tìm ảnh đệ quy nên thư mục con có thể giữ nguyên. Cần ảnh có tên gốc `train_00000001.jpg`, `test_00000001.jpg`, ...; không dùng bản mirror đổi tên/crop ảnh. Không cần các file nhãn `.mat` gốc cho quy trình này.

Ví dụ:

```text
datasets/mafa/
  annotations/
    train_mafa_annotation.csv
    test_mafa_annotation_full.csv
  raw/
    train-images/
      train_00000001.jpg
      ...
    test-images/
      test_00000001.jpg
      ...
```

Tại PowerShell trong Lab3:

```powershell
Set-Location 'D:\PTIT\N4\HK1\IoT\Lab3'
New-Item -ItemType Directory -Force datasets/mafa/raw
```

Nếu cần tải lại nhãn, dùng hai lệnh sau (không dùng nhãn MAFA gốc thay thế):

```powershell
Invoke-WebRequest 'https://raw.githubusercontent.com/ElenaRyumina/AnnotationMAFA/main/train_mafa_annotation.csv' -OutFile datasets/mafa/annotations/train_mafa_annotation.csv
Invoke-WebRequest 'https://raw.githubusercontent.com/ElenaRyumina/AnnotationMAFA/main/test_mafa_annotation_full.csv' -OutFile datasets/mafa/annotations/test_mafa_annotation_full.csv
```

## 2. Hiểu mã lớp

CSV thực tế có cột `name_file,x,y,w,h,label`. Một ảnh có thể có nhiều dòng tương ứng nhiều khuôn mặt.

| AnnotationMAFA | Ý nghĩa nguồn | Lớp của dự án |
|---|---|---|
| 0 | Không đeo | 1: without_mask |
| 1 | Đeo đúng | 0: with_mask |
| 2 | Đeo sai | 2: mask_incorrect |
| 3 | Vật khác che mặt | 1: without_mask, chỉ sau khi xem và xác nhận không có khẩu trang |

Lớp 3 không chỉ có tay mà còn có điện thoại, sách, đồ ăn… Không đổi toàn bộ lớp 3 một cách mù quáng. Nguồn còn coi một số khăn che mặt là khẩu trang; loại ảnh không phù hợp định nghĩa khẩu trang trong bài lab.

Quy trình giữ ba lớp hiện tại, không thêm lớp 4. Chỉ giữ ảnh có tay che mũi/miệng, xác định được không có khẩu trang; tất cả box và nhãn trong ảnh phải đúng. Nếu che quá nhiều, có khả năng đang đeo khẩu trang phía sau tay, thiếu box ở người nền hoặc nhãn sai, loại ảnh khỏi lượt này. Công cụ duyệt không sửa box; chỉ giúp giữ hoặc loại ảnh đã được gán nhãn.

## 3. Duyệt ảnh

```powershell
.\.venv\Scripts\python.exe scripts/prepare_mafa.py review --source train
```

Công cụ chỉ duyệt các ảnh có ít nhất một mặt nhãn 3. Box nhãn 3 màu cam; chữ trên ảnh là mã lớp NGUỒN, chưa phải mã lớp YOLO mới.

- K: giữ ảnh sau khi kiểm tra **tất cả** khuôn mặt. Với mọi box nhãn 3, cần xác nhận không có khẩu trang thực tế. Ít nhất một trường hợp tay che mũi/miệng phù hợp mục tiêu.
- X: loại ảnh không phù hợp, mơ hồ hoặc nhãn/box không đầy đủ.
- Q: thoát. Quyết định lưu ngay vào `datasets/mafa/review.json`; chạy lại tiếp tục phần chưa duyệt.

Bắt đầu khoảng 200–400 ảnh train nguồn được giữ, đa dạng người/góc/ánh sáng. Con số là mục tiêu thử nghiệm, không bảo đảm độ chính xác. Nếu có ảnh gần giống nhau giữa các tập, loại bớt trước khi train.

Duyệt riêng ảnh nguồn test và giữ khoảng 50–100 ảnh phù hợp:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_mafa.py review --source test
```

Ảnh test nguồn không được đưa vào train hoặc val. Cần kiểm tra nguồn huấn luyện của model cũ: nếu đã train bằng MAFA test thì tập này không phải test độc lập; dùng thêm ảnh webcam chưa từng train để đánh giá.

## 4. Chuyển sang YOLO và trộn dữ liệu

```powershell
.\.venv\Scripts\python.exe scripts/prepare_mafa.py convert
.\.venv\Scripts\python.exe scripts/merge_hard_cases.py --extra datasets/mafa_hand --output datasets/mask_mafa
```

Lệnh đầu tạo `datasets/mafa_hand`: 85% ảnh train nguồn được giữ vào train, 15% vào val với seed 42; ảnh test nguồn được giữ vào test. Chia theo ảnh, không chia từng box. Giữ mọi box của ảnh đã duyệt.

YOLO dùng `(class, x_center/W, y_center/H, box_width/W, box_height/H)`. Script không tự dịch tọa độ nguồn một pixel; cần nhìn box hiển thị khi duyệt. Box vượt biên tối đa 1 pixel được clip; sai lớn hơn bị chặn để kiểm tra.

Lệnh thứ hai trộn với dataset ba lớp cũ `datasets/mask`, giữ nguyên các split. Cả hai lệnh không ghi đè thư mục output tồn tại. Nếu thử lại, dùng tên output mới và sửa đường dẫn train tương ứng. Khi báo ảnh trùng, cần loại ảnh trùng khỏi tập bổ sung trước khi xuất lại; không bỏ cơ chế kiểm tra.

Ảnh trùng byte bị chặn, nhưng ảnh gần trùng hoặc cùng người vẫn cần kiểm tra thủ công. File `manifest.json` ghi nguồn và split của ảnh bổ sung. Xem vài file labels để chắc chắn class mới là 0/1/2 và box khớp ảnh.

## 5. Fine-tune từ model hiện tại

Đóng client và dừng server để giảm tải CPU/RAM. Máy hiện không có CUDA. Lượt thử CPU 3 epoch chỉ để kiểm tra pipeline:

```powershell
.\.venv\Scripts\python.exe scripts/train.py --data datasets/mask_mafa/data.yaml --weights models/best.pt --epochs 3 --batch 2 --device cpu --optimizer AdamW --lr0 0.0003 --name mafa_hand_smoke
```

Sau khi pipeline chạy được, train chính từ model ban đầu:

```powershell
.\.venv\Scripts\python.exe scripts/train.py --data datasets/mask_mafa/data.yaml --weights models/best.pt --epochs 30 --batch 2 --device cpu --optimizer AdamW --lr0 0.0003 --name mafa_hand_finetune
```

AdamW và lr0=0.0003 là cấu hình khởi đầu để thử, không phải giá trị tối ưu đã đo. Điều chỉnh bằng tập val. `optimizer=auto` có thể bỏ qua lr0 nên ở đây chọn optimizer tường minh. Script đặt imgsz=640, seed=42, workers=0, patience=10. Đây là lượt fine-tune mới, không dùng resume=True. Giữ model ba lớp từ weights cũ.

Model thường lưu tại `artifacts/training/mafa_hand_finetune/weights/best.pt`. Dùng đúng đường dẫn console báo nếu thư mục có hậu tố. `best.pt` mới được chọn trên val, không đồng nghĩa chắc chắn hết lỗi tay che mặt.

Để train trên notebook GPU: upload dataset trộn và weights cũ, sửa `path` trong data.yaml theo vị trí thật của notebook; dùng script với `--device 0`, có thể bắt đầu `--batch 8`. Không dùng đường dẫn D:/ của Windows trên notebook Linux. Có thể dùng CLI Ultralytics hoặc cùng script; không cần train lại từ COCO.

## 6. So sánh model cũ và mới

Đánh giá trên cùng tập MAFA test bổ sung, rồi đánh giá lại test cũ:

```powershell
.\.venv\Scripts\yolo.exe detect val model=models/best.pt data=datasets/mafa_hand/data.yaml split=test conf=0.4 iou=0.5 agnostic_nms=True plots=True project=artifacts/evaluation name=old_mafa_hand
.\.venv\Scripts\yolo.exe detect val model=artifacts/training/mafa_hand_finetune/weights/best.pt data=datasets/mafa_hand/data.yaml split=test conf=0.4 iou=0.5 agnostic_nms=True plots=True project=artifacts/evaluation name=new_mafa_hand
.\.venv\Scripts\yolo.exe detect val model=models/best.pt data=datasets/mask/data.yaml split=test conf=0.4 iou=0.5 agnostic_nms=True plots=True project=artifacts/evaluation name=old_original
.\.venv\Scripts\yolo.exe detect val model=artifacts/training/mafa_hand_finetune/weights/best.pt data=datasets/mask/data.yaml split=test conf=0.4 iou=0.5 agnostic_nms=True plots=True project=artifacts/evaluation name=new_original
```

Tập MAFA bổ sung được lọc quanh tay che mặt có thể ít hoặc không có một số lớp; không dùng riêng mAP tổng của nó để kết luận khả năng nhận cả ba lớp. Xem confusion matrix, ảnh dự đoán và precision/recall; so sánh test cũ để kiểm tra khả năng nhận khẩu trang thật có giảm không. Precision/recall tóm tắt trong thư viện có thể được chọn theo điểm max-F1, không nhất thiết là số đo trực tiếp ở ngưỡng hiển thị 0.4.

Xuất ảnh dự đoán để đếm riêng lỗi tay che mặt ở cấu hình inference giống server:

```powershell
.\.venv\Scripts\yolo.exe detect predict model=models/best.pt source=datasets/mafa_hand/images/test conf=0.4 iou=0.5 agnostic_nms=True save=True save_txt=True save_conf=True project=artifacts/evaluation name=old_hand_predictions
.\.venv\Scripts\yolo.exe detect predict model=artifacts/training/mafa_hand_finetune/weights/best.pt source=datasets/mafa_hand/images/test conf=0.4 iou=0.5 agnostic_nms=True save=True save_txt=True save_conf=True project=artifacts/evaluation name=new_hand_predictions
```

Với từng khuôn mặt mục tiêu, ghi một trong: báo đeo đúng sai / báo không đeo đúng / báo đeo sai sai / không phát hiện. Không coi việc model ngừng phát hiện khuôn mặt là đã sửa lỗi. Tính số mặt tay che bị báo with_mask / tổng số mặt tay che đã gán nhãn, đồng thời tính số mặt được nhận đúng without_mask. Kiểm tra webcam mới với người, tư thế và ánh sáng khác. Chỉ điều chỉnh model/ngưỡng trên val; nếu tiếp tục dùng test để chỉnh thì cần một test mới độc lập.

## 7. Thay model khi đánh giá đạt

Không có bước thay model tự động. Sau khi model mới giảm lỗi và không làm chất lượng khẩu trang thật giảm đáng kể, dừng server và sao lưu:

```powershell
Copy-Item models/best.pt models/best_before_mafa.pt
Copy-Item artifacts/training/mafa_hand_finetune/weights/best.pt models/best.pt
.\.venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --workers 1 --ws-max-size 6000000
```

Chỉ chạy lệnh sao lưu nếu `best_before_mafa.pt` chưa tồn tại; nếu có, chọn tên khác để giữ bản sao trước. Mở client ở terminal thứ hai như bình thường. Server phải ready=true và names vẫn là 0 with_mask, 1 without_mask, 2 mask_incorrect.

Nguồn tài liệu train/val: https://docs.ultralytics.com/modes/train/ và https://docs.ultralytics.com/modes/val/.
