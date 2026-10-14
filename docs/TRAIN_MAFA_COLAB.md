# Cải thiện model tay che mặt bằng GPU Google Colab

## Trạng thái hiện tại

- Model ba lớp hiện có: `models/best.pt`.
- Dataset cũ đã chuẩn bị: `datasets/mask`.
- Hai CSV AnnotationMAFA đã tải: `datasets/mafa/annotations`.
- Chưa có ảnh MAFA, quyết định duyệt, dataset `mafa_hand` hoặc `mask_mafa`.
- Notebook sẵn có: `notebooks/MAFA_Hand_Finetune_Colab.ipynb`.
- Toàn bộ train chạy trên hosted GPU của Colab. Laptop chỉ tải/duyệt/chuyển/đóng gói dữ liệu và chạy ứng dụng sau train.

## Bước 1. Tải ảnh MAFA về laptop

Nguồn ảnh do repo nghiên cứu [FMLD](https://github.com/borutb-fri/FMLD) dẫn:

- [Google Drive MAFA](https://drive.google.com/drive/folders/1nbtM1n0--iZ3VVbNGhocxbnBGhMau_OG).
- [Mirror Kaggle](https://www.kaggle.com/datasets/rahulmangalampalli/mafa-data).
- Nhãn được dùng: [AnnotationMAFA](https://github.com/ElenaRyumina/AnnotationMAFA).

Tải ảnh train và test, giải nén dưới `datasets/mafa/raw`. Giữ tên `train_00000001.jpg`, `test_00000001.jpg`, ... và ảnh gốc chưa resize/crop để CSV khớp. Có thể giữ các thư mục con `train-images`, `test-images`; script tìm đệ quy. Không cần nhãn `.mat` gốc. Nếu link nguồn cần đăng nhập hoặc không truy cập được, thử nguồn còn lại và kiểm tra tên ảnh.

```powershell
Set-Location 'D:\PTIT\N4\HK1\IoT\Lab3'
New-Item -ItemType Directory -Force datasets/mafa/raw
```

## Bước 2. Duyệt ảnh thay vì vẽ lại box

```powershell
.\.venv\Scripts\python.exe scripts/prepare_mafa.py review --source train
```

Công cụ chỉ hiển thị ảnh có mặt bị vật khác che (nhãn nguồn 3). K giữ, X loại, Q thoát; quyết định tự lưu và lần sau tiếp tục. Box nhãn 3 màu cam, chữ hiển thị là nhãn nguồn:

| Mã nguồn | Ý nghĩa | Mã model hiện tại |
|---|---|---|
| 0 | Không đeo | 1 without_mask |
| 1 | Đeo đúng | 0 with_mask |
| 2 | Đeo sai | 2 mask_incorrect |
| 3 | Vật khác che | 1 without_mask chỉ sau khi kiểm tra |

Giữ ảnh có tay che mũi/miệng mà xác định được không có khẩu trang. Kiểm tra mọi khuôn mặt đều có box và mọi nhãn trong ảnh đúng. Với mọi mặt nhãn 3, xác nhận không có khẩu trang thực tế; không giữ mặt bị che mơ hồ hoặc có khẩu trang sau tay. Loại ảnh thiếu box, box sai và khăn che mặt không phù hợp quy ước bài lab. Lớp 3 còn chứa điện thoại/sách/đồ ăn, không chỉ tay.

Bắt đầu 200–400 ảnh nguồn train đa dạng (người, tư thế, khoảng cách, ánh sáng); đây là mục tiêu thử, không bảo đảm chất lượng. Sau đó duyệt riêng nguồn test và giữ 50–100 ảnh phù hợp:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_mafa.py review --source test
```

Test nguồn không được đưa vào train. Loại ảnh gần trùng xuất hiện ở nhiều nguồn. Nếu model cũ đã học MAFA test, cần tập webcam độc lập chưa từng train để đánh giá thực sự.

## Bước 3. Chuyển nhãn, trộn và đóng gói

```powershell
.\.venv\Scripts\python.exe scripts/prepare_mafa.py convert
.\.venv\Scripts\python.exe scripts/merge_hard_cases.py --extra datasets/mafa_hand --output datasets/mask_mafa
.\.venv\Scripts\python.exe scripts/pack_mafa_colab.py
```

Lệnh 1 xuất `datasets/mafa_hand` dạng YOLO, chia ảnh train nguồn 85/15 thành train/val, giữ test nguồn riêng. Lệnh 2 trộn với dữ liệu cũ và giữ mọi split. Lệnh 3 tạo `artifacts/mafa_colab_bundle.zip`.

Nếu convert báo box vượt biên lớn, có thể chạy `scripts/prepare_mafa.py convert --skip-invalid-images` để bỏ toàn bộ ảnh có box không hợp lệ. Ảnh bị bỏ được liệt kê trong `datasets/mafa_hand/conversion_report.json`; CSV và lựa chọn duyệt không thay đổi. Không tự cắt box vượt biên lớn để tránh xuất nhãn sai. Nếu output đã tồn tại, dùng thư mục output mới thay vì ghi đè.

Bundle gồm dataset trộn để train, MAFA đã duyệt để đánh giá trường hợp khó, dataset cũ để kiểm tra khả năng nhận khẩu trang thật, model gốc để so sánh và thông tin nguồn. Không đóng gói toàn bộ ảnh MAFA chưa chọn. Lưu data cũ trong bundle giúp so sánh, không có nghĩa là đưa test vào train: YOLO chỉ dùng đường dẫn train/val chỉ định cho training.

Script chặn output tồn tại và ảnh trùng byte; chọn tên mới khi thử lại để giữ các phiên bản. Dữ liệu phải kiểm tra ảnh gần trùng thủ công. Đừng train chỉ ảnh tay che mặt vì có thể làm model mất khả năng nhận khẩu trang.

## Bước 4. Đưa file lên Drive và mở notebook

1. Trong Google Drive tạo thư mục `lab3_mafa` ở My Drive.
2. Upload `artifacts/mafa_colab_bundle.zip` vào đó. Đợi upload hoàn tất.
3. Mở https://colab.research.google.com/.
4. Chọn File → Upload notebook, chọn `notebooks/MAFA_Hand_Finetune_Colab.ipynb` trên laptop. Có thể mở qua hộp thoại Upload lúc mới vào Colab.
5. Chọn Runtime → Change runtime type → GPU; T4 là đủ cho lượt YOLOv8n đầu nếu được cấp.
6. Dùng hosted runtime Colab. Không chọn Connect to local runtime.
7. Chạy từng cell từ trên xuống, không bỏ qua kiểm tra nhãn và val.

Colab có thể giới hạn GPU và thời lượng phiên; không có bảo đảm GPU luôn sẵn. Xem [Colab FAQ](https://research.google.com/colaboratory/faq.html). Quy trình copy archive về `/content` trước khi giải nén giúp giảm việc đọc hàng nghìn file ảnh trực tiếp từ Drive.

## Bước 5. Chuẩn bị runtime theo notebook

- Cài `ultralytics==8.4.175` để khớp môi trường hiện tại của dự án.
- Xác nhận `torch.cuda.is_available()` là True và có tên GPU.
- Mount Google Drive, hoàn tất bước cấp quyền mà Colab hiển thị.
- Notebook tìm file `/content/drive/MyDrive/lab3_mafa/mafa_colab_bundle.zip`. Nếu bạn đặt nơi khác, sửa ARCHIVE.
- Copy và giải nén sang `/content/lab3_mafa`.
- Sửa đường dẫn YAML sang Linux. Không dùng đường dẫn D:/ của Windows trên Colab.
- Kiểm tra nhãn, số ảnh/mặt từng lớp, ảnh trùng giữa split của dataset trộn; hiển thị ảnh và box mẫu.

Nếu ảnh tay vẫn bị gán with_mask hoặc box sai, quay lại bước duyệt/sửa dữ liệu; đừng tiếp tục train. Đây là lỗi dữ liệu, tăng epoch không khắc phục nhãn sai.

## Bước 6. Chạy fine-tune trên GPU

Notebook đã có cell:

```python
model = YOLO(str(BASELINE))
model.train(
    data=DATA['mask_mafa'],
    epochs=30, batch=8, imgsz=640,
    device=0, workers=2, cache=False,
    optimizer='AdamW', lr0=0.0003,
    seed=42, patience=10,
    save=True, save_period=5,
    project=str(RUN_PROJECT), name=RUN_NAME,
)
```

- BASELINE là `best.pt` hiện tại, không phải COCO weights.
- imgsz 640 giữ kích thước inference như server.
- batch 8 là khởi đầu cho GPU; nếu hết VRAM giảm 4 hoặc 2.
- AdamW và learning rate 0.0003 là cấu hình khởi đầu để thử, điều chỉnh bằng val. Chọn optimizer cụ thể vì optimizer=auto có thể bỏ qua lr0.
- 30 epoch là giới hạn ban đầu; patience 10 cho phép dừng sớm khi val không cải thiện. Không kết luận tăng epoch chắc chắn sửa được lỗi.
- Không dùng resume=True ở lượt fine-tune mới với dữ liệu bổ sung.

Để smoke, có thể đặt EPOCHS=3 trước. Sau đó đặt lại 30 và chạy lượt mới từ BASELINE; RUN_NAME tự có timestamp. Lượt 3 epoch chỉ kiểm tra pipeline, không dùng để kết luận chất lượng.

Weights/biểu đồ được lưu theo timestamp ở `MyDrive/lab3_mafa/runs/<tên-lượt>/`. `best.pt` được chọn theo val của dataset trộn, `last.pt` phục vụ resume lượt chưa hoàn thành. Lưu model không tự bảo đảm tập tay che mặt đã tốt hơn. Theo dõi log train, results.csv và results.png.

Nguồn: [Ultralytics train](https://docs.ultralytics.com/modes/train/).

## Bước 7. So sánh bằng val và điều chỉnh nếu cần

Notebook chạy cả model cũ và mới trên:

- `mafa_hand` val: kiểm tra trường hợp tay/vật che mặt.
- `mask` val: kiểm tra còn nhận tốt khẩu trang đúng, sai và không đeo.

Sau đó xuất ảnh dự đoán ở conf 0.4, iou 0.5, max_det 100 và agnostic_nms=True như server. Mở ảnh old/new cạnh nhau trong notebook; toàn bộ ảnh dự đoán nằm trong Drive. 8 ảnh hiển thị mẫu chưa đủ để đánh giá toàn bộ.

Ghi bảng theo khuôn mặt tay che thực tế, không theo số ảnh (một ảnh có nhiều mặt):

| Chỉ số trên cùng bộ ảnh val | Model cũ | Model mới |
|---|---:|---:|
| Tổng mặt tay che cần đánh giá | N | N |
| Bị báo with_mask sai | ... | ... |
| Được báo without_mask đúng | ... | ... |
| Bị báo mask_incorrect sai | ... | ... |
| Không phát hiện | ... | ... |

Tỷ lệ báo đeo đúng sai = số mặt tay che bị báo with_mask / N. Tỷ lệ nhận đúng = số được báo without_mask đúng / N. Đừng coi việc model bỏ luôn box là nhận đúng.

Xem thêm precision/recall từng lớp và confusion matrix, nhưng tập khó có thể thiếu một số lớp nên mAP tổng không đủ. Chỉ số precision/recall tóm tắt Ultralytics có thể ở điểm max-F1, không đồng nhất với đếm prediction ở conf=0.4. Nguồn: [Ultralytics val](https://docs.ultralytics.com/modes/val/).

Nếu chưa tốt: kiểm tra nhãn trước; bổ sung ảnh train đa dạng đúng tình huống lỗi; điều chỉnh learning rate/epoch bằng val; train một lượt mới từ baseline để so sánh. Không chuyển những ảnh test đã xem sang train rồi tiếp tục gọi cùng tập đó là test độc lập. Nếu dùng val để bổ sung train, cần một val khác độc lập cho vòng tiếp theo.

## Bước 8. Chốt cấu hình rồi chạy test cuối

Đổi CONFIRM_FINAL_TEST=False thành True trong cell test. Notebook so sánh old/new trên MAFA test đã giữ riêng và test gốc, xuất ảnh dự đoán tương ứng. Đếm lại cùng bảng như val.

Model đạt mục tiêu khi số tay che bị báo đeo đúng giảm, số nhận đúng without_mask tăng, không chỉ tăng bỏ sót; đồng thời chất lượng khẩu trang thật không giảm rõ rệt. Không có cam kết dataset này tự động bảo đảm mức cải thiện cụ thể. Nếu tiếp tục chỉnh dựa vào test cuối, cần tập test mới để đánh giá công bằng.

## Bước 9. Tải weights và thử ứng dụng

Chạy cell export. Nó lưu `best_mafa_hand_<tên-lượt>.pt` lên Drive và tải xuống laptop. Giữ `experiment.json`, args.yaml, results.csv, biểu đồ và bảng đánh giá để ghi báo cáo.

Copy weights tải về thành `models/best_mafa_hand.pt` trong dự án. Không ghi đè model cũ ngay. Dừng server đang chạy, mở PowerShell tại Lab3:

```powershell
$env:MASK_MODEL_PATH = 'models/best_mafa_hand.pt'
.\.venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --workers 1 --ws-max-size 6000000
```

Ở terminal thứ hai:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
.\.venv\Scripts\python.exe -m client.app --config configs/webcam.json
```

Health phải ready=True và đúng ba nhãn. Thử không đeo, tay che miệng, tay che cả mũi/miệng, khẩu trang đúng, lộ mũi, kéo xuống cằm; đổi người/khoảng cách/ánh sáng. Ghi số lỗi và tốc độ CPU. Train trên GPU không đồng nghĩa ứng dụng laptop inference trên GPU; server hiện chạy CPU.

Nếu cần quay lại model cũ, dừng server, chạy `Remove-Item Env:MASK_MODEL_PATH -ErrorAction SilentlyContinue` trong terminal server rồi khởi động lại. Biến môi trường chỉ áp dụng terminal hiện tại; mở terminal mới sẽ cần đặt lại nếu dùng weights mới.

## Khi phiên Colab bị ngắt

Kết nối GPU lại, chạy các cell chuẩn bị môi trường/Drive/dataset. Dùng cùng bundle và giữ đường dẫn `/content/lab3_mafa`. Chạy cell mới với đúng last.pt:

```python
from ultralytics import YOLO
from datetime import datetime
from zoneinfo import ZoneInfo
BATCH = 8
LAST = Path('/content/drive/MyDrive/lab3_mafa/runs/TEN_LUOT/weights/last.pt')
resumed_model = YOLO(str(LAST))
resumed_model.train(resume=True)
BEST = Path(resumed_model.trainer.best)
RUN_DIR = Path(resumed_model.trainer.save_dir)
```

Chỉ resume lượt chưa hoàn thành với checkpoint hợp lệ. Nếu chưa hết epoch đầu thì có thể chưa có last.pt. Không dùng cell tạo lượt mới để resume. Notebook có hướng dẫn ở cuối.

Notebook đã được kiểm tra cú pháp cell và các script đã có kiểm tra tự động; chưa chạy train/GPU Colab thực tế vì dataset ảnh MAFA chưa được tải và duyệt.
