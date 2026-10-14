# Công việc và cách làm — Lab3 khẩu trang

> Bổ sung sau khi làm rõ yêu cầu: dùng `05_STREAMING_VA_ESP32_CAM.md` cho camera adapter, streaming/WebSocket, nhận dạng N mặt và tích hợp ESP32-CAM qua gateway. W09/W10/W11/W12/W13 cần áp dụng phần cập nhật đó; POST predict vẫn có thể giữ để debug và benchmark ảnh.

Đọc cùng `01_DAC_TA_DE_TAI.md` và `03_TEST_VA_NGHIEM_THU.md` trong thư mục này.

**Hiện có tài liệu kế hoạch, chưa có các script/module mô tả dưới đây.** Đường dẫn đầu ra là quy hoạch cho bước triển khai, không phải artifact đã tồn tại. Lệnh với script dự kiến chỉ chạy sau khi script đó được viết. Lệnh setup/train bằng thư viện là ví dụ thực hiện, chưa được chạy ở lượt lập kế hoạch.

## 1. Thứ tự và phụ thuộc

```text
W00 đặc tả → W01 môi trường → W02 nguồn dữ liệu
W02 → W03 chụp bổ sung → W04 nhãn YOLO → W05 chia và kiểm tra
W05 → W06 train thử → W07 train chính → W08 đánh giá và đóng gói
W08 → W09 API local → W10 GUI local → W11 Cloud
W11 → W12 test đầu-cuối → W13 đo hiệu năng → W14 báo cáo và demo
```

Có thể thiết kế schema API/giao diện khi đang chuẩn bị dữ liệu. Chưa có model thật thì kết quả giả chỉ dùng thử giao diện, phải ghi rõ; không coi là bằng chứng nhận dạng.

| ID | Công việc | Output chính | Điều kiện hoàn thành |
|---|---|---|---|
| W00 | Đặc tả đề tài | Spec, bộ nhãn và tiêu chí | Phạm vi thống nhất, đúng ba lớp |
| W01 | Môi trường | Venv, phiên bản, smoke report | Import/camera/model runtime hoạt động |
| W02 | Kiểm kê dataset nguồn | Raw data, sources, inventory | Nhãn/bbox/nguồn đã kiểm tra |
| W03 | Thu ảnh webcam bổ sung | Ảnh, metadata buổi chụp | Đa dạng, có lớp 2, giữ test riêng |
| W04 | Gán/chuyển nhãn YOLO | TXT labels, data YAML, overlay | Bbox đúng, class ID đúng |
| W05 | Chia tập và audit | Manifest, data audit | Không trùng split, đủ ba lớp |
| W06 | Train thử | Checkpoint nhỏ, smoke log | Pipeline train/val hoạt động |
| W07 | Train chính thức | best.pt, logs, curves | Có model lựa chọn bằng validation |
| W08 | Test model và đóng gói | Metrics, model metadata/hash | Test độc lập, artifact nạp lại được |
| W09 | API local | Server, JSON, API tests | Input/output/lỗi/timing đúng |
| W10 | GUI local | Client webcam, ảnh và CSV | Đúng frame, UI hoạt động khi API chậm |
| W11 | Deploy Cloud | Remote endpoint, Docker/logs | Cloud nạp đúng weights và suy luận |
| W12 | Test đầu-cuối | Functional report/evidence | Ba lớp cùng khung hình, lỗi được xử lý |
| W13 | Thí nghiệm độ trễ | Raw CSV, summaries/plots | Phép đo có định nghĩa, có số lỗi |
| W14 | Bài nộp | README, báo cáo, video, index | Tái chạy và đối chiếu yêu cầu được |

## 2. Quy hoạch thư mục triển khai

```text
Lab3/
  docs/                       đặc tả, kế hoạch, test và tiến độ
  datasets/
    raw/                      dữ liệu gốc giữ nguyên
    webcam_raw/               ảnh bổ sung theo người/buổi chụp
    mask_yolo/
      images/{train,val,test}/
      labels/{train,val,test}/
      data.yaml
    manifests/                nguồn, hash, group và split
  scripts/
    check_environment.py
    capture_dataset.py
    convert_voc_to_yolo.py
    audit_dataset.py
    split_dataset.py
    train.py
    evaluate.py
    benchmark.py
    summarize_benchmark.py
  server/
    app.py
    inference.py
    schemas.py
  client/
    app.py
    camera.py
    api_client.py
    metrics.py
  models/
    mask_v1/best.pt
    mask_v1/metadata.json
  tests/
    fixtures/
    test_dataset.py
    test_api_contract.py
    test_metrics.py
  deploy/
    Dockerfile
    README.md                 metadata và hướng dẫn Docker Space
  artifacts/
    environment/
    dataset/
    training/
    evaluation/
    tests/
    benchmark/
    demo/
  requirements-client.txt
  requirements-server.txt
  requirements-train.txt
  requirements-dev.txt
  README.md
```

Client không cần cài Ultralytics/PyTorch. Tách dependencies để chứng minh laptop chỉ chụp/gửi/hiển thị. Raw data, weights, venv và ảnh cá nhân không tự động đẩy lên kho công khai.

## W00 — Chốt đặc tả

**Đầu vào:** slide và lựa chọn khẩu trang của người học.

**Cách làm:** dùng spec đã lập, khóa mapping 0/1/2; xác định bounding box quanh mặt, cảnh ba người, Cloud thật, phép đo và tiêu chí. Ghi riêng câu hỏi độ trễ một chiều để trao đổi với giảng viên nếu cần.

**Output hiện có:** `docs/01_DAC_TA_DE_TAI.md`.

**Test:** đọc đối chiếu R01–R12; không còn lựa chọn trái cây/classification làm hướng chính. W00 pass ở mức kế hoạch, chưa phải nghiệm thu hệ thống.

## W01 — Chuẩn bị môi trường

**Đầu vào:** laptop Windows, webcam. Hiện kiểm tra được Python 3.13.9 và Docker executable; chưa xác nhận camera/GPU/dependencies.

**Cách làm:**

1. Chọn Python 3.11 cho dự án để thống nhất local/container; giữ Python hiện tại. Kiểm tra có runtime 3.11 trước khi dùng lệnh bên dưới.
2. Tạo venv trong workspace; gọi trực tiếp python của venv để tránh lỗi activation PowerShell.
3. Viết requirements cho client/server/train/dev. Client: opencv-python, Pillow, requests; Tkinter đi kèm bản Python phù hợp. Server: fastapi, uvicorn, python-multipart, ultralytics, Pillow, numpy và backend PyTorch. Dev: pytest, httpx. Dùng opencv-python-headless ở Cloud; không cài đồng thời hai biến thể OpenCV cùng namespace trong một env.
4. Cài PyTorch theo cấu hình CPU/GPU được hỗ trợ từ trang chính thức; không đoán bản CUDA dựa trên tên card.
5. Viết `check_environment.py`: ghi Python/package versions, import modules, GPU availability, camera index 0..2; kiểm tra mở/đọc và release camera.
6. Sau smoke pass mới khóa versions; ghi riêng dependencies client và server, không freeze tất cả gói của máy vào một file chung.

Ví dụ setup sau khi đã có Python và requirements:

```powershell
Set-Location 'D:\PTIT\N4\HK1\IoT\Lab3'
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-client.txt
.\.venv\Scripts\python.exe scripts/check_environment.py
```

**Output:** venv; `artifacts/environment/environment.json`; `camera_sample.jpg`; versions/lock files sau kiểm tra.

**Test:** T01–T03. Camera frame width/height > 0; đóng/mở camera lại được; model runtime CPU khởi tạo tensor và chạy được. Chỉ có executable Docker không tính Docker đã sẵn sàng.

**Nếu lỗi:** tắt ứng dụng đang giữ webcam, kiểm tra quyền camera Windows, thử camera index/backend khác; lỗi torch import kiểm tra wheel/Python/architecture; không có GPU thì train thử CPU và dùng GPU notebook cho lượt train chính.

## W02 — Chọn và kiểm kê dataset nguồn

**Cách làm:**

1. Xem nguồn ứng viên: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection. Nội dung thực tế, số ảnh, tên lớp và điều kiện sử dụng cần kiểm tra sau khi tải; trang nguồn không trả nội dung dataset trong phiên đọc hiện tại. Không điền số lượng theo trí nhớ.
2. Tải dataset do người học có quyền truy cập. Lưu nguyên bản dưới raw; ghi URL, tác giả, ngày tải và điều kiện sử dụng trong `sources.md`.
3. Kiểm tra annotation có bbox cho cả ba lớp. Nếu là Pascal VOC XML, dùng quy trình chuyển nhãn W04; nếu YOLO, kiểm tra mapping trước khi sử dụng.
4. Thống kê **số ảnh** và **số mặt mỗi lớp** riêng biệt. Một ảnh nhiều người không được tính là một mẫu cho từng người mà không đếm instances.
5. Đếm lớp 2, ảnh hỏng, nhãn lạ và bbox bất hợp lệ. Không bỏ lớp `mask_incorrect` để tăng điểm chung.
6. Xem ảnh mỗi lớp và chính sách nhãn nguồn; lọc hoặc sửa các ví dụ không phù hợp định nghĩa đã chốt.

**Output:** raw dataset; `artifacts/dataset/inventory.json`; `datasets/manifests/sources.md`; danh sách lỗi/cần review.

**Test:** T04–T05. Nguồn và nhãn có bằng chứng; cả ba lớp có instances hợp lệ; số lượng tính từ files thực tế. Nếu lớp 2 ít, bổ sung W03 trước train chính.

## W03 — Thu ảnh webcam và gán metadata

**Cách làm:**

1. Viết script capture lưu frame, không cần model. Tên file chứa session và số thứ tự; metadata có person/group ID, session, trạng thái dự kiến, điều kiện sáng, góc và khoảng cách.
2. Mỗi người chụp đeo đúng, không đeo, hở mũi, kéo xuống cằm; mặt phía trước và lệch nhẹ, hai nền, hai điều kiện sáng, hai khoảng cách khả dụng.
3. Thu ảnh khi tư thế thay đổi; không coi burst frame gần như giống nhau là dữ liệu đa dạng. Có thể chụp mỗi 1–2 giây nhưng vẫn phải xem và loại ảnh gần trùng.
4. Mục tiêu thu bổ sung ban đầu: khoảng 100–150 ảnh hữu ích mỗi lớp từ nhiều buổi; đây là mục tiêu kế hoạch, không bảo đảm chất lượng. Tăng lớp 2 nếu thống kê nguồn thiếu.
5. Dành riêng người hoặc buổi chụp chưa dùng train để đánh giá webcam cuối. Ví dụ bộ độc lập 30 ảnh mỗi lớp và 10 cảnh ba người; tránh dùng chúng để chọn threshold.
6. Thu 20 ảnh không có mặt để kiểm tra false positives. Thu nhiều người cùng ảnh cho bài detection; ảnh ghép có thể thử trước nhưng không thay thế demo camera thật.
7. Khi có người tham gia, thống nhất việc dùng ảnh cho bài lab. Không tự đưa ảnh cá nhân lên repository public; server mặc định không lưu ảnh nhận được.

**Output:** webcam_raw; capture manifest; ảnh độc lập giữ riêng, chưa đụng trong tune.

**Test:** T06. Metadata đủ để chia theo session/person; ảnh test không phải bản augmentation của train; có ví dụ lớp 2 hợp định nghĩa.

## W04 — Gán nhãn hoặc chuyển nhãn

**Cách làm:**

1. Vẽ bbox bao quanh từng khuôn mặt và gán class ID bằng công cụ annotation xuất YOLO. Gán tất cả mặt đủ rõ trong ảnh, không chỉ mặt chính.
2. Nếu XML: đọc width/height ảnh thực và bbox xmin/ymin/xmax/ymax. Kiểm tra quy ước nguồn là tọa độ 0-based hay 1-based trước khi chuyển.
3. Chuyển theo công thức với tọa độ đã chuẩn hóa đúng quy ước:

```text
xc = (xmin + xmax) / (2 * image_width)
yc = (ymin + ymax) / (2 * image_height)
w  = (xmax - xmin) / image_width
h  = (ymax - ymin) / image_height
```

4. Mỗi object một dòng `class_id xc yc w h`; class ID thuộc 0..2, giá trị hữu hạn, w/h > 0, bốn cạnh trong [0,1]. Không im lặng clamp mọi lỗi annotation.
5. Nhãn lạ hoặc XML lỗi: log và review, không tự bỏ object. Ảnh background đã review dùng file TXT rỗng để phân biệt với ảnh bị thiếu nhãn.
6. Kiểm tra ví dụ chuyển đổi bằng tay và overlay. Mục tiêu review ban đầu 30 ảnh mỗi lớp cùng mọi ảnh có cảnh báo.

Ví dụ một bbox [100,50,300,250] trên ảnh 640×480:

```text
0 0.312500 0.312500 0.312500 0.416667
```

Nội dung YAML dự kiến; `path` phải là đường dẫn dataset đúng trên máy hoặc notebook:

```yaml
path: D:/PTIT/N4/HK1/IoT/Lab3/datasets/mask_yolo
train: images/train
val: images/val
test: images/test
names:
  0: with_mask
  1: without_mask
  2: mask_incorrect
```

**Output:** label TXT, annotation policy, data.yaml, ảnh overlay và conversion report.

**Test:** T07–T10. Chuyển bbox khớp phép tính; mapping đúng; phát hiện class 3, NaN, zero area và nhãn thiếu; các mặt được gán đúng phạm vi.

## W05 — Chia train/validation/test và audit

**Cách làm:**

1. Tạo manifest gồm relative path, SHA256, source, person/session/group, split và counts mỗi lớp.
2. Group ảnh theo nguồn trùng, near-duplicate, video/session hoặc person nếu biết. Không cho cùng group sang nhiều split.
3. Chia hướng tới 70/15/15 bằng seed 42, nhưng giữ nguyên group và đủ support từng lớp quan trọng hơn tỷ lệ tuyệt đối.
4. Với public dataset không biết identity, ghi rõ giới hạn không chứng minh person-disjoint; tối thiểu kiểm tra hash/near-duplicates và giữ webcam test riêng.
5. Perceptual hash dùng phát hiện ứng viên gần trùng để review, không đủ để tự khẳng định không rò rỉ dữ liệu.
6. Test chưa đủ lớp 2: bổ sung nhóm dữ liệu mới và chia lại **trước khi** bắt đầu tune. Không copy một ảnh yếu sang cả train và test.
7. Audit ảnh/label tương ứng, file mở được, bounds, distribution và overlap; chỉ augment train sau khi chia.

**Output:** split_manifest; data.yaml; `data_audit.json`; bảng counts và ảnh minh họa.

**Test:** T11–T13. Không cùng hash/group xuyên split; val/test đều có cả ba lớp; mọi ảnh train có label đã review; dataset audit không có lỗi chặn.

## W06 — Train thử để kiểm tra pipeline

**Cách làm:**

1. Tạo subset nhỏ đã audit có đủ lớp; không dùng test cuối.
2. Chạy 1–3 epoch, batch nhỏ, workers 0 trên Windows nếu gặp multiprocessing.
3. Kiểm tra log có đúng ba names, samples tìm thấy, loss hữu hạn, weights sinh ra.
4. Load checkpoint và predict một ảnh. Nếu output nghèo sau vài epoch, chưa kết luận model sai; lượt này chỉ kiểm tra pipeline.

Ví dụ CLI sau khi đã cài thư viện/data:

```powershell
.\.venv\Scripts\yolo.exe detect train model=yolov8n.pt data=datasets/mask_yolo/data.yaml epochs=3 imgsz=640 batch=4 workers=0 device=cpu project=artifacts/training name=smoke
```

**Output:** smoke log/checkpoint. Đây chưa phải model dùng nghiệm thu.

**Test:** T14. Train và validation kết thúc, checkpoint load lại được, `names` khớp ID 0/1/2. CUDA OOM: giảm batch trước, không tự đổi test data.

## W07 — Huấn luyện chính thức và chọn cấu hình

**Cách làm:**

1. Bắt đầu `yolov8n.pt`, epochs tối đa 50, imgsz 640, batch 8 nếu GPU đủ, seed 42, patience 10. Các giá trị là baseline; lưu cấu hình thực sự.
2. Có GPU notebook thì chuyển dataset và sửa path đúng notebook; tải đầy đủ kết quả cuối về laptop. Không trộn package versions và weights giữa các lượt.
3. Theo dõi validation theo từng lớp, loss, precision/recall và mAP; `mask_incorrect` là lớp cần xem riêng.
4. Nếu chưa đạt validation: xem nhãn sai, thiếu dữ liệu, domain shift; bổ sung dữ liệu train/val phù hợp, rồi tạo run mới.
5. Chọn best checkpoint dựa trên validation. Chọn confidence trong khoảng thử 0.25/0.40/0.50 cũng trên validation; lưu quyết định và không dùng test chọn threshold.
6. Không cần chuyển ONNX ngay. Bắt đầu deployment bằng PT; tối ưu định dạng chỉ khi đo thật cho thấy cần và phải đánh giá lại model export.

Ví dụ GPU CLI, chỉ dùng `device=0` khi đã xác nhận GPU khả dụng:

```powershell
.\.venv\Scripts\yolo.exe detect train model=yolov8n.pt data=datasets/mask_yolo/data.yaml epochs=50 imgsz=640 batch=8 patience=10 seed=42 workers=0 device=0 project=artifacts/training name=mask_v1
```

**Output:** best.pt, last.pt, args.yaml, results.csv, curves/plots, model selection notes. Ghi `project/name` thực tế vì thư viện có thể tạo hậu tố nếu tên đã tồn tại.

**Test:** T15. Loss không NaN; names đúng; validation đủ ba lớp; biết lý do chọn checkpoint. Chưa gọi kết quả validation là test accuracy.

## W08 — Đánh giá model và đóng gói

**Cách làm:**

1. Khóa weights, preprocessing, imgsz, threshold và NMS setting đã chọn.
2. Evaluate split test bằng Ultralytics; lưu mAP50, mAP50-95, precision/recall theo lớp, counts và confusion matrix.
3. Lưu ảnh false positives, false negatives và nhầm class; giải thích ít nhất vài lỗi cụ thể.
4. Chạy thêm bộ webcam test độc lập bằng cấu hình deployment, match một-một IoU >= 0.5 và ghi FP/FN. Không dùng cấu hình mAP chuẩn để ngầm thay cho kiểm tra threshold của ứng dụng.
5. Copy best.pt vào models/mask_v1; metadata chứa SHA256, names, model architecture, package versions, dataset version, seed, imgsz, conf, iou, training/evaluation references.
6. Load lại đúng file đóng gói, predict cùng fixture để kiểm tra tính toàn vẹn. Dùng weights tự tạo hoặc nguồn đáng tin khi nạp PT.

Ví dụ evaluate test:

```powershell
.\.venv\Scripts\yolo.exe detect val model=artifacts/training/mask_v1/weights/best.pt data=datasets/mask_yolo/data.yaml split=test imgsz=640 device=cpu project=artifacts/evaluation name=test_mask_v1
Get-FileHash -LiteralPath 'models\mask_v1\best.pt' -Algorithm SHA256
```

**Output:** metrics_test.json, report/errors, weights và metadata.

**Test:** T16–T18. Test độc lập, names/hash đúng, file đóng gói có cùng dự đoán trong tolerance. Ngưỡng chất lượng xem spec; nếu chưa đạt phải ghi fail/mục cần cải thiện.

## W09 — Viết API và kiểm tra local

**Cách làm:**

1. Viết schemas theo spec; names lấy từ model, kiểm tra bằng mapping đã chốt.
2. Lifespan nạp weights và warm-up trước ready; lỗi thiếu model không chạy ngầm bằng model COCO mặc định.
3. Viết GET health và POST predict; kiểm tra file decode, kích thước bytes/pixels, MIME chỉ là gợi ý chứ không đủ để tin file.
4. Lock model, CPU batch 1, Uvicorn một worker; ghi timing bao quanh các đoạn đã định nghĩa và Results.speed.
5. Trả bbox theo original input image; counts tính từ detections cuối. Không round timing quá sớm.
6. Viết API contract tests cho ảnh hợp lệ, ảnh trống, lỗi đầu vào và model không sẵn sàng. Unit test có thể mock detector để cô lập protocol; integration test bắt buộc dùng weights thật.
7. API có thể nhận token từ biến môi trường nếu dùng endpoint cần hạn chế truy cập. Không ghi token vào CSV/log/source.

Lệnh dự kiến sau khi module được viết:

```powershell
.\.venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --workers 1
curl.exe http://127.0.0.1:8000/health
curl.exe -F "file=@tests/fixtures/three_people.jpg" -F "request_id=local-001" http://127.0.0.1:8000/predict
.\.venv\Scripts\python.exe -m pytest tests/test_api_contract.py -q
```

**Output:** API code, tests, response fixtures, local test report.

**Test:** T19–T25. Handler không crash với ảnh lỗi; no-detection HTTP200; class/count/bbox hợp lệ; restart model cùng hash. Mốc timing thành phần không bị gọi sai tên.

## W10 — Giao diện webcam local

**Cách làm:**

1. Camera worker giữ frame mới nhất, copy frame khi tạo request; lấy L0/L1/L2/L3 đúng định nghĩa.
2. GUI có preview camera và ảnh kết quả riêng; màu xanh/đỏ/vàng cho ba nhãn kèm chữ để không phụ thuộc màu.
3. Network worker nhận frame, encode JPEG, gửi multipart với ID. Chỉ tối đa một request pending; cập nhật kết quả qua queue cho GUI.
4. Khi response về, verify request_id và kích thước ảnh; vẽ bbox lên frame đã gửi. Mirror/resize phải áp dụng đồng nhất hoặc tắt trong bản đầu.
5. Hiển thị timing, counts, model version, trạng thái. Nút Stop không nhận thêm frame mới; kết quả đến sau khi đã stop/đổi run bị bỏ nếu không còn hợp lệ.
6. CSV ghi cả success và failure, timing không có thì dùng null/ô trống. Không tạo thread mới vô hạn sau timeout.
7. Đóng app giải phóng camera và requests session; không để worker cập nhật Tk widget đã bị hủy.

Lệnh dự kiến: `.\.venv\Scripts\python.exe -m client.app`.

**Output:** client code; snapshot; demo local; CSV mẫu thật.

**Test:** T26–T30. API chậm 5 giây vẫn thao tác GUI được; queue không tăng; đúng frame; stop/restart và camera failure không treo; export đủ rows và ID.

## W11 — Đưa server và model lên Cloud

**Cách làm:**

1. Chỉ triển khai sau khi API local đã pass. Kiểm tra tài khoản/service đáp ứng Docker và tài nguyên cần dùng.
2. Docker dùng Python 3.11, requirements-server đã khóa; cài biến thể headless và system libs cần thiết; chạy non-root phù hợp dịch vụ.
3. Copy server và model hoặc tải từ kho model có version cố định lúc startup. Không copy datasets, client, ảnh cá nhân hay secrets vào image.
4. Uvicorn bind 0.0.0.0:7860, một worker. Metadata Space `sdk: docker`, `app_port: 7860`.
5. Kiểm tra Docker daemon trước build. Lệnh build ở workspace: `docker build -f deploy/Dockerfile -t lab3-mask:dev .`; run và kiểm tra health local trước upload.
6. Deploy vào Docker Space; nếu repo public, kiểm tra những file được công khai. Token/secrets dùng cấu hình secrets, không ghi vào Dockerfile.
7. Lấy HTTPS URL thật từ service; client gọi health, so names/version/hash; gửi fixture như local.
8. Restart Cloud để xác nhận model vẫn có và nạp đúng. Lưu log/load time, hardware, region nếu biết và giới hạn sleep/cold-start.

**Output:** Dockerfile, deployment README, URL thật trong config phù hợp, health JSON, startup log và remote prediction.

**Test:** T31–T33. Cloud khớp hash local; có suy luận thật; reboot không mất model. Không coi một mock endpoint có HTTP200 là test Cloud pass.

## W12 — Kiểm tra đầu-cuối

**Cách làm:**

1. Dùng webcam, GUI và endpoint remote thật; không dùng route local khi ghi video nghiệm thu Cloud.
2. Test từng lớp 30 ảnh độc lập, 10 cảnh đủ ba người và 20 cảnh không có mặt theo spec. Lưu input/response/overlay/ground truth đã chuẩn bị.
3. Test người đổi vị trí và thay khẩu trang; khung bao phải đi theo ảnh mới, không tái dùng kết quả cũ.
4. Test timeout, network disconnect, server503 và ảnh lỗi. Tắt stream sau test, kiểm tra không giữ camera.
5. Ghi pass/fail/blocked với expected/actual và evidence path. Chưa có ba người thì test camera đa lớp đánh dấu blocked; ảnh ghép chỉ tính fixture integration.

**Output:** functional_report.md; evidence index; ảnh và video ngắn.

**Test:** T34–T37. Dùng mẫu và ngưỡng đã chốt; lỗi mạng không làm kết quả cũ trông như mới; demo đủ ba lớp trên camera thật.

## W13 — Đo hiệu năng và tổng hợp

**Cách làm:**

1. Ghi model hash, package versions, hardware CPU/GPU, network, API endpoint, conf/iou, imgsz, JPEG quality và concurrency.
2. Tách hai bài đo: A) API benchmark dùng cùng bộ ảnh cố định để so sánh upload; B) GUI/webcam để đo end_to_end trong ứng dụng. API benchmark không có chụp/render thì end_to_end để trống, không điền request_ms thay thế.
3. Mỗi cấu hình warm-up 10 request rồi đo 100 **lần thử** tuần tự; lưu mọi failure. Mục tiêu >=99 success/100 thử. Tách cold-start khỏi thống kê steady-state.
4. So ảnh upload 320×240 và 640×480 JPEG80; giữ model imgsz640. Thử thêm imgsz320 là thí nghiệm riêng vì thay đổi cả độ chính xác model.
5. Đối với GUI, thu ít nhất 100 lượt tương ứng mỗi cấu hình; ghi frame_id và user-visible latency. Theo dõi missing data.
6. Raw CSV có run_id, request_id, mode, resolution, bytes, model config, timestamps/durations, status và error_code. Failed request vẫn có elapsed và status, model timing trống.
7. Tổng hợp trên samples thành công: mean/median/p95/min/max, n_success; báo n_attempts và failure_rate riêng. Dùng numpy.percentile(method='linear') và ghi phương pháp.
8. Nếu model/network quá chậm, phân tách bottleneck trước tối ưu. Giảm JPEG payload/độ phân giải có thể giảm chất lượng; phải kiểm tra lại fixture và metrics.
9. Lưu biểu đồ theo cấu hình và phân bố latency bằng matplotlib. Không dùng số trong ví dụ API làm số liệu thực nghiệm.

Lệnh dự kiến sau khi viết scripts:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py --config experiments/cloud_640.json --attempts 100 --warmup 10
.\.venv\Scripts\python.exe scripts/summarize_benchmark.py --input artifacts/benchmark/run01.csv
```

**Output:** experiment config, raw CSV, summary JSON, bảng/biểu đồ và ghi chú cold-start.

**Test:** T38–T41. Unit test durations/percentile bằng dữ liệu nhỏ biết đáp án; csv không mất lỗi; timing thành công hợp lý, không timestamp khác máy; mỗi summary truy ra raw dataset.

## W14 — Báo cáo, demo và tái chạy

**Cách làm:**

1. Viết README từ môi trường mới: tạo venv, dependencies, tải/đặt model, chạy API local, GUI, thay remote URL và benchmark.
2. Báo cáo theo spec: bài toán, nguồn dữ liệu, split, model/train, kiến trúc, API/UI, chất lượng, phương pháp đo, kết quả thật, hạn chế.
3. Video: mở GUI và health/version; demo từng lớp, cả ba lớp, thay đổi trạng thái, hiển thị timing, một lỗi mạng có xử lý. Video 3–5 phút là gợi ý, theo yêu cầu lớp nếu khác.
4. Đối chiếu R01–R12 với evidence paths, đánh dấu còn thiếu. Có PDF/DOCX/bộ slide chỉ khi cần định dạng nộp; hiện chưa tạo các artifact này.
5. Thử theo README trong venv sạch, có thể cùng máy nhưng folder/env riêng. Kiểm tra startup không cần mạng để tải weights ngầm nếu model đã đóng gói.

**Output:** README, report, video, evidence index, reproducibility result.

**Test:** T42–T43. Tái chạy không dựa vào paths/secrets riêng trên máy tác giả; bài nộp dùng số đo thật, có đủ requirement mapping.

## 3. Các cổng kiểm tra trước khi chuyển giai đoạn

- G1 sau W05: data audit pass, lớp 2 hiện diện và tập test được giữ riêng.
- G2 sau W08: model nạp được, names/hash đúng, biết chất lượng từng lớp.
- G3 sau W10: API/GUI local pass; đúng frame và timing.
- G4 sau W12: Cloud và demo ba lớp pass; lỗi được ghi rõ.
- G5 sau W14: đủ dữ liệu thô, báo cáo, demo, tái chạy và đối chiếu yêu cầu.

Không ấn định thời gian train hoặc ngày hoàn thành khi chưa biết hardware/dataset. W00 hiện đã lập; W01 là bước thực thi đầu tiên tiếp theo.

## 4. Phần bổ sung nếu phải đo local tới Cloud một chiều

Ba số đo chính đã chốt ở spec. Nếu cần số đo một chiều riêng, thực hiện phần bổ sung dưới đây trước kết luận R10 đã bao phủ cách hiểu đó.

**Output dự kiến:** endpoint `/time-sync`, offset log, arrival timestamps, `upload_to_handler_estimate_ms` cùng khoảng bất định và mô tả ranh giới đo.

**Cách thực hiện:**

1. Endpoint time-sync ghi s2 khi vào handler và s3 ngay trước chuẩn bị trả response, bằng UTC epoch time nanosecond. Client ghi l1 ngay trước gọi HTTP và l4 ngay sau nhận đủ response. Payload sync nhỏ; vẫn chịu proxy/serialization.
2. Với mỗi probe, offset server-minus-client nằm xấp xỉ trong khoảng [s3-l4, s2-l1] nếu wall clocks không nhảy/drift đáng kể và các điểm timestamp đúng thứ tự.
3. Dùng midpoint làm ước lượng offset; nửa độ rộng khoảng là bất định tối thiểu theo mô hình đo. Tính nhiều probe, khoảng 20 lần trước và sau run; lưu từng probe và chọn probe có khoảng hẹp, không bỏ raw data.
4. Với request ảnh, l_send là thời gian local ngay trước gọi HTTP; s_received là timestamp Cloud sau khi đã đọc xong bytes ảnh trong handler. Ước lượng `s_received - offset_estimate - l_send`; báo khoảng từ `s_received - offset_upper - l_send` tới `s_received - offset_lower - l_send`.
5. Điểm s_received này gồm upload, proxy, multipart parsing và chờ endpoint; tên số đo phải ghi rõ local gửi tới ứng dụng nhận xong ảnh. Không gọi là thời gian đường truyền thuần túy.
6. So wall-clock elapsed với perf_counter elapsed để phát hiện đồng hồ nhảy; drift trước/sau run lớn hoặc interval không hợp lệ thì đánh dấu sample không hợp lệ, không âm thầm sửa thành0.
7. Nếu bất định quá lớn so với độ trễ hoặc giảng viên yêu cầu timestamp network/proxy chính xác, dùng server cho phép instrumentation ở ingress/đồng bộ đồng hồ thích hợp. Docker Space thông thường không tự cung cấp đầy đủ ranh giới hạ tầng đó.

**Test bổ sung:** với l1=0,s2=130,s3=140,l4=70ms và offset thật100ms, khoảng offset=[70,130]ms, midpoint100ms, half-width30ms. Nếu request ảnh có l_send=200,s_received=350ms, estimate=50ms, khoảng[20,80]ms. Ghi đây là test toán của công thức bằng dữ liệu giả, không phải độ trễ Cloud đã đo.

Phần này được coi là nghiệm thu bổ sung chỉ khi có dữ liệu thật, clock stability check và báo cáo uncertainty; chưa triển khai ở giai đoạn lập kế hoạch.
