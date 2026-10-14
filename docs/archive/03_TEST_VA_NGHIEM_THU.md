# Ma trận test và nghiệm thu — Lab3 khẩu trang

> Bổ sung streaming/IoT: các test S01–S08 và H01–H06 nằm trong `05_STREAMING_VA_ESP32_CAM.md`. Cảnh ba người là kiểm tra tối thiểu đủ nhãn; phải thêm cảnh nhiều người và kiểm tra chạy liên tục. Điều chỉnh transport/timing tests theo WebSocket khi triển khai.

Trạng thái: **đã lập test plan, chưa chạy test hệ thống**. Không có dataset/model/API/GUI thật trong các artifact được tạo ở lượt này.

## 1. Cách ghi kết quả test

Mỗi test có:

```text
test_id:
run_id / ngày giờ:
model_sha256 / code version / environment:
input hoặc fixture:
cấu hình:
expected:
actual:
status: NOT_RUN | PASS | FAIL | BLOCKED
evidence: đường dẫn ảnh / JSON / log / video
ghi chú và hướng xử lý:
```

- NOT_RUN: chưa thực hiện.
- PASS: actual đáp ứng expected, có bằng chứng.
- FAIL: đã thực hiện nhưng không đạt; giữ bằng chứng lỗi.
- BLOCKED: thiếu điều kiện thật, như chưa có model hoặc chưa có ba người tham gia.
- Các test model/camera dùng fixture thật. Mock chỉ dùng test giao thức/lỗi có thể kiểm soát; không thay thế integration và nghiệm thu model.
- Khi model, nhãn, preprocessing hoặc threshold thay đổi, ghi version mới và chạy lại các test chịu ảnh hưởng.
- Khi thay đổi API schema, chạy lại contract và client integration. Chỉ sửa văn bản kế hoạch không cần chạy lại train/model.

## 2. Fixture cần chuẩn bị

| Fixture | Nội dung | Dùng cho |
|---|---|---|
| face_correct.jpg | Một mặt đeo đúng | API/model cơ bản |
| face_none.jpg | Một mặt không đeo | API/model cơ bản |
| face_nose_exposed.jpg | Khẩu trang hở mũi | Kiểm tra lớp 2 |
| face_chin.jpg | Khẩu trang kéo xuống cằm | Kiểm tra lớp 2 theo policy |
| three_people.jpg | Ba mặt, mỗi lớp một mặt | Integration ba nhãn |
| two_same_class.jpg | Hai mặt cùng lớp | Detection không phải single-classification |
| no_face_01.jpg ... no_face_20.jpg | Cảnh không có mặt | False positives |
| corrupt.jpg | Bytes không phải ảnh hợp lệ | API decode error |
| empty.jpg | 0 bytes | API validation |
| oversized.bin | File vượt giới hạn bytes | API413 |
| too_many_pixels.png | Ảnh nén nhỏ nhưng pixel vượt giới hạn | API pixel guard |
| three_people_gt.json | Nhãn/bbox ground truth độc lập | Matching IoU |
| synthetic_api_result.json | JSON giả có schema rõ | Unit test GUI/contract, không nghiệm thu model |

Ảnh unit test cấu trúc có thể được tạo chỉ để kiểm tra decode/kích thước. Ảnh nghiệm thu nhận dạng phải có nội dung thực tế và ground truth đã review. Các fixture minh họa và webcam test độc lập tách riêng để không tune bằng bộ nghiệm thu cuối.

## 3. Ma trận test chi tiết

### Môi trường và dữ liệu

| ID | Phương pháp | Cách test cụ thể | Expected | Evidence |
|---|---|---|---|---|
| T01 | Tự động | Import dependencies đúng env; ghi Python, torch, ultralytics, CPU/GPU | Import thành công; phiên bản tái lập được; GPU báo đúng availability | environment.json và console log |
| T02 | Webcam thật | Mở camera, đọc 10 frame, kiểm tra shape, save một frame, release | Có frame hợp lệ, ảnh nhìn được; release xong mở lại được | camera_sample.jpg và log |
| T03 | Local/Docker | CPU tensor op; nạp runtime model; nếu dùng Docker chạy docker info và container smoke | Runtime chạy được; Docker daemon sẵn sàng nếu cần build local | smoke log |
| T04 | Tự động + review | Đếm file ảnh/annotation và instances từng class | Số thực tính từ files; đủ cả ba lớp; lỗi không bị bỏ im lặng | inventory.json |
| T05 | Review | Xem nguồn/license và 10 ví dụ mỗi lớp; đối chiếu annotation policy | Nguồn ghi được; nhãn nguồn phù hợp hoặc có map/sửa được giải thích | sources.md, review log |
| T06 | Review + audit | Kiểm tra metadata ảnh webcam và reserved test | Có person/session/group; test riêng; lớp2 có hở mũi và cằm nếu demo cả hai | capture manifest |
| T07 | Unit | Bbox [100,50,300,250] trên 640×480, class0 chuyển YOLO | xc=.3125, yc=.3125, w=.3125, h=.4166667 trong tolerance 1e-6 | pytest log |
| T08 | Unit | Map with_mask, without_mask, mask_weared_incorrect và tên lạ | IDs0/1/2; tên lạ báo lỗi cần review | pytest log |
| T09 | Unit | Label class3, NaN, w=0, cạnh ngoài ảnh, label thiếu | Audit phân biệt từng lỗi, không cho dữ liệu hỏng qua gate | audit error fixtures và report |
| T10 | Review | Overlay ít nhất 30 ảnh mỗi lớp và toàn bộ ảnh warning | Box quanh mặt đúng; nhãn khớp; tất cả mặt đủ rõ được gán | overlay gallery và checklist |
| T11 | Unit + dataset audit | Cố ý đặt cùng SHA256/group vào train và test, rồi kiểm tra bộ thật | Bộ giả bị chặn; bộ thật không có overlap exact/group | split audit |
| T12 | Dataset audit | Đếm instances từng lớp mỗi split và background đã review | Train/val/test đều có cả ba lớp, đủ số mẫu để tính từng lớp | distribution table |
| T13 | Dataset audit | Quét near-duplicates, augmentation gốc và session; kiểm tra quyền tồn tại paths | Các cặp nghi ngờ đã review; không rò rỉ group biết được; YAML paths đúng | manifest, review decisions |

T03 không tự suy ra model nhận dạng tốt. T12 kiểm tra support, không đặt một số nhỏ tùy ý rồi khẳng định đủ tổng quát. Nếu chưa đủ, bổ sung dữ liệu trước train chính.

### Model

| ID | Phương pháp | Cách test cụ thể | Expected | Evidence |
|---|---|---|---|---|
| T14 | Integration train | Train thử 1–3 epoch với subset đủ lớp; reload checkpoint | Train/val không lỗi, loss hữu hạn, names đúng | smoke log, weights |
| T15 | Review train chính | Kiểm tra args, logs, curves, validation support và checkpoint selection | Cấu hình đầy đủ, best chọn bằng validation; không dùng test tune | train artifacts và selection.md |
| T16 | Model test độc lập | Evaluate split test, xem mAP/precision/recall từng lớp | Báo đủ metrics/support; mục tiêu mAP50>=.80, từng lớp P/R>=.70 | metrics_test.json và plots |
| T17 | Artifact integrity | Hash file trước/sau đóng gói; load model và names | SHA256 và class mapping khớp metadata, không lẫn COCO weights | model metadata + hash log |
| T18 | Model regression | Predict cùng fixture với weights trước/sau copy và runtime deployment | Kết quả tương đương theo tolerance đã chọn; không mất lớp2 | JSON comparison |

T16 có thể FAIL mục tiêu định lượng dù pipeline hoạt động. Khi đó phân tích dữ liệu/nhầm lớp và quyết định làm run mới; không trình bày validation thay test. Với T18, so tolerance phù hợp CPU/GPU và runtime; class/count thay đổi do sát threshold phải được review, không yêu cầu confidence float khớp tuyệt đối mọi phần cứng.

### API

| ID | Phương pháp | Cách test cụ thể | Expected | Evidence |
|---|---|---|---|---|
| T19 | Contract + integration | GET health sau startup; POST ảnh thật/request_id | HTTP200; names/hash/version đúng; ID trả lại khớp | health và response JSON |
| T20 | Contract unit | Detector stub trả []; gửi ảnh hợp lệ; sau đó test ảnh no-face với model thật | Protocol200 + []; model FP đánh giá riêng; count zero khi list rỗng | unit log + integration JSON |
| T21 | Contract tự động | Thiếu file, file0bytes, ảnh hỏng, MIME giả | Thiếu file422; ảnh hỏng400; không crash; MIME giả không vượt decode check | HTTP/error report |
| T22 | Contract tự động | File >5MiB và ảnh >4096×4096 pixels | HTTP413, tránh chạy model cho input bị từ chối | HTTP/error log |
| T23 | Contract tự động | Xác thực từng bbox/class/confidence/count trên response | IDs0..2, confidence0..1 hữu hạn, box dương trong ảnh, counts khớp detections | schema assertion log |
| T24 | Timing integration | 10 predict, ghi Results.speed và timers handler/pipeline | Không âm/NaN; tên đúng phạm vi; inference không bị gán request wall time | timing JSON và code reference |
| T25 | Lỗi/integration | Model thiếu hoặc names sai khi khởi động; trạng thái unready nếu server còn phục vụ; token sai nếu bật auth | Startup từ chối model sai, không fallback COCO; health503 khi unready; auth lỗi như spec | startup/error log |

Giới hạn bytes tại handler chỉ chứng minh handler chặn inference; proxy/framework có thể đã nhận file trước đó. Nếu cần chặn upload trước parsing phải cấu hình ingress/body-size limit riêng. Test không suy diễn phạm vi bảo vệ vượt phần đã triển khai.

### Giao diện và camera

| ID | Phương pháp | Cách test cụ thể | Expected | Evidence |
|---|---|---|---|---|
| T26 | GUI/integration | GUI gửi frameA, di chuyển người để camera thành frameB trước response | Kết quả vẽ trên frameA có ID tương ứng, không overlay nhầm frameB | input/response/overlay hoặc video |
| T27 | GUI/unit + review | Resize ảnh có bbox biết trước; bật mirror nếu có | Box biến đổi cùng ảnh, nhãn không bị cắt khỏi vùng hiển thị | coordinate assertions và screenshot |
| T28 | GUI/stub chậm | Endpoint delay5s; bấm Stop, resize, đóng/mở app; quan sát pending | UI phản hồi; tối đa một request pending; không queue vô hạn | video, pending/worker log |
| T29 | GUI/lỗi thật | Camera không mở, timeout, network error, server503, JSON lỗi | Trạng thái rõ; không hiển thị kết quả cũ như mới; có retry chủ động | video/screenshots và error CSV |
| T30 | GUI/log | Start/stop/restart2 lần; export CSV; close rồi mở camera lại | Không worker dư/camera bị giữ; ID không trùng trong run; mọi attempt có record | CSV audit + log |

### Cloud và nhận dạng đầu-cuối

| ID | Phương pháp | Cách test cụ thể | Expected | Evidence |
|---|---|---|---|---|
| T31 | Remote integration | Laptop gọi HTTPS health/predict; so hash/names local | Remote thật, đúng artifact, response đúng schema | URL/config, JSON và Cloud log |
| T32 | Cloud restart | Restart container/service, gọi health rồi predict | Model nạp lại cùng hash; ready sau warm-up; cold-start ghi riêng | startup log trước/sau |
| T33 | Kiến trúc + integration | Chạy client env không chứa Ultralytics/torch, request có ID được ghi ở Cloud | Client vẫn nhận diện qua Cloud; không inference trên laptop | client deps, Cloud request log, response |
| T34 | Webcam độc lập | Mỗi lớp30 ảnh, đã định ground truth; matching IoU>=.5 một-một | Mục tiêu>=27/30 ảnh đúng mỗi lớp; ghi thêm TP/FP/FN | image/GT/prediction index, report |
| T35 | Webcam ba người | 10 ảnh/cảnh độc lập có ba lớp; tất cả mặt có GT | Mục tiêu>=9/10 cảnh phát hiện/nhãn đúng cả ba, không mặt thừa | 10 bộ input/GT/JSON/overlay + video |
| T36 | Negative scene | 20 cảnh không có mặt, nhiều nền/vật khác nhau | Mục tiêu tối đa1/20 ảnh có FP; không ép một nhãn khi rỗng | raw images và predictions |
| T37 | Robustness và lỗi | Thử nghiêng/xa/tối nhẹ, che khuất; ngắt mạng rồi kết nối lại | Báo kết quả thật kể cả bỏ sót; no crash; error state/recovery đúng | failure gallery và video |

T34: từng ảnh single-face pass nếu có đúng một detection match đúng lớp và IoU>=.5, không detection thừa. T35: mỗi scene pass nếu cả ba mặt match đúng lớp một-một và không detection thừa. Nếu frame có thêm mặt ngoài ba người, phải gán GT cho tất cả hoặc không dùng làm fixture ba mặt.

T37 không hứa nhận dạng đúng mọi ánh sáng/góc nhìn; ghi điều kiện thất bại và phạm vi ứng dụng. Nếu dùng ảnh ghép để thử T35 sơ bộ, giữ kết quả đó ở integration fixture; webcam demo cuối vẫn chưa pass cho tới khi có cảnh thật.

### Đo thời gian và hoàn thiện bài

| ID | Phương pháp | Cách test cụ thể | Expected | Evidence |
|---|---|---|---|---|
| T38 | Unit timing | Mốc local tương đương L0=0,L1=20,L2=120,L3=150ms; đơn vị thực trong ns | request100ms; toàn trình150ms; không trừ đồng hồ server | pytest log |
| T39 | Unit thống kê | Dãy thành công [100,200,300,400,500]ms; percentile linear | mean300, median300, p95=480; n_success5 | pytest log |
| T40 | Benchmark audit | N_attempts100 gồm98 success2 timeout; raw CSV đủ100 row | failure_rate2%; stats model chỉ98success; errors không mất | raw/summary comparison |
| T41 | Thực nghiệm | warm-up riêng,100attempts/cấu hình; GUI và API mode riêng; model cố định | Metadata đầy đủ; null đúng trường; request<=toàn trình với samples GUI hợp lệ | configs, CSV, summary/plots |
| T42 | Tái chạy | Venv sạch theo README; đặt model; chạy API/client/fixture | Không phụ thuộc package/path ẩn; names/hash đúng; fixture pass | reproducibility log |
| T43 | Review bài nộp | Đối chiếu R01–R12, số đo/metrics về raw data, video và sources | Bằng chứng truy được; không claim output chưa tạo hoặc test chưa chạy | requirement mapping/evidence index |

Với T41, thời gian clock/timer có nhiễu và các timer thành phần khác ranh giới; chỉ đặt tolerance phù hợp, không đòi tổng thành phần bằng chính xác wall time. Nếu đo `request < server_processing` đáng kể, review ranh giới/unit/ID và loại khỏi summary hợp lệ sau khi lưu nguyên bản cùng lý do, không xóa lỗi im lặng.

## 4. Quy trình matching và chất lượng

IoU là diện tích giao hai box chia diện tích hợp. IoU>=0.5 là quy ước kiểm tra demo, không phải confidence>=0.5.

1. Đọc GT theo ảnh và prediction sau threshold/NMS đã khóa.
2. Ghép prediction-GT một-một theo nhãn và IoU; mỗi prediction và GT chỉ dùng một lần.
3. Prediction đúng lớp và đủ IoU là TP.
4. Prediction không match là FP; GT không match là FN. Sai nhãn có thể tạo một FP và một FN trong đánh giá từng lớp.
5. Tính precision=TP/(TP+FP), recall=TP/(TP+FN); nếu mẫu số0 ghi không xác định theo quy ước đánh giá, không tự điền1.
6. Báo support mỗi lớp để thấy chỉ số trên ít mẫu không ổn định.

mAP dùng bộ evaluator tiêu chuẩn của thư viện, không thay bằng phần trăm ảnh demo đúng. Với detection, không gọi confidence hoặc mAP là classification accuracy.

## 5. Thí nghiệm hiệu năng

| Run | Input upload | Model imgsz | Mode | Warm-up | Lượt thử chính |
|---|---|---|---|---|---|
| P01 | 640×480 JPEG80 | 640 | API cố định ảnh | 10 | 100 |
| P02 | 320×240 JPEG80 | 640 | API cùng bộ ảnh resize | 10 | 100 |
| P03 | 640×480 JPEG80 | 640 | Webcam + GUI | 10 | 100 |
| P04 | 320×240 JPEG80 | 640 | Webcam + GUI | 10 | 100 |

Giữ model/hash, CPU/GPU, network, concurrency1 và threshold/NMS. So sánh P01/P02 chủ yếu payload; không khẳng định model chính xác như nhau khi ảnh bị giảm độ phân giải. Mỗi run lưu số bytes từng JPEG.

Mục tiêu trải nghiệm: lỗi<=1% và GUI p95 toàn trình<=2000ms trong điều kiện ghi nhận. Đây là mục tiêu kế hoạch; kết quả thực phải ghi cả khi không đạt. Không có mục tiêu FPS webcam30 ở phiên bản này.

CSV dự kiến:

```text
run_id,request_id,mode,model_sha256,imgsz,confidence,iou,width,height,jpeg_bytes,
decode_ms,preprocess_ms,inference_ms,postprocess_ms,model_pipeline_ms,
server_processing_ms,request_ms,end_to_end_ms,attempt_elapsed_ms,status,error_code
```

Không có model timing ở timeout; để trống. API-only benchmark không có end_to_end GUI; để trống. Cột UTC/local date dành ghi log, không dùng tính thời gian qua máy.

## 6. Kiểm tra hoàn thành cuối

### Phải có trước khi gọi hệ thống hoàn chỉnh

- [ ] R01–R12 có evidence tương ứng.
- [ ] Dataset audit/nhãn/split pass.
- [ ] Weights đã train, names/hash đúng.
- [ ] Test model độc lập có đủ ba lớp và kết quả thật.
- [ ] Cloud chạy đúng model, client không inference.
- [ ] Webcam demo cùng khung hình ba trạng thái.
- [ ] API/UI xử lý timeout, empty detection, ảnh lỗi và stop/close.
- [ ] Ba số đo inference/request/toàn trình có định nghĩa và dữ liệu thô.
- [ ] Tình trạng đo local->Cloud một chiều được ghi rõ theo yêu cầu giảng viên.
- [ ] Chỉ tiêu chất lượng và hiệu năng được đánh dấu đạt/chưa đạt đúng thực tế.
- [ ] README tái chạy được, báo cáo/video hoàn chỉnh.

### Khi gặp fail

- Dữ liệu fail: sửa trước train chính, không giải quyết bằng đổi nhãn hiển thị.
- Lớp2 recall yếu: review annotation và bổ sung nhóm dữ liệu train; không đổi ID để dùng nhãn khác.
- Nhiều box cùng một mặt: review NMS/tune trên validation, kiểm tra mặt gần nhau không bị mất.
- Cloud slow: tách decode/model/network; giảm payload hoặc optimize runtime, rồi retest chất lượng.
- GUI treo: sửa threading/queue và test chậm; không che lỗi bằng bỏ hiển thị timing.
- Không đủ người demo: đánh dấu camera multi-person BLOCKED; giữ fixture sơ bộ và bố trí cảnh thật sau.

## 7. Test bổ sung cho độ trễ một chiều khi áp dụng

Ngoài T01–T43 cốt lõi, phần one-way có các kiểm tra điều kiện riêng, chưa chạy:

- OW01: unit test offset bằng ví dụ trong mục4 của tài liệu công việc; kết quả midpoint100ms, khoảng[70,130]ms, estimate nhận ảnh50ms và khoảng[20,80]ms.
- OW02: fixture mô phỏng clock nhảy khiến interval sai/thời gian wall lệch monotonic; sample bị đánh dấu invalid, không clamp0.
- OW03: trên Cloud thật chạy probes trước/sau run, lưu tất cả timestamps/interval và ước lượng drift. Interval và clock stability phải được báo cáo.
- OW04: mỗi request có upload-to-handler estimate/uncertainty, đúng request_id; báo cáo nêu cả proxy/parsing nằm trong số đo và không tuyên bố latency mạng thuần túy.

Nếu không triển khai phần này, bài báo cáo chỉ được claim inference, request khứ hồi và toàn trình, kèm giới hạn đo một chiều.
