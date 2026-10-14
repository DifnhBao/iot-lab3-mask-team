# Train mới cân bằng ba trạng thái — không cần chụp thêm

## Đã làm xong tại laptop

- Tải nhãn [FMLD](https://github.com/borutb-fri/FMLD) và dùng phần ảnh MAFA đã có. Chưa dùng ảnh WIDER FACE nên đây là một subset của FMLD.
- Đọc/kiểm tra toàn bộ 30.811 XML phần MAFA, bỏ ảnh có nhãn invalid/difficult, kích thước không khớp, box sai lớn hoặc mặt quá nhỏ.
- Khóa holdout cũ khỏi train mới bằng hash và tên ảnh nguồn. Val/test FMLD mới loại ảnh đã dùng ở các lượt trước. Không đảm bảo tách hết danh tính người hoặc ảnh gần trùng.
- Chọn ảnh có một mặt được gán nhãn để cân bằng **số khuôn mặt**, không nhân bản ảnh hay chỉ giữ nhãn thiểu số trong ảnh nhiều người.
- Một số ảnh nguồn còn người nền thiếu nhãn: tạo crop ngữ cảnh quanh mặt được gán, giữ thêm vùng cằm. Crop giữ nguyên split của ảnh cha.
- Loại thêm khuôn mặt nguồn có cạnh dưới 48 pixel vì quá ít chi tiết để phân biệt mũi/miệng; cân bằng lại sau bước này.
- Dataset cuối: `datasets/mask_balanced_context_hq`.

| Split | Đeo đúng | Không đeo | Đeo sai | Tổng ảnh |
|---|---:|---:|---:|---:|
| train | 735 | 735 | 735 | 2205 |
| val | 72 | 72 | 72 | 216 |
| test | 74 | 74 | 74 | 222 |

Labels 0 with_mask, 1 without_mask, 2 mask_incorrect. Xem `summary.json` và `manifest.json` trong dataset để truy nguồn/ảnh bị bỏ. Bộ nguồn nguyên khung trước crop nằm ở `datasets/mask_balanced_fmld`, giữ để kiểm tra.

20 ảnh webcam train chỉ có đeo sai không được đưa vào train mới. Các dataset cũ và weights cũ vẫn giữ nguyên. Webcam val/test đã có chỉ dùng đánh giá/chọn checkpoint và hồi quy, không yêu cầu bạn tự chụp tiếp.

## Bạn làm trên Colab

1. Tạo thư mục `lab3_balanced` ở My Drive.
2. Upload `artifacts/mask_balanced_colab_bundle.zip` vào thư mục đó. Đợi upload hoàn tất.
3. Mở https://colab.research.google.com/ và Upload notebook `notebooks/Balanced_Three_State_Colab.ipynb`.
4. Runtime → Change runtime type → GPU (T4 nếu được cấp). Dùng hosted runtime, không local runtime.
5. Chạy từng cell: cài Ultralytics, kiểm tra GPU, mount Drive, giải nén, sửa YAML, kiểm tra số lớp và xem ảnh/box mẫu.

Bundle có dataset cân bằng đủ train/val/test, chỉ phần val/test của bốn dataset hồi quy, ba model cũ và script đo F1. Ảnh train cũ không cần upload vì không được dùng để train lượt mới. Alias `datasets/mask_mafa` bên trong bundle chính là dataset cân bằng mới; notebook ghi rõ điều này.

## Cách train khác hai lượt trước

- Khởi tạo YOLOv8n từ COCO weights, tải tự động trong Colab. Không dùng model lượt 2 đang lệch lớp để khởi tạo.
- Train mới trên dataset cân bằng, không resume trạng thái của lượt trước.
- Mặc định 80 epoch, batch 16, imgsz 640, AdamW lr0 0.001, patience 15. Đây là cấu hình khởi đầu để đo, không bảo đảm tối ưu.
- Nếu CUDA out of memory, giảm BATCH xuống 8 hoặc 4 rồi tạo lượt mới. Không tự tăng model size; cần giữ tốc độ CPU laptop.
- Checkpoints lưu Drive mỗi 5 epoch và best/last. GPU/quota Colab không được bảo đảm; thời gian phụ thuộc runtime thực tế.

## Chọn model theo đủ ba lớp

Notebook đo ba model cũ trên cùng val trước. Sau train, so sánh best, last và tối đa ba checkpoint định kỳ có mAP val cao nhất.

Phép đo chính dùng conf=0.4 như server, IoU ghép box=0.5, một prediction chỉ ghép với một ground-truth face. Nhầm lớp tính lỗi, mất box tính false negative, box dư tính false positive. Đây là F1 ở ngưỡng cố định, không phải precision/recall tại max-F1 mà log val Ultralytics thường báo.

Đánh giá đồng thời:

- Val FMLD mới có crop ngữ cảnh, đủ 3 lớp.
- Val webcam cũ, ảnh nguyên khung.
- Val tay/vật che mặt MAFA, ảnh nguyên khung.
- Val MAFA đeo sai.
- Val dataset khẩu trang ban đầu.

Score = 0.5 × trung bình macro-F1 của các bộ + 0.5 × F1 thấp nhất của một lớp đủ mẫu. Lớp dưới 3 mẫu vẫn hiển thị số liệu nhưng không dùng chọn checkpoint. Score có trọng số bằng nhau giữa các bộ để bộ nhiều ảnh không che khuất lỗi webcam. Các bộ nhỏ vẫn có độ bất định; cần xem ảnh dự đoán.

Không chọn chỉ bằng mAP tổng. Notebook báo `VAL_IMPROVED` nếu checkpoint mới có score val vượt cả ba model cũ. Nếu không vượt, giữ model cũ và gửi notebook outputs để phân tích; không tăng epoch mù quáng hoặc gọi candidate là bản nâng cấp.

## Test và export

Sau khi chốt checkpoint bằng val:

```python
RUN_FINAL_TEST = True
```

Chạy cell test. FMLD test mới chưa được dùng trong các lượt trước theo hash/tên local; các test cũ đã từng xem là bộ hồi quy. Dữ liệu công khai có thể có ảnh gần trùng hoặc cùng người chưa phát hiện, nên không coi đây là chứng minh tách theo danh tính.

Xem bảng precision/recall/F1 từng lớp và ảnh old/new. Đeo sai phải giảm nhầm thành không đeo; không đeo/tay che không được bị báo đeo sai hàng loạt; đeo đúng không được mất box. Một lớp mất toàn bộ detection là thất bại dù mAP tổng tăng.

Khi kết quả đạt và đã xem test, bật:

```python
EXPORT_CANDIDATE = True
```

Notebook chặn export nếu val không vượt ba model cũ hoặc có lớp webcam test (từ 3 mẫu) recall=0. Check này chỉ chặn lỗi rõ ràng, không tự chứng minh mọi tiêu chí đều đạt. Bạn vẫn cần xem bảng so sánh và ảnh dự đoán trước khi thay ứng dụng.

Lưu weights tải về thành `models/best_balanced.pt`. Dừng server bằng tay và thử:

```powershell
$env:MASK_MODEL_PATH = 'models/best_balanced.pt'
.\.venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --workers 1 --ws-max-size 6000000
```

Giữ weights cũ. Chưa tự đổi cấu hình server hoặc deploy model mới trong bước chuẩn bị này.

## Giới hạn và phần thủ công tối thiểu

Bạn không phải chụp thêm hoặc vẽ lại box hàng trăm ảnh. Chỉ upload, chạy notebook, xem một số ảnh/box đại diện và xem kết quả ba lớp. Nhãn FMLD công khai vẫn có trường hợp khăn/buff, mặt bị che và định nghĩa dưới cằm khác mong muốn; nếu mẫu rõ ràng sai, ghi tên ảnh để lọc hoặc sửa trước train.

Crop giảm người nền chưa có nhãn nhưng không bảo đảm loại hết. Vì vậy kết quả crop phải đi cùng webcam/ảnh nguyên khung. Cân bằng số lượng không tự bảo đảm cân bằng độ khó, màu khẩu trang, tuổi hoặc góc mặt. Đánh giá thực tế quyết định model có tốt hơn hay không.

Chưa có weights mới ở thời điểm chuẩn bị: phần GPU cần chạy trên Colab. Notebook đã kiểm tra cú pháp cell, script kiểm tra bằng unit tests; chưa chạy GPU Colab thực tế.
