"""Generate the portable, documented Colab notebook without notebook dependencies."""
import json
from pathlib import Path

cells = []


def md(source):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': source.strip().splitlines(keepends=True)})


def code(source):
    cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                  'source': source.strip().splitlines(keepends=True)})


md('''# Lab3 — Fine-tune tay che mặt trên Google Colab

Chuẩn bị trên laptop: tải ảnh MAFA → duyệt bằng prepare_mafa.py → convert → merge_hard_cases.py → pack_mafa_colab.py.
Upload `mafa_colab_bundle.zip` lên Google Drive trong thư mục `lab3_mafa`. Bundle chứa dataset trộn, dữ liệu MAFA đã duyệt, dataset cũ và model gốc. **Chưa có bundle thì chưa chạy train.**

Trong Colab chọn Runtime → Change runtime type → GPU (T4 nếu có). Dùng hosted runtime của Colab. Chạy từng cell theo thứ tự, đọc kết quả trước khi tiếp tục. Khả năng cấp GPU phụ thuộc tài nguyên/quota Colab.

Giữ 3 lớp: 0 with_mask, 1 without_mask, 2 mask_incorrect. Những mặt nhãn 3 nguồn MAFA chỉ được đổi thành without_mask sau khi duyệt ảnh. Không thêm lớp mới trong notebook.

Tập val dùng để lựa chọn cấu hình. Tập test chỉ dùng sau khi chốt. Ngoài MAFA, cần kiểm tra webcam mới chưa từng dùng train. Nếu model gốc từng train trên MAFA test thì tập MAFA test không còn độc lập.''')
code('%pip install -q ultralytics==8.4.175')
md('''## 1. Xác nhận GPU
Nếu `CUDA=False`, đổi runtime sang GPU và chạy lại. Notebook dừng để tránh vô tình train trên CPU laptop/runtime CPU.''')
code('''import torch
import ultralytics
print('Ultralytics:', ultralytics.__version__)
print('Torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'Chọn hosted GPU runtime trong Colab trước khi train.'
print('GPU:', torch.cuda.get_device_name(0))''')
md('''## 2. Kết nối Google Drive
Colab sẽ hiện bước cấp quyền cho Drive. Dữ liệu ảnh sẽ được copy vào `/content` để đọc nhanh; checkpoints/biểu đồ lưu trong Drive. `/content` có thể mất khi runtime bị reset.''')
code('''from google.colab import drive
drive.mount('/content/drive')''')
md('''## 3. Copy và giải nén bundle
Mặc định file ở `MyDrive/lab3_mafa/mafa_colab_bundle.zip`. Nếu bạn đặt nơi khác, sửa `DRIVE_BASE` hoặc `ARCHIVE`. Không dùng đường dẫn `D:/...` của laptop trên Colab.''')
code('''from pathlib import Path
import hashlib, json, shutil, zipfile

DRIVE_BASE = Path('/content/drive/MyDrive/lab3_mafa')
ARCHIVE = DRIVE_BASE / 'mafa_colab_bundle.zip'
LOCAL_ROOT = Path('/content/lab3_mafa')
assert ARCHIVE.is_file(), f'Chưa thấy bundle: {ARCHIVE}'
local_zip = LOCAL_ROOT.parent / f'{LOCAL_ROOT.name}_input.zip'
shutil.copy2(ARCHIVE, local_zip)
def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
archive_hash = sha256_file(local_zip)
marker = LOCAL_ROOT / '.archive_sha256'
if LOCAL_ROOT.exists() and any(LOCAL_ROOT.iterdir()):
    assert marker.is_file() and marker.read_text().strip() == archive_hash, 'Thư mục local chứa bundle khác; chọn LOCAL_ROOT mới.'
else:
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local_zip) as archive:
        for member in archive.infolist():
            target = (LOCAL_ROOT / member.filename).resolve()
            assert target.is_relative_to(LOCAL_ROOT.resolve()), 'Archive có đường dẫn không hợp lệ.'
            assert (member.external_attr >> 16) & 0o170000 != 0o120000, 'Không chấp nhận symlink trong bundle.'
        archive.extractall(LOCAL_ROOT)
    marker.write_text(archive_hash)
metadata = json.loads((LOCAL_ROOT / 'bundle.json').read_text())
BASELINE = LOCAL_ROOT / 'models/best.pt'
assert sha256_file(BASELINE) == metadata['baseline_sha256'], 'Hash model gốc không khớp.'
print(json.dumps(metadata['counts'], indent=2))''')
md('''## 4. Sửa YAML và kiểm tra nhãn
`mask_mafa` dùng train chính; `mafa_hand` dùng đánh giá trường hợp khó; `mask` dùng kiểm tra chất lượng trên dữ liệu cũ. Việc cùng ảnh xuất hiện trong dataset trộn và dataset đánh giá là có chủ ý, nhưng train/val/test trong dataset trộn phải tách biệt.
Cell kiểm tra hình học và ảnh trùng byte giữa các split của dataset trộn; không phát hiện mọi ảnh gần trùng hoặc cùng người.''')
code('''import yaml, math
from collections import Counter
EXPECTED = {0: 'with_mask', 1: 'without_mask', 2: 'mask_incorrect'}
DATA = {}
for name in ['mask_mafa', 'mafa_hand', 'mask']:
    folder = LOCAL_ROOT / 'datasets' / name
    cfg = {'path': str(folder), 'train': 'images/train', 'val': 'images/val', 'test': 'images/test', 'names': EXPECTED}
    yaml_path = folder / 'data.yaml'
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    DATA[name] = str(yaml_path)
    for split in ['train', 'val', 'test']:
        images = sorted(p for p in (folder / 'images' / split).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        assert images, f'Tập trống: {name}/{split}'
        count = Counter()
        for image in images:
            label = folder / 'labels' / split / f'{image.stem}.txt'
            assert label.is_file() and label.read_text().strip(), f'Thiếu/rỗng nhãn: {label}'
            for line in label.read_text().splitlines():
                parts = line.split()
                assert len(parts) == 5, label
                cid, x, y, w, h = map(float, parts)
                assert cid in EXPECTED and all(math.isfinite(v) for v in [x,y,w,h]), label
                assert w > 0 and h > 0 and x-w/2 >= -1e-6 and y-h/2 >= -1e-6 and x+w/2 <= 1+1e-6 and y+h/2 <= 1+1e-6, label
                count[int(cid)] += 1
        print(name, split, 'images=', len(images), 'faces=', dict(count))
        if name == 'mask_mafa':
            assert all(count[cid] > 0 for cid in EXPECTED), f'Tập trộn {split} thiếu lớp; xem lại dữ liệu.'
seen = {}
for split in ['train', 'val', 'test']:
    folder = LOCAL_ROOT / 'datasets/mask_mafa/images' / split
    for image in folder.iterdir():
        if image.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        digest = sha256_file(image)
        if digest in seen:
            assert seen[digest][0] == split, f'Ảnh trùng giữa split: {image}, {seen[digest]}'
        seen[digest] = (split, str(image))
print('Dataset checks passed')''')
md('''## 5. Xem box/nhãn trước khi train
Xem ảnh train trộn và val khó. Nếu tay bị gán with_mask hoặc box sai, quay lại dữ liệu và sửa trước. Không dùng test để điều chỉnh nhãn/cấu hình theo dự đoán model.''')
code('''import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

def show_labels(dataset, split, limit=6):
    folder = LOCAL_ROOT / 'datasets' / dataset
    images = sorted(p for p in (folder / 'images' / split).iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
    sample = random.Random(42).sample(images, min(limit, len(images)))
    fig, axes = plt.subplots(1, len(sample), figsize=(4*len(sample), 4), squeeze=False)
    for ax, image in zip(axes[0], sample):
        im = Image.open(image).convert('RGB')
        width, height = im.size
        ax.imshow(im)
        for line in (folder / 'labels' / split / f'{image.stem}.txt').read_text().splitlines():
            cid, x, y, w, h = map(float, line.split())
            x1, y1 = (x-w/2)*width, (y-h/2)*height
            color = ['lime', 'red', 'orange'][int(cid)]
            ax.add_patch(Rectangle((x1,y1), w*width,h*height, fill=False, edgecolor=color, linewidth=2))
            ax.text(x1, y1, EXPECTED[int(cid)], color=color, backgroundcolor='black', fontsize=8)
        ax.set_title(image.name, fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
show_labels('mask_mafa', 'train')
show_labels('mafa_hand', 'val')''')
md('''## 6. Fine-tune trên GPU
Bắt đầu 30 epoch, batch 8, imgsz 640. AdamW với lr0=0.0003 là cấu hình thử ban đầu, chọn bằng val. Không dùng resume=True cho lượt bổ sung dữ liệu mới. Weights mới lưu vào Drive; `models/best.pt` gốc trong bundle được giữ nguyên.

Nếu CUDA out of memory, giảm BATCH xuống 4 hoặc 2 và chạy lại cell với RUN_NAME mới. Không tăng model size ở lượt đầu. Train có thể dừng sớm sau 10 epoch không cải thiện val. GPU Colab không được bảo đảm luôn sẵn có.''')
code('''from datetime import datetime
from zoneinfo import ZoneInfo
from ultralytics import YOLO

EPOCHS = 30
BATCH = 8
LR0 = 0.0003
RUN_NAME = 'mafa_hand_' + datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%Y%m%d_%H%M%S')
RUN_PROJECT = DRIVE_BASE / 'runs'
RUN_PROJECT.mkdir(parents=True, exist_ok=True)
RUN_DIR = RUN_PROJECT / RUN_NAME
model = YOLO(str(BASELINE))
assert {int(k):v for k,v in model.names.items()} == EXPECTED, 'Sai lớp của model gốc.'
train_result = model.train(
    data=DATA['mask_mafa'], epochs=EPOCHS, batch=BATCH, imgsz=640,
    device=0, workers=2, cache=False, optimizer='AdamW', lr0=LR0,
    seed=42, patience=10, save=True, save_period=5,
    project=str(RUN_PROJECT), name=RUN_NAME, exist_ok=False,
)
BEST = Path(model.trainer.best)
RUN_DIR = Path(model.trainer.save_dir)
assert BEST.is_file(), 'Chưa có best.pt; xem log train.'
run_info = {'run_dir': str(RUN_DIR), 'best': str(BEST), 'bundle_sha256': archive_hash,
            'baseline_sha256': metadata['baseline_sha256'], 'ultralytics': ultralytics.__version__,
            'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
(RUN_DIR / 'experiment.json').write_text(json.dumps(run_info, indent=2))
print('Best:', BEST)
print('Last:', RUN_DIR / 'weights/last.pt')''')
md('''## 7. Xem kết quả train
Loss giảm chưa đủ chứng minh hết lỗi tay che mặt. `best.pt` được chọn trên val của dataset trộn, cần đánh giá riêng tập khó và tập gốc.''')
code('''import pandas as pd
from IPython.display import display
display(pd.read_csv(RUN_DIR / 'results.csv').tail())
if (RUN_DIR / 'results.png').is_file():
    display(Image.open(RUN_DIR / 'results.png'))''')
md('''## 8. So sánh trên VAL trước
Chạy model cũ và mới trên val MAFA khó và val gốc. Giữ cùng imgsz/conf/iou. Chỉ số precision/recall tóm tắt của thư viện có thể ở điểm max-F1; ảnh dự đoán cell sau dùng đúng conf=0.4 của server.

Không đánh giá riêng mAP tập khó để kết luận cả ba lớp vì tập này có thể thiếu lớp. Điều chỉnh nếu cần bằng val; sau khi chốt mới chạy cell test.''')
code('''EVAL_ROOT = DRIVE_BASE / 'evaluation' / RUN_DIR.name
EVAL_ROOT.mkdir(parents=True, exist_ok=True)
MODELS = {'old': BASELINE, 'new': BEST}
def evaluate(split):
    rows = []
    stamp = datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%Y%m%d_%H%M%S')
    for tag, weights in MODELS.items():
        for dataset in ['mafa_hand', 'mask']:
            metrics = YOLO(str(weights)).val(data=DATA[dataset], split=split, device=0,
                imgsz=640, batch=BATCH, conf=0.4, iou=0.5, agnostic_nms=True, max_det=100, plots=True,
                project=str(EVAL_ROOT), name=f'{tag}_{dataset}_{split}_{stamp}')
            rows.append({'model': tag, 'dataset': dataset, 'split': split,
                         'precision_summary': float(metrics.box.mp), 'recall_summary': float(metrics.box.mr),
                         'mAP50': float(metrics.box.map50), 'mAP50_95': float(metrics.box.map)})
    table = pd.DataFrame(rows)
    table.to_csv(EVAL_ROOT / f'metrics_{split}_{stamp}.csv', index=False)
    display(table)
    return table
val_metrics = evaluate('val')''')
md('''## 9. Xem dự đoán cũ/mới ở conf=0.4
Ghi riêng số mặt tay che bị báo with_mask, số nhận đúng without_mask và số không phát hiện. Không coi việc biến mất box là nhận đúng.
Cell lưu ảnh dự đoán cho toàn bộ val khó và hiển thị 8 ảnh mẫu; cần xem phần còn lại trong Drive.''')
code('''PRED_DIRS = {}
def make_predictions(split):
    source = LOCAL_ROOT / 'datasets/mafa_hand/images' / split
    stamp = datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%Y%m%d_%H%M%S')
    for tag, weights in MODELS.items():
        out = EVAL_ROOT / f'{tag}_pred_{split}_{stamp}'
        results = YOLO(str(weights)).predict(source=str(source), stream=True, device=0,
            imgsz=640, conf=0.4, iou=0.5, agnostic_nms=True, max_det=100,
            save=True, save_txt=True, save_conf=True, verbose=False,
            project=str(EVAL_ROOT), name=out.name)
        for result in results:
            pass
        PRED_DIRS[(tag, split)] = out
    return source

def compare_images(split, limit=8):
    source = LOCAL_ROOT / 'datasets/mafa_hand/images' / split
    images = sorted(p for p in source.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
    sample = random.Random(42).sample(images, min(limit, len(images)))
    for image in sample:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, title, path in zip(axes, ['Original', 'Old model', 'New model'],
                [image, PRED_DIRS[('old', split)] / image.name, PRED_DIRS[('new', split)] / image.name]):
            assert path.is_file(), f'Chưa thấy ảnh dự đoán: {path}'
            ax.imshow(Image.open(path).convert('RGB'))
            ax.set_title(title)
            ax.axis('off')
        fig.suptitle(image.name)
        plt.tight_layout()
        plt.show()
make_predictions('val')
compare_images('val')''')
md('''## 10. Test cuối — chỉ bật sau khi đã chọn model
Đổi `CONFIRM_FINAL_TEST=True` khi đã chốt cấu hình bằng val. Không tiếp tục chỉnh model theo kết quả test; nếu làm vậy cần test độc lập mới. Kết quả test không bảo đảm thành công trên webcam mới.

Đếm theo khuôn mặt mục tiêu: tổng số mặt tay che, báo with_mask sai, báo without_mask đúng, báo mask_incorrect sai, không phát hiện. So sánh cùng ảnh và cùng ngưỡng. Nếu người đó thực sự có khẩu trang sau tay thì không dùng làm mẫu without_mask.''')
code('''CONFIRM_FINAL_TEST = False
if CONFIRM_FINAL_TEST:
    test_metrics = evaluate('test')
    make_predictions('test')
    compare_images('test')
else:
    print('Test chưa chạy. Chọn model bằng val, sau đó đổi CONFIRM_FINAL_TEST=True.')''')
md('''## 11. Tải weights về laptop
Cell xuất weights sang tên riêng trên Drive và tải xuống. Đây là candidate để thử; không tự thay model đang chạy.
Giữ thêm experiment.json, args.yaml, results.csv, biểu đồ và bảng đánh giá trong Drive.''')
code('''from google.colab import files
EXPORT = DRIVE_BASE / f'best_mafa_hand_{RUN_DIR.name}.pt'
assert BEST.is_file()
assert {int(k):v for k,v in YOLO(str(BEST)).names.items()} == EXPECTED
shutil.copy2(BEST, EXPORT)
print('Export:', EXPORT, 'SHA256:', sha256_file(EXPORT))
files.download(str(EXPORT))''')
md('''## Nếu runtime bị ngắt khi đang train

Kết nối GPU lại, chạy lại các cell cài thư viện, mount Drive, giải nén và sửa YAML. Checkpoint nằm trong `MyDrive/lab3_mafa/runs/<tên-lượt>/weights/last.pt`.

Chạy cell mới, thay đúng tên lượt (không dùng cell tạo lượt train mới):

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

Resume chỉ áp dụng cho lượt train chưa hoàn thành có checkpoint hợp lệ. Dùng cùng bundle, phiên bản thư viện và đường dẫn dataset local như lượt ban đầu. Nếu chưa hoàn thành epoch đầu tiên thì có thể chưa có last.pt.

## Đưa về ứng dụng

Lưu weights mới thành `models/best_mafa_hand.pt`. Dừng server, chạy thử bằng biến `MASK_MODEL_PATH` trỏ tới file này; giữ `models/best.pt` cũ. Kiểm tra health ready=true rồi thử tay che mặt, khẩu trang đúng/sai ở các góc/ánh sáng mới. Đo cả số lỗi lẫn số mặt bị bỏ sót và tốc độ CPU.

Nguồn: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [Ultralytics train](https://docs.ultralytics.com/modes/train/), [Ultralytics val](https://docs.ultralytics.com/modes/val/), [AnnotationMAFA](https://github.com/ElenaRyumina/AnnotationMAFA).''')


if __name__ == '__main__':
    notebook = {'cells': cells, 'metadata': {'colab': {'name': 'MAFA_Hand_Finetune_Colab.ipynb'},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU'}, 'nbformat': 4, 'nbformat_minor': 5}
    for i, cell in enumerate(cells):
        cell['id'] = f'cell-{i:03d}'
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if not source.startswith('%pip'):
                compile(source, f'cell-{i}', 'exec')
    output = Path(__file__).resolve().parents[1] / 'notebooks/MAFA_Hand_Finetune_Colab.ipynb'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Generated {len(cells)} cells: {output}')
