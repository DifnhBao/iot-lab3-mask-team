"""Generate a clean three-class training notebook with weakest-class selection."""
import json
from pathlib import Path

from create_mafa_colab_notebook import cells as round_one_cells

cells = []


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip().splitlines(keepends=True)})


def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': text.strip().splitlines(keepends=True),
                  'execution_count': None, 'outputs': []})


md('''# Lab3 — Train mới, cân bằng ba trạng thái từ FMLD

Upload `mask_balanced_colab_bundle.zip` vào MyDrive/lab3_balanced. Chọn hosted GPU runtime (T4 nếu có).
Không cần chụp ảnh mới. Dữ liệu mới dùng ảnh MAFA với nhãn FMLD, chọn các ảnh có đúng một mặt được gán nhãn và cân bằng số khuôn mặt ba lớp.

Ảnh train/val/test mới là crop có ngữ cảnh quanh một mặt đã có nhãn, giữ vùng cằm và giữ split ảnh nguồn; giảm người nền chưa có box. Cân bằng vẫn theo mặt, không nhân bản ảnh để tăng số lượng. Các bộ webcam/MAFA cũ vẫn nguyên khung để kiểm tra ứng dụng thực tế.
Train mới từ COCO YOLOv8n để không kế thừa thiên lệch đeo sai của lượt 2. Ba model trước nằm trong bundle để so sánh. Không tự đổi weights đang chạy.
Val/test mới loại ảnh đã sử dụng trước theo hash và tên nguồn; test webcam cũ chỉ là regression test. Không bảo đảm tách hết danh tính người/ảnh gần trùng.

Chạy từng cell. Kiểm tra box mẫu trước khi train. Chọn checkpoint bằng val và chỉ bật test sau khi chốt. Nếu chưa tốt ở cả 3 lớp, không gọi model này là bản nâng cấp.''')
code('%pip install -q ultralytics==8.4.175')
code('''import torch, ultralytics
print('Ultralytics', ultralytics.__version__, 'Torch', torch.__version__)
assert torch.cuda.is_available(), 'Chọn GPU runtime trong Colab.'
print(torch.cuda.get_device_name(0))
from google.colab import drive
drive.mount('/content/drive')''')
md('''## 1. Giải nén và xác minh
Drive lưu ZIP/checkpoints; ảnh train đọc từ /content. Nếu upload bundle phiên bản khác, đổi LOCAL_ROOT sang tên mới.''')
code('''from pathlib import Path
import json, hashlib, shutil, zipfile, sys, yaml
DRIVE_BASE = Path('/content/drive/MyDrive/lab3_balanced')
ARCHIVE = DRIVE_BASE / 'mask_balanced_colab_bundle.zip'
LOCAL_ROOT = Path('/content/lab3_balanced')
assert ARCHIVE.is_file(), ARCHIVE
def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
local_zip = Path('/content/mask_balanced_input.zip')
shutil.copy2(ARCHIVE, local_zip)
archive_hash = sha256_file(local_zip)
marker = LOCAL_ROOT / '.archive_sha256'
if LOCAL_ROOT.exists() and any(LOCAL_ROOT.iterdir()):
    assert marker.is_file() and marker.read_text().strip() == archive_hash, 'Bundle khác; đổi LOCAL_ROOT.'
else:
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local_zip) as z:
        for member in z.infolist():
            assert (LOCAL_ROOT/member.filename).resolve().is_relative_to(LOCAL_ROOT.resolve())
            assert (member.external_attr >> 16) & 0o170000 != 0o120000
        z.extractall(LOCAL_ROOT)
    marker.write_text(archive_hash)
metadata = json.loads((LOCAL_ROOT/'bundle.json').read_text())
BASELINE = LOCAL_ROOT/'models/best.pt'  # round 1, chỉ để so sánh
assert sha256_file(BASELINE) == metadata['baseline_sha256']
REFS = {'round1_hand': BASELINE}
for tag, reference in metadata['reference_models'].items():
    path = LOCAL_ROOT/reference['path']
    assert sha256_file(path) == reference['sha256']
    REFS[tag] = path
sys.path.insert(0, str(LOCAL_ROOT/'scripts'))
from balanced_colab_runtime import evaluate_groups, save_report
print(json.dumps(metadata['counts'], indent=2))''')
md('''## 2. Kiểm tra nhãn và cân bằng
Alias `mask_mafa` trong bundle chính là dataset FMLD cân bằng mới, không phải dataset lượt 1. Bốn bộ khác chỉ để đánh giá; không trộn vào train.
Nếu có lớp trống hoặc tỷ lệ class quá lệch, dừng và sửa dataset trước.''')
code('''from collections import Counter
EXPECTED = {0:'with_mask',1:'without_mask',2:'mask_incorrect'}
DATA = {}
for name in metadata['counts']:
    folder = LOCAL_ROOT/'datasets'/name
    cfg = {'path':str(folder),'train':'images/train','val':'images/val','test':'images/test','names':EXPECTED}
    yaml_path = folder/'data.yaml'
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    DATA[name] = str(yaml_path)
    for split in ['train','val','test']:
        if metadata['counts'][name][split]==0:
            print(name,split,'not included: evaluation-only dataset')
            continue
        counts = Counter()
        images = [p for p in (folder/'images'/split).iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}]
        assert images
        for image in images:
            label = folder/'labels'/split/(image.stem+'.txt')
            assert label.is_file() and label.read_text().strip()
            for line in label.read_text().splitlines():
                cid,x,y,w,h = map(float,line.split())
                assert cid in EXPECTED and w>0 and h>0
                assert x-w/2>=-1e-6 and y-h/2>=-1e-6 and x+w/2<=1+1e-6 and y+h/2<=1+1e-6
                counts[int(cid)] += 1
        print(name, split, len(images), 'images; faces=',dict(counts))
        if name=='mask_mafa':
            assert min(counts.get(c,0) for c in EXPECTED)>0
            assert max(counts.values())/min(counts.values()) <= 1.3
train_seen = set()
for split in ['train','val','test']:
    for image in (LOCAL_ROOT/'datasets/mask_mafa/images'/split).iterdir():
        if image.suffix.lower() not in {'.jpg','.jpeg','.png'}: continue
        digest=sha256_file(image)
        assert digest not in train_seen, f'Duplicate content: {image}'
        train_seen.add(digest)
print('Checks passed. Summary:')
summary = json.loads((LOCAL_ROOT/'datasets/mask_mafa/summary.json').read_text())
print(json.dumps({k:summary[k] for k in ['source','splits','new_validation_unseen','new_test_unseen']},indent=2))''')
md('''## 3. Xem ảnh và box mẫu
Kiểm tra đủ ba trạng thái. Có khẩu trang dưới cằm phải phù hợp quy ước đeo sai của dự án; dữ liệu công khai có thể khác quy ước, nên xem các trường hợp này nếu xuất hiện. Không sửa nhãn để khớp dự đoán model.
Nếu mẫu nhãn rõ ràng sai, dừng và báo tên ảnh; không cần tự chụp thêm.''')
preview = ''.join(round_one_cells[11]['source'])
preview = preview.replace("show_labels('mafa_hand', 'val')", "show_labels('mask_mafa', 'val')")
code(preview)
md('''## 4. Đo ba model trước trên cùng val
Đo ở conf=0.4 và IoU ghép box=0.5, mỗi box chỉ ghép một lần. Nhầm lớp và mất box đều bị tính lỗi.
Score = 0.5*trung bình macro-F1 các bộ + 0.5*F1 thấp nhất của một lớp đủ mẫu. Lớp có dưới 3 mẫu vẫn được báo, chỉ không dùng chọn checkpoint. Không phải accuracy toàn ứng dụng.''')
code('''import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
STAMP = datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%Y%m%d_%H%M%S')
EVAL_ROOT = DRIVE_BASE/'evaluation'/STAMP
GROUPS = ['mask_mafa','webcam_incorrect','mafa_hand','mafa_incorrect','mask']
def class_table(results):
    rows=[]
    for tag,result in results.items():
        for report in result['reports']:
            for row in report['classes']:
                rows.append({'model':tag,'dataset':report['dataset'],'split':report['split'],**row})
    return pd.DataFrame(rows)
REFERENCE_VAL = {}
for tag,path in REFS.items():
    print('Evaluating',tag)
    REFERENCE_VAL[tag] = evaluate_groups(path,LOCAL_ROOT,GROUPS,'val',device=0)
save_report(REFERENCE_VAL,EVAL_ROOT/'reference_val.json')
display(class_table(REFERENCE_VAL))
display(pd.DataFrame([{'model':tag,'score':r['score'],'worst_f1':r['worst_supported_class_f1']} for tag,r in REFERENCE_VAL.items()]))''')
md('''## 5. Train mới từ COCO YOLOv8n
Tải yolov8n.pt tự động trong Colab, không dùng weights lệch lớp của lượt 2 để khởi tạo. Giữ model nhỏ để suy luận CPU laptop.
80 epoch là giới hạn đầu, patience=15 dừng sớm. Batch 16 có thể giảm 8/4 nếu hết VRAM. AdamW lr0=0.001 là cấu hình ban đầu, chưa phải tối ưu đã đo. Checkpoints lưu Drive mỗi 5 epoch.
Không resume=True ở lượt mới. Nếu runtime ngắt, chạy lại cell chuẩn bị rồi resume đúng last.pt của lượt này.''')
code('''from ultralytics import YOLO
RUN_PROJECT = DRIVE_BASE/'runs'
RUN_NAME = 'balanced_' + STAMP
EPOCHS = 80
BATCH = 16
model = YOLO('yolov8n.pt')
model.train(data=DATA['mask_mafa'],epochs=EPOCHS,batch=BATCH,imgsz=640,
    device=0,workers=2,cache=False,optimizer='AdamW',lr0=0.001,
    seed=42,patience=15,save=True,save_period=5,
    project=str(RUN_PROJECT),name=RUN_NAME,exist_ok=False)
RUN_DIR=Path(model.trainer.save_dir)
BEST=Path(model.trainer.best)
assert BEST.is_file()
save_report({'bundle_sha256':archive_hash,'run_dir':str(RUN_DIR),'ultralytics':ultralytics.__version__,
             'initialization':'COCO yolov8n.pt','names':EXPECTED},RUN_DIR/'experiment.json')
display(pd.read_csv(RUN_DIR/'results.csv').tail())
display(Image.open(RUN_DIR/'results.png'))
print('Training best:',BEST)''')
md('''## 6. Chọn checkpoint bằng cả ba lớp trên val
So sánh best/last và tối đa 3 checkpoint định kỳ có mAP val cao. Chọn bằng score cố định ở cell 4, không chỉ mAP tổng.
Phần này có thêm thời gian inference. Nếu candidate không vượt các model trước, notebook báo rõ và không tự xuất như một bản nâng cấp.''')
code('''log=pd.read_csv(RUN_DIR/'results.csv')
log.columns=log.columns.str.strip()
periodic=[]
for path in (RUN_DIR/'weights').glob('epoch*.pt'):
    epoch=int(path.stem.replace('epoch',''))+1
    values=log.loc[log['epoch']==epoch,'metrics/mAP50-95(B)']
    periodic.append((float(values.iloc[0]) if len(values) else -1.,path))
CANDIDATES={'training_best':BEST,'last':RUN_DIR/'weights/last.pt'}
for _,path in sorted(periodic,key=lambda item:item[0],reverse=True)[:3]:
    CANDIDATES[path.stem]=path
CANDIDATE_VAL={}
for tag,path in CANDIDATES.items():
    if not path.is_file(): continue
    print('Evaluating checkpoint',tag)
    CANDIDATE_VAL[tag]=evaluate_groups(path,LOCAL_ROOT,GROUPS,'val',device=0)
save_report(CANDIDATE_VAL,EVAL_ROOT/'candidate_val.json')
SELECTED_TAG=max(CANDIDATE_VAL,key=lambda tag:CANDIDATE_VAL[tag]['score'])
SELECTED=CANDIDATES[SELECTED_TAG]
reference_best=max(r['score'] for r in REFERENCE_VAL.values())
VAL_IMPROVED=CANDIDATE_VAL[SELECTED_TAG]['score']>reference_best
display(class_table(CANDIDATE_VAL))
display(pd.DataFrame([{'checkpoint':tag,'score':r['score'],'worst_f1':r['worst_supported_class_f1']} for tag,r in CANDIDATE_VAL.items()]))
print('Selected:',SELECTED,'Val score beats all previous models:',VAL_IMPROVED)
if not VAL_IMPROVED: print('Chưa chứng minh candidate tốt hơn; giữ model cũ và xem lớp còn yếu trước khi gọi đây là bản nâng cấp.')''')
md('''## 7. Chốt rồi test
Đổi RUN_FINAL_TEST=True sau khi chọn bằng val. FMLD test mới là bộ chưa dùng trong các lượt trước theo dữ liệu local; các bộ test cũ là regression checks vì đã từng xem.
Không tiếp tục chỉnh theo test rồi gọi test đó độc lập. Nếu có lớp webcam mất box hoặc F1=0, chưa đạt mục tiêu đủ 3 trạng thái.''')
code('''RUN_FINAL_TEST=False
TEST_RESULTS={}
if RUN_FINAL_TEST:
    for tag,path in {**REFS,'balanced_candidate':SELECTED}.items():
        print('Final test',tag)
        TEST_RESULTS[tag]=evaluate_groups(path,LOCAL_ROOT,GROUPS,'test',device=0)
    save_report(TEST_RESULTS,EVAL_ROOT/'final_test.json')
    table=class_table(TEST_RESULTS)
    table.to_csv(EVAL_ROOT/'final_test_classes.csv',index=False)
    display(table)
else:
    print('Test chưa chạy; chốt checkpoint bằng val trước.')''')
md('''## 8. Xem dự đoán webcam
So sánh với model lượt 1. Muốn xem test chỉ chạy sau khi bật cell test. Chú ý không đeo/tay che có bị báo đeo sai không; mất box cũng là lỗi.''')
code('''from PIL import Image
import matplotlib.pyplot as plt
PRED_SPLIT='val'  # đổi test chỉ sau khi chốt
PRED_SOURCE=LOCAL_ROOT/'datasets/webcam_incorrect/images'/PRED_SPLIT
PRED_OUTPUTS={}
for tag,path in {'round1_hand':BASELINE,'balanced_candidate':SELECTED}.items():
    out=EVAL_ROOT/f'{tag}_predict_{PRED_SPLIT}'
    for result in YOLO(str(path)).predict(source=str(PRED_SOURCE),stream=True,device=0,
        imgsz=640,conf=.4,iou=.5,agnostic_nms=True,max_det=100,save=True,save_txt=True,save_conf=True,
        project=str(EVAL_ROOT),name=out.name,exist_ok=True,verbose=False): pass
    PRED_OUTPUTS[tag]=out
images=sorted(p for p in PRED_SOURCE.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
for image in images[:10]:
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,title,path in zip(axes,['Original','Round 1','Balanced candidate'],
        [image,PRED_OUTPUTS['round1_hand']/image.name,PRED_OUTPUTS['balanced_candidate']/image.name]):
        ax.imshow(Image.open(path)); ax.set_title(title); ax.axis('off')
    plt.show()''')
md('''## 9. Export candidate khi đã xem kết quả
Không tự thay weights ứng dụng. Bật EXPORT_CANDIDATE=True sau khi đã chạy test và xem ảnh. Nếu val không tốt hơn, hoặc test cho thấy một lớp bị mất, dừng để phân tích.
Giữ toàn bộ report JSON/CSV và notebook outputs để đối chiếu. Không có cam kết trước rằng một lần train sẽ đạt mục tiêu.''')
code('''EXPORT_CANDIDATE=False
if EXPORT_CANDIDATE:
    assert RUN_FINAL_TEST and TEST_RESULTS, 'Chạy test trước khi export.'
    assert VAL_IMPROVED, 'Candidate chưa vượt các model cũ theo score val; xem kết quả trước.'
    webcam_test=next(r for r in TEST_RESULTS['balanced_candidate']['reports'] if r['dataset']=='webcam_incorrect')
    assert all(row['recall']>0 for row in webcam_test['classes'] if row['support']>=3), 'Một lớp webcam test vẫn recall=0; chưa đạt đủ ba trạng thái.'
    target=DRIVE_BASE/f'best_balanced_{RUN_DIR.name}_{SELECTED_TAG}.pt'
    shutil.copy2(SELECTED,target)
    print('Candidate:',target,'SHA256:',sha256_file(target))
    from google.colab import files
    files.download(str(target))
else:
    print('Chưa export. Đánh giá đủ ba lớp trước khi bật EXPORT_CANDIDATE=True.')''')
md('''## Nếu Colab bị ngắt khi đang train
Chạy lại các cell cài thư viện, Drive, giải nén, kiểm tra YAML và reference val. Thêm cell:

```python
LAST=Path('/content/drive/MyDrive/lab3_balanced/runs/TEN_LUOT/weights/last.pt')
resumed=YOLO(str(LAST))
resumed.train(resume=True)
RUN_DIR=Path(resumed.trainer.save_dir)
BEST=Path(resumed.trainer.best)
```

Import YOLO trước nếu chưa có. Resume chỉ áp dụng lượt chưa hoàn thành có checkpoint hợp lệ; giữ cùng bundle/đường dẫn local. Lượt đã hoàn thành chỉ cần đặt RUN_DIR và BEST để đánh giá, không train lại.

Lưu weights tải về thành models/best_balanced.pt, thử server bằng MASK_MODEL_PATH trước khi thay model chính thức. Dữ liệu công khai khác miền webcam và có thể khác định nghĩa dưới cằm, nên cần xem kết quả theo từng trạng thái.
Nguồn dữ liệu: https://github.com/borutb-fri/FMLD. Nguồn train: https://docs.ultralytics.com/modes/train/.''')


if __name__ == '__main__':
    for i, cell in enumerate(cells):
        cell['id']=f'balanced-cell-{i:03d}'
        source=''.join(cell['source'])
        if cell['cell_type']=='code' and not source.startswith('%pip'): compile(source,cell['id'],'exec')
    path=Path(__file__).resolve().parents[1]/'notebooks/Balanced_Three_State_Colab.ipynb'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'cells':cells,'metadata':{'accelerator':'GPU','colab':{'name':path.name},
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
        'nbformat':4,'nbformat_minor':5},ensure_ascii=False,indent=2),encoding='utf-8')
    print(path)
