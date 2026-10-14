"""Build round 2 from the verified Colab workflow; keep round 1 unchanged."""
import json
from pathlib import Path

from create_mafa_colab_notebook import cells


def build():
    result = json.loads(json.dumps(cells))
    for i, cell in enumerate(result):
        source = ''.join(cell['source'])
        source = source.replace('lab3_mafa', 'lab3_mafa_v2')
        source = source.replace('mafa_colab_bundle.zip', 'mask_incorrect_colab_bundle.zip')
        source = source.replace("for name in ['mask_mafa', 'mafa_hand', 'mask']:", "for name in metadata['counts']:")
        source = source.replace("for dataset in ['mafa_hand', 'mask']:",
                                "for dataset in ['mafa_incorrect', 'webcam_incorrect', 'mafa_hand', 'mask']:")
        source = source.replace("LR0 = 0.0003", "LR0 = 0.0001")
        source = source.replace("RUN_NAME = 'mafa_hand_'", "RUN_NAME = 'mask_incorrect_'")
        source = source.replace('best_mafa_hand_', 'best_mask_incorrect_')
        source = source.replace("show_labels('mafa_hand', 'val')", "show_labels('mafa_incorrect', 'val')\nshow_labels('webcam_incorrect', 'val')")
        source = source.replace("'datasets/mafa_hand/images'", "f'datasets/{PRED_DATASET}/images'")
        if cell['cell_type'] == 'code' and source.startswith('PRED_DIRS = {}'):
            source = "PRED_DATASET = 'webcam_incorrect'  # đổi thành 'mafa_incorrect' hoặc 'mafa_hand' để xem bộ khác\n" + source
        if cell['cell_type'] == 'code' and source.startswith('EVAL_ROOT ='):
            source = source.replace("rows.append({'model': tag, 'dataset': dataset, 'split': split,",
                                    "rows.append({'model': tag, 'dataset': dataset, 'split': split, 'class': 'all',")
            source = source.replace("'mAP50_95': float(metrics.box.map)})", """'mAP50_95': float(metrics.box.map)})
            for position, cid in enumerate(metrics.box.ap_class_index):
                rows.append({'model': tag, 'dataset': dataset, 'split': split, 'class': EXPECTED[int(cid)],
                    'precision_summary': float(metrics.box.p[position]),
                    'recall_summary': float(metrics.box.r[position]),
                    'mAP50': float(metrics.box.ap50[position]),
                    'mAP50_95': float(metrics.box.ap[position])})""")
        source = source.replace('CPU laptop/runtime CPU', 'runtime CPU')
        if i == 0:
            source = '''# Lab3 — Lượt 2: khẩu trang đeo sai, giữ khả năng nhận tay che mặt

Đọc docs/TRAIN_MASK_INCORRECT_COLAB.md trước. Duyệt MAFA nhãn 2 bằng review_incorrect.json; thu thêm webcam khẩu trang đen lộ mũi/kéo xuống cằm và đủ ba trạng thái.
Trộn với dataset vòng 1, đóng gói bằng pack_mafa_colab.py theo hướng dẫn. Upload mask_incorrect_colab_bundle.zip vào MyDrive/lab3_mafa_v2.

Bundle dùng **models/best_mafa_hand.pt hiện tại** làm baseline và khởi tạo train, không phải model gốc trước MAFA. Giữ 3 lớp 0 with_mask, 1 without_mask, 2 mask_incorrect.
Candidate MAFA đã dùng ở vòng trước bị loại khỏi lượt duyệt mới để tránh đổi split. Không dùng test làm train. Các bộ test vòng 1 chỉ là bộ hồi quy nếu đã từng xem; webcam test buổi mới là phép kiểm tra độc lập bổ sung.

Runtime → Change runtime type → GPU. Chạy từng cell, kiểm tra nhãn trước khi train. Checkpoints nằm trong Drive. Test cuối tắt mặc định.
Notebook so sánh baseline hiện tại và model mới trên MAFA đeo sai, webcam mới, tay che mặt và dữ liệu gốc. Không kết luận thành công chỉ bằng mAP tổng.'''
        elif cell['cell_type'] == 'markdown':
            source = source.replace('AdamW với lr0=0.0003', 'AdamW với lr0=0.0001')
            source = source.replace('train chính; `mafa_hand` dùng đánh giá trường hợp khó;',
                                    'train chính; `mafa_incorrect` và `webcam_incorrect` kiểm tra đeo sai; `mafa_hand` kiểm tra tay che mặt;')
            source = source.replace('val MAFA khó và val gốc', 'val MAFA đeo sai, val webcam, val tay che mặt và val gốc')
            source = source.replace('val khó', 'val webcam')
            source = source.replace('mặt tay che', 'mặt đeo sai')
            source = source.replace('báo with_mask', 'báo with_mask/without_mask')
            source = source.replace('báo without_mask đúng', 'báo mask_incorrect đúng')
            source = source.replace('tổng số mặt tay che', 'tổng số mặt đeo sai')
            source = source.replace('models/best_mafa_hand.pt', 'models/best_mask_incorrect_v2.pt')
        cell['source'] = source.splitlines(keepends=True)
        cell['id'] = f'incorrect-cell-{i:03d}'
        if cell['cell_type'] == 'code' and not source.startswith('%pip'):
            compile(source, f'incorrect-cell-{i}', 'exec')
    output = Path(__file__).resolve().parents[1] / 'notebooks/Mask_Incorrect_Finetune_Colab.ipynb'
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = {'cells': result, 'metadata': {
        'colab': {'name': output.name}, 'accelerator': 'GPU',
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    build()
