"""Fixed-threshold, one-to-one detection evaluation for the balanced Colab run."""
import json
from pathlib import Path

import numpy as np

NAMES = {0: 'with_mask', 1: 'without_mask', 2: 'mask_incorrect'}


def match_confusion(truth, prediction, threshold=.5):
    """Rows = ground truth; columns = prediction; class 3 = missing/background.

    truth: class,x1,y1,x2,y2. prediction: class,confidence,x1,y1,x2,y2.
    All matches are one-to-one and independent of class, so class mistakes count.
    """
    truth = np.asarray(truth, dtype=float).reshape(-1, 5)
    prediction = np.asarray(prediction, dtype=float).reshape(-1, 6)
    confusion = np.zeros((4, 4), dtype=int)
    used_gt, used_pred = set(), set()
    if len(truth) and len(prediction):
        gt, pred = truth[:, 1:], prediction[:, 2:]
        left = np.maximum(gt[:, None, :2], pred[None, :, :2])
        right = np.minimum(gt[:, None, 2:], pred[None, :, 2:])
        intersection = np.prod(np.maximum(0, right-left), axis=2)
        gt_area = np.prod(np.maximum(0, gt[:, 2:]-gt[:, :2]), axis=1)
        pred_area = np.prod(np.maximum(0, pred[:, 2:]-pred[:, :2]), axis=1)
        iou = intersection / np.maximum(1e-9, gt_area[:, None] + pred_area[None, :] - intersection)
        pairs = [(float(iou[g, p]), g, p) for g, p in zip(*np.where(iou >= threshold))]
        for _, g, p in sorted(pairs, reverse=True):
            if g in used_gt or p in used_pred:
                continue
            used_gt.add(g)
            used_pred.add(p)
            confusion[int(truth[g, 0]), int(prediction[p, 0])] += 1
    for g in range(len(truth)):
        if g not in used_gt:
            confusion[int(truth[g, 0]), 3] += 1
    for p in range(len(prediction)):
        if p not in used_pred:
            confusion[3, int(prediction[p, 0])] += 1
    return confusion


def metrics_from_confusion(confusion, minimum_support=3):
    confusion = np.asarray(confusion)
    rows = []
    for cid, name in NAMES.items():
        tp = int(confusion[cid, cid])
        support = int(confusion[cid].sum())
        predicted = int(confusion[:, cid].sum())
        precision = tp/predicted if predicted else 0.
        recall = tp/support if support else 0.
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
        rows.append({'class': name, 'support': support, 'tp': tp,
                     'fp': predicted-tp, 'fn': support-tp, 'precision': precision,
                     'recall': recall, 'f1': f1, 'used_in_score': support >= minimum_support})
    supported = [row['f1'] for row in rows if row['used_in_score']]
    return rows, float(np.mean(supported)) if supported else 0.


def evaluate_model(weights, dataset_root, split='val', device=0, conf=.4):
    from ultralytics import YOLO
    folder = Path(dataset_root)
    images = sorted(p for p in (folder/'images'/split).iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
    if not images:
        raise ValueError(f'No images: {folder}/{split}')
    model = YOLO(str(weights))
    if {int(k): v for k, v in model.names.items()} != NAMES:
        raise ValueError(f'Model class mapping differs: {weights}')
    matrix = np.zeros((4, 4), dtype=int)
    predictions = model.predict(source=[str(p) for p in images], stream=True, batch=1,
        device=device, imgsz=640, rect=True, conf=conf, iou=.5, agnostic_nms=True,
        max_det=100, verbose=False)
    for result in predictions:
        label = folder/'labels'/split/(Path(result.path).stem+'.txt')
        height, width = result.orig_shape
        truth = []
        for line in label.read_text().splitlines():
            cid, x, y, w, h = map(float, line.split())
            truth.append([cid, (x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
        boxes = result.boxes
        prediction = np.column_stack([boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.xyxy.cpu().numpy()])
        matrix += match_confusion(truth, prediction)
    rows, macro_f1 = metrics_from_confusion(matrix)
    return {'weights': str(weights), 'dataset': folder.name, 'split': split,
            'conf': conf, 'match_iou': .5, 'images': len(images), 'classes': rows,
            'macro_f1_supported': macro_f1, 'confusion': matrix.tolist(),
            'note': 'Fixed threshold; missing boxes count as false negatives. Support<3 excluded only from selection score.'}


def evaluate_groups(weights, root, datasets, split='val', device=0):
    reports = [evaluate_model(weights, Path(root)/'datasets'/name, split, device) for name in datasets]
    f1s = [row['f1'] for report in reports for row in report['classes'] if row['used_in_score']]
    group_average = float(np.mean([report['macro_f1_supported'] for report in reports]))
    worst_class = min(f1s) if f1s else 0.
    # Both mean quality and weakest supported class must matter.
    score = .5*group_average + .5*worst_class
    return {'score': score, 'mean_group_macro_f1': group_average, 'worst_supported_class_f1': worst_class,
            'reports': reports}


def save_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
