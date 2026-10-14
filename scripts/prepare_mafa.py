"""Review AnnotationMAFA object-occluded faces, then export three-class YOLO data."""
import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

CLASS_MAP = {0: 1, 1: 0, 2: 2, 3: 1}
NAMES = {0: 'with_mask', 1: 'without_mask', 2: 'mask_incorrect'}
SOURCE_NAMES = {0: 'no_mask', 1: 'mask_correct', 2: 'mask_incorrect', 3: 'other_object'}


def read_annotations(folder):
    groups = {}
    for origin, filename in [('train', 'train_mafa_annotation.csv'),
                             ('test', 'test_mafa_annotation_full.csv')]:
        result = defaultdict(list)
        with (Path(folder) / filename).open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if not {'name_file', 'x', 'y', 'w', 'h', 'label'}.issubset(reader.fieldnames or []):
                raise ValueError(f'Unexpected CSV columns: {filename}')
            for row in reader:
                name = row['name_file']
                if Path(name).name != name or '/' in name or '\\' in name:
                    raise ValueError(f'Unexpected filename: {name}')
                x, y, w, h = (float(row[key]) for key in ['x', 'y', 'w', 'h'])
                label = int(row['label'])
                if label not in CLASS_MAP or not all(math.isfinite(v) for v in [x, y, w, h]) or w <= 0 or h <= 0:
                    raise ValueError(f'Invalid annotation: {row}')
                result[name].append((label, x, y, w, h))
        groups[origin] = dict(result)
    return groups


def image_index(folder):
    index = {}
    for path in Path(folder).rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            if path.name in index:
                raise ValueError(f'Duplicate image filename: {path.name}')
            index[path.name] = path
    return index


def load_review(path):
    if path.is_file():
        return json.loads(path.read_text(encoding='utf-8'))
    return {'train': {}, 'test': {}}


def excluded_hashes(folders):
    """Do not reintroduce already-used images into a different split."""
    hashes = set()
    for folder in folders:
        image_folder = Path(folder) / 'images'
        if not image_folder.is_dir():
            raise ValueError(f'Missing exclusion dataset: {image_folder}')
        for path in image_folder.rglob('*'):
            if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
    return hashes


def candidate_names(groups, index, source, target_class, exclusions):
    names, excluded = [], []
    for name, boxes in groups[source].items():
        if not any(box[0] == target_class for box in boxes):
            continue
        if name in index and hashlib.sha256(index[name].read_bytes()).hexdigest() in exclusions:
            excluded.append(name)
        else:
            names.append(name)
    return names, excluded


def review(args):
    import cv2
    groups, index = read_annotations(args.annotations), image_index(args.images)
    review_path = Path(args.review_file)
    decisions = load_review(review_path)
    target_class = getattr(args, 'target_class', 3)
    exclusions = excluded_hashes(getattr(args, 'exclude_dataset', []))
    candidates, excluded = candidate_names(groups, index, args.source, target_class, exclusions)
    print(f'Target class {target_class}: {len(candidates)} candidates; {len(excluded)} already-used images excluded.', flush=True)
    random.Random(42).shuffle(candidates)
    remaining = [name for name in candidates if name not in decisions[args.source]]
    missing = [name for name in remaining if name not in index]
    if missing:
        raise ValueError(f'{len(missing)} candidate images missing; first: {missing[0]}. Check extracted MAFA folders.')
    print('K = KEEP after checking all boxes/classes; X = exclude; Q = quit. Decisions save immediately.', flush=True)
    if target_class == 2:
        print('Keep actual incorrectly worn masks: exposed nose, exposed mouth, or mask on chin.', flush=True)
    else:
        print('Keep only images with hand covering nose/mouth and no actual mask on that face.', flush=True)
    print('Exclude ambiguous faces, wrong/incomplete boxes or any class 3 face with an actual mask.', flush=True)
    window = 'MAFA review - K keep, X exclude, Q quit'
    try:
        for position, name in enumerate(remaining, 1):
            image = cv2.imread(str(index[name]))
            if image is None:
                raise ValueError(f'Unreadable image: {index[name]}')
            height, width = image.shape[:2]
            scale = min(1, 1100 / width, 700 / height)
            display = cv2.resize(image, (max(1, round(width*scale)), max(1, round(height*scale))))
            for label, x, y, w, h in groups[args.source][name]:
                color = (0, 160, 255) if label == target_class else (0, 220, 0)
                p1, p2 = (round(x*scale), round(y*scale)), (round((x+w)*scale), round((y+h)*scale))
                cv2.rectangle(display, p1, p2, color, 2)
                cv2.putText(display, f'{label}:{SOURCE_NAMES[label]}', (p1[0], max(15, p1[1]-5)), 0, .45, color, 1)
            kept = sum(value == 'keep' for value in decisions[args.source].values())
            banner = cv2.copyMakeBorder(display, 45, 0, 0, 0, cv2.BORDER_CONSTANT)
            box_error = None
            try:
                for box in groups[args.source][name]:
                    convert_box(box, width, height)
            except ValueError as error:
                box_error = str(error)
            cv2.putText(banner, f'{name} | {position}/{len(remaining)} | kept={kept} | K keep / X exclude / Q quit',
                        (8, 25), 0, .5, (255, 255, 255), 1)
            if box_error:
                cv2.putText(banner, 'INVALID BOX - press X to exclude', (8, 42), 0, .45, (0, 0, 255), 1)
            cv2.imshow(window, banner)
            while True:
                key = cv2.waitKey(100) & 255
                if key in (ord('q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return
                if key in (ord('k'), ord('x')):
                    if key == ord('k') and box_error:
                        print(f'Cannot keep {name}: {box_error}. Press X.', flush=True)
                        continue
                    decisions[args.source][name] = 'keep' if key == ord('k') else 'exclude'
                    review_path.parent.mkdir(parents=True, exist_ok=True)
                    review_path.write_text(json.dumps(decisions, indent=2), encoding='utf-8')
                    break
    finally:
        cv2.destroyAllWindows()


def convert_box(box, width, height):
    label, x, y, w, h = box
    # Preserve source x/y convention; review overlays before exporting.
    overflow = max(0, -x, -y, x+w-width, y+h-height)
    if overflow > 1:
        raise ValueError(f'Box exceeds image boundary by {overflow:g}px: {box}')
    x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x+w), min(height, y+h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f'Empty box: {box}')
    return f'{CLASS_MAP[label]} {(x1+x2)/(2*width):.7f} {(y1+y2)/(2*height):.7f} {(x2-x1)/width:.7f} {(y2-y1)/height:.7f}'


def convert(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError('Output already exists; choose a new --output.')
    groups, index = read_annotations(args.annotations), image_index(args.images)
    decisions = load_review(Path(args.review_file))
    target_class = getattr(args, 'target_class', 3)
    exclusions = excluded_hashes(getattr(args, 'exclude_dataset', []))
    selected = {origin: sorted(name for name, decision in decisions[origin].items() if decision == 'keep')
                for origin in ['train', 'test']}
    if len(selected['train']) < 10 or len(selected['test']) < 5:
        raise ValueError('Review at least 10 train and 5 test images; more diverse data is recommended.')
    random.Random(42).shuffle(selected['train'])
    n_val = max(1, round(len(selected['train']) * .15))
    assignments = [('val', 'train', name) for name in selected['train'][:n_val]]
    assignments += [('train', 'train', name) for name in selected['train'][n_val:]]
    assignments += [('test', 'test', name) for name in selected['test']]
    seen, records, skipped = {}, [], []
    for split, origin, name in assignments:
        if name not in groups[origin] or name not in index:
            raise ValueError(f'Review references a missing image/annotation: {name}')
        if not any(box[0] == target_class for box in groups[origin][name]):
            raise ValueError(f'Image has no source class {target_class}: {name}')
        path = index[name]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in exclusions:
            raise ValueError(f'Already-used image selected: {name}; exclude it from this review to preserve splits.')
        if digest in seen:
            raise ValueError(f'Duplicate content: {name} and {seen[digest]}')
        seen[digest] = name
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        try:
            lines = [convert_box(box, width, height) for box in groups[origin][name]]
        except ValueError as error:
            reason = f'{name} ({width}x{height}): {error}'
            if not getattr(args, 'skip_invalid_images', False):
                raise ValueError(reason + '; review this image or use --skip-invalid-images to exclude the entire image.') from error
            skipped.append({'file': name, 'split': split, 'source_split': origin,
                            'image_size': [width, height], 'reason': str(error)})
            print(f'SKIP {reason}')
            continue
        records.append((split, origin, path, lines, digest))
    for split in ['train', 'val', 'test']:
        if not any(record[0] == split for record in records):
            raise ValueError(f'No valid images remain in {split}; review more images before exporting.')
    manifest, counts = [], Counter()
    for split, origin, path, lines, digest in records:
        image_dir, label_dir = output / 'images' / split, output / 'labels' / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, image_dir / path.name)
        (label_dir / f'{path.stem}.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        counts[split] += 1
        manifest.append({'file': path.name, 'split': split, 'source_split': origin, 'sha256': digest})
    (output / 'data.yaml').write_text(
        f"path: '{output.resolve().as_posix()}'\ntrain: images/train\nval: images/val\ntest: images/test\n"
        'names:\n  0: with_mask\n  1: without_mask\n  2: mask_incorrect\n', encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output / 'conversion_report.json').write_text(json.dumps(
        {'target_source_class': target_class, 'excluded_datasets': getattr(args, 'exclude_dataset', []),
         'selected_counts': {source: len(names) for source, names in selected.items()},
         'exported_counts': dict(counts), 'skipped_images': skipped}, indent=2), encoding='utf-8')
    print(f'Exported {dict(counts)} to {output}. Class 3 mapped to no mask ONLY for reviewed images.')
    print(f'Skipped {len(skipped)} invalid images; details: {output / "conversion_report.json"}. Source CSV/review unchanged.')
    print('Exact duplicates checked; review similar images across splits manually.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['review', 'convert'])
    parser.add_argument('--annotations', default='datasets/mafa/annotations')
    parser.add_argument('--images', default='datasets/mafa/raw')
    parser.add_argument('--review-file', default=None)
    parser.add_argument('--source', choices=['train', 'test'], default='train')
    parser.add_argument('--output', default=None)
    parser.add_argument('--target-class', type=int, choices=[2, 3], default=3)
    parser.add_argument('--exclude-dataset', action='append', default=[],
                        help='Exclude content already used in this dataset; repeat for multiple datasets')
    parser.add_argument('--skip-invalid-images', action='store_true',
                        help='Exclude an entire image with invalid boxes and record it in conversion_report.json')
    args = parser.parse_args()
    args.review_file = args.review_file or ('datasets/mafa/review_incorrect.json' if args.target_class == 2 else 'datasets/mafa/review.json')
    args.output = args.output or ('datasets/mafa_incorrect' if args.target_class == 2 else 'datasets/mafa_hand')
    (review if args.mode == 'review' else convert)(args)


if __name__ == '__main__':
    main()
