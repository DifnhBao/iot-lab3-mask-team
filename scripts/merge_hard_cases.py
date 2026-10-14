"""Build a new dataset, preserving split assignments and original files."""
import argparse
import hashlib
import math
import shutil
from pathlib import Path

from PIL import Image


def merge(base, extra, output):
    base, extra, output = map(Path, (base, extra, output))
    if output.exists():
        raise ValueError('Output exists; choose a new folder to preserve previous data.')
    records, seen, extra_counts = [], {}, {}
    for source_name, source in [('base', base), ('extra', extra)]:
        for split in ['train', 'val', 'test']:
            image_dir = source / 'images' / split
            if not image_dir.is_dir():
                raise ValueError(f'Missing {image_dir}; collect a separate session for each split.')
            subset = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
            if not subset:
                raise ValueError(f'No images in {image_dir}')
            if source_name == 'extra':
                extra_counts[split] = len(subset)
            for image in subset:
                with Image.open(image) as im:
                    im.verify()
                label = source / 'labels' / split / (image.stem + '.txt')
                if not label.is_file():
                    raise ValueError(f'Missing label: {label}')
                text = label.read_text(encoding='utf-8')
                if not text.strip():
                    raise ValueError(f'Empty face labels: {label}')
                for line in text.splitlines():
                    values = line.split()
                    if len(values) != 5:
                        raise ValueError(f'Invalid YOLO label: {label}')
                    cid, x, y, w, h = map(float, values)
                    if (not all(math.isfinite(v) for v in [cid, x, y, w, h]) or cid not in (0, 1, 2)
                            or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1)
                            or x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6):
                        raise ValueError(f'Invalid class/coordinates: {label}')
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                if digest in seen:
                    raise ValueError(f'Duplicate image: {image} and {seen[digest]}')
                seen[digest] = image
                records.append((source_name, split, image, label))
    for source_name, split, image, label in records:
        image_dir = output / 'images' / split
        label_dir = output / 'labels' / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        stem = f'{source_name}_{image.stem}'
        shutil.copyfile(image, image_dir / (stem + image.suffix.lower()))
        shutil.copyfile(label, label_dir / (stem + '.txt'))
    (output / 'data.yaml').write_text(
        f"path: '{output.resolve().as_posix()}'\ntrain: images/train\nval: images/val\ntest: images/test\n"
        'names:\n  0: with_mask\n  1: without_mask\n  2: mask_incorrect\n', encoding='utf-8')
    (extra / 'data.yaml').write_text(
        f"path: '{extra.resolve().as_posix()}'\ntrain: images/train\nval: images/val\ntest: images/test\n"
        'names:\n  0: with_mask\n  1: without_mask\n  2: mask_incorrect\n', encoding='utf-8')
    print(f'Merged {len(records)} images; added {extra_counts}; YAML: {output / "data.yaml"}')
    print('Exact duplicates checked; visually similar frames still require manual review.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='datasets/mask')
    parser.add_argument('--extra', default='datasets/hard_cases')
    parser.add_argument('--output', default='datasets/mask_hard_cases')
    args = parser.parse_args()
    merge(args.base, args.extra, args.output)
