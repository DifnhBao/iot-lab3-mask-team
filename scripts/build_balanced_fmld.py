"""Build balanced full-image YOLO data from the available MAFA part of FMLD.

Existing holdouts stay out of training. New val/test also exclude all previously
used images. Every box in a selected image is retained; no minority-only labels.
"""
import argparse
import hashlib
import json
import math
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

if __package__:
    from .prepare_mafa import image_index
    from .prepare_dataset import clip_boundary_box
else:
    from prepare_mafa import image_index
    from prepare_dataset import clip_boundary_box

NAMES = {0: 'with_mask', 1: 'without_mask', 2: 'mask_incorrect'}
CLASS_IDS = {'masked_face': 0, 'unmasked_face': 1, 'incorrectly_masked_face': 2}


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def old_assignments(folder):
    seen_hashes, holdout_hashes, seen_names, holdout_names = set(), set(), set(), set()
    for split in ['train', 'val', 'test']:
        for image in (Path(folder) / 'images' / split).iterdir():
            if image.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            digest = fingerprint(image)
            seen_hashes.add(digest)
            match = re.search(r'((?:train|test)_\d+\.(?:jpg|jpeg|png))$', image.name, re.I)
            if match:
                seen_names.add(match.group(1))
            if split != 'train':
                holdout_hashes.add(digest)
                if match:
                    holdout_names.add(match.group(1))
    return seen_hashes, holdout_hashes, seen_names, holdout_names


def read_record(xml_bytes, image):
    tree = ET.fromstring(xml_bytes)
    with Image.open(image) as im:
        width, height = im.size
        im.verify()
    size = tree.find('size')
    if size is None or (int(size.findtext('width')), int(size.findtext('height'))) != (width, height):
        raise ValueError('Annotation/image dimensions differ')
    objects = tree.findall('object')
    if not objects:
        raise ValueError('No annotated faces')
    lines, counts, clipped = [], [0, 0, 0], 0
    for obj in objects:
        label = obj.findtext('name')
        if label not in CLASS_IDS or obj.findtext('difficult', '0') != '0':
            raise ValueError('Image has uncertain/invalid/difficult face labels')
        box = obj.find('bndbox')
        if box is None:
            raise ValueError('Missing box')
        values = [float(box.findtext(key)) for key in ['xmin', 'ymin', 'xmax', 'ymax']]
        x1, y1, x2, y2 = clip_boundary_box(values, width, height)
        if min(x2-x1, y2-y1) < 16:
            raise ValueError('Image has a face smaller than 16px')
        clipped += tuple(values) != (x1, y1, x2, y2)
        cid = CLASS_IDS[label]
        counts[cid] += 1
        lines.append(f'{cid} {(x1+x2)/(2*width):.7f} {(y1+y2)/(2*height):.7f} {(x2-x1)/width:.7f} {(y2-y1)/height:.7f}')
    return {'path': image, 'file': image.name, 'counts': counts, 'lines': lines,
            'sha256': fingerprint(image), 'clipped_boxes': clipped}


def balanced_select(records, target, seed=42, cap_ratio=1.15):
    """Keep complete images, including every class, while capping class counts."""
    available = [sum(record['counts'][cid] for record in records) for cid in range(3)]
    target = min(target, min(available))
    if target < 1:
        raise ValueError(f'Not enough classes for a balanced split: {available}')
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    shuffled.sort(key=lambda record: sum(record['counts']))
    buckets = [[record for record in shuffled if record['counts'][cid] > 0] for cid in range(3)]
    positions, counts, selected, used, blocked = [0, 0, 0], [0, 0, 0], [], set(), set()
    cap = math.ceil(target * cap_ratio)
    while True:
        needed = [cid for cid in range(3) if counts[cid] < target and cid not in blocked]
        if not needed:
            break
        cid = min(needed, key=lambda c: counts[c])
        found = None
        while positions[cid] < len(buckets[cid]):
            candidate = buckets[cid][positions[cid]]
            positions[cid] += 1
            if candidate['sha256'] in used:
                continue
            if any(counts[c] + candidate['counts'][c] > cap for c in range(3)):
                continue
            found = candidate
            break
        if found is None:
            blocked.add(cid)
            continue
        used.add(found['sha256'])
        selected.append(found)
        counts = [counts[c] + found['counts'][c] for c in range(3)]
    if min(counts) < .9*target or max(counts) > 1.3*min(counts):
        raise ValueError(f'Cannot balance full images safely: {counts}, target={target}. Lower target or add data.')
    return selected, counts, target


def build(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError('Output exists; choose a new --output to preserve previous data.')
    index = image_index(args.images)
    seen, holdout, seen_names, holdout_names = old_assignments(args.previous)
    pools = {'train': [], 'test': []}
    skipped, observed, all_digests = [], Counter(), {}
    with ZipFile(args.annotations) as archive:
        members = [name for name in archive.namelist() if name.endswith('.xml')
                   and Path(name).stem.startswith(('train_', 'test_'))]
        for i, member in enumerate(members, 1):
            filename = Path(member).stem + '.jpg'
            origin = 'train' if '/train/' in member else 'test'
            if filename not in index:
                skipped.append({'file': filename, 'reason': 'missing_image'})
                continue
            try:
                record = read_record(archive.read(member), index[filename])
            except (ValueError, OSError, ET.ParseError, TypeError) as error:
                skipped.append({'file': filename, 'reason': str(error)})
                continue
            digest = record['sha256']
            if digest in all_digests:
                skipped.append({'file': filename, 'reason': f'duplicate_of:{all_digests[digest]}'})
                continue
            all_digests[digest] = filename
            record['origin'] = origin
            record['seen_before'] = digest in seen or filename in seen_names
            if origin == 'train' and (digest in holdout or filename in holdout_names):
                skipped.append({'file': filename, 'reason': 'previous_holdout_locked'})
                continue
            if origin == 'test' and record['seen_before']:
                skipped.append({'file': filename, 'reason': 'previously_seen_test_excluded'})
                continue
            pools[origin].append(record)
            observed[origin] += 1
            if i % 3000 == 0:
                print(f'Checked {i}/{len(members)} annotations', flush=True)
    # New val must not have been seen by any previous model.
    val_pool = [record for record in pools['train'] if not record['seen_before']]
    val, val_counts, val_target = balanced_select(val_pool, args.val_per_class, seed=43)
    val_hashes = {record['sha256'] for record in val}
    train_pool = [record for record in pools['train'] if record['sha256'] not in val_hashes]
    train, train_counts, train_target = balanced_select(train_pool, args.train_per_class)
    test, test_counts, test_target = balanced_select(pools['test'], args.test_per_class, seed=44)
    selected = {'train': train, 'val': val, 'test': test}
    assignments = {}
    for split, records in selected.items():
        for record in records:
            if record['sha256'] in assignments:
                raise ValueError(f'Duplicate between splits: {record["file"]}')
            assignments[record['sha256']] = split
    manifest = []
    for split, records in selected.items():
        image_dir, label_dir = output / 'images' / split, output / 'labels' / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            image = record['path']
            shutil.copyfile(image, image_dir / image.name)
            (label_dir / f'{image.stem}.txt').write_text('\n'.join(record['lines']) + '\n', encoding='utf-8')
            manifest.append({key: record[key] for key in ['file', 'origin', 'sha256', 'seen_before', 'clipped_boxes']} | {'split': split})
    (output / 'data.yaml').write_text(
        f"path: '{output.resolve().as_posix()}'\ntrain: images/train\nval: images/val\ntest: images/test\n"
        'names:\n  0: with_mask\n  1: without_mask\n  2: mask_incorrect\n', encoding='utf-8')
    summary = {'source': 'MAFA images with FMLD annotations; WIDER images not included',
               'selection': 'Whole images, every box preserved; target/cap per face class',
               'previous_dataset': args.previous, 'available_images': dict(observed),
               'new_validation_unseen': True, 'new_test_unseen': True,
               'requested_targets': {'train': args.train_per_class, 'val': args.val_per_class, 'test': args.test_per_class},
               'actual_targets': {'train': train_target, 'val': val_target, 'test': test_target},
               'splits': {split: {'images': len(selected[split]), 'faces': dict(enumerate(counts))}
                          for split, counts in [('train', train_counts), ('val', val_counts), ('test', test_counts)]},
               'skipped_reason_counts': dict(Counter(item['reason'].split(':')[0] for item in skipped)),
               'skipped_images': skipped,
               'limitations': ['Exact hashes and source filenames checked, not all near duplicates or identities.',
                               'Public labels require representative visual QA; difficult/unknown images excluded.',
                               'No guarantee of improved accuracy; compare all classes and existing regression sets.']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(summary['splits'], indent=2))
    print('Dataset:', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', default='datasets/fmld/source/FMLD_annotations.zip')
    parser.add_argument('--images', default='datasets/mafa/raw')
    parser.add_argument('--previous', default='datasets/mask_mafa_v2')
    parser.add_argument('--output', default='datasets/mask_balanced_fmld')
    parser.add_argument('--train-per-class', type=int, default=800)
    parser.add_argument('--val-per-class', type=int, default=80)
    parser.add_argument('--test-per-class', type=int, default=80)
    args = parser.parse_args()
    build(args)
