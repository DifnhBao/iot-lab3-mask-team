"""Package prepared datasets and baseline weights for the Colab notebook."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def pack(root, output, training_dataset='mask_mafa', baseline='models/best.pt', extra_datasets=None,
         reference_models=None, runtime_scripts=None, evaluation_only=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Archive already exists; choose a new --output to preserve it.')
    entries, counts = [], {}
    datasets = [('mask_mafa', training_dataset), ('mafa_hand', 'mafa_hand'), ('mask', 'mask')]
    datasets += [(name, name) for name in (extra_datasets or [])]
    if len({alias for alias, source in datasets}) != len(datasets):
        raise ValueError('Dataset aliases must be unique.')
    for name, source_name in datasets:
        if Path(source_name).name != source_name or '/' in source_name or '\\' in source_name:
            raise ValueError(f'Expected dataset folder name: {source_name}')
        folder = root / 'datasets' / source_name
        if not (folder / 'data.yaml').is_file():
            raise ValueError(f'Missing {folder / "data.yaml"}; prepare/review/merge data first.')
        counts[name] = {}
        for split in ['train', 'val', 'test']:
            if evaluation_only and name != 'mask_mafa' and split == 'train':
                counts[name][split] = 0
                continue
            images = folder / 'images' / split
            labels = folder / 'labels' / split
            subset = sorted(p for p in images.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png'])
            if not subset:
                raise ValueError(f'Empty/missing images: {images}')
            counts[name][split] = len(subset)
            for image in subset:
                label = labels / (image.stem + '.txt')
                if not label.is_file() or not label.read_text(encoding='utf-8').strip():
                    raise ValueError(f'Missing/empty label: {label}')
                entries.extend([(image, f'datasets/{name}/images/{split}/{image.name}'),
                                (label, f'datasets/{name}/labels/{split}/{label.name}')])
        for filename in ['data.yaml', 'manifest.json', 'summary.json', 'conversion_report.json']:
            if (folder / filename).is_file():
                entries.append((folder / filename, f'datasets/{name}/{filename}'))
    weights = (root / baseline).resolve()
    if not weights.is_relative_to(root):
        raise ValueError('Baseline weights must be inside the project.')
    if not weights.is_file():
        raise ValueError(f'Missing baseline model: {weights}')
    entries.append((weights, 'models/best.pt'))
    references = {}
    for spec in reference_models or []:
        tag, separator, relative = spec.partition('=')
        if not separator or not tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in tag):
            raise ValueError('Use reference format: simple_tag=models/file.pt')
        if tag in references:
            raise ValueError(f'Duplicate reference tag: {tag}')
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Missing/invalid reference model: {relative}')
        archive_name = f'models/references/{tag}.pt'
        entries.append((path, archive_name))
        references[tag] = {'path': archive_name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for relative in runtime_scripts or []:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Missing/invalid runtime script: {relative}')
        entries.append((path, path.relative_to(root).as_posix()))
    for filename in ['train_mafa_annotation.csv', 'test_mafa_annotation_full.csv']:
        path = root / 'datasets' / 'mafa' / 'annotations' / filename
        if path.is_file():
            entries.append((path, path.relative_to(root).as_posix()))
    for filename in ['review.json', 'review_incorrect.json']:
        path = root / 'datasets' / 'mafa' / filename
        if path.is_file():
            entries.append((path, path.relative_to(root).as_posix()))
    metadata = {'baseline_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
                'counts': counts, 'names': {0: 'with_mask', 1: 'without_mask', 2: 'mask_incorrect'},
                'ultralytics_version': '8.4.175',
                'baseline_source': baseline,
                'dataset_sources': dict(datasets),
                'reference_models': references,
                'evaluation_only_reference_datasets': evaluation_only,
                'note': 'Colab notebook rewrites absolute Windows YAML paths after extraction.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'w', compression=ZIP_DEFLATED, compresslevel=3) as archive:
        for path, archive_name in entries:
            archive.write(path, archive_name)
        archive.writestr('bundle.json', json.dumps(metadata, indent=2))
    print(f'Created: {output} ({output.stat().st_size / 1024**2:.1f} MiB)')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument('--output', default='artifacts/mafa_colab_bundle.zip')
    parser.add_argument('--training-dataset', default='mask_mafa')
    parser.add_argument('--baseline', default='models/best.pt')
    parser.add_argument('--extra-dataset', action='append', default=[])
    parser.add_argument('--reference-model', action='append', default=[])
    parser.add_argument('--runtime-script', action='append', default=[])
    parser.add_argument('--evaluation-only', action='store_true',
                        help='Keep only val/test images for reference datasets; training dataset stays complete')
    args = parser.parse_args()
    pack(args.root, args.output, args.training_dataset, args.baseline, args.extra_dataset,
         args.reference_model, args.runtime_script, args.evaluation_only)
