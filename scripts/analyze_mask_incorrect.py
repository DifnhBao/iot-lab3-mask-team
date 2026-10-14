"""Read-only audit; never approves labels or changes review decisions."""
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from prepare_mafa import read_annotations, image_index, excluded_hashes, candidate_names, convert_box


def main():
    groups = read_annotations('datasets/mafa/annotations')
    index = image_index('datasets/mafa/raw')
    exclusions = excluded_hashes(['datasets/mask_mafa'])
    training = Counter(int(line.split()[0]) for path in Path('datasets/mask_mafa/labels/train').glob('*.txt')
                       for line in path.read_text().splitlines() if line.strip())
    report = {'current_training_faces': dict(training), 'source_candidates': {},
              'note': 'Counts are geometry checks only; all class meanings and visible face boxes require manual review.'}
    for origin in ['train', 'test']:
        names, excluded = candidate_names(groups, index, origin, 2, exclusions)
        invalid, valid, missing = [], [], []
        for name in names:
            if name not in index:
                missing.append(name)
                continue
            try:
                with Image.open(index[name]) as image:
                    width, height = image.size
                    image.verify()
                for box in groups[origin][name]:
                    convert_box(box, width, height)
            except (ValueError, OSError) as error:
                invalid.append({'file': name, 'reason': str(error)})
                continue
            valid.append(name)
        report['source_candidates'][origin] = {
            'candidate_images': len(names), 'valid_geometry_images': len(valid),
            'already_used_images': excluded, 'invalid_images': invalid,
            'missing_images': missing, 'review_candidate_filenames': valid}
        print(origin, 'candidates=', len(names), 'valid_geometry=', len(valid),
              'excluded_already_used=', len(excluded), 'bad_geometry=', len(invalid), 'missing=', len(missing))
    output = Path('artifacts/mask_incorrect_analysis.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Current train faces:', dict(training))
    print('Report:', output)


if __name__ == '__main__':
    main()
