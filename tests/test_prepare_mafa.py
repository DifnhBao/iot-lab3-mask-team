import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from PIL import Image
from scripts.prepare_mafa import convert, convert_box, candidate_names, read_annotations, image_index, excluded_hashes


class MafaTests(unittest.TestCase):
    def test_class_mapping_and_xywh(self):
        for old, new in [(0, 1), (1, 0), (2, 2), (3, 1)]:
            values = convert_box((old, 10, 20, 30, 40), 100, 100).split()
            self.assertEqual(int(values[0]), new)
            self.assertEqual(list(map(float, values[1:])), [.25, .4, .3, .4])
        with self.assertRaises(ValueError):
            convert_box((3, 90, 20, 30, 40), 100, 100)

    def fixture(self, root):
        annotations, images = root / 'annotations', root / 'images'
        annotations.mkdir()
        images.mkdir()
        decisions = {'train': {}, 'test': {}}
        for origin, count in [('train', 10), ('test', 5)]:
            filename = 'train_mafa_annotation.csv' if origin == 'train' else 'test_mafa_annotation_full.csv'
            with (annotations / filename).open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['name_file', 'x', 'y', 'w', 'h', 'label'])
                for i in range(count):
                    name = f'{origin}_{i:08d}.png'
                    Image.new('RGB', (100, 100), (i*15, 10 if origin == 'train' else 220, 0)).save(images / name)
                    writer.writerow([name, 10, 20, 30, 40, 3])
                    writer.writerow([name, 50, 20, 30, 40, 1])
                    decisions[origin][name] = 'keep'
        review = root / 'review.json'
        review.write_text(json.dumps(decisions))
        return SimpleNamespace(annotations=annotations, images=images, review_file=review, output=root / 'export')

    def test_source_test_held_out_and_all_boxes_kept(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            convert(args)
            output = args.output
            self.assertEqual(len(list((output / 'images' / 'test').glob('*.png'))), 5)
            self.assertFalse(list((output / 'images' / 'train').glob('test_*')))
            lines = next((output / 'labels' / 'test').glob('*.txt')).read_text().splitlines()
            self.assertEqual([int(line.split()[0]) for line in lines], [1, 0])

    def test_duplicate_rejected_before_export(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            (args.images / 'test_00000000.png').write_bytes((args.images / 'train_00000000.png').read_bytes())
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                convert(args)
            self.assertFalse(args.output.exists())

    def test_bad_boundary_has_filename_and_no_partial_export(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            csv_path = args.annotations / 'test_mafa_annotation_full.csv'
            csv_path.write_text(csv_path.read_text().replace('test_00000000.png,10,20,30,40,3',
                                                              'test_00000000.png,90,20,30,40,3'))
            with self.assertRaisesRegex(ValueError, 'test_00000000.png'):
                convert(args)
            self.assertFalse(args.output.exists())

    def test_skip_invalid_excludes_whole_image_and_preserves_review(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            original_review = args.review_file.read_bytes()
            csv_path = args.annotations / 'test_mafa_annotation_full.csv'
            csv_path.write_text(csv_path.read_text().replace('test_00000000.png,10,20,30,40,3',
                                                              'test_00000000.png,90,20,30,40,3'))
            args.skip_invalid_images = True
            convert(args)
            self.assertFalse((args.output / 'images/test/test_00000000.png').exists())
            self.assertFalse((args.output / 'labels/test/test_00000000.txt').exists())
            self.assertEqual(args.review_file.read_bytes(), original_review)
            report = json.loads((args.output / 'conversion_report.json').read_text())
            self.assertEqual(len(report['skipped_images']), 1)
            self.assertEqual(report['exported_counts']['test'], 4)

    def test_incorrect_target_exports_class_two_without_touching_old_review(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            for csv_path in args.annotations.glob('*.csv'):
                csv_path.write_text(csv_path.read_text().replace(',30,40,3', ',30,40,2'))
            args.target_class = 2
            original_review = args.review_file.read_bytes()
            convert(args)
            lines = next((args.output / 'labels/train').glob('*.txt')).read_text().splitlines()
            self.assertEqual(int(lines[0].split()[0]), 2)
            self.assertEqual(args.review_file.read_bytes(), original_review)

    def test_already_used_content_excluded_from_new_candidates(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixture(root)
            csv_path = args.annotations / 'train_mafa_annotation.csv'
            csv_path.write_text(csv_path.read_text().replace(',30,40,3', ',30,40,2'))
            reserved = root / 'reserved/images/val'
            reserved.mkdir(parents=True)
            (reserved / 'renamed.png').write_bytes((args.images / 'train_00000000.png').read_bytes())
            exclusions = excluded_hashes([root / 'reserved'])
            names, excluded = candidate_names(read_annotations(args.annotations), image_index(args.images), 'train', 2, exclusions)
            self.assertNotIn('train_00000000.png', names)
            self.assertEqual(excluded, ['train_00000000.png'])
            args.target_class = 2
            args.exclude_dataset = [str(root / 'reserved')]
            with self.assertRaisesRegex(ValueError, 'Already-used'):
                convert(args)
            self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
