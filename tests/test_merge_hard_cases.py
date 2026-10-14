import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
from scripts.merge_hard_cases import merge


class MergeTests(unittest.TestCase):
    def make_dataset(self, root):
        for source_index, source in enumerate(['base', 'extra']):
            for split_index, split in enumerate(['train', 'val', 'test']):
                images = root / source / 'images' / split
                labels = root / source / 'labels' / split
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                Image.new('RGB', (20, 20), (source_index*100, split_index*70, 10)).save(images / 'a.png')
                (labels / 'a.txt').write_text('1 0.5 0.5 0.5 0.5\n')

    def test_merge_preserves_splits_and_originals(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            merge(root / 'base', root / 'extra', root / 'merged')
            for split in ['train', 'val', 'test']:
                self.assertEqual(len(list((root / 'merged' / 'images' / split).glob('*.png'))), 2)
                self.assertTrue((root / 'base' / 'images' / split / 'a.png').is_file())

    def test_invalid_label_rejected_before_writing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            (root / 'extra' / 'labels' / 'train' / 'a.txt').write_text('1 2 0.5 0.5 0.5')
            with self.assertRaises(ValueError):
                merge(root / 'base', root / 'extra', root / 'merged')
            self.assertFalse((root / 'merged').exists())

    def test_duplicate_between_splits_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            (root / 'extra' / 'images' / 'test' / 'a.png').write_bytes(
                (root / 'extra' / 'images' / 'train' / 'a.png').read_bytes())
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                merge(root / 'base', root / 'extra', root / 'merged')
            self.assertFalse((root / 'merged').exists())


if __name__ == '__main__':
    unittest.main()
