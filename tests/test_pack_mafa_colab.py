import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts.pack_mafa_colab import pack


class PackTests(unittest.TestCase):
    def test_bundle_contains_prepared_data_and_baseline(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset in ['mask_mafa', 'mafa_hand', 'mask']:
                folder = root / 'datasets' / dataset
                for split in ['train', 'val', 'test']:
                    images, labels = folder / 'images' / split, folder / 'labels' / split
                    images.mkdir(parents=True)
                    labels.mkdir(parents=True)
                    (images / 'a.jpg').write_bytes(b'fixture-only')
                    (labels / 'a.txt').write_text('1 0.5 0.5 0.5 0.5')
                (folder / 'data.yaml').write_text('path: placeholder')
            (root / 'models').mkdir()
            (root / 'models/best.pt').write_bytes(b'baseline-fixture')
            output = root / 'bundle.zip'
            pack(root, output)
            with ZipFile(output) as archive:
                self.assertIn('models/best.pt', archive.namelist())
                self.assertNotIn('datasets/mafa/raw', archive.namelist())
                metadata = json.loads(archive.read('bundle.json'))
                self.assertEqual(metadata['counts']['mafa_hand']['test'], 1)
                self.assertEqual(archive.read('models/best.pt'), b'baseline-fixture')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                pack(root, output)
            (root / 'models/round1.pt').write_bytes(b'round-one-fixture')
            # Use an actual different source folder but retain the portable alias.
            import shutil
            shutil.copytree(root / 'datasets/mask_mafa', root / 'datasets/round2')
            pack(root, root / 'v2.zip', training_dataset='round2', baseline='models/round1.pt')
            with ZipFile(root / 'v2.zip') as archive:
                self.assertEqual(archive.read('models/best.pt'), b'round-one-fixture')
                self.assertIn('datasets/mask_mafa/images/train/a.jpg', archive.namelist())
                metadata = json.loads(archive.read('bundle.json'))
                self.assertEqual(metadata['dataset_sources']['mask_mafa'], 'round2')
            (root/'scripts').mkdir()
            (root/'scripts/runtime.py').write_text('print(1)')
            pack(root,root/'references.zip',reference_models=['original=models/best.pt'],
                 runtime_scripts=['scripts/runtime.py'])
            with ZipFile(root/'references.zip') as archive:
                self.assertIn('models/references/original.pt',archive.namelist())
                self.assertIn('scripts/runtime.py',archive.namelist())
                metadata=json.loads(archive.read('bundle.json'))
                self.assertIn('original',metadata['reference_models'])
            pack(root,root/'compact.zip',evaluation_only=True)
            with ZipFile(root/'compact.zip') as archive:
                self.assertIn('datasets/mask_mafa/images/train/a.jpg',archive.namelist())
                self.assertNotIn('datasets/mask/images/train/a.jpg',archive.namelist())
                self.assertIn('datasets/mask/images/test/a.jpg',archive.namelist())
                metadata=json.loads(archive.read('bundle.json'))
                self.assertEqual(metadata['counts']['mask']['train'],0)

    def test_missing_prepared_data_rejected_without_archive(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'Missing'):
                pack(root, root / 'bundle.zip')
            self.assertFalse((root / 'bundle.zip').exists())


if __name__ == '__main__':
    unittest.main()
