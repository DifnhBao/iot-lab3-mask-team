import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from scripts.build_balanced_fmld import balanced_select, read_record, old_assignments
from scripts.balanced_colab_runtime import match_confusion, metrics_from_confusion
from scripts.make_balanced_context import crop_bounds, make
import json


class BalancedTests(unittest.TestCase):
    def test_context_preserves_parent_split_and_rebalances_after_filter(self):
        with TemporaryDirectory() as tmp:
            source=Path(tmp)/'source'
            manifest=[]
            for s,split in enumerate(['train','val','test']):
                images,labels=source/'images'/split,source/'labels'/split
                images.mkdir(parents=True); labels.mkdir(parents=True)
                for cid in range(3):
                    for i in range(20):
                        name=f'{split}_{cid}_{i}.jpg'
                        Image.new('RGB',(100,100),(cid*70,s*70,i)).save(images/name)
                        (labels/(Path(name).stem+'.txt')).write_text(f'{cid} .5 .5 .6 .6\n')
                        manifest.append({'file':name,'split':split,'sha256':name})
            Image.new('RGB',(100,100),'red').save(source/'images/train/small.jpg')
            (source/'labels/train/small.txt').write_text('0 .5 .5 .1 .1\n')
            manifest.append({'file':'small.jpg','split':'train','sha256':'small'})
            (source/'manifest.json').write_text(json.dumps(manifest))
            (source/'summary.json').write_text(json.dumps({'limitations':[]}))
            make(source,Path(tmp)/'output')
            out=Path(tmp)/'output'
            self.assertFalse((out/'images/train/small.jpg').exists())
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['splits']['train']['faces'],{'0':20,'1':20,'2':20})
            for r in json.loads((out/'manifest.json').read_text()):
                self.assertTrue(r['file'].startswith(r['split']+'_'))
                self.assertEqual(r['parent_sha256'],r['file'])
            coords=list(map(float,next((out/'labels/train').glob('*.txt')).read_text().split()[1:]))
            self.assertTrue(all(0<v<=1 for v in coords))

    def test_context_crop_keeps_face_and_extra_chin_area(self):
        bounds=crop_bounds((30,20,70,70),100,100)
        self.assertLessEqual(bounds[0],30)
        self.assertLessEqual(bounds[1],20)
        self.assertGreaterEqual(bounds[2],70)
        self.assertGreater(bounds[3],70)
        self.assertGreaterEqual(bounds[0],0)
        self.assertLessEqual(bounds[3],100)

    def test_selector_preserves_all_boxes_and_caps_classes(self):
        records=[]
        for cid in range(3):
            for i in range(12):
                counts=[0,0,0]
                counts[cid]=1
                records.append({'sha256':f'{cid}_{i}','counts':counts})
        records.append({'sha256':'crowd','counts':[20,0,1]})
        selected,counts,target=balanced_select(records,10)
        self.assertEqual(counts,[10,10,10])
        self.assertEqual(target,10)
        self.assertNotIn('crowd',[r['sha256'] for r in selected])
        self.assertEqual(len({r['sha256'] for r in selected}),len(selected))

    def test_xml_unknown_face_rejects_entire_image(self):
        with TemporaryDirectory() as tmp:
            image=Path(tmp)/'train_00000001.jpg'
            Image.new('RGB',(100,100),'white').save(image)
            xml='''<annotation><size><width>100</width><height>100</height></size>
            <object><name>incorrectly_masked_face</name><difficult>0</difficult><bndbox>
            <xmin>20</xmin><ymin>20</ymin><xmax>80</xmax><ymax>80</ymax></bndbox></object></annotation>'''
            record=read_record(xml.encode(),image)
            self.assertEqual(record['counts'],[0,0,1])
            self.assertEqual(record['lines'][0].split()[0],'2')
            with self.assertRaisesRegex(ValueError,'uncertain'):
                read_record(xml.replace('incorrectly_masked_face','invalid_face').encode(),image)

    def test_old_holdout_locked_by_hash_and_original_name(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            for split in ['train','val','test']:
                (root/'images'/split).mkdir(parents=True)
            image=root/'images/val/base_extra_train_00000001.jpg'
            image.write_bytes(b'heldout')
            seen,heldout,names,heldout_names=old_assignments(root)
            self.assertEqual(seen,heldout)
            self.assertIn('train_00000001.jpg',heldout_names)

    def test_wrong_class_counts_as_error_not_detection_success(self):
        truth=[[1,0,0,100,100]]
        pred=[[2,.9,0,0,100,100]]
        matrix=match_confusion(truth,pred)
        self.assertEqual(matrix[1,2],1)
        rows,_=metrics_from_confusion(matrix,minimum_support=1)
        self.assertEqual(rows[1]['fn'],1)
        self.assertEqual(rows[2]['fp'],1)
        self.assertEqual(rows[1]['f1'],0)

    def test_missing_and_duplicate_boxes_are_penalized(self):
        matrix=match_confusion([[2,0,0,100,100]],[])
        self.assertEqual(matrix[2,3],1)
        matrix=match_confusion([[0,0,0,100,100]],[[0,.9,0,0,100,100],[0,.8,0,0,100,100]])
        self.assertEqual(matrix[0,0],1)
        self.assertEqual(matrix[3,0],1)
        rows,_=metrics_from_confusion(matrix,minimum_support=1)
        self.assertEqual(rows[0]['precision'],.5)
        self.assertEqual(rows[0]['recall'],1.)


if __name__ == '__main__':
    unittest.main()
