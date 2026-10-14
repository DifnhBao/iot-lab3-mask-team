"""Reduce unlabeled background faces using contextual crops, preserving splits."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image


def crop_bounds(box, width, height):
    x1,y1,x2,y2 = box
    w,h=x2-x1,y2-y1
    return (max(0,int(x1-.3*w)),max(0,int(y1-.2*h)),
            min(width,int(x2+.3*w+1)),min(height,int(y2+.4*h+1)))


def make(source, output, minimum_face_side=48):
    source,output=Path(source),Path(output)
    if output.exists(): raise ValueError('Output exists; choose a new output.')
    source_manifest={r['file']:r for r in json.loads((source/'manifest.json').read_text())}
    records=[]
    small_faces=Counter()
    # Validate everything before creating the output.
    for split in ['train','val','test']:
        for image in sorted((source/'images'/split).glob('*.jpg')):
            lines=(source/'labels'/split/(image.stem+'.txt')).read_text().splitlines()
            if len(lines)!=1: raise ValueError(f'Expected one annotated face: {image}')
            cid,x,y,w,h=map(float,lines[0].split())
            with Image.open(image) as im: width,height=im.size
            box=((x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height)
            if min(box[2]-box[0],box[3]-box[1]) < minimum_face_side:
                small_faces[split]+=1
                continue
            bounds=crop_bounds(box,width,height)
            records.append((split,image,int(cid),box,bounds))
    # Keep the same count per class after removing faces too small to show nose/mouth.
    selected=[]
    for split in ['train','val','test']:
        buckets={cid:[r for r in records if r[0]==split and r[2]==cid] for cid in range(3)}
        target=min(len(bucket) for bucket in buckets.values())
        if target<20: raise ValueError(f'Too few clear faces in {split}: {target}')
        for cid in range(3):
            random.Random(42+cid).shuffle(buckets[cid])
            selected.extend(buckets[cid][:target])
    records=selected
    counts={s:Counter() for s in ['train','val','test']}
    manifest=[]
    for split,image,cid,box,bounds in records:
        image_dir,label_dir=output/'images'/split,output/'labels'/split
        image_dir.mkdir(parents=True,exist_ok=True); label_dir.mkdir(parents=True,exist_ok=True)
        with Image.open(image) as im: crop=im.convert('RGB').crop(bounds)
        target=image_dir/image.name
        crop.save(target,'JPEG',quality=95)
        width,height=crop.size
        x1,y1,x2,y2=box
        x1-=bounds[0]; x2-=bounds[0]; y1-=bounds[1]; y2-=bounds[1]
        line=f'{cid} {(x1+x2)/(2*width):.7f} {(y1+y2)/(2*height):.7f} {(x2-x1)/width:.7f} {(y2-y1)/height:.7f}\n'
        (label_dir/(image.stem+'.txt')).write_text(line)
        counts[split][cid]+=1
        parent=source_manifest[image.name]
        manifest.append({**parent,'parent_sha256':parent['sha256'],
                         'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                         'crop_xyxy':list(bounds)})
    summary=json.loads((source/'summary.json').read_text())
    summary['selection']='One contextual crop per source image; parent split unchanged; every source annotation retained.'
    summary['context_crop']={'side_padding':.3,'top_padding':.2,'bottom_padding':.4,
                             'minimum_source_face_side':minimum_face_side,
                             'small_faces_excluded':dict(small_faces),
                             'reason':'Reduce known unlabeled background faces; preserve chin area; filter low resolution.'}
    summary['limitations'].append('Context crops reduce background label noise, but cannot guarantee all nearby faces are annotated. Full-frame regression tests remain necessary.')
    summary['splits']={s:{'images':sum(c.values()),'faces':dict(c)} for s,c in counts.items()}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'data.yaml').write_text(f"path: '{output.resolve().as_posix()}'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: with_mask\n  1: without_mask\n  2: mask_incorrect\n")
    print(json.dumps(summary['splits'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='datasets/mask_balanced_fmld')
    parser.add_argument('--output',default='datasets/mask_balanced_context_hq')
    parser.add_argument('--minimum-face-side',type=int,default=48)
    args=parser.parse_args()
    make(args.source,args.output,args.minimum_face_side)
