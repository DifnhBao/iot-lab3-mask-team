"""Render representative source annotations for manual visual QA."""
import random
import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',default='datasets/mask_balanced_fmld')
    parser.add_argument('--output',default='artifacts/balanced_qa')
    args=parser.parse_args()
    dataset = Path(args.dataset)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    names = ['with_mask', 'without_mask', 'mask_incorrect']
    colors = ['lime', 'red', 'orange']
    for split in ['train', 'val', 'test']:
        by_class = {cid: [] for cid in range(3)}
        for label in sorted((dataset/'labels'/split).glob('*.txt')):
            cid = int(label.read_text().split()[0])
            by_class[cid].append(label)
        canvas = Image.new('RGB', (1200, 960), '#20242b')
        for cid in range(3):
            sample = random.Random(43+cid).sample(by_class[cid], min(4, len(by_class[cid])))
            for col, label in enumerate(sample):
                source = dataset/'images'/split/(label.stem+'.jpg')
                im = Image.open(source).convert('RGB')
                width, height = im.size
                draw = ImageDraw.Draw(im)
                for line in label.read_text().splitlines():
                    cls,x,y,w,h = map(float,line.split())
                    draw.rectangle(((x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height),
                                   outline=colors[int(cls)],width=3)
                im.thumbnail((290,270))
                origin=(col*300+5,cid*320+40)
                canvas.paste(im,origin)
                d=ImageDraw.Draw(canvas)
                d.text((col*300+5,cid*320+5),names[cid],fill=colors[cid])
                d.text((col*300+5,cid*320+23),source.name,fill='white')
        path=out/f'{split}_labels.png'
        canvas.save(path)
        print(path)


if __name__ == '__main__':
    main()
