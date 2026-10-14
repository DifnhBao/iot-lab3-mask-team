"""Capture and manually label every visible face, without model-generated labels."""
import argparse
import sys
import uuid
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--output', default='datasets/hard_cases')
    parser.add_argument('--class-id', type=int, choices=[0, 1, 2], default=1,
                        help='Initial class for new boxes; keys 0/1/2 still change it')
    args = parser.parse_args()
    folder = Path(args.output)
    images = folder / 'images' / args.split
    labels = folder / 'labels' / args.split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.index, cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY)
    window = 'Collect faces - SPACE freeze, Q quit'
    names = ['with_mask', 'without_mask', 'mask_incorrect']
    colors = [(0, 220, 0), (0, 0, 255), (0, 170, 255)]
    saved = 0
    print('Close the recognition app first to release the webcam.', flush=True)
    print('SPACE: freeze. On frozen image: 0/1/2 choose class; R draw face box; U undo; S save; N discard; Q quit.', flush=True)
    print('1 = no mask, including a HAND covering nose/mouth. Label ALL visible faces before saving.', flush=True)
    print('Use different people or recording sessions for train / val / test.', flush=True)
    frozen = None
    boxes = []
    class_id = args.class_id
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open webcam. Close other camera apps or try --index 1.')
        while True:
            if frozen is None:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Cannot read webcam frame.')
                display = frame.copy()
                hint = f'SPACE freeze | Q quit | saved {saved} | {args.split}'
            else:
                display = frozen.copy()
                for cid, x, y, w, h in boxes:
                    cv2.rectangle(display, (x, y), (x+w, y+h), colors[cid], 2)
                    cv2.putText(display, names[cid], (x, max(40, y-5)), 0, .5, colors[cid], 1)
                hint = f'Class {class_id}: {names[class_id]} | R box | U undo | S save | N discard'
            cv2.putText(display, hint, (8, 22), 0, .5, (255, 255, 255), 2)
            cv2.imshow(window, display)
            key = cv2.waitKey(30) & 255
            if key in (ord('q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if frozen is None:
                if key == 32:
                    frozen = frame.copy()
                    boxes = []
                continue
            if key in (ord('0'), ord('1'), ord('2')):
                class_id = key - ord('0')
            elif key == ord('r'):
                x, y, w, h = map(int, cv2.selectROI('Draw entire face - ENTER confirm, C cancel', frozen, False, False))
                cv2.destroyWindow('Draw entire face - ENTER confirm, C cancel')
                if w > 0 and h > 0:
                    boxes.append((class_id, x, y, w, h))
            elif key == ord('u') and boxes:
                boxes.pop()
            elif key == ord('n'):
                frozen = None
            elif key == ord('s'):
                if not boxes:
                    print('Draw and label all face boxes before saving.', flush=True)
                    continue
                height, width = frozen.shape[:2]
                stem = uuid.uuid4().hex
                if not cv2.imwrite(str(images / f'{stem}.jpg'), frozen):
                    raise RuntimeError('Could not save image')
                lines = [f'{cid} {(x+w/2)/width:.7f} {(y+h/2)/height:.7f} {w/width:.7f} {h/height:.7f}'
                         for cid, x, y, w, h in boxes]
                (labels / f'{stem}.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
                saved += 1
                print(f'Saved {saved}: {images / (stem + ".jpg")}', flush=True)
                frozen = None
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
