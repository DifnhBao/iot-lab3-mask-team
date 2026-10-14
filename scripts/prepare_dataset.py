"""Convert a Pascal VOC face-mask dataset to three-class YOLO layout."""
import argparse
import hashlib
import json
import math
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image

CLASS_IDS = {"with_mask": 0, "without_mask": 1,
             "mask_incorrect": 2, "mask_weared_incorrect": 2}


def clip_boundary_box(values, width, height, tolerance=1.0):
    """Clip tiny boundary overflows, without changing raw XML or shifting all boxes."""
    xmin, ymin, xmax, ymax = values
    if not all(math.isfinite(value) for value in values) or width <= 0 or height <= 0:
        raise ValueError(f"Non-finite box or invalid image size: {values} in {(width, height)}")
    if not (xmin < xmax and ymin < ymax):
        raise ValueError(f"Degenerate/reversed box: {values}")
    overflow = max(0, -xmin, -ymin, xmax - width, ymax - height)
    if overflow > tolerance:
        raise ValueError(f"Box exceeds image by {overflow:g} pixels (limit {tolerance:g}): {values}")
    clipped = (max(0.0, xmin), max(0.0, ymin), min(float(width), xmax), min(float(height), ymax))
    if not (0 <= clipped[0] < clipped[2] <= width and 0 <= clipped[1] < clipped[3] <= height):
        raise ValueError(f"Box is empty after boundary clipping: {values}")
    return clipped


def convert_box(class_name, xmin, ymin, xmax, ymax, width, height):
    if class_name not in CLASS_IDS:
        raise ValueError(f"Unknown class: {class_name}")
    if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
        raise ValueError(f"Invalid box: {(xmin, ymin, xmax, ymax)} in {(width, height)}")
    return f"{CLASS_IDS[class_name]} {(xmin+xmax)/(2*width):.7f} {(ymin+ymax)/(2*height):.7f} {(xmax-xmin)/width:.7f} {(ymax-ymin)/height:.7f}"


def prepare(raw_dir, output_dir):
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    if output_dir.exists():
        raise ValueError("Output directory already exists; choose a new path to preserve earlier data")
    images = raw_dir / "images"
    annotations = raw_dir / "annotations"
    if not images.is_dir() or not annotations.is_dir():
        raise ValueError("Raw directory must contain images/ and annotations/ (Pascal VOC XML)")
    records, seen, corrections = [], set(), []
    for annotation in sorted(annotations.glob("*.xml")):
        tree = ET.parse(annotation).getroot()
        filename = tree.findtext("filename")
        candidates = [images / Path(filename).name] if filename else []
        candidates += [images / (annotation.stem + extension) for extension in (".png", ".jpg", ".jpeg")]
        image_path = next((path for path in candidates if path.is_file()), None)
        if image_path is None:
            raise ValueError(f"Image missing for {annotation.name}")
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        declared = tree.find("size")
        if declared is not None:
            if (int(declared.findtext("width")), int(declared.findtext("height"))) != (width, height):
                raise ValueError(f"Annotation/image size mismatch: {annotation.name}")
        lines, counts = [], Counter()
        for object_index, obj in enumerate(tree.findall("object"), start=1):
            name = obj.findtext("name")
            box = obj.find("bndbox")
            values = [float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")]
            try:
                clipped = clip_boundary_box(values, width, height)
                lines.append(convert_box(name, *clipped, width, height))
            except ValueError as error:
                raise ValueError(f"{annotation.name}, object {object_index}: {error}") from error
            if tuple(values) != clipped:
                corrections.append({"annotation": annotation.name, "object": object_index,
                                    "label": name, "original_box": values,
                                    "clipped_box": list(clipped), "image_size": [width, height]})
            counts[CLASS_IDS[name]] += 1
        records.append({"path": image_path, "sha256": digest, "labels": lines, "counts": counts})
        seen.add(digest)
    if len(records) < 3:
        raise ValueError("At least three unique images are needed for train/val/test")
    random.Random(42).shuffle(records)
    n_val = max(1, int(len(records) * .15))
    n_test = max(1, int(len(records) * .15))
    n_train = len(records) - n_val - n_test
    splits = {"train": records[:n_train], "val": records[n_train:n_train+n_val],
              "test": records[n_train+n_val:]}
    report = {"unique_images": len(records), "seed": 42, "splits": {}, "manifest": [],
              "boundary_clip_tolerance_pixels": 1.0, "corrected_boxes": len(corrections),
              "box_corrections": corrections}
    for split, subset in splits.items():
        image_folder = output_dir / "images" / split
        label_folder = output_dir / "labels" / split
        image_folder.mkdir(parents=True)
        label_folder.mkdir(parents=True)
        counts = Counter()
        for index, record in enumerate(subset):
            stem = f"{split}_{index:05d}"
            new_name = stem + record["path"].suffix.lower()
            shutil.copyfile(record["path"], image_folder / new_name)
            (label_folder / (stem + ".txt")).write_text("\n".join(record["labels"]), encoding="utf-8")
            counts.update(record["counts"])
            report["manifest"].append({"file": new_name, "source": record["path"].name,
                                       "split": split, "sha256": record["sha256"]})
        report["splits"][split] = {"images": len(subset), "faces": {str(i): counts[i] for i in range(3)}}
    dataset_path = output_dir.resolve().as_posix()
    (output_dir / "data.yaml").write_text(
        f"path: '{dataset_path}'\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: with_mask\n  1: without_mask\n  2: mask_incorrect\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for split, info in report["splits"].items():
        print(f"{split}: {info['images']} images, class face counts: {info['faces']}")
        if any(count == 0 for count in info["faces"].values()):
            print(f"WARNING: {split} is missing a class; add suitable data or review split before training")
    print(f"Dataset YAML: {output_dir / 'data.yaml'}")
    print(f"Clipped {len(corrections)} boundary boxes by at most 1 pixel; details in summary.json. Raw XML unchanged.")
    print("Pixel coordinates are interpreted as image coordinates; review box overlays before training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="datasets/raw")
    parser.add_argument("--output", default="datasets/mask")
    args = parser.parse_args()
    prepare(args.raw, args.output)
