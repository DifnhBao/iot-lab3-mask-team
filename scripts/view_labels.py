from pathlib import Path
import random

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

dataset = Path("datasets/mask")
names = ["Đeo đúng", "Không đeo", "Đeo chưa đúng"]
colors = ["lime", "red", "orange"]

images = list((dataset / "images/train").glob("*"))
samples = random.sample(images, min(6, len(images)))

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for ax, image_path in zip(axes.flat, samples):
    image = Image.open(image_path)
    width, height = image.size
    ax.imshow(image)

    label_path = dataset / "labels/train" / f"{image_path.stem}.txt"

    for line in label_path.read_text().splitlines():
        class_id, xc, yc, bw, bh = map(float, line.split())
        class_id = int(class_id)

        x = (xc - bw / 2) * width
        y = (yc - bh / 2) * height
        box_width = bw * width
        box_height = bh * height

        ax.add_patch(Rectangle(
            (x, y), box_width, box_height,
            fill=False, edgecolor=colors[class_id], linewidth=2
        ))
        ax.text(
            x, max(0, y - 3), names[class_id],
            color=colors[class_id], fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.5}
        )

    ax.set_title(image_path.name)
    ax.axis("off")

plt.tight_layout()
plt.show()