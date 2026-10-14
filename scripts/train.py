"""Baseline training: run on a GPU notebook, or CPU for a small smoke run."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/mask/data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--weights", default="yolov8n.pt", help="Use models/best.pt for fine-tuning")
    parser.add_argument("--name", default="mask_lab3")
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    args = parser.parse_args()
    if not Path(args.data).is_file():
        parser.error("Dataset YAML not found; prepare the three-class dataset first")
    import torch
    from ultralytics import YOLO

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, batch=args.batch, imgsz=640,
                device=device, workers=0, seed=42, patience=10,
                optimizer=args.optimizer, lr0=args.lr0,
                project="artifacts/training", name=args.name)
    print("Copy the best.pt from this run to models/best.pt before starting the server.")


if __name__ == "__main__":
    main()
