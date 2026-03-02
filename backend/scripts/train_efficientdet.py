"""Fine-tuning entrypoint for EfficientDet on laboratory instrument datasets.

This script is intentionally lightweight and production-oriented:
- expects YOLO-format or COCO-format exports in a local dataset path
- supports transfer learning from a selected EfficientDet backbone
- exports a checkpoint that the API can load via DETECTOR_WEIGHTS_PATH

Usage example:
  python3 scripts/train_efficientdet.py \
      --dataset ./data/lab-instruments \
      --model tf_efficientdet_d2 \
      --epochs 40 \
      --batch-size 8 \
      --output ./models/efficientdet_lab_instruments.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune EfficientDet for lab instrument detection")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset root (train/val split expected)")
    parser.add_argument("--model", type=str, default="tf_efficientdet_d2", help="EfficientDet variant")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output", type=Path, required=True, help="Output checkpoint path")
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=None,
        help="Optional JSON file containing ordered class names for training/inference alignment.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classes when --labels-file is not supplied.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset path not found: {args.dataset}")
    if args.labels_file and not args.labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {args.labels_file}")

    # Import lazily so the backend runtime does not require these heavy deps.
    try:
        import torch
        from effdet import create_model
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing training dependencies. Install torch + effdet before running training."
        ) from exc

    labels: list[str] = []
    if args.labels_file:
        content = json.loads(args.labels_file.read_text(encoding="utf-8"))
        if not isinstance(content, list) or not all(isinstance(item, str) for item in content):
            raise RuntimeError("--labels-file must be a JSON array of class names")
        labels = [item.strip() for item in content if item.strip()]
        if not labels:
            raise RuntimeError("--labels-file did not contain valid class names")

    num_classes = len(labels) if labels else args.num_classes
    if num_classes <= 0:
        raise RuntimeError("num_classes must be > 0")

    # Skeleton transfer-learning flow. Extend this with your preferred dataloader + augmentations.
    model = create_model(args.model, bench_task="train", pretrained=True, num_classes=num_classes)

    # This file intentionally leaves the data pipeline open because dataset formats differ across labs.
    # Recommended: implement a PyTorch Dataset that yields EfficientDet-compatible targets and train here.
    print("Model initialized:", args.model)
    print("Dataset:", args.dataset)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("LR:", args.lr)
    print("Classes:", num_classes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print("Saved checkpoint:", args.output)

    if labels:
        label_map_path = args.output.parent / "class_labels.json"
        label_map_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
        print("Saved class label map:", label_map_path)


if __name__ == "__main__":
    main()
