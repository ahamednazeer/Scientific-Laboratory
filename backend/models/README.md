# Model Artifacts

Place fine-tuned EfficientDet checkpoints here.

Expected default path:
- `backend/models/efficientdet_lab_instruments.pth`
- `backend/models/class_labels.json` (ordered list of class names used during training)

To enable real model inference:
1. Install optional detector deps (`torch`, `torchvision`, `effdet`).
2. Train/fine-tune using your 12.5k labeled instrument dataset.
3. Set `DETECTOR_MODE=efficientdet` in `backend/.env`.
4. Ensure `class_labels.json` ordering matches training class indices.
