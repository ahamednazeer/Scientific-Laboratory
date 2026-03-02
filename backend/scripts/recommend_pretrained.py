"""Select a pretrained EfficientDet backbone based on target FPS and accuracy preference."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EfficientDetProfile:
    name: str
    coco_map: float
    expected_fps_gpu: float
    params_m: float
    notes: str


CANDIDATES = [
    EfficientDetProfile("tf_efficientdet_d0", coco_map=34.6, expected_fps_gpu=70.0, params_m=3.9, notes="fastest"),
    EfficientDetProfile("tf_efficientdet_d1", coco_map=40.5, expected_fps_gpu=48.0, params_m=6.6, notes="balanced"),
    EfficientDetProfile("tf_efficientdet_d2", coco_map=43.0, expected_fps_gpu=33.0, params_m=8.1, notes="best web-lab tradeoff"),
    EfficientDetProfile("tf_efficientdet_d3", coco_map=45.8, expected_fps_gpu=24.0, params_m=12.0, notes="higher accuracy"),
]


def select_profile(*, min_fps: float = 25.0, prefer_accuracy: bool = True) -> EfficientDetProfile:
    eligible = [item for item in CANDIDATES if item.expected_fps_gpu >= min_fps]
    if not eligible:
        return CANDIDATES[0]
    if prefer_accuracy:
        return sorted(eligible, key=lambda item: item.coco_map, reverse=True)[0]
    return sorted(eligible, key=lambda item: item.expected_fps_gpu, reverse=True)[0]


if __name__ == "__main__":
    choice = select_profile(min_fps=25.0, prefer_accuracy=True)
    print(f"Recommended pretrained backbone: {choice.name}")
    print(f"COCO mAP: {choice.coco_map}")
    print(f"Expected FPS (GPU): {choice.expected_fps_gpu}")
    print(f"Reason: {choice.notes}")
