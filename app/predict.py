"""
predict.py
==========
Clean inference wrapper around the winning benchmarked model
(assumed here: EfficientNet-B0 fine-tuned head). Swap the model
builder + weights path if a different architecture won your benchmark.

This is the "Script" step in the course pipeline: Notebook -> Script.
"""

from pathlib import Path
from io import BytesIO

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = Path(__file__).parent / "style_classifier_effnet_b0.pt"

# Must match the class order used during training (ImageFolder sorts
# alphabetically).
CLASS_NAMES = [
    "coastal_tropical",
    "contemporary_scandinavian",
    "eclectic_industrial",
    "mid_century_modern",
    "rustic_farmhouse",
    "traditional_classic",
]

IMG_SIZE = 224
_eval_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


DROPOUT_P = 0.3  # must match the value used during training


def _build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    # Matches the Dropout -> Linear head used in the training notebook so
    # the saved state_dict loads cleanly.
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_P),
        nn.Linear(in_features, num_classes),
    )
    return model


_model = None  # lazy-loaded singleton so Flask doesn't reload weights per-request


def load_model() -> nn.Module:
    global _model
    if _model is None:
        model = _build_model(len(CLASS_NAMES))
        state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE).eval()
        _model = model
    return _model


def predict_style(image_bytes: bytes) -> dict:
    """
    Takes raw image bytes (e.g. from a Flask file upload), returns the
    predicted style label + confidence + full probability distribution.
    """
    model = load_model()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = _eval_tfms(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    top_idx = int(probs.argmax())
    return {
        "style": CLASS_NAMES[top_idx],
        "confidence": round(float(probs[top_idx]), 4),
        "all_probabilities": {
            CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)
        },
    }


if __name__ == "__main__":
    # quick manual test: python predict.py path/to/room.jpg
    import sys
    img_path = sys.argv[1]
    with open(img_path, "rb") as f:
        result = predict_style(f.read())
    print(result)
