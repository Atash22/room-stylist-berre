"""
Room Style Classifier - Training & Benchmarking
=================================================
Compares ResNet-18, EfficientNet-B0, and CLIP zero-shot for classifying
room photos into interior design styles, for the Berre.ca style-matching
recommender project.

Expected folder structure (after downloading the Kaggle dataset):

    data/
      train/
        mid_century_modern/
        boucle_glam/
        traditional/
        minimalist/
        industrial_scandinavian/
      val/
        <same subfolders>
      test/
        <same subfolders>

Run cells top to bottom in Jupyter/Colab, or run as a script.
"""

import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data")
BATCH_SIZE = 32
EPOCHS = 8
LR = 3e-4
IMG_SIZE = 224

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def make_loaders():
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=eval_tfms)
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=eval_tfms)

    class_names = train_ds.classes
    print(f"Classes ({len(class_names)}): {class_names}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader, class_names


# ---------------------------------------------------------------------------
# 2. Model builders (transfer learning: freeze backbone, replace head)
# ---------------------------------------------------------------------------

def build_resnet18(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# 3. Train / evaluate loop (shared across CNN models)
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # only train the newly added head (params with requires_grad=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                out = model(x)
                preds = out.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(y.numpy())
        val_acc = accuracy_score(val_true, val_preds)
        print(f"  epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

    return model


def evaluate_model(model, test_loader, class_names):
    model.eval()
    preds, true = [], []
    start = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            out = model(x)
            batch_preds = out.argmax(dim=1).cpu().numpy()
            preds.extend(batch_preds)
            true.extend(y.numpy())
    elapsed = time.time() - start
    per_image_ms = (elapsed / len(test_loader.dataset)) * 1000

    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average="macro")
    cm = confusion_matrix(true, preds)
    report = classification_report(true, preds, target_names=class_names)

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "inference_ms_per_image": per_image_ms,
        "confusion_matrix": cm.tolist(),
        "report": report,
    }


# ---------------------------------------------------------------------------
# 4. CLIP zero-shot baseline (no training needed)
# ---------------------------------------------------------------------------

def evaluate_clip_zero_shot(test_loader_raw, class_names):
    """
    test_loader_raw should yield PIL images (not normalized tensors) — see
    the `ImageFolder` + no-transform variant used in the notebook cell below.
    Requires: pip install open_clip_torch --break-system-packages
    """
    import open_clip
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(DEVICE).eval()

    # style class -> natural language prompt (tune these for accuracy)
    prompt_map = {
        "mid_century_modern": "a photo of a mid-century modern style living room",
        "boucle_glam": "a photo of a glamorous boucle and velvet living room",
        "traditional": "a photo of a traditional, ornate, classic living room",
        "minimalist": "a photo of a minimalist, clean-lined living room",
        "industrial_scandinavian": "a photo of an industrial scandinavian style living room",
    }
    prompts = [prompt_map.get(c, f"a photo of a {c.replace('_', ' ')} living room") for c in class_names]
    text_tokens = tokenizer(prompts).to(DEVICE)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    preds, true = [], []
    start = time.time()
    with torch.no_grad():
        for img_paths, labels in test_loader_raw:  # yields (PIL image list, label list)
            for img, label in zip(img_paths, labels):
                image_input = preprocess(img).unsqueeze(0).to(DEVICE)
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                sims = (image_features @ text_features.T).squeeze(0)
                pred = sims.argmax().item()
                preds.append(pred)
                true.append(label)
    elapsed = time.time() - start
    per_image_ms = (elapsed / len(true)) * 1000

    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1, "inference_ms_per_image": per_image_ms}


# ---------------------------------------------------------------------------
# 5. Run the full benchmark
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = make_loaders()
    results = {}

    print("\n=== ResNet-18 ===")
    resnet = build_resnet18(len(class_names))
    resnet = train_model(resnet, train_loader, val_loader)
    results["resnet18"] = evaluate_model(resnet, test_loader, class_names)

    print("\n=== EfficientNet-B0 ===")
    effnet = build_efficientnet_b0(len(class_names))
    effnet = train_model(effnet, train_loader, val_loader)
    results["efficientnet_b0"] = evaluate_model(effnet, test_loader, class_names)

    # NOTE: CLIP zero-shot needs a raw-PIL test loader (no tensor transform).
    # See the notebook markdown cell for how to construct one — kept separate
    # here to avoid mixing tensor and PIL batches in one function.

    print("\n=== Summary ===")
    for name, r in results.items():
        print(f"{name:20s} acc={r['accuracy']:.4f}  f1={r['macro_f1']:.4f}  "
              f"latency={r['inference_ms_per_image']:.1f}ms/img")

    with open("benchmark_results.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"}
                   for k, v in results.items()}, f, indent=2)

    # Save the winning model's weights for deployment, e.g.:
    # torch.save(effnet.state_dict(), "style_classifier_effnet_b0.pt")
    print("\nSaved benchmark_results.json — pick the winner and export weights for app/predict.py")
