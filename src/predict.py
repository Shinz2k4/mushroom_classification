import pathlib
import torch
import torch.nn.functional as F
from PIL import Image

from config import DEVICE, IMG_SIZE, MUSHROOM_NET_PATH, RESNET50_PATH
from data_loader import get_eval_transform


def load_model(model, weights_path):
    weights_path = pathlib.Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Không tìm thấy weights: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(model, image_path: str | pathlib.Path,
                  class_names: list[str]) -> dict:
    transform = get_eval_transform()
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top_idx = probs.argmax().item()
    return {
        "file": str(image_path),
        "class": class_names[top_idx],
        "confidence": float(probs[top_idx]) * 100,
        "top5": [
            {"class": class_names[i], "prob": float(probs[i]) * 100}
            for i in probs.topk(min(5, len(class_names))).indices.tolist()
        ],
    }


def predict_directory(model, test_dir: str | pathlib.Path,
                      class_names: list[str],
                      extensions=("*.jpg", "*.jpeg", "*.png")) -> list[dict]:
    test_dir = pathlib.Path(test_dir)
    files = []
    for ext in extensions:
        files.extend(list(test_dir.glob(ext)))
        files.extend(list(test_dir.glob(ext.upper())))

    if not files:
        print(f"Không tìm thấy ảnh trong {test_dir}")
        return []

    results = []
    for f in sorted(files):
        r = predict_image(model, f, class_names)
        results.append(r)
        print(f"{f.name:40s} → {r['class']:25s} ({r['confidence']:.1f}%)")
    return results
