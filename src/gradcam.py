import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config import DEVICE, IMG_SIZE, MODEL_DIR
from data_loader import get_eval_transform


class GradCAM:
    def __init__(self, model):
        self.model = model
        self._fmap = None   # feature map sau conv_last
        self._grad = None   # gradient w.r.t. feature map
        self._h_fwd = model.conv_last.register_forward_hook(self._save_fmap)
        self._h_bwd = model.conv_last.register_full_backward_hook(self._save_grad)

    def _save_fmap(self, module, inp, out):
        self._fmap = out.detach()                   # (1, 960, H, W)

    def _save_grad(self, module, grad_in, grad_out):
        self._grad = grad_out[0].detach()           # (1, 960, H, W)

    def generate(self, img_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.eval()
        logits = self.model(img_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global average pooling của gradient → trọng số kênh
        weights = self._grad.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self._fmap).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam).squeeze().cpu().numpy()              # (H, W)

        # Normalize về [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def remove_hooks(self):
        self._h_fwd.remove()
        self._h_bwd.remove()


def visualize_gradcam(model, image_path: str, class_names: list,
                      save_path: str = None) -> str:
    transform = get_eval_transform()
    orig = Image.open(image_path).convert("RGB")
    tensor = transform(orig).unsqueeze(0).to(DEVICE)

    gcam = GradCAM(model)
    cam  = gcam.generate(tensor)
    gcam.remove_hooks()

    with torch.no_grad():
        pred_idx = model(tensor).argmax(dim=1).item()
    pred_cls = class_names[pred_idx]

    # Resize ảnh gốc về 224×224 để overlay
    orig_arr  = np.array(orig.resize((IMG_SIZE[1], IMG_SIZE[0]))) / 255.0
    # Resize heatmap lên 224×224
    cam_large = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR
        )
    ) / 255.0

    heatmap = plt.cm.jet(cam_large)[:, :, :3]
    overlay = np.clip(0.5 * orig_arr + 0.5 * heatmap, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_arr);  axes[0].set_title("Original");     axes[0].axis("off")
    axes[1].imshow(cam_large, cmap="jet");
    axes[1].set_title("Heat Map (Grad-CAM)"); axes[1].axis("off")
    axes[2].imshow(overlay);   axes[2].set_title(f"Overlay\n→ {pred_cls}"); axes[2].axis("off")
    plt.tight_layout()

    if save_path is None:
        import pathlib
        save_path = str(MODEL_DIR / f"gradcam_{pathlib.Path(image_path).stem}.png")

    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Grad-CAM saved: {save_path}")
    return save_path
