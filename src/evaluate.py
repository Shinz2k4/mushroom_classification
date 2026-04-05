"""
Đánh giá đầy đủ theo bài báo (Section 3):
  - Accuracy, Precision, F1, Recall cho từng loài (Eq. 4-7)
  - Confusion Matrix (Figure 7/10)
  - ROC Curve (Figure 6/9)
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from config import DEVICE, MODEL_DIR


# ── Full evaluation ────────────────────────────────────────────────

def full_evaluate(model, test_loader, class_names: list) -> dict:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = float((y_true == y_pred).mean())
    n = len(class_names)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(n)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))

    # ── In bảng kết quả ───────────────────────────
    W = 72
    print(f"\n{'='*W}")
    print(f"{'EVALUATION RESULTS':^{W}}")
    print(f"{'='*W}")
    print(f"Overall Accuracy: {acc:.4f}  ({acc*100:.2f}%)\n")
    hdr = f"{'ID':<4}{'Class':<28}{'Correct':>8}{'Total':>7}{'Acc%':>7}{'Prec%':>7}{'F1%':>7}{'Recall%':>8}"
    print(hdr)
    print("-" * W)

    for i, cls in enumerate(class_names):
        correct = int(cm[i, i])
        total   = int(support[i])
        cls_acc = correct / total * 100 if total > 0 else 0.0
        print(f"{i:<4}{cls:<28}{correct:>8}{total:>7}"
              f"{cls_acc:>6.2f}%{precision[i]*100:>6.2f}%"
              f"{f1[i]*100:>6.2f}%{recall[i]*100:>7.2f}%")

    print("-" * W)
    print(f"{'':4}{'Mean':28}"
          f"{'':>8}{'':>7}"
          f"{acc*100:>6.2f}%"
          f"{precision.mean()*100:>6.2f}%"
          f"{f1.mean()*100:>6.2f}%"
          f"{recall.mean()*100:>7.2f}%")
    print("=" * W)

    return {
        "accuracy": acc, "precision": precision,
        "recall": recall, "f1": f1,
        "confusion_matrix": cm,
        "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
    }


# ── Confusion Matrix ────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path=None):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 2)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ticks = np.arange(n)
    short = [c[:12] for c in class_names]
    ax.set_xticks(ticks); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(ticks); ax.set_yticklabels(short, fontsize=8)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=7)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    out = save_path or (MODEL_DIR / "confusion_matrix.png")
    plt.savefig(out, dpi=120); plt.close()
    print(f"[INFO] Confusion matrix saved: {out}")


# ── ROC Curve ────────────────────────────────────────────────────

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   class_names: list, save_path=None):
    n = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n)))

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, lw=1.2, label=f"{cls} (AUC={auc(fpr,tpr):.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC Curve")
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()

    out = save_path or (MODEL_DIR / "roc_curve.png")
    plt.savefig(out, dpi=120); plt.close()
    print(f"[INFO] ROC curve saved: {out}")
