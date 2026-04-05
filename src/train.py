import time
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DEVICE, LEARNING_RATE,
    EPOCHS_STAGE2, EPOCHS_STAGE3,
    MUSHROOM_NET_PATH,
)


# ── Một epoch ────────────────────────────────────────────────────────────────

def _run_epoch(model, loader, criterion, optimizer, is_train: bool):
    model.train(is_train)
    total_loss = total_correct = total = 0

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss    += loss.item() * imgs.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total         += imgs.size(0)

    return total_loss / total, total_correct / total


# ── Vòng lặp train ────────────────────────────────────────────────────────────

def train_loop(model, train_loader, val_loader, epochs,
               save_path=MUSHROOM_NET_PATH,
               lr=LEARNING_RATE,
               init_best_val_acc: float = 0.0):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = init_best_val_acc

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, True)
        vl_loss, vl_acc = _run_epoch(model, val_loader,   criterion, None,      False)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        mark = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), save_path)
            mark = " ← best"

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
            f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}  "
            f"({time.time()-t0:.1f}s){mark}"
        )

    print(f"\nBest val_acc: {best_val_acc:.4f}  → {save_path}")
    return history


# ── 3-stage training ──────────────────────────────────────────────────────────

def stage2_train(model, train_loader, val_loader,
                 epochs=EPOCHS_STAGE2, save_path=MUSHROOM_NET_PATH):
    print("\n" + "=" * 60)
    print("STAGE 2 – Fine-tune toàn bộ mạng trên dataset nấm")
    print("=" * 60)
    for p in model.parameters():
        p.requires_grad = True
    return train_loop(model, train_loader, val_loader, epochs, save_path)


def stage3_train(model, train_loader, val_loader,
                 epochs=EPOCHS_STAGE3, save_path=MUSHROOM_NET_PATH):
    print("\n" + "=" * 60)
    print("STAGE 3 – Train attention modules (SE + ECANet)")
    print("=" * 60)

    model.load_state_dict(
        torch.load(save_path, map_location=DEVICE, weights_only=False)
    )
    model.freeze_for_stage3()

    criterion = nn.CrossEntropyLoss()
    _, stage2_best_val_acc = _run_epoch(model, val_loader, criterion, None, False)
    print(f"Stage-2 best val_acc (baseline): {stage2_best_val_acc:.4f}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    return train_loop(model, train_loader, val_loader, epochs, save_path,
                      init_best_val_acc=stage2_best_val_acc)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    loss, acc = _run_epoch(model, test_loader, criterion, None, False)
    print(f"\n[TEST]  loss={loss:.4f}  acc={acc:.4f} ({acc*100:.2f}%)")
    return loss, acc


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_history(histories: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"Stage 2": ("steelblue", "dodgerblue"),
              "Stage 3": ("firebrick", "tomato")}

    for stage_name, h in histories.items():
        c_tr, c_vl = colors.get(stage_name, ("gray", "silver"))
        ep = range(1, len(h["train_acc"]) + 1)
        axes[0].plot(ep, h["train_acc"],  color=c_tr, label=f"{stage_name} train")
        axes[0].plot(ep, h["val_acc"],    color=c_vl, linestyle="--",
                     label=f"{stage_name} val")
        axes[1].plot(ep, h["train_loss"], color=c_tr, label=f"{stage_name} train")
        axes[1].plot(ep, h["val_loss"],   color=c_vl, linestyle="--",
                     label=f"{stage_name} val")

    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].set_xlabel("Epoch")
    axes[1].set_title("Loss");     axes[1].legend(); axes[1].set_xlabel("Epoch")
    plt.tight_layout()

    from config import MODEL_DIR
    out_path = MODEL_DIR / "training_history.png"
    plt.savefig(out_path, dpi=120)
    print(f"\n[INFO] Biểu đồ đã lưu: {out_path}")
    plt.close()
