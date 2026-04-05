import time
import numpy as np
import torch
import torch.nn.functional as F

from config import DEVICE, LEARNING_RATE
from genetic_distance import get_dist_matrix, get_target_tensor, identify_by_distance


# ── Một epoch ────────────────────────────────────────────────────────────────

def _run_epoch(model, loader, target_matrix_t, activation, optimizer, is_train):
    model.train(is_train)
    total_loss = total = 0

    reduction = "sum" if activation == "mse-sum" else "mean"

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            raw = model(imgs)                              # (B, N)

            pred = F.softmax(raw, dim=1) if activation == "softmax" else raw
            targets = target_matrix_t[labels]             # (B, N)
            loss = F.mse_loss(pred, targets, reduction=reduction)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

    return total_loss / total


# ── Vòng lặp huấn luyện chính ────────────────────────────────────────────────

def train_genetic(
    model,
    train_loader,
    val_loader,
    activation: str = "mse-mean",
    diagonal_val: float = 0.0,
    epochs: int = 30,
    lr: float = LEARNING_RATE,
    save_path=None,
):
    train_mat = get_target_tensor(get_dist_matrix(diagonal_val), DEVICE)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    best_val = float("inf")
    print(f"\n{'='*60}")
    print(f"GENETIC DISTANCE TRAINING")
    print(f"  activation  = {activation}")
    print(f"  diagonal    = {diagonal_val}")
    print(f"  epochs      = {epochs}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = _run_epoch(model, train_loader, train_mat, activation, optimizer, True)
        vl_loss = _run_epoch(model, val_loader,   train_mat, activation, None,      False)

        mark = ""
        if vl_loss < best_val:
            best_val = vl_loss
            if save_path:
                torch.save(model.state_dict(), save_path)
            mark = " ← best"

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  "
            f"({time.time()-t0:.1f}s){mark}"
        )

    print(f"\nBest val_loss: {best_val:.4f}")
    if save_path:
        print(f"Model đã lưu tại: {save_path}")


# ── Đánh giá xác định loài qua genetic distance ───────────────────────────────

def evaluate_genetic(model, test_loader, metric: str = "cosine") -> float: 
    model.eval()
    gt_matrix = get_dist_matrix(diagonal_val=0.0)    # numpy (N, N)

    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            preds = model(imgs.to(DEVICE)).cpu().numpy()     # (B, N)
            for pred_vec, true_cls in zip(preds, labels.numpy()):
                if identify_by_distance(pred_vec, gt_matrix, metric) == true_cls:
                    correct += 1
                total += 1

    acc = correct / total
    print(f"[GENETIC ID]  metric={metric}  acc={acc:.4f}  ({acc*100:.2f}%)")
    return acc


def evaluate_genetic_distance_error(model, test_loader) -> float:
    model.eval()
    gt_matrix_t = get_target_tensor(get_dist_matrix(0.0), DEVICE)

    total_mse = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds   = model(imgs)
            targets = gt_matrix_t[labels]
            mse = F.mse_loss(preds, targets, reduction="sum").item()
            total_mse += mse
            total += imgs.size(0)

    avg_mse = total_mse / total
    print(f"[GENETIC MSE]  avg_mse={avg_mse:.4f}")
    return avg_mse
