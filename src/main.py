import argparse
import pathlib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import torch
from config import (
    DEVICE, EPOCHS_STAGE2, EPOCHS_STAGE3, EPOCHS_GENETIC,
    MUSHROOM_NET_PATH, RESNET50_PATH, GENETIC_MODEL_PATH,
)
from data_loader import get_train_loader, get_val_loader, get_test_loader, get_class_names
from model import build_mushroomnet, build_resnet50
from train import stage2_train, stage3_train, evaluate, plot_history
from predict import load_model, predict_image, predict_directory
from genetic_train import train_genetic, evaluate_genetic, evaluate_genetic_distance_error


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_model_and_path(model_name, class_names, weights=None):
    n = len(class_names)
    if model_name == "mushroomnet":
        default_path = MUSHROOM_NET_PATH
        model = build_mushroomnet(n).to(DEVICE)
    else:
        default_path = RESNET50_PATH
        model = build_resnet50(n).to(DEVICE)
    save_path = pathlib.Path(weights) if weights else default_path
    return model, save_path


# ── Subcommands ───────────────────────────────────────────────────────────────

def cmd_train(args):
    print(f"[INFO] Device: {DEVICE}")
    train_dir = pathlib.Path(args.train_dir) if args.train_dir else None
    val_dir   = pathlib.Path(args.val_dir)   if args.val_dir   else None
    class_names = get_class_names(train_dir)
    print(f"[INFO] {len(class_names)} lớp: {class_names}")

    train_loader = get_train_loader(train_dir)
    val_loader   = get_val_loader(val_dir)
    model, save_path = _get_model_and_path(args.model, class_names, args.weights)
    print(f"[INFO] Model: {args.model}  →  {save_path}")

    histories = {}
    run_all = args.stage is None

    if run_all or args.stage == 2:
        if args.model != "mushroomnet":
            for p in model.parameters():
                p.requires_grad = True
        histories["Stage 2"] = stage2_train(
            model, train_loader, val_loader, args.epochs2, save_path
        )

    if run_all or args.stage == 3:
        if args.model != "mushroomnet":
            print("[WARN] Stage 3 chỉ áp dụng cho MushroomNet.")
        else:
            if args.stage == 3:
                model.load_state_dict(
                    torch.load(save_path, map_location=DEVICE, weights_only=False)
                )
            histories["Stage 3"] = stage3_train(
                model, train_loader, val_loader, args.epochs3, save_path
            )

    if histories:
        plot_history(histories)


def cmd_test(args):
    test_dir  = pathlib.Path(args.test_dir)  if args.test_dir  else None
    train_dir = pathlib.Path(args.train_dir) if args.train_dir else None
    class_names = get_class_names(train_dir)
    model, save_path = _get_model_and_path(args.model, class_names, args.weights)
    print(f"[INFO] Loading weights: {save_path}  ({len(class_names)} lớp)")
    model = load_model(model, save_path)
    test_loader = get_test_loader(test_dir)

    if args.report:
        from evaluate import full_evaluate, plot_confusion_matrix, plot_roc_curve
        results = full_evaluate(model, test_loader, class_names)
        plot_confusion_matrix(results["confusion_matrix"], class_names)
        plot_roc_curve(results["y_true"], results["y_prob"], class_names)
    else:
        evaluate(model, test_loader)


def cmd_predict(args):
    class_names = get_class_names()
    model, save_path = _get_model_and_path(args.model, class_names)
    model = load_model(model, save_path)

    if args.image:
        r = predict_image(model, args.image, class_names)
        print(f"\nKết quả: {r['class']}  (confidence: {r['confidence']:.1f}%)")
        print("Top-5:")
        for t in r["top5"]:
            print(f"  {t['class']:25s} {t['prob']:.2f}%")

        if args.gradcam:
            from gradcam import visualize_gradcam
            visualize_gradcam(model, args.image, class_names)
    else:
        predict_directory(model, args.dir, class_names)


def cmd_genetic(args):
    from genetic_distance import LOCAL_SPECIES
    n_species = len(LOCAL_SPECIES)
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Genetic distance — {n_species} loài (Local Dataset)")

    model = build_mushroomnet(n_species).to(DEVICE)
    diagonal_val = -1.0 if args.diagonal_minus1 else 0.0

    if args.eval:
        model = load_model(model, GENETIC_MODEL_PATH)
        test_loader = get_test_loader()
        evaluate_genetic_distance_error(model, test_loader)
        evaluate_genetic(model, test_loader, metric=args.metric)
        return

    train_loader = get_train_loader()
    val_loader   = get_val_loader()
    train_genetic(
        model, train_loader, val_loader,
        activation=args.activation, diagonal_val=diagonal_val,
        epochs=args.epochs, save_path=GENETIC_MODEL_PATH,
    )
    model = load_model(model, GENETIC_MODEL_PATH)
    test_loader = get_test_loader()
    evaluate_genetic_distance_error(model, test_loader)
    evaluate_genetic(model, test_loader, metric="cosine")
    evaluate_genetic(model, test_loader, metric="euclidean")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MushroomNet")
    sub = p.add_subparsers(dest="command")

    tr = sub.add_parser("train")
    tr.add_argument("--model", choices=["mushroomnet", "resnet50"], default="mushroomnet")
    tr.add_argument("--stage", type=int, choices=[2, 3])
    tr.add_argument("--epochs2", type=int, default=EPOCHS_STAGE2)
    tr.add_argument("--epochs3", type=int, default=EPOCHS_STAGE3)
    tr.add_argument("--train-dir", type=str, default=None,
                    help="Thư mục train (mặc định: dùng config.py)")
    tr.add_argument("--val-dir", type=str, default=None,
                    help="Thư mục validation (mặc định: dùng config.py)")
    tr.add_argument("--weights", type=str, default=None,
                    help="Đường dẫn file .pth để lưu (mặc định: models/mushroomnet.pth)")

    ts = sub.add_parser("test")
    ts.add_argument("--model", choices=["mushroomnet", "resnet50"], default="mushroomnet")
    ts.add_argument("--report", action="store_true",
                    help="In Acc/Prec/F1/Recall + lưu Confusion Matrix và ROC Curve")
    ts.add_argument("--test-dir", type=str, default=None,
                    help="Thư mục test (mặc định: dùng config.py)")
    ts.add_argument("--train-dir", type=str, default=None,
                    help="Thư mục train để lấy class_names (mặc định: dùng config.py)")
    ts.add_argument("--weights", type=str, default=None,
                    help="Đường dẫn file .pth để load (mặc định: models/mushroomnet.pth)")

    pr = sub.add_parser("predict")
    pr.add_argument("--model", choices=["mushroomnet", "resnet50"], default="mushroomnet")
    grp = pr.add_mutually_exclusive_group(required=True)
    grp.add_argument("--image", type=str)
    grp.add_argument("--dir",   type=str)
    pr.add_argument("--gradcam", action="store_true",
                    help="Lưu Grad-CAM heat map (Figure 8/11 bài báo)")

    gn = sub.add_parser("genetic")
    gn.add_argument("--activation", choices=["mse-sum", "mse-mean", "softmax"],
                    default="mse-mean")
    gn.add_argument("--diagonal-minus1", action="store_true",
                    help="Improved: diagonal target = -1 (Section 3.4.2)")
    gn.add_argument("--epochs", type=int, default=EPOCHS_GENETIC)
    gn.add_argument("--eval", action="store_true")
    gn.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")

    return p.parse_args()


def main():
    args = parse_args()
    if   args.command == "train":   cmd_train(args)
    elif args.command == "test":    cmd_test(args)
    elif args.command == "predict": cmd_predict(args)
    elif args.command == "genetic": cmd_genetic(args)
    else:
        print("Dùng: python src/main.py {train|test|predict|genetic} --help")


if __name__ == "__main__":
    main()
