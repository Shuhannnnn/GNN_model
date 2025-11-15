"""
Entry point for the 2025 E.SUN AI Challenge GNN baseline.

This script wires together the preprocessing and modeling modules:

1. Load raw CSV files.
2. Build account-level features.
3. Construct the transaction graph.
4. Train a GraphSAGE model with full-batch training.
5. Select a robust decision threshold.
6. Generate `result.csv` for submission.

The folder structure is:

- Preprocess/data_preprocess.py  : data loading & feature engineering
- Model/tree_based_model.py      : GNN model, graph builder, training loop
"""

import argparse
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from Preprocess.data_preprocess import (
    build_account_features,
    load_csvs,
    resolve_columns,
    set_seed,
)
from Model.graph_based_model import (
    SAGEModel,
    build_graph,
    select_conservative_threshold,
    train_model_fullbatch,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the training script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with training and I/O configuration.
    """
    parser = argparse.ArgumentParser(description="Full-batch GraphSAGE baseline for E.SUN 2025 challenge")
    parser.add_argument("--data_dir", type=str, default="../preliminary_data")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--eval_every", type=int, default=10)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--out_csv", type=str, default="result.csv")
    parser.add_argument("--save_model", type=str, default="")
    parser.add_argument("--force_thr", type=float, default=None)

    parser.add_argument(
        "--grad_ckpt",
        action="store_true",
        help="Enable gradient checkpointing to save GPU memory",
    )
    return parser.parse_args()


def choose_device(args: argparse.Namespace) -> torch.device:
    """
    Decide which device (CPU or GPU) to use based on user arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    torch.device
        Selected device for training and inference.
    """
    if args.device == "cuda" and torch.cuda.is_available():
        return torch.device(f"cuda:{args.device_id}")
    if args.device == "auto" and torch.cuda.is_available():
        return torch.device(f"cuda:{args.device_id}")
    return torch.device("cpu")


def main() -> None:
    """
    Main execution function.

    It orchestrates the whole pipeline: preprocessing, model training and
    test prediction generation. The final predictions are saved to the CSV
    file specified by `--out_csv`.
    """
    args = parse_args()
    set_seed(args.seed)

    device_pref = choose_device(args)
    use_amp = device_pref.type == "cuda"
    prefer_bf16 = torch.cuda.is_bf16_supported() if use_amp else False
    amp_dtype = torch.bfloat16 if (use_amp and prefer_bf16) else (torch.float16 if use_amp else torch.float32)

    tx, alert, predict = load_csvs(args.data_dir)
    col = resolve_columns(tx, alert, predict)

    feat = build_account_features(tx, col)
    feat["acct"] = feat["acct"].astype(str)

    pos_set = set(alert[col["alert_acct"]].astype(str).tolist())
    pred_list: List[str] = predict[col["predict_acct"]].astype(str).tolist()
    pred_set = set(pred_list)

    train_df = feat[(~feat["acct"].isin(pred_set)) & (feat["is_esun"] == 1)].copy()
    y_train = train_df["acct"].map(lambda a: 1 if a in pos_set else 0).astype(np.int64).values

    all_accts = pd.Index(feat["acct"].astype(str).unique())
    acct2idx = {a: i for i, a in enumerate(all_accts)}

    feat_cols = [c for c in feat.columns if c != "acct"]
    X_all = feat.set_index("acct").loc[all_accts][feat_cols].astype(np.float32).values

    train_idx = np.array([acct2idx[a] for a in train_df["acct"].astype(str)], dtype=np.int64)
    test_idx = np.array([acct2idx[a] for a in pred_list if a in acct2idx], dtype=np.int64)

    scaler = StandardScaler()
    X_all[train_idx] = scaler.fit_transform(X_all[train_idx])
    mean = scaler.mean_
    scale = getattr(scaler, "scale_", np.sqrt(getattr(scaler, "var_", np.ones_like(mean)) + 1e-9))
    mask_rest = np.ones(len(X_all), dtype=bool)
    mask_rest[train_idx] = False
    X_all[mask_rest] = np.clip((X_all[mask_rest] - mean) / (scale + 1e-12), -5.0, 5.0)

    y_all = np.zeros(len(all_accts), dtype=np.float32)
    for i, a in enumerate(all_accts):
        if (a in pos_set) and (a not in pred_set):
            y_all[i] = 1.0
    if y_train.sum() > 0:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        tr_idx, va_idx = next(skf.split(train_idx, y_train))
        train_nodes = train_idx[tr_idx]
        val_nodes = train_idx[va_idx]
    else:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(train_idx)
        k = max(1, int(len(train_idx) * 0.15))
        val_nodes = train_idx[:k]
        train_nodes = train_idx[k:]

    print(f"[Split] train_nodes={len(train_nodes)}, val_nodes={len(val_nodes)}, test_nodes={len(test_idx)}")
 
    A = build_graph(tx, col, acct2idx, undirected=True)
    x_all_t = torch.from_numpy(X_all)
    y_all_t = torch.from_numpy(y_all)

    hid = args.hidden
    if device_pref.type == "cuda" and args.hidden == 128:
        hid = 96

    model = SAGEModel(
        in_dim=X_all.shape[1],
        hidden_dim=hid,
        num_layers=args.layers,
        dropout=args.dropout,
        use_ckpt=args.grad_ckpt,
    )

    try:
        model, best_val_thr, best_val_f1 = train_model_fullbatch(
            model=model, x_all=x_all_t, A=A, y_all=y_all_t,
            train_idx=train_nodes, val_idx=val_nodes,
            lr=args.lr, weight_decay=args.weight_decay,
            epochs=args.epochs, patience=args.patience,
            label_smoothing=args.label_smoothing, eval_every=args.eval_every,
            device=device_pref, amp_dtype=amp_dtype)
    except Exception as e:
        if "out of memory" in str(e).lower() and device_pref.type == "cuda":
            print("[Warn] CUDA OOM, falling back to CPU.]")
            try: torch.cuda.empty_cache()
            except Exception: pass
            model = SAGEModel(in_dim=X_all.shape[1], hidden_dim=args.hidden,
                              num_layers=args.layers, dropout=args.dropout, use_ckpt=False)
            model, best_val_thr, best_val_f1 = train_model_fullbatch(
                model=model, x_all=x_all_t, A=A, y_all=y_all_t,
                train_idx=train_nodes, val_idx=val_nodes, lr=args.lr,
                weight_decay=args.weight_decay, epochs=args.epochs, patience=args.patience,
                label_smoothing=args.label_smoothing, eval_every=args.eval_every,
                device=torch.device("cpu"), amp_dtype=torch.float32)
        else:
            raise

    if args.save_model:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "best_val_thr": best_val_thr,
                "best_val_f1": best_val_f1,
                "feat_cols": feat_cols,
            },
            args.save_model,
        )
        print(f"[Info] Best model saved to {args.save_model}")

    model.eval()
    A_cpu = A.to_sparse_csr()
    x_cpu = torch.from_numpy(X_all).float()
    model = model.float()

    with torch.no_grad():
        logits_all = model(x_cpu, A_cpu)
        prob_all = torch.sigmoid(logits_all).cpu().numpy()

    train_probs = prob_all[train_nodes]
    val_probs = prob_all[val_nodes]
    train_true = y_all[train_nodes].astype(int)
    val_true = y_all[val_nodes].astype(int)

    robust_thr = (
        float(args.force_thr)
        if args.force_thr is not None
        else select_conservative_threshold(
            train_true, train_probs, val_true, val_probs,
            thr_min=0.3, thr_max=0.7, steps=200, min_recall=0.35,
        )
    )
    if args.force_thr is not None:
        print(f"[Thresh] Use force_thr={robust_thr:.3f}")
    else:
        print(f"[Thresh] conservative_thr={robust_thr:.3f}")

    val_pred = (val_probs >= robust_thr).astype(int)
    tp = ((val_pred == 1) & (val_true == 1)).sum()
    fp = ((val_pred == 1) & (val_true == 0)).sum()
    fn = ((val_pred == 0) & (val_true == 1)).sum()
    denom = 2 * tp + fp + fn
    val_f1_robust = 2 * tp / denom if denom > 0 else 0.0
    print(f"[Valid] F1_robust={val_f1_robust:.4f} @thr={robust_thr:.3f} (search_best={best_val_f1:.4f})")

    acct_out, label_out = [], []
    for a in pred_list:
        acct_out.append(a)
        label_out.append(int(prob_all[acct2idx[a]] >= robust_thr) if a in acct2idx else 0)

    out_df = pd.DataFrame({"acct": acct_out, "label": label_out})
    out_df.to_csv(args.out_csv, index=False)
    print(f"[Test] Threshold-only @thr={robust_thr:.3f}, Pred_pos={sum(label_out)}")
    print(f"(Finish) Output saved to {args.out_csv}")


if __name__ == "__main__":
    main()