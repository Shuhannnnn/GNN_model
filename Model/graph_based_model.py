"""
Graph-based modeling utilities for the 2025 E.SUN AI Challenge.

Despite the filename, this module implements a GraphSAGE-like neural
network instead of a tree-based model. It contains:

- AMP compatibility helpers for mixed-precision training.
- Threshold search utilities for F1 optimization.
- Graph construction from account-to-account transactions.
- A multi-layer GraphSAGE model with residual connections.
- A full-batch training loop with early stopping and AMP.

The main script (`main.py`) imports this module to train the model and
generate predictions.
"""

import math
from contextlib import nullcontext
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- AMP compatibility (new torch.amp first, fallback to torch.cuda.amp) ----
try:
    from torch.amp import autocast as _autocast_new
except Exception:  
    _autocast_new = None

try:
    from torch.amp import GradScaler as _GradScalerNew  
except Exception:  
    _GradScalerNew = None

try:
    from torch.cuda.amp import autocast as _autocast_legacy, GradScaler as _GradScalerLegacy
except Exception: 
    _autocast_legacy, _GradScalerLegacy = None, None

# Public shims
def AMP_autocast():
    """
    Return the best available autocast function reference.

    Prefers :mod:`torch.amp` when available and otherwise falls back to
    :mod:`torch.cuda.amp`. The function itself is not used directly; see
    :func:`amp_autocast_ctx` for the recommended high-level API.
    """
    return _autocast_new if _autocast_new is not None else _autocast_legacy


def AMP_GradScaler():
    """
    Return the best available GradScaler class reference.

    Returns
    -------
    type or None
        A GradScaler class if either :mod:`torch.amp` or
        :mod:`torch.cuda.amp` provides it, otherwise ``None``.
    """
    return _GradScalerNew if _GradScalerNew is not None else _GradScalerLegacy


def amp_autocast_ctx(device_type: str, dtype: torch.dtype, enabled: bool):
    """
    Create an autocast context manager that works across PyTorch versions.

    Parameters
    ----------
    device_type : {"cpu", "cuda"}
        Device type on which the model runs.
    dtype : torch.dtype
        The computation dtype to use inside autocast (e.g. bf16 or fp16).
    enabled : bool
        If False, returns a no-op context manager.

    Returns
    -------
    contextlib.AbstractContextManager
        A context manager that enables autocast if supported, or a dummy
        context if not.
    """
    if not enabled:
        return nullcontext()

    if _autocast_new is not None:
        try:
            return _autocast_new(device_type=device_type, dtype=dtype, enabled=True)
        except TypeError:
            pass

    if device_type == "cuda" and _autocast_legacy is not None:
        return _autocast_legacy(dtype=dtype, enabled=True)

    return nullcontext()


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, steps: int = 400) -> Tuple[float, float]:
    """
    Brute-force search for the threshold that maximizes F1 score.

    Parameters
    ----------
    y_true : numpy.ndarray
        Binary ground-truth labels (0 or 1).
    y_prob : numpy.ndarray
        Predicted probabilities in [0, 1].
    steps : int, optional
        Number of thresholds to evaluate between 0.01 and 0.99. Default is 400.

    Returns
    -------
    best_thr : float
        Threshold that maximizes F1.
    best_f1 : float
        Best F1 score obtained at `best_thr`.
    """
    assert y_true.shape == y_prob.shape
    best_thr, best_f1 = 0.5, -1.0
    grid = np.linspace(0.01, 0.99, steps)
    for thr in grid:
        pred = (y_prob >= thr).astype(int)
        tp = np.logical_and(pred == 1, y_true == 1).sum()
        fp = np.logical_and(pred == 1, y_true == 0).sum()
        fn = np.logical_and(pred == 0, y_true == 1).sum()
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1


def select_conservative_threshold(train_y: np.ndarray, train_p: np.ndarray,
                                  val_y: np.ndarray, val_p: np.ndarray,
                                  thr_min=0.3, thr_max=0.7, steps=200,
                                  min_recall=0.35) -> float:
    """
    Select a classification threshold that trades off F1 and recall.

    The function scans thresholds in `[thr_min, thr_max]` and enforces a
    minimum recall on both the training and validation sets. Among all
    thresholds that satisfy the recall constraint, it chooses the one with
    the highest validation F1 (with a slight preference for lower
    thresholds when F1 ties).

    Parameters
    ----------
    train_y, val_y : numpy.ndarray
        Ground-truth labels for the training and validation sets.
    train_p, val_p : numpy.ndarray
        Predicted probabilities for the respective sets.
    thr_min : float, optional
        Lower bound of the threshold search interval.
    thr_max : float, optional
        Upper bound of the threshold search interval.
    steps : int, optional
        Number of thresholds to evaluate.
    min_recall : float, optional
        Minimum recall that must be satisfied on both sets.

    Returns
    -------
    float
        The selected threshold. If no valid threshold is found, 0.5 is used.
    """
    grid = np.linspace(thr_min, thr_max, steps)
    best_thr, best_f1 = 0.5, -1.0

    for thr in grid:
        ok = True
        for (y_true, y_prob) in [(train_y, train_p), (val_y, val_p)]:
            pred = (y_prob >= thr).astype(int)
            tp = ((pred == 1) & (y_true == 1)).sum()
            fn = ((pred == 0) & (y_true == 1)).sum()
            den = tp + fn
            recall = tp / den if den > 0 else 0.0
            if recall < min_recall:
                ok = False
                break
        if not ok:
            continue

        pred_val = (val_p >= thr).astype(int)
        tp = ((pred_val == 1) & (val_y == 1)).sum()
        fp = ((pred_val == 1) & (val_y == 0)).sum()
        fn = ((pred_val == 0) & (val_y == 1)).sum()
        den = 2 * tp + fp + fn
        f1 = 2 * tp / den if den > 0 else 0.0

        if (f1 > best_f1 + 1e-6) or (abs(f1 - best_f1) <= 1e-6 and thr < best_thr):
            best_f1, best_thr = f1, float(thr)

    return best_thr if best_f1 >= 0 else 0.5


def build_graph(tx: torch.Tensor, col: Dict[str, str], acct2idx: Dict[str, int], undirected: bool = True) -> torch.Tensor:
    """
    Build a sparse, row-normalized adjacency matrix for the account graph.

    Each unique sender-to-receiver pair is treated as a directed edge.
    Optionally, undirected edges are constructed by mirroring each edge.
    Self-loops are always added so that each node contributes its own
    features during message passing.

    Parameters
    ----------
    tx : pandas.DataFrame
        Transaction table.
    col : dict
        Column mapping dictionary containing `src` and `dst` keys.
    acct2idx : dict
        Mapping from account ID (string) to node index.
    undirected : bool, optional
        If True, every edge `u -> v` is mirrored as `v -> u`.

    Returns
    -------
    torch.Tensor
        Sparse CSR adjacency matrix `A` with shape (N, N), where N is the
        number of nodes. Each row is normalized by the out-degree so that
        it sums roughly to 1.
    """
    import pandas as pd  

    src, dst = col["src"], col["dst"]
    edges = tx[[src, dst]].dropna().drop_duplicates()
    keys = set(acct2idx.keys())
    edges = edges[edges[src].astype(str).isin(keys) & edges[dst].astype(str).isin(keys)]

    n = len(acct2idx)
    if edges.empty:
        idx = torch.arange(n, dtype=torch.long)
        A = torch.sparse_coo_tensor(
            torch.stack([idx, idx]),
            torch.ones(n, dtype=torch.float32),
            (n, n),
        ).coalesce()
        return A.to_sparse_csr()

    u = edges[src].astype(str).map(acct2idx).astype(np.int64).values
    v = edges[dst].astype(str).map(acct2idx).astype(np.int64).values

    if undirected:
        uu = np.concatenate([u, v], axis=0)
        vv = np.concatenate([v, u], axis=0)
    else:
        uu, vv = u, v

    self_idx = np.arange(n, dtype=np.int64)
    uu = np.concatenate([uu, self_idx], axis=0)
    vv = np.concatenate([vv, self_idx], axis=0)

    deg = np.bincount(uu, minlength=n).astype(np.float32)
    deg[deg == 0.0] = 1.0
    vals = 1.0 / deg[uu]

    indices = torch.from_numpy(np.stack([uu, vv], axis=0)).long()
    values = torch.from_numpy(vals.astype(np.float32))
    A_coo = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    return A_coo.to_sparse_csr()


class SAGELayer(nn.Module):
    """
    Single GraphSAGE layer with separate linear transforms for self and neighbors.

    This layer computes:
    .. math::

        h_i' = W_{self} h_i + W_{neigh} \\sum_j A_{ij} h_j

    where A is a row-normalized adjacency matrix.
    """
    def __init__(self, in_dim: int, out_dim: int) -> None:
        """
        Initialize the GraphSAGE layer.

        Parameters
        ----------
        in_dim : int
            Input feature dimension.
        out_dim : int
            Output feature dimension.
        """
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Apply the GraphSAGE layer.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape (N, D_in).
        A : torch.Tensor
            Sparse CSR adjacency matrix of shape (N, N).

        Returns
        -------
        torch.Tensor
            Updated node features of shape (N, D_out).
        """
        neigh = torch.matmul(A, x)
        return self.lin_self(x) + self.lin_neigh(neigh)


class SAGEBlock(nn.Module):
    """
    GraphSAGE block with LayerNorm, PReLU, dropout and residual connection.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        """
        Initialize the SAGE block.

        Parameters
        ----------
        in_dim : int
            Input feature dimension.
        out_dim : int
            Output feature dimension.
        dropout : float
            Dropout rate applied after the activation.
        """
        super().__init__()
        self.sage = SAGELayer(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.PReLU()
        self.drop = nn.Dropout(dropout)
        self.res = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Apply the SAGE block to node features.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape (N, D_in).
        A : torch.Tensor
            Sparse CSR adjacency matrix of shape (N, N).

        Returns
        -------
        torch.Tensor
            Updated node features of shape (N, D_out).
        """
        h = self.sage(x, A)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        return h + self.res(x)


class SAGEModel(nn.Module):
    """
    Multi-layer GraphSAGE model for node-level binary classification.

    The model stacks several :class:`SAGEBlock` modules and then applies a
    small MLP head that outputs a single logit per node.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_ckpt: bool = False,
    ) -> None:
        """
        Initialize the GraphSAGE model.

        Parameters
        ----------
        in_dim : int
            Input feature dimension.
        hidden_dim : int, optional
            Hidden dimension used for all blocks. Default is 128.
        num_layers : int, optional
            Number of stacked SAGE blocks. Default is 2.
        dropout : float, optional
            Dropout rate used in all blocks and in the final head. Default is 0.3.
        use_ckpt : bool, optional
            If True, enable gradient checkpointing for each block to save GPU
            memory at the cost of extra compute.
        """
        super().__init__()
        from torch.utils.checkpoint import checkpoint as _checkpoint

        self._checkpoint = _checkpoint
        self.use_ckpt = use_ckpt

        dims = [in_dim] + [hidden_dim] * num_layers
        self.blocks = nn.ModuleList(
            SAGEBlock(dims[i], dims[i + 1], dropout) for i in range(num_layers)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass of the GraphSAGE model.

        Parameters
        ----------
        x : torch.Tensor
            Input node features of shape (N, D_in).
        A : torch.Tensor
            Sparse CSR adjacency matrix of shape (N, N).

        Returns
        -------
        torch.Tensor
            Logits for each node, shape (N,).
        """
        h = x
        for blk in self.blocks:
            if self.use_ckpt and self.training:
                h = self._checkpoint(lambda _h, _blk=blk: _blk(_h, A), h, use_reentrant=False)
            else:
                h = blk(h, A)
        return self.head(h).squeeze(-1)


def train_model_fullbatch(model: nn.Module, x_all: torch.Tensor, A: torch.Tensor, y_all: torch.Tensor,
                          train_idx: np.ndarray, val_idx: np.ndarray, lr: float, weight_decay: float,
                          epochs: int, patience: int, label_smoothing: float, eval_every: int,
                          device: torch.device, amp_dtype: torch.dtype) -> Tuple[nn.Module, float, float]:
    """
    Train the GraphSAGE model in full-batch fashion.

    The whole graph is loaded on a single device (CPU or GPU) and optimized
    using AdamW with an F1-based early stopping criterion.

    Parameters
    ----------
    model : torch.nn.Module
        GraphSAGE model to be trained.
    x_all : torch.Tensor
        Node feature matrix of shape (N, D).
    A : torch.Tensor
        Sparse CSR adjacency matrix of shape (N, N).
    y_all : torch.Tensor
        Binary labels for all nodes (only indices in `train_idx` are used
        for computing the loss).
    train_idx, val_idx : numpy.ndarray
        Arrays of node indices used for training and validation.
    lr : float
        Learning rate for AdamW.
    weight_decay : float
        Weight decay coefficient.
    epochs : int
        Maximum number of training epochs.
    patience : int
        Early stopping patience in terms of evaluation steps.
    label_smoothing : float
        Amount of label smoothing to apply in BCE.
    eval_every : int
        Evaluate on the validation set every `eval_every` epochs.
    device : torch.device
        Target device for training (CPU or CUDA).
    amp_dtype : torch.dtype
        Mixed-precision dtype (e.g. bf16 or fp16) to be used with AMP.

    Returns
    -------
    model : torch.nn.Module
        Best-performing model loaded back to CPU.
    best_val_thr : float
        Validation threshold used during training (fixed at 0.5 here).
    best_val_f1 : float
        Best validation F1 score obtained.
    """
    model = model.to(device)
    x_all = x_all.to(device, non_blocking=True)
    A = A.to(device)
    y_all = y_all.to(device, non_blocking=True)

    if device.type == "cuda":
        try:
            x_all = x_all.to(dtype=amp_dtype)
            model = model.to(dtype=amp_dtype)
        except Exception:
            pass

    train_idx_t = torch.from_numpy(train_idx).long().to(device)
    val_idx_t = torch.from_numpy(val_idx).long().to(device)

    with torch.no_grad():
        pos = y_all[train_idx_t].sum().item()
    neg = float(len(train_idx) - pos)
    pos_weight = 1.0 if pos <= 0 else math.sqrt(max(neg / pos, 1.0))
    print(f"[Info] train_pos={pos:.0f}, train_neg={neg:.0f}, pos_weight={pos_weight:.3f}")
    pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)

    def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Binary cross-entropy loss with optional label smoothing.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model outputs.
        target : torch.Tensor
            Ground-truth labels (0 or 1).

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        t = target.float()
        if label_smoothing > 0.0:
            eps = label_smoothing
            t = t * (1.0 - eps) + 0.5 * eps
        return F.binary_cross_entropy_with_logits(logits, t, pos_weight=pos_weight_t)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=max(patience // 4, 3)
    )

    use_amp = device.type == "cuda"
    use_scaler = use_amp and (amp_dtype is torch.float16) and (AMP_GradScaler() is not None)
    ScalerCls = AMP_GradScaler()
    scaler = ScalerCls(device.type if use_scaler else "cpu", enabled=use_scaler) if ScalerCls else None

    best_state, best_val_f1, best_val_thr, no_improve = None, -1.0, 0.5, 0
    max_grad_norm = 5.0
    val_true_np = y_all[val_idx_t].detach().cpu().numpy().astype(int)

    try:  
        torch.backends.cuda.matmul.fp32_precision = "high"
    except Exception:
        pass

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        with amp_autocast_ctx(device.type if use_amp else "cpu", amp_dtype if use_amp else torch.float32, use_amp):
            logits_all = model(x_all, A)
            loss = loss_fn(logits_all[train_idx_t], y_all[train_idx_t])

        if use_scaler and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

        if epoch == 1 or (epoch % eval_every == 0):
            model.eval()
            with torch.no_grad():
                with amp_autocast_ctx(
                    device.type if use_amp else "cpu",
                    amp_dtype if use_amp else torch.float32,
                    use_amp,
                ):
                    logits_val = model(x_all, A)[val_idx_t]
                    prob_val = torch.sigmoid(logits_val).float().cpu().numpy()

            thr = 0.5
            val_pred_fix = (prob_val >= thr).astype(int)
            tp = ((val_pred_fix == 1) & (val_true_np == 1)).sum()
            fp = ((val_pred_fix == 1) & (val_true_np == 0)).sum()
            fn = ((val_pred_fix == 0) & (val_true_np == 1)).sum()
            den = 2 * tp + fp + fn
            f1 = 2 * tp / den if den > 0 else 0.0

            sched.step(f1)

            if f1 > best_val_f1 + 1e-6:
                best_val_f1, best_val_thr = f1, thr
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            print(
                f"[Epoch {epoch:04d}] loss={loss.item():.4f} "
                f"valF1={f1:.4f} @thr={thr:.3f} "
                f"(best={best_val_f1:.4f} @thr={best_val_thr:.3f})"
            )

            if no_improve >= patience:
                print(f"[EarlyStop] epoch={epoch}, best_valF1={best_val_f1:.4f} @thr={best_val_thr:.3f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("[Warn] No val improvement tracked; using last epoch weights.")

    # Always move the final model back to CPU and full precision.
    model = model.to(torch.device("cpu")).float()
    return model, best_val_thr, best_val_f1
