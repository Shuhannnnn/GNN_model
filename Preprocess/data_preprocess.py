"""
Data preprocessing utilities for the 2025 E.SUN AI Challenge.

This module is responsible for:
- Setting the random seed for reproducibility.
- Loading the raw CSV files from disk.
- Resolving column names that may differ between datasets.
- Aggregating transaction-level records into account-level features.

The output features are later consumed by the GNN model defined in
`Model/graph_based_model.py`.
"""

import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch

def load_csvs(data_dir: str):
    """
    Load the three datasets provided in the competition: transaction data, 
    alert account labels, and the list of accounts to be predicted.

    Parameters
    ----------
    data_dir : str
        Directory that contains the three CSV files.

    Returns
    -------
    tx : pandas.DataFrame
        Transaction-level records.
    alert : pandas.DataFrame
        Accounts that have been labeled as alert in the future.
    predict : pandas.DataFrame
        List of accounts to be scored by the model.
    """
    tx = pd.read_csv(os.path.join(data_dir, "acct_transaction.csv"))
    alert = pd.read_csv(os.path.join(data_dir, "acct_alert.csv"))
    predict = pd.read_csv(os.path.join(data_dir, "acct_predict.csv"))
    print("(Finish) Load Dataset.")
    return tx, alert, predict

def set_seed(seed: int = 42):
    """
    Set random seeds for Python, NumPy and PyTorch.

    Parameters
    ----------
    seed : int, optional
        Seed value to use for all RNGs. Default is 42.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Heuristically find a column name in a DataFrame given several candidates.

    The function tries:
    1. Exact match (case-sensitive).
    2. Exact match (case-insensitive).
    3. Substring match (case-insensitive).

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    candidates : list of str
        Possible column names for the target field.

    Returns
    -------
    str or None
        The best-matching column name, or None if nothing is found.
    """
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        cand_lower = cand.lower()
        if cand_lower in lower:
            return lower[cand_lower]
    for cand in candidates:
        cand_lower = cand.lower()
        for c in cols:
            if cand_lower in c.lower():
                return c
    return None


def resolve_columns(tx: pd.DataFrame, alert: pd.DataFrame, predict: pd.DataFrame) -> Dict[str, str]:
    """
    Resolve column names for key fields used in the pipeline.

    Different versions of the dataset may use slightly different names for
    sender account, receiver account, transaction amount and account IDs.
    This helper tries to map them into a unified dictionary.

    Parameters
    ----------
    tx : pandas.DataFrame
        Transaction table.
    alert : pandas.DataFrame
        Alert account table.
    predict : pandas.DataFrame
        Prediction account table.

    Returns
    -------
    dict
        Dictionary containing the resolved column names with keys:
        `src`, `dst`, `amt`, `from_type`, `to_type`,
        `alert_acct`, `predict_acct`.

    Raises
    ------
    ValueError
        If mandatory columns (src/dst or alert/predict account) cannot be found.
    """
    src = find_col(tx, ["from_acct", "payer_acct_id", "src", "src_acct"])
    dst = find_col(tx, ["to_acct", "receiver_acct_id", "dst", "dst_acct"])
    amt = find_col(tx, ["txn_amt", "amount", "amt", "tx_amt", "trade_amount"])
    if src is None or dst is None:
        raise ValueError("Cannot find src/dst in acct_transaction.csv")
    if amt is None:
        tx["_AMT_"] = 1.0
        amt = "_AMT_"
    from_type = find_col(tx, ["from_acct_type", "payer_type", "src_type", "is_esun_from"])
    to_type = find_col(tx, ["to_acct_type", "receiver_type", "dst_type", "is_esun_to"])
    alert_acct = find_col(alert, ["acct", "acct_id", "account_id"])
    predict_acct = find_col(predict, ["acct", "acct_id", "account_id"])
    if alert_acct is None or predict_acct is None:
        raise ValueError("Cannot find acct in alert/predict")
    col = {
        "src": src,
        "dst": dst,
        "amt": amt,
        "from_type": from_type,
        "to_type": to_type,
        "alert_acct": alert_acct,
        "predict_acct": predict_acct,
    }
    print("[Column mapping]", col)
    return col


def build_account_features(tx: pd.DataFrame, col: Dict[str, str]) -> pd.DataFrame:
    """
    Aggregate transaction-level data into account-level features.

    The function produces degree-like statistics, amount summaries and
    log-transformed versions of skewed features. It also tries to infer
    whether an account belongs to E.SUN bank (is_esun) using optional
    `from_type` and `to_type` columns.

    Parameters
    ----------
    tx : pandas.DataFrame
        Transaction-level table.
    col : dict
        Column mapping dictionary returned by :func:`resolve_columns`.

    Returns
    -------
    pandas.DataFrame
        Account-level feature table. Each row corresponds to one account and
        contains:
        - basic send/receive summary statistics,
        - out-degree / in-degree,
        - transaction counts,
        - log1p-transformed versions of skewed features,
        - `is_esun` flag.
    """
    src, dst, amt = col["src"], col["dst"], col["amt"]

    send_sum = tx.groupby(src)[amt].sum().rename("total_send_amt")
    send_max = tx.groupby(src)[amt].max().rename("max_send_amt")
    send_min = tx.groupby(src)[amt].min().rename("min_send_amt")
    send_avg = tx.groupby(src)[amt].mean().rename("avg_send_amt")
    out_deg = tx.groupby(src)[dst].nunique().rename("out_deg")
    out_cnt = tx.groupby(src)[dst].count().rename("out_tx_count")

    recv_sum = tx.groupby(dst)[amt].sum().rename("total_recv_amt")
    recv_max = tx.groupby(dst)[amt].max().rename("max_recv_amt")
    recv_min = tx.groupby(dst)[amt].min().rename("min_recv_amt")
    recv_avg = tx.groupby(dst)[amt].mean().rename("avg_recv_amt")
    in_deg = tx.groupby(dst)[src].nunique().rename("in_deg")
    in_cnt = tx.groupby(dst)[src].count().rename("in_tx_count")

    df_out = pd.concat([send_sum, send_max, send_min, send_avg, out_deg, out_cnt], axis=1)
    df_in = pd.concat([recv_sum, recv_max, recv_min, recv_avg, in_deg, in_cnt], axis=1)

    idx = pd.Index(df_out.index).union(df_in.index)
    feat = (pd.DataFrame(index=idx).join(df_out, how="left").join(df_in, how="left")
            .fillna(0.0).reset_index().rename(columns={"index": "acct"}))

    feat["acct"] = feat["acct"].astype(str)
    feat["is_esun"] = 1

    from_type = col.get("from_type"); to_type = col.get("to_type"); esun_accts = set()
    if from_type and from_type in tx.columns:
        tmp = tx[[src, from_type]].dropna().drop_duplicates()
        if tmp[from_type].dtype != object: esun_accts.update(tmp.loc[tmp[from_type] == 1, src].astype(str).tolist())
    if to_type and to_type in tx.columns:
        tmp = tx[[dst, to_type]].dropna().drop_duplicates()
        if tmp[to_type].dtype != object: esun_accts.update(tmp.loc[tmp[to_type] == 1, dst].astype(str).tolist())
    if esun_accts:
        feat["is_esun"] = feat["acct"].isin(esun_accts).astype(int)

    skew_cols = [
        "total_send_amt", "total_recv_amt",
        "max_send_amt", "min_send_amt", "avg_send_amt",
        "max_recv_amt", "min_recv_amt", "avg_recv_amt",
        "out_tx_count", "in_tx_count",
    ]
    for c in skew_cols: feat[f"log1p_{c}"] = np.log1p(feat[c].astype(float))

    keep_cols = (["acct", "is_esun",
                  "total_send_amt", "total_recv_amt",
                  "max_send_amt", "min_send_amt", "avg_send_amt",
                  "max_recv_amt", "min_recv_amt", "avg_recv_amt",
                  "out_deg", "in_deg",
                  "out_tx_count", "in_tx_count"]
                 + [f"log1p_{c}" for c in skew_cols])
    return feat[keep_cols].fillna(0.0)