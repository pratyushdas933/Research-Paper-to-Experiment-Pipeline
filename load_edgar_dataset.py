"""
load_edgar_dataset.py

Turns ma_deals.csv (built by edgar_dataset_builder.py) into feature/label
tensors matching what train.py's train_model() expects.

Honest limitations of this data, carried over from edgar_dataset_builder.py:
  - Labels are noisy (Item 1.02 termination in the follow-up window is used
    as a proxy for "deal failed," which can be triggered by an unrelated
    contract termination -- see the README written alongside ma_deals.csv).
  - `mentions_cash` is dropped entirely: it's constant (always False) across
    every row, because EDGAR's full-text search API only returns filing
    METADATA in search hits, not the actual filing body text, so the
    keyword search never had real text to check against. Keeping a
    constant column would add nothing and could look misleadingly like a
    real feature.
  - `mentions_stock` is kept but is true for only 1/100 rows -- essentially
    uninformative at this sample size, included for completeness rather
    than because it's expected to carry real signal.
  - Only 100 rows total: features are kept deliberately low-dimensional
    (bucketed industry + one continuous feature) to avoid a feature count
    that swamps the sample size, which the paper's 65-feature setup
    (17,440 rows) could afford but this dataset cannot.
"""

from __future__ import annotations

import pandas as pd
import torch


def load_edgar_ma_dataset(
    csv_path: str,
    top_n_industries: int = 8,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """
    Returns a dict with X_train, y_train, X_val, y_val tensors, ready for
    train.train_model().

    Feature construction:
      - sic_code: bucketed to the top_n_industries most frequent categories
        plus a single "other" bucket, then one-hot encoded. This keeps
        dimensionality sane for a 100-row dataset (rather than one-hot
        encoding all 53 raw categories, most of which appear only once
        or twice and would just be memorized rather than learned from).
      - days_to_resolution: min-max normalized to [0, 1].
      - mentions_stock: kept as-is (0/1), despite being nearly constant --
        included for completeness, not because it's expected to help.

    Label: 0 = completed, 1 = failed (same convention as train.py/paper).
    """
    df = pd.read_csv(csv_path)

    top_industries = df["sic_code"].value_counts().nlargest(top_n_industries).index.tolist()
    df["sic_bucket"] = df["sic_code"].apply(lambda x: x if x in top_industries else "other")
    sic_dummies = pd.get_dummies(df["sic_bucket"], prefix="sic").astype(float)

    days = df["days_to_resolution"].astype(float)
    days_norm = (days - days.min()) / (days.max() - days.min() + 1e-8)

    mentions_stock = df["mentions_stock"].astype(float)

    X = pd.concat([sic_dummies, days_norm.rename("days_norm"), mentions_stock], axis=1)
    y = df["label"].astype(float)

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(-1)

    torch.manual_seed(seed)
    n = X_tensor.shape[0]
    perm = torch.randperm(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    return {
        "X_train": X_tensor[train_idx],
        "y_train": y_tensor[train_idx],
        "X_val": X_tensor[val_idx],
        "y_val": y_tensor[val_idx],
        "feature_names": list(X.columns),
    }
