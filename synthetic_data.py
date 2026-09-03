"""
synthetic_data.py

Generates a SYNTHETIC dataset that matches the statistical shape described
in Karatas & Hirsa (2021) "Predicting Status of Pre and Post M&A Deals..."
Section 3 -- NOT a reproduction of their actual FactSet data, which is
proprietary and not publicly available.

What we match from the paper's stated numbers:
  - 65 total features after dimensionality reduction (20 PCA dims from
    52 numerical variables + 45 MCA dims from 108 one-hot categorical vars)
  - ~80% completed / ~20% cancelled class balance (paper states training
    data is 80.84% completed, test data 79.89% completed)
  - Train/test split sizes: 16,525 train / 915 test (paper's actual counts)

What we do NOT and CANNOT match:
  - The real underlying financial signal / feature-label relationship.
    We generate labels from a synthetic linear-ish rule with noise, which
    means the model's achievable accuracy on this data has no necessary
    relationship to the paper's reported 88%. Any accuracy number obtained
    by training on this data is a PIPELINE VALIDATION metric only ("does
    the extract -> codegen -> train loop run correctly end to end"), not
    evidence of reproducing the paper's actual result.

This distinction should be stated explicitly wherever results from this
data are reported (e.g. in a portfolio writeup or a diagnostic agent's
verdict) -- conflating "pipeline works" with "paper reproduced" would be
a misleading claim about a proprietary dataset we don't have access to.
"""

from __future__ import annotations

import torch


def generate_ma_deal_dataset(
    n_train: int = 16525,
    n_test: int = 915,
    n_features: int = 65,
    positive_rate: float = 0.808,  # "completed" deals, matches paper's train split stat
    test_positive_rate: float = 0.799,  # matches paper's test split stat
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """
    Returns a dict with X_train, y_train, X_val, y_val tensors.

    Label convention matches the paper: 0 = completed deal, 1 = cancelled
    deal (paper labels completed as 0, cancelled as 1). y tensors are
    shape (N, 1) float, matching what train.py's train_model() expects.

    The label-generating function is a fixed random linear combination of
    features plus noise -- deliberately simple, since the point is to
    exercise the pipeline (extraction -> codegen -> training -> metrics),
    not to simulate real M&A dynamics.
    """
    torch.manual_seed(seed)

    # A fixed "true" weight vector so labels have SOME learnable structure
    # (otherwise even a correctly-implemented model couldn't beat chance,
    # which would make it impossible to tell "pipeline bug" apart from
    # "no signal in the data" when debugging).
    true_weights = torch.randn(n_features, 1)
    noise_scale = 1.5  # tuned so the synthetic task is learnable but not trivial

    def _make_split(n: int, target_positive_rate: float) -> tuple[torch.Tensor, torch.Tensor]:
        X = torch.randn(n, n_features)
        logits = X @ true_weights + torch.randn(n, 1) * noise_scale
        # Threshold chosen per-split to hit the paper's stated class balance
        # (completed=0 is the MAJORITY class at ~80%, so cancelled=1 is the
        # ~20% minority -- threshold set high so few instances are labeled 1).
        k = int(round(n * (1 - target_positive_rate)))  # number of cancelled(=1) cases... see note below
        # NOTE: target_positive_rate here means "fraction completed" per the
        # paper's phrasing ("80.84% of these deals are completed"), so the
        # minority/cancelled class is (1 - target_positive_rate) of the data.
        threshold = torch.quantile(logits.squeeze(), 1 - (1 - target_positive_rate))
        y = (logits.squeeze() > threshold).float().unsqueeze(-1)
        return X, y

    X_train, y_train = _make_split(n_train, positive_rate)
    X_val, y_val = _make_split(n_test, test_positive_rate)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }
