"""
train.py

Takes a ModelIR (already extracted/validated) plus a dataset, and:
  1. patches in dataset-derived dimensions the paper text didn't state
     (e.g. in_features on the first linear layer) -- this is the
     legitimate place for that patching to happen, not a manual notebook
     step each time.
  2. builds the model via codegen.build_model()
  3. trains it using whatever TrainingSpec fields the extraction agent
     found, falling back to clearly-logged defaults for anything left
     null -- every fallback is recorded in TrainingRunResult.assumptions
     so the diagnostic/reproducibility agent can later tell "mismatch
     because paper never specified LR" apart from "mismatch because
     something is actually wrong."
  4. returns a TrainingRunResult with the trained model, per-epoch
     metrics, and the final metrics to compare against ir.reported_results.

This module still contains NO LLM calls -- it's the deterministic training
counterpart to codegen.py's deterministic architecture building.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ir_schema import ModelIR, LayerSpec, LayerType
from codegen import build_model, GeneratedModel


# Fallback defaults used ONLY when the field is null in the extracted
# TrainingSpec. Every use of a fallback is logged into
# TrainingRunResult.assumptions so it's never silently indistinguishable
# from a paper-stated value.
DEFAULT_OPTIMIZER = "adam"
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
DEFAULT_LOSS_FN = "binary_cross_entropy"


_LOSS_FN_MAP = {
    "binary_cross_entropy": nn.BCELoss,
    "binary cross-entropy": nn.BCELoss,
    "binary crossentropy": nn.BCELoss,
    "bce": nn.BCELoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "cross-entropy": nn.CrossEntropyLoss,
}


def _resolve_loss_fn(name: str | None, assumptions: list[str]) -> nn.Module:
    if name is None:
        assumptions.append(f"loss_fn not specified in paper text -- defaulted to {DEFAULT_LOSS_FN}")
        name = DEFAULT_LOSS_FN
    key = name.strip().lower()
    if key not in _LOSS_FN_MAP:
        assumptions.append(
            f"loss_fn '{name}' from paper is not implemented in this training harness "
            f"(only BCE/cross-entropy supported) -- defaulted to {DEFAULT_LOSS_FN}"
        )
        key = DEFAULT_LOSS_FN
    return _LOSS_FN_MAP[key]()


def _resolve_optimizer(name: str | None, params, lr: float, assumptions: list[str]) -> torch.optim.Optimizer:
    if name is None:
        assumptions.append(f"optimizer not specified in paper text -- defaulted to {DEFAULT_OPTIMIZER}")
        name = DEFAULT_OPTIMIZER
    key = name.strip().lower()
    if key == "adam":
        return torch.optim.Adam(params, lr=lr)
    elif key == "sgd":
        return torch.optim.SGD(params, lr=lr)
    elif key == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    else:
        assumptions.append(f"optimizer '{name}' not recognized -- defaulted to {DEFAULT_OPTIMIZER}")
        return torch.optim.Adam(params, lr=lr)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float | None = None
    val_accuracy: float | None = None


@dataclass
class TrainingRunResult:
    model: GeneratedModel
    history: list[EpochMetrics] = field(default_factory=list)
    final_train_accuracy: float = 0.0
    final_val_accuracy: float | None = None
    # Every place a value was missing from the paper and a fallback default
    # was substituted, logged in plain language -- this is what the
    # diagnostic agent reads to separate "paper never said" from "genuinely
    # wrong extraction/reproduction."
    assumptions: list[str] = field(default_factory=list)


def _accuracy(logits_or_probs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = (logits_or_probs.squeeze(-1) > 0.5).float()
    return (preds == targets.squeeze(-1)).float().mean().item()


def patch_input_dim(ir: ModelIR, in_features: int) -> list[str]:
    """
    Fill in in_features (linear) / in_channels (conv2d) on the FIRST layer
    of the IR if it's missing -- this is the dataset-derived value the
    paper text often doesn't restate explicitly (e.g. "65" being implied
    by PCA(20)+MCA(45) mentioned in a different section than the one that
    was extracted).

    Also forward-propagates in_features/in_channels for any LATER layer
    the extraction agent left blank, inferring it from the previous
    layer's out_features/out_channels -- this is a genuine architectural
    necessity (consecutive linear/conv layers must have matching
    dimensions to be valid), not a guess about paper content, so it's
    safe to do deterministically rather than requiring it to have been
    explicitly extracted.

    Deeper missing params that AREN'T inferable this way (e.g. a
    completely absent out_features) are left alone so build_model() still
    raises loudly on them.

    Returns a list of assumption strings describing what was patched.
    """
    assumptions = []
    if not ir.layers:
        return assumptions

    first = ir.layers[0]
    if first.type.value == "linear" and "in_features" not in first.params:
        first.params["in_features"] = in_features
        assumptions.append(
            f"Layer '{first.id}': in_features not stated in extracted text -- "
            f"patched with dataset's actual feature count ({in_features})"
        )
    elif first.type.value == "conv2d" and "in_channels" not in first.params:
        first.params["in_channels"] = in_features
        assumptions.append(
            f"Layer '{first.id}': in_channels not stated in extracted text -- "
            f"patched with dataset's actual channel count ({in_features})"
        )

    prev_out = None
    for layer in ir.layers:
        if layer.type.value == "linear":
            if "in_features" not in layer.params and prev_out is not None:
                layer.params["in_features"] = prev_out
                assumptions.append(
                    f"Layer '{layer.id}': in_features not stated -- inferred "
                    f"from previous layer's output size ({prev_out})"
                )
            prev_out = layer.params.get("out_features", prev_out)
        elif layer.type.value == "conv2d":
            if "in_channels" not in layer.params and prev_out is not None:
                layer.params["in_channels"] = prev_out
                assumptions.append(
                    f"Layer '{layer.id}': in_channels not stated -- inferred "
                    f"from previous layer's output size ({prev_out})"
                )
            prev_out = layer.params.get("out_channels", prev_out)

    return assumptions


def ensure_binary_output_head(ir: ModelIR) -> list[str]:
    """
    If the IR's last layer doesn't produce a single-value sigmoid output
    (i.e. it's not shaped for binary classification), append a
    Linear(->1) + Sigmoid head.

    This covers a real gap: a paper's text often states the hidden-layer
    sizes but never explicitly restates "and a final layer maps to 1
    output with sigmoid" -- that's implied by "this is a binary
    classifier" rather than spelled out, so the extraction agent
    correctly leaves it out per the never-guess rule (see
    extraction_agent.py). Some output head has to exist for training to
    be possible at all, so this harness adds one deterministically and
    logs it as an assumption -- it is NOT invented by the LLM.
    """
    assumptions = []
    if not ir.layers:
        return assumptions

    last = ir.layers[-1]
    already_binary_head = (
        last.type.value == "activation" and last.params.get("fn") == "sigmoid"
    )
    if already_binary_head:
        return assumptions

    # Find the output size of the last linear/conv layer to size the new head.
    prev_out = None
    for layer in ir.layers:
        if layer.type.value == "linear":
            prev_out = layer.params.get("out_features", prev_out)
        elif layer.type.value == "conv2d":
            prev_out = layer.params.get("out_channels", prev_out)
    if prev_out is None:
        return assumptions  # nothing sane to attach a head to; let build_model() fail loudly instead

    new_id_base = f"auto_output_head_{len(ir.layers)}"
    ir.layers.append(LayerSpec(
        id=f"{new_id_base}_linear",
        type=LayerType.linear,
        params={"in_features": prev_out, "out_features": 1},
    ))
    ir.layers.append(LayerSpec(
        id=f"{new_id_base}_sigmoid",
        type=LayerType.activation,
        params={"fn": "sigmoid"},
    ))
    assumptions.append(
        f"Paper text described hidden layers ending at {prev_out} units but never explicitly "
        f"stated a final output layer -- appended Linear({prev_out}->1) + Sigmoid as the binary "
        f"classification head, since one is required for training but wasn't stated in the "
        f"extracted text."
    )
    return assumptions


def train_model(
    ir: ModelIR,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
    verbose: bool = True,
) -> TrainingRunResult:
    """
    Train a model built from `ir` on the given tensors.

    X_train: (N, in_features) float tensor
    y_train: (N, 1) float tensor with values in {0, 1} (binary classification)
    X_val / y_val: optional held-out set, same shape convention.

    Automatically patches the first layer's input dimension from
    X_train.shape[1] if the IR doesn't already specify it -- this is the
    dataset-derived patch described in patch_input_dim().
    """
    assumptions: list[str] = []
    assumptions += patch_input_dim(ir, in_features=X_train.shape[1])
    assumptions += ensure_binary_output_head(ir)

    model = build_model(ir)

    ts = ir.training
    loss_fn = _resolve_loss_fn(ts.loss_fn, assumptions)
    lr = ts.learning_rate if ts.learning_rate is not None else DEFAULT_LEARNING_RATE
    if ts.learning_rate is None:
        assumptions.append(f"learning_rate not specified in paper text -- defaulted to {DEFAULT_LEARNING_RATE}")
    optimizer = _resolve_optimizer(ts.optimizer, model.parameters(), lr, assumptions)

    batch_size = ts.batch_size if ts.batch_size is not None else DEFAULT_BATCH_SIZE
    if ts.batch_size is None:
        assumptions.append(f"batch_size not specified in paper text -- defaulted to {DEFAULT_BATCH_SIZE}")

    epochs = ts.epochs if ts.epochs is not None else DEFAULT_EPOCHS
    if ts.epochs is None:
        assumptions.append(f"epochs not specified in paper text -- defaulted to {DEFAULT_EPOCHS}")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    history: list[EpochMetrics] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += _accuracy(preds.detach(), yb)
            n_batches += 1
        train_loss = epoch_loss / n_batches
        train_acc = epoch_acc / n_batches

        val_loss = val_acc = None
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val)
                val_loss = loss_fn(val_preds, y_val).item()
                val_acc = _accuracy(val_preds, y_val)

        history.append(EpochMetrics(epoch, train_loss, train_acc, val_loss, val_acc))
        if verbose:
            msg = f"Epoch {epoch}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f}"
            if val_acc is not None:
                msg += f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
            print(msg)

    return TrainingRunResult(
        model=model,
        history=history,
        final_train_accuracy=history[-1].train_accuracy,
        final_val_accuracy=history[-1].val_accuracy,
        assumptions=assumptions,
    )
