"""
generate_dashboard_data.py

Runs the full paper-to-experiment pipeline once (extraction -> codegen ->
training -> diagnosis) and dumps a single dashboard_data.json that the
static HTML dashboard (dashboard.html) reads. Follows the same
notebook/script -> JSON -> HTML/JS pattern used across the other projects
in this portfolio.

Run this after you have:
  - paper text ready (paste inline or load from a .txt file)
  - GROQ_API_KEY set (via .env, loaded automatically)
  - ma_deals.csv (or another dataset CSV) available for the training step

Usage:
    python generate_dashboard_data.py
"""

from __future__ import annotations

import json

from extraction_agent import extract_ir
from codegen import build_model
from load_edgar_dataset import load_edgar_ma_dataset
from train import train_model, ensure_binary_output_head
from diagnose_agent import diagnose


PAPER_TITLE = "Predicting Status of Pre and Post M&A Deals Using ML and DL Techniques"
PAPER_SOURCE = "Karatas & Hirsa (2021), arXiv:2110.09315"

SECTION_TEXT = """
NN-Accuracy model is defined by two hidden layers with 128 and 8 neurons, respectively.
Relu (rectified linear unit) activation function is used in both layers, and binary
cross-entropy function is chosen as the loss function.
"""

DATASET_CSV = "ma_deals.csv"
DATASET_LABEL = "Real SEC EDGAR merger-agreement filings (self-collected, noisy labels -- see ma_deals_README.txt)"
USED_SYNTHETIC_DATA = False


def main():
    print("Step 1/5: Extracting IR from paper text...")
    ir = extract_ir(SECTION_TEXT, paper_title=PAPER_TITLE)

    print("Step 2/5: Building model from IR...")
    # Codegen needs in_features on every linear layer before it can build.
    # The dataset-derived value for the FIRST layer gets patched in
    # automatically inside train_model(); here we build a throwaway
    # summary copy, so we also forward-propagate in_features/in_channels
    # for any LATER layer the extraction agent left blank (inferable from
    # the previous layer's out_features/out_channels) -- this is a
    # genuine architectural inference, not a guess, since consecutive
    # linear layers must match dimensions to be valid.
    ir_for_summary = ir.model_copy(deep=True)
    if ir_for_summary.layers and "in_features" not in ir_for_summary.layers[0].params:
        ir_for_summary.layers[0].params["in_features"] = 11  # matches load_edgar_ma_dataset's feature count

    prev_out = None
    for layer in ir_for_summary.layers:
        if layer.type.value == "linear":
            if "in_features" not in layer.params and prev_out is not None:
                layer.params["in_features"] = prev_out
            prev_out = layer.params.get("out_features", prev_out)
        elif layer.type.value == "conv2d":
            if "in_channels" not in layer.params and prev_out is not None:
                layer.params["in_channels"] = prev_out
            prev_out = layer.params.get("out_channels", prev_out)
    ensure_binary_output_head(ir_for_summary)

    model = build_model(ir_for_summary)
    architecture_summary = str(model)

    print("Step 3/5: Loading dataset...")
    data = load_edgar_ma_dataset(DATASET_CSV)

    print("Step 4/5: Training...")
    result = train_model(
        ir, data["X_train"], data["y_train"], data["X_val"], data["y_val"], verbose=True
    )

    print("Step 5/5: Running diagnostic verdict...")
    verdict = diagnose(ir, result, used_synthetic_data=USED_SYNTHETIC_DATA)

    dashboard_data = {
        "paper": {
            "title": ir.paper_title or PAPER_TITLE,
            "source": PAPER_SOURCE,
            "excerpt": SECTION_TEXT.strip(),
        },
        "extracted_ir": {
            "layers": [
                {
                    "id": layer.id,
                    "type": layer.type.value,
                    "params": layer.params,
                    "source_span": layer.source_span,
                }
                for layer in ir.layers
            ],
            "training_spec": ir.training.model_dump(),
            "unsupported_elements": ir.unsupported_elements,
            "reported_results": [r.model_dump() for r in ir.reported_results],
        },
        "model_architecture": architecture_summary,
        "dataset": {
            "label": DATASET_LABEL,
            "n_train": data["X_train"].shape[0],
            "n_val": data["X_val"].shape[0],
            "feature_names": data["feature_names"],
        },
        "training_history": [
            {
                "epoch": e.epoch,
                "train_loss": e.train_loss,
                "train_accuracy": e.train_accuracy,
                "val_loss": e.val_loss,
                "val_accuracy": e.val_accuracy,
            }
            for e in result.history
        ],
        "assumptions": result.assumptions,
        "final_train_accuracy": result.final_train_accuracy,
        "final_val_accuracy": result.final_val_accuracy,
        "verdict": {
            "label": verdict.label,
            "achieved_metric": verdict.achieved_metric,
            "reported_metric": verdict.reported_metric,
            "metric_name": verdict.metric_name,
            "contributing_factors": verdict.contributing_factors,
            "summary": verdict.summary,
        },
    }

    with open("dashboard_data.json", "w") as f:
        json.dump(dashboard_data, f, indent=2)

    print("\nDone. Wrote dashboard_data.json -- open dashboard.html to view it.")


if __name__ == "__main__":
    main()
