"""
diagnose_agent.py

The reproducibility-verdict agent. Takes a ModelIR (with its
unsupported_elements) and a TrainingRunResult (with its assumptions and
metrics), and produces a structured verdict on whether the paper's result
was reproduced -- and if not, WHY, distinguishing between:

  (a) "not comparable" -- e.g. synthetic/substitute data was used instead
      of the paper's real dataset, so no metric comparison is meaningful
      regardless of the numbers
  (b) "extraction gap" -- something in unsupported_elements plausibly
      explains the mismatch (missing preprocessing steps, skipped model
      variants, unmapped architecture pieces)
  (c) "assumption gap" -- a fallback default (train.py's assumptions list)
      was substituted for a hyperparameter the paper never stated, and
      that's the more likely explanation
  (d) "reproduced" -- metrics are within tolerance and no confounding
      gaps were logged
  (e) "not reproduced, unexplained" -- metrics don't match and neither
      (a)/(b)/(c) obviously explains it -- this is the honest "something
      is actually wrong" case, which is the most useful one to surface
      clearly rather than paper over.

This is implemented as a LangGraph graph because the routing between these
cases is genuinely conditional -- the right next check depends on the
outcome of the previous one, which is exactly the case where a simple
linear script would end up as a pile of nested if/else that's harder to
extend later (e.g. adding a retry-with-different-hyperparameters branch).

No LLM call is required for the comparison logic itself (it's arithmetic
+ list inspection), but the FINAL verdict summary is phrased through an
LLM call so it reads as a clear, human-readable diagnostic report rather
than a dumped data structure. That LLM call is instructed to only
summarize the structured findings already computed -- not to invent new
judgments about numbers it wasn't given.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional, TypedDict

from dotenv import load_dotenv
from groq import Groq

from ir_schema import ModelIR
from train import TrainingRunResult

load_dotenv()

VerdictLabel = Literal[
    "not_comparable",
    "reproduced",
    "not_reproduced_extraction_gap",
    "not_reproduced_assumption_gap",
    "not_reproduced_unexplained",
]

# How close achieved accuracy must be to the paper's reported accuracy to
# count as "reproduced" -- a fixed, disclosed tolerance rather than a vague
# judgment call.
ACCURACY_TOLERANCE = 0.05


@dataclass
class DiagnosticVerdict:
    label: VerdictLabel
    achieved_metric: Optional[float]
    reported_metric: Optional[float]
    metric_name: Optional[str]
    contributing_factors: list[str] = field(default_factory=list)
    summary: str = ""


class DiagnosisState(TypedDict, total=False):
    ir: ModelIR
    result: TrainingRunResult
    used_synthetic_data: bool
    metric_name: Optional[str]
    reported_metric: Optional[float]
    achieved_metric: Optional[float]
    metrics_within_tolerance: Optional[bool]
    contributing_factors: list[str]
    label: VerdictLabel
    summary: str


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def node_check_data_authenticity(state: DiagnosisState) -> DiagnosisState:
    """
    First gate: if synthetic/substitute data was used, no metric comparison
    against the paper is meaningful, full stop. This check happens before
    anything else so we never accidentally present a synthetic-data accuracy
    number as evidence about the paper's real result.
    """
    if state.get("used_synthetic_data"):
        state["label"] = "not_comparable"
        state["contributing_factors"] = [
            "Training was run on synthetic/substitute data, not the paper's "
            "original (proprietary) dataset. Any accuracy figure obtained "
            "validates the pipeline mechanics only, not the paper's finding."
        ]
    return state


def node_compare_metrics(state: DiagnosisState) -> DiagnosisState:
    """
    Compares TrainingRunResult's achieved metric against ir.reported_results.
    Only runs meaningfully if we didn't already short-circuit to
    'not_comparable'.
    """
    ir: ModelIR = state["ir"]
    result: TrainingRunResult = state["result"]

    if not ir.reported_results:
        state["metric_name"] = None
        state["reported_metric"] = None
        state["achieved_metric"] = result.final_val_accuracy or result.final_train_accuracy
        state["metrics_within_tolerance"] = None
        return state

    # Use the first reported accuracy-like metric found; a more thorough
    # version could match by metric_name against multiple achieved metrics,
    # but accuracy is the only one train.py currently tracks.
    reported = next(
        (r for r in ir.reported_results if r.metric_name.lower() in ("accuracy", "acc")),
        ir.reported_results[0],
    )
    achieved = result.final_val_accuracy if result.final_val_accuracy is not None else result.final_train_accuracy

    state["metric_name"] = reported.metric_name
    state["reported_metric"] = reported.metric_value
    state["achieved_metric"] = achieved
    state["metrics_within_tolerance"] = abs(achieved - reported.metric_value) <= ACCURACY_TOLERANCE
    return state


def node_check_extraction_gaps(state: DiagnosisState) -> DiagnosisState:
    """
    If metrics didn't match, check whether ir.unsupported_elements plausibly
    explains it (e.g. missing preprocessing steps like PCA/MCA/SMOTE that
    were never applied to the training data since codegen can't build them).
    """
    if state.get("label") == "not_comparable":
        return state
    if state.get("metrics_within_tolerance"):
        state["label"] = "reproduced"
        return state

    ir: ModelIR = state["ir"]
    if ir.unsupported_elements:
        state["label"] = "not_reproduced_extraction_gap"
        state["contributing_factors"] = state.get("contributing_factors", []) + [
            f"Unsupported/unmapped element from extraction: {e}" for e in ir.unsupported_elements
        ]
    return state


def node_check_assumption_gaps(state: DiagnosisState) -> DiagnosisState:
    """
    If not already explained by an extraction gap, check whether
    TrainingRunResult.assumptions (fallback defaults substituted for
    hyperparameters the paper never stated) plausibly explains the mismatch.
    """
    if state.get("label") in ("not_comparable", "reproduced", "not_reproduced_extraction_gap"):
        return state

    result: TrainingRunResult = state["result"]
    if result.assumptions:
        state["label"] = "not_reproduced_assumption_gap"
        state["contributing_factors"] = state.get("contributing_factors", []) + [
            f"Fallback assumption used (paper didn't specify): {a}" for a in result.assumptions
        ]
    else:
        state["label"] = "not_reproduced_unexplained"
        state["contributing_factors"] = state.get("contributing_factors", []) + [
            "No extraction gaps or fallback assumptions were logged, yet the "
            "achieved metric does not match the paper's reported value within "
            f"tolerance ({ACCURACY_TOLERANCE:.0%}). This suggests a genuine "
            "discrepancy -- e.g. an extraction error that wasn't self-flagged, "
            "a codegen bug, or a real difference between the training setup "
            "and the paper's actual procedure -- and warrants manual review."
        ]
    return state


def node_summarize(state: DiagnosisState) -> DiagnosisState:
    """
    Phrases the structured verdict as a short, clear human-readable summary.
    Uses an LLM call, but ONLY to summarize the already-computed structured
    findings -- explicitly told not to add new judgments about the numbers.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        state["summary"] = _fallback_summary(state)
        return state

    client = Groq(api_key=api_key)
    prompt = f"""Summarize this reproducibility-check result in 2-4 plain sentences. \
Do not invent any numbers or claims beyond what's given below. Do not soften \
or hedge an "unexplained" verdict -- state it plainly if that's the label.

Verdict label: {state.get('label')}
Metric name: {state.get('metric_name')}
Paper's reported value: {state.get('reported_metric')}
Achieved value: {state.get('achieved_metric')}
Contributing factors:
{chr(10).join('- ' + f for f in state.get('contributing_factors', []))}
"""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        state["summary"] = response.choices[0].message.content.strip()
    except Exception:
        state["summary"] = _fallback_summary(state)
    return state


def _fallback_summary(state: DiagnosisState) -> str:
    """Deterministic summary used if no LLM call is available/succeeds."""
    label = state.get("label")
    if label == "not_comparable":
        return "Not comparable: synthetic/substitute data was used, so this run cannot validate or refute the paper's reported result."
    if label == "reproduced":
        return (
            f"Reproduced within tolerance: achieved {state.get('achieved_metric'):.3f} "
            f"vs. paper's reported {state.get('reported_metric'):.3f} ({state.get('metric_name')})."
        )
    factors = "; ".join(state.get("contributing_factors", [])) or "no specific factors logged"
    return f"Not reproduced ({label}): {factors}"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def _build_graph():
    from langgraph.graph import StateGraph, END

    graph = StateGraph(DiagnosisState)
    graph.add_node("check_data_authenticity", node_check_data_authenticity)
    graph.add_node("compare_metrics", node_compare_metrics)
    graph.add_node("check_extraction_gaps", node_check_extraction_gaps)
    graph.add_node("check_assumption_gaps", node_check_assumption_gaps)
    graph.add_node("summarize", node_summarize)

    graph.set_entry_point("check_data_authenticity")

    def route_after_authenticity(state: DiagnosisState) -> str:
        return "summarize" if state.get("label") == "not_comparable" else "compare_metrics"

    graph.add_conditional_edges("check_data_authenticity", route_after_authenticity, {
        "summarize": "summarize",
        "compare_metrics": "compare_metrics",
    })
    graph.add_edge("compare_metrics", "check_extraction_gaps")

    def route_after_extraction_check(state: DiagnosisState) -> str:
        return "summarize" if state.get("label") in ("reproduced",) else "check_assumption_gaps"

    graph.add_conditional_edges("check_extraction_gaps", route_after_extraction_check, {
        "summarize": "summarize",
        "check_assumption_gaps": "check_assumption_gaps",
    })
    graph.add_edge("check_assumption_gaps", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


def diagnose(ir: ModelIR, result: TrainingRunResult, used_synthetic_data: bool) -> DiagnosticVerdict:
    """
    Public entry point. Runs the LangGraph diagnostic graph and returns a
    DiagnosticVerdict.
    """
    app = _build_graph()
    initial_state: DiagnosisState = {
        "ir": ir,
        "result": result,
        "used_synthetic_data": used_synthetic_data,
        "contributing_factors": [],
    }
    final_state = app.invoke(initial_state)

    return DiagnosticVerdict(
        label=final_state["label"],
        achieved_metric=final_state.get("achieved_metric"),
        reported_metric=final_state.get("reported_metric"),
        metric_name=final_state.get("metric_name"),
        contributing_factors=final_state.get("contributing_factors", []),
        summary=final_state.get("summary", ""),
    )
