# Research-Paper-to-Experiment Pipeline

**An agentic system that reads a machine learning paper's methodology, reconstructs the described model, trains it on real data, and produces an honest verdict on whether the paper's results reproduce.**

---

## Why I built this

Most portfolio ML projects show a model that works. This one is deliberately different: it shows a *system that investigates whether a claimed result holds up* — and is explicit about the difference between "this pipeline ran successfully" and "this paper's finding was reproduced." That distinction is the whole point. A system that quietly conflates the two would be more impressive to look at and less trustworthy to use.

I chose an M&A deal-outcome prediction paper (Karatas & Hirsa, 2021, arXiv:2110.09315) deliberately, since it connects directly to the kind of transaction advisory and due-diligence work Aritra Partners does — the pipeline extracts and reproduces a neural network that predicts whether a merger deal completes or fails, the same question a deal-advisory team cares about.

## What it does

Given a paper's methodology text, the system:

1. **Extracts** the described model architecture and training setup into a structured, validated specification (an "IR" — intermediate representation) using an LLM call with strict rules against hallucinating unstated details.
2. **Generates** a real PyTorch model deterministically from that specification — no LLM involvement in code generation itself, so the same spec always produces the same model.
3. **Trains** the model on data, automatically resolving dataset-derived gaps (like input dimensions) the paper's text didn't restate, while logging every fallback assumption used along the way.
4. **Diagnoses** the result with a LangGraph agent that compares achieved metrics against the paper's reported results and explains *why* they match or don't — distinguishing a genuine discrepancy from an explainable gap (missing preprocessing steps, an unstated hyperparameter, or non-comparable data).
5. **Presents** the full trail — extracted spec, generated architecture, training curve, and verdict — in a dashboard.

## Architecture decisions worth explaining

**Extraction and code generation are deliberately separated.** The LLM only ever fills in a structured specification; a separate, pure Python module turns that specification into a PyTorch model. This means the same extracted spec always produces the same model — architecture generation is fully reproducible and debuggable, and if something's wrong with the model, I know immediately whether to look at what was extracted or how it was built.

**The system is built to say "I don't know" instead of guessing.** Every field in the extracted specification can be left blank rather than filled with a plausible-sounding default — a missing learning rate stays missing, not defaulted silently. When the training harness *does* need to fill a gap to make training possible at all (like inferring a layer's input size from the dataset, or appending an output layer that "two hidden layers of 128 and 8 neurons" implies but doesn't state), that assumption is logged in plain language and carried through to the final verdict. The diagnostic agent's job is largely to read that assumption log and decide whether it plausibly explains a mismatch.

**The reproducibility verdict has five distinct outcomes, not two.** Rather than a binary "reproduced / not reproduced," the diagnostic agent distinguishes: not comparable (substitute data was used), reproduced within tolerance, not reproduced due to a flagged extraction gap, not reproduced due to a disclosed hyperparameter assumption, and — the most important one to get right — not reproduced with no explanation found, which is stated plainly rather than papered over.

## Working with real, imperfect data

The paper's original dataset (FactSet M&A records) is proprietary and unavailable. Rather than quietly substituting synthetic data and presenting results as if they meant something they didn't, I built a second path: a real dataset sourced from SEC EDGAR's public filings, using merger-agreement announcement 8-Ks and subsequent termination filings to label deal outcomes.

This real-data path surfaced a genuine, useful lesson: the labeling heuristic (inferring deal failure from any termination filing in a follow-up window) is noisy — it can't distinguish termination of the merger itself from an unrelated contract termination — and produces a higher failure rate than realistic M&A base rates suggest. Rather than hide this, it's documented directly alongside the dataset (an auto-generated limitations file) and factored into how the pipeline's own diagnostic agent interprets results trained on it.

I consider this a feature of the project, not a shortfall: a system that surfaces its own data-quality limitations, rather than a system that produces a clean number I can't fully stand behind, is the more defensible thing to bring into a due-diligence-adjacent context.

## What a run actually looks like

For the paper's "NN-Accuracy" model — a two-hidden-layer network (128, then 8 neurons, ReLU) — the pipeline:
- Correctly extracted the described layers without inventing an unstated output layer
- Correctly flagged that gap, and the training harness appended a disclosed, logged output head to make training possible
- Trained on 200 real EDGAR-derived deals, reaching ~80% validation accuracy
- Returned a `not_reproduced_assumption_gap` verdict — explicitly naming every fallback default used (learning rate, optimizer, batch size, epochs, output head) as the reason the ~80% result isn't directly comparable to the paper's reported 88%, rather than claiming either success or failure

## Stack

Python, PyTorch, Pydantic v2 (schema validation), LangGraph (diagnostic agent), Groq API (extraction LLM calls), SEC EDGAR full-text search and submissions APIs, pandas, Chart.js (dashboard).

## How to run it

```bash
pip install torch pydantic groq python-dotenv langgraph pandas requests

# .env file with your Groq key
echo "GROQ_API_KEY=your_key_here" > .env

# (optional) rebuild the real dataset from SEC EDGAR
export EDGAR_CONTACT_EMAIL="you@example.com"
python edgar_dataset_builder.py --target-deals 200 --out ma_deals.csv

# run the full pipeline and generate dashboard data
python generate_dashboard_data.py

# open dashboard.html in a browser
```

`generate_dashboard_data.py` runs all five stages — extraction, codegen, data loading, training, diagnosis — against the paper text and dataset path set at the top of the file. Swap in different paper text or a different CSV to run it against another paper.

## What I'd extend next

- A second, architecturally different paper (e.g. one using convolutional or residual layers) to test extraction generality beyond feedforward networks
- Tighter EDGAR labeling (verifying termination filings reference the *same* merger agreement, not just any termination in the window)
- A richer real feature set — the current EDGAR-derived features are intentionally minimal (industry, resolution timing) compared to the paper's 65 financial-ratio features, since building genuine financial-ratio features from public filings is a substantial project in its own right
