"""
extraction_agent.py

Takes raw methodology-section text from a paper and extracts a ModelIR
(see ir_schema.py) using an LLM call to the Groq API.

Design principles:
- This is the ONLY module that calls an LLM in the pipeline (codegen.py is
  pure/deterministic). Keep it that way -- it makes debugging much easier:
  if something's wrong with the model, check codegen; if something's wrong
  with what got extracted, check here.
- Extraction is treated as structured extraction, not open QA: we give the
  LLM the exact JSON schema to fill and validate the response against
  ModelIR via Pydantic. No free-form prose response is accepted.
- We NEVER let the LLM silently guess a hyperparameter it isn't sure about.
  The prompt explicitly instructs it to leave a field null / add to
  unsupported_elements rather than invent a plausible-looking number --
  this matters because a hallucinated learning rate would corrupt the
  eventual reproducibility verdict.
- One retry on invalid JSON / schema validation failure, with the
  validation error fed back to the model, before giving up.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from dotenv import load_dotenv
from groq import Groq
from pydantic import ValidationError

from ir_schema import ModelIR

# Load GROQ_API_KEY (and any other vars) from a .env file in the project
# root, if present. Safe to call even if no .env exists or the vars are
# already set some other way (e.g. exported in the shell) -- load_dotenv()
# won't override existing environment variables by default.
load_dotenv()


EXTRACTION_SYSTEM_PROMPT = """You are a precise information-extraction system. You read a machine learning \
paper's methodology section and extract the described model architecture and \
training setup into a strict JSON schema.

CRITICAL RULES:
1. Only extract what is EXPLICITLY stated in the text. Never infer or guess a \
value that is not written.
2. If a field is not stated in the text, set it to null (or, for a layer \
"params" dict, simply omit that key). Do NOT fill in a "typical" or \
"reasonable" default value. A missing learning rate must stay null. A missing \
in_features must be omitted, not guessed -- see rule 7 for the one exception.
3. If the text describes an architectural element you cannot map to the \
provided layer types (conv2d, linear, batchnorm2d, layernorm, activation, \
pooling, dropout, flatten, residual_block, attention, embedding), do NOT force \
it into one of these types. Instead add a short description of it to the \
unsupported_elements list and omit it from layers.
4. For every layer you DO extract, copy the exact sentence or phrase you took \
it from into source_span, so the extraction can be audited.
5. Respond with ONLY the JSON object. No markdown fences, no preamble, no \
explanation outside the JSON.
6. layers must be a flat, ordered list representing the forward pass through \
ONE single model. If the text describes MULTIPLE model variants (e.g. "Model \
A uses X neurons... Model B uses Y neurons..."), extract ONLY the first \
variant that is fully and unambiguously described, and add a note to \
unsupported_elements naming the other variants you skipped (e.g. "additional \
model variant 'NN-Recall' described in text but not extracted -- run \
separately"). NEVER merge layers from different named model variants into one \
layers list.
7. in_features (for linear) and in_channels (for conv2d) may be left out of \
params if not explicitly stated AND not unambiguously computable from an \
adjacent stated layer's output size in the SAME model variant. Do not guess \
these from unrelated context (e.g. a dataset's total feature count mentioned \
elsewhere) unless the text directly connects that number to this layer as its \
input.

PARAMS KEY REFERENCE -- use these exact key names, no others, per layer type:
- linear: in_features (int, omit if unknown), out_features (int, required), \
bias (bool, optional)
- conv2d: in_channels (int, omit if unknown), out_channels (int, required), \
kernel_size (int, required), stride (int, optional), padding (int, optional)
- batchnorm2d: num_features (int, required)
- layernorm: normalized_shape (int, required)
- activation: fn (string, required -- one of: relu, gelu, tanh, sigmoid, \
softmax, leaky_relu; map selu/elu to the closest of these and note the \
substitution in source_span, e.g. "selu approximated as gelu")
- pooling: kind (string, one of: max, avg, adaptive_avg), kernel_size (int), \
stride (int, optional), output_size (int, only for adaptive_avg)
- dropout: p (float, required)
- flatten: start_dim (int, optional, default 1)
- embedding: num_embeddings (int, required), embedding_dim (int, required)
- residual_block: sub_layers (list of layer objects using this same key \
reference), skip_projection (a single layer object or null), \
activation_after_add (string, optional)
Do not invent params outside this reference. Do not use synonyms (e.g. never \
"function" for activation -- the key is exactly "fn").

You will be given the JSON schema to fill and the paper text. Extract only \
from the text provided -- do not use outside knowledge of the paper or model \
family."""


def _build_user_prompt(paper_text: str, schema_json: dict) -> str:
    return f"""JSON SCHEMA TO FILL (Pydantic model dumped as JSON schema):
{json.dumps(schema_json, indent=2)}

PAPER METHODOLOGY TEXT:
\"\"\"
{paper_text}
\"\"\"

Extract the model architecture, training setup, and any reported results \
described in this text into an object matching the schema above. Follow all \
rules in the system prompt, especially: never invent a value that isn't \
stated, and route anything you can't map to the layer types into \
unsupported_elements."""


class ExtractionError(Exception):
    """Raised when extraction fails after retry (invalid JSON or schema mismatch)."""


def extract_ir(
    paper_text: str,
    paper_title: Optional[str] = None,
    model: str = "openai/gpt-oss-120b",
    client: Optional[Groq] = None,
) -> ModelIR:
    """
    Extract a ModelIR from raw paper methodology text.

    Args:
        paper_text: the methodology section text (or relevant excerpt).
        paper_title: optional, seeds ModelIR.paper_title if the model doesn't
            extract one itself.
        model: Groq model name.
        client: optional pre-built Groq client (mainly for testing); if not
            given, one is constructed from GROQ_API_KEY in the environment.

    Returns:
        A validated ModelIR instance.

    Raises:
        ExtractionError: if the LLM's output can't be parsed/validated even
            after one retry with the validation error fed back to it.
    """
    if client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ExtractionError("GROQ_API_KEY not set in environment.")
        client = Groq(api_key=api_key)

    schema_json = ModelIR.model_json_schema()
    user_prompt = _build_user_prompt(paper_text, schema_json)

    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    ir = _call_and_validate(client, model, messages)
    if ir is not None:
        if paper_title and not ir.paper_title:
            ir.paper_title = paper_title
        return ir

    # Retry once, feeding back what went wrong.
    raise ExtractionError(
        "Failed to extract a valid ModelIR after retry. "
        "Inspect the raw model output for details."
    )


def _call_and_validate(client: Groq, model: str, messages: list[dict], attempt: int = 1) -> Optional[ModelIR]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        if attempt >= 2:
            return None
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"Your response was not valid JSON: {e}. "
                                          "Respond again with ONLY the corrected JSON object."},
        ]
        return _call_and_validate(client, model, messages, attempt=attempt + 1)

    try:
        return ModelIR(**data)
    except ValidationError as e:
        if attempt >= 2:
            return None
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"Your JSON did not match the schema: {e}. "
                                          "Respond again with ONLY the corrected JSON object "
                                          "that matches the schema exactly."},
        ]
        return _call_and_validate(client, model, messages, attempt=attempt + 1)
