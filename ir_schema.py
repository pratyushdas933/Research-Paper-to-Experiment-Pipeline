"""
ir_schema.py

The IR (intermediate representation) contract between:
  - the extraction agent (paper text -> ModelIR)
  - codegen.py (ModelIR -> nn.Module)
  - the training harness (ModelIR.training -> training loop config)
  - the diagnostic agent (ModelIR.reported_results vs actual results)

schema_version is bumped whenever fields change shape, so downstream code
can branch on it if needed.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class LayerType(str, Enum):
    conv2d = "conv2d"
    linear = "linear"
    batchnorm2d = "batchnorm2d"
    layernorm = "layernorm"
    activation = "activation"
    pooling = "pooling"
    dropout = "dropout"
    flatten = "flatten"
    residual_block = "residual_block"   # composite
    attention = "attention"             # composite, placeholder for later
    embedding = "embedding"


class ActivationFn(str, Enum):
    relu = "relu"
    gelu = "gelu"
    tanh = "tanh"
    sigmoid = "sigmoid"
    softmax = "softmax"
    leaky_relu = "leaky_relu"


class LayerSpec(BaseModel):
    id: str
    type: LayerType
    params: dict = Field(default_factory=dict)
    source_confidence: float = 1.0
    source_span: Optional[str] = None


class TrainingSpec(BaseModel):
    optimizer: Optional[str] = None
    learning_rate: Optional[float] = None
    lr_schedule: Optional[str] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    loss_fn: Optional[str] = None
    weight_decay: Optional[float] = None
    dataset: Optional[str] = None
    train_val_split: Optional[str] = None


class ReportedResult(BaseModel):
    metric_name: str
    metric_value: float
    dataset_split: str


class ModelIR(BaseModel):
    schema_version: str = "v1"
    paper_title: Optional[str] = None
    layers: list[LayerSpec]
    training: TrainingSpec
    reported_results: list[ReportedResult] = Field(default_factory=list)
    unsupported_elements: list[str] = Field(default_factory=list)
