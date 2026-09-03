"""
codegen.py

Deterministic translation from ModelIR (see ir_schema.py) into a working
PyTorch nn.Module.

Design principle: the LLM only ever fills in the IR (a JSON-like spec).
This module contains ZERO LLM calls — it's pure, deterministic code that
turns that spec into an actual nn.Module. That's what makes results
reproducible and debuggable: given the same IR, you always get the same
model.

Supported in v1:
- Simple sequential stacks (conv2d, linear, batchnorm2d, layernorm,
  activation, pooling, dropout, flatten, embedding)
- residual_block as a composite that wraps a sub-list of layers and adds
  a skip connection from a named source layer

Anything not covered should end up in ir.unsupported_elements upstream —
this module will raise a clear error rather than silently guessing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ir_schema import ModelIR, LayerSpec, LayerType, ActivationFn


class UnsupportedLayerError(Exception):
    """Raised when the IR asks for something codegen can't build yet."""


# ---------------------------------------------------------------------------
# Per-layer-type builders
# ---------------------------------------------------------------------------

def _build_activation(params: dict) -> nn.Module:
    fn = params.get("fn", "relu")
    fn = ActivationFn(fn) if not isinstance(fn, ActivationFn) else fn
    return {
        ActivationFn.relu: nn.ReLU,
        ActivationFn.gelu: nn.GELU,
        ActivationFn.tanh: nn.Tanh,
        ActivationFn.sigmoid: nn.Sigmoid,
        ActivationFn.softmax: lambda: nn.Softmax(dim=params.get("dim", -1)),
        ActivationFn.leaky_relu: lambda: nn.LeakyReLU(params.get("negative_slope", 0.01)),
    }[fn]()


def _build_conv2d(params: dict) -> nn.Module:
    return nn.Conv2d(
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        kernel_size=params.get("kernel_size", 3),
        stride=params.get("stride", 1),
        padding=params.get("padding", 0),
    )


def _build_linear(params: dict) -> nn.Module:
    return nn.Linear(
        in_features=params["in_features"],
        out_features=params["out_features"],
        bias=params.get("bias", True),
    )


def _build_batchnorm2d(params: dict) -> nn.Module:
    return nn.BatchNorm2d(num_features=params["num_features"])


def _build_layernorm(params: dict) -> nn.Module:
    return nn.LayerNorm(normalized_shape=params["normalized_shape"])


def _build_pooling(params: dict) -> nn.Module:
    kind = params.get("kind", "max")
    kernel_size = params.get("kernel_size", 2)
    stride = params.get("stride", kernel_size)
    if kind == "max":
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    elif kind == "avg":
        return nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
    elif kind == "adaptive_avg":
        return nn.AdaptiveAvgPool2d(output_size=params.get("output_size", 1))
    raise UnsupportedLayerError(f"Unknown pooling kind: {kind}")


def _build_dropout(params: dict) -> nn.Module:
    return nn.Dropout(p=params.get("p", 0.5))


def _build_flatten(params: dict) -> nn.Module:
    return nn.Flatten(start_dim=params.get("start_dim", 1))


def _build_embedding(params: dict) -> nn.Module:
    return nn.Embedding(
        num_embeddings=params["num_embeddings"],
        embedding_dim=params["embedding_dim"],
    )


_SIMPLE_BUILDERS = {
    LayerType.conv2d: _build_conv2d,
    LayerType.linear: _build_linear,
    LayerType.batchnorm2d: _build_batchnorm2d,
    LayerType.layernorm: _build_layernorm,
    LayerType.activation: _build_activation,
    LayerType.pooling: _build_pooling,
    LayerType.dropout: _build_dropout,
    LayerType.flatten: _build_flatten,
    LayerType.embedding: _build_embedding,
}


# ---------------------------------------------------------------------------
# Residual block support
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Wraps a sequence of sub-layers and adds a skip connection.

    IR contract for a residual_block LayerSpec:
        params = {
            "sub_layers": [LayerSpec, LayerSpec, ...],   # the main path
            "skip_projection": Optional[LayerSpec],       # e.g. 1x1 conv if
                                                            # channel/stride
                                                            # mismatch
            "activation_after_add": "relu"  (optional)
        }
    """

    def __init__(self, sub_layers: list[LayerSpec], skip_projection: LayerSpec | None,
                 activation_after_add: str | None):
        super().__init__()
        self.main_path = nn.Sequential(*[build_layer(spec) for spec in sub_layers])
        self.projection = build_layer(skip_projection) if skip_projection else nn.Identity()
        self.post_activation = (
            _build_activation({"fn": activation_after_add}) if activation_after_add else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.projection(x)
        out = self.main_path(x)
        out = out + identity
        return self.post_activation(out)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def build_layer(spec: LayerSpec) -> nn.Module:
    """Build a single nn.Module from one LayerSpec (recursing for composites)."""
    if spec.type == LayerType.residual_block:
        sub_layers_raw = spec.params.get("sub_layers", [])
        sub_layers = [
            s if isinstance(s, LayerSpec) else LayerSpec(**s) for s in sub_layers_raw
        ]
        skip_raw = spec.params.get("skip_projection")
        skip = (skip_raw if isinstance(skip_raw, LayerSpec) or skip_raw is None
                else LayerSpec(**skip_raw))
        return ResidualBlock(
            sub_layers=sub_layers,
            skip_projection=skip,
            activation_after_add=spec.params.get("activation_after_add"),
        )

    if spec.type == LayerType.attention:
        raise UnsupportedLayerError(
            f"Layer '{spec.id}': attention codegen not implemented in v1. "
            "This should have been routed to unsupported_elements upstream."
        )

    builder = _SIMPLE_BUILDERS.get(spec.type)
    if builder is None:
        raise UnsupportedLayerError(f"Layer '{spec.id}': no codegen builder for type {spec.type}")

    try:
        return builder(spec.params)
    except KeyError as e:
        raise UnsupportedLayerError(
            f"Layer '{spec.id}' (type={spec.type}): missing required param {e}"
        ) from e


class GeneratedModel(nn.Module):
    """The final model built from a ModelIR — a thin sequential wrapper."""

    def __init__(self, ir: ModelIR):
        super().__init__()
        self.ir = ir  # keep a reference for provenance/debugging
        self.net = nn.Sequential(*[build_layer(spec) for spec in ir.layers])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(ir: ModelIR) -> GeneratedModel:
    """
    Public entry point: ModelIR -> instantiated, ready-to-train nn.Module.

    Raises UnsupportedLayerError with a clear message if the IR references
    something codegen can't build — this is intentional; better to fail
    loudly than silently produce a wrong architecture.
    """
    if ir.unsupported_elements:
        raise UnsupportedLayerError(
            f"IR has {len(ir.unsupported_elements)} unsupported element(s) "
            f"flagged during extraction: {ir.unsupported_elements}. "
            "Resolve these before codegen."
        )
    return GeneratedModel(ir)
