#!/usr/bin/env python3
"""
Chapter 10: Parameter-Efficient Fine-Tuning (LoRA) & Model Quantization (参数高效微调LoRA与模型量化)
SJTU Hands-on AI Agent (动手学大模型智能体) — Part 3: Agent Post-Training

Implements from scratch:
1. `LoRALinear`: Low-Rank Adaptation layer (Hu et al., 2021):
     h = x W_0^T + (alpha / r) * x A^T B^T
   including Kaiming/Gaussian initialization of A, zero initialization of B,
   backward gradients (dA, dB) while freezing W_0, and lossless weight merging `W_merged`.
2. Symmetric AbsMax INT8 Quantization and Group-Wise Asymmetric INT4 Quantization
   (`quantize_int8_absmax`, `quantize_int4_groupwise`) with dequantization error (SQNR / MSE).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple
import numpy as np


class LoRALinear:
    """
    Low-Rank Adaptation (LoRA) wrapper around a frozen pretrained weight matrix W_0.
    W_0 has shape (out_features, in_features).
    A has shape (r, in_features), initialized with N(0, sigma^2).
    B has shape (out_features, r), initialized to 0 so delta_W = 0 at step 0.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        lora_alpha: float = 8.0,
        seed: int = 42,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        rng = np.random.default_rng(seed)
        self.W_0 = rng.normal(0.0, 1.0 / math.sqrt(in_features), size=(out_features, in_features))
        self.A = rng.normal(0.0, 0.02, size=(r, in_features))
        self.B = np.zeros((out_features, r), dtype=np.float64)

    def parameter_counts(self) -> Dict[str, Any]:
        base_params = self.in_features * self.out_features
        lora_params = self.r * (self.in_features + self.out_features)
        return {
            "base_params": base_params,
            "trainable_lora_params": lora_params,
            "compression_ratio": round(base_params / lora_params, 2),
            "trainable_pct": round((lora_params / base_params) * 100.0, 2),
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: x shape (batch, in_features) -> (batch, out_features)."""
        base_out = x @ self.W_0.T
        lora_out = (x @ self.A.T) @ self.B.T * self.scaling
        return base_out + lora_out

    def backward_and_step(
        self, x: np.ndarray, grad_out: np.ndarray, lr: float = 0.05
    ) -> Dict[str, float]:
        """
        Computes exact analytical gradients with respect to trainable matrices A and B
        (keeping W_0 strictly frozen) and applies one SGD step.
        """
        # h_lora = scaling * (x @ A.T) @ B.T
        xa = x @ self.A.T  # (batch, r)
        grad_B = self.scaling * (grad_out.T @ xa)  # (out_features, r)
        grad_A = self.scaling * ((grad_out @ self.B).T @ x)  # (r, in_features)

        self.B -= lr * grad_B
        self.A -= lr * grad_A
        return {
            "grad_A_norm": float(np.linalg.norm(grad_A)),
            "grad_B_norm": float(np.linalg.norm(grad_B)),
        }

    def merged_weight(self) -> np.ndarray:
        """Merges LoRA adapter into base weights for zero-latency inference: W = W_0 + (alpha/r) B A."""
        return self.W_0 + self.scaling * (self.B @ self.A)


def quantize_int8_absmax(weight: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Symmetric INT8 AbsMax quantization into [-127, 127]:
      scale = max(|W|) / 127
      W_q = clip(round(W / scale), -127, 127)
      W_deq = W_q * scale
    """
    abs_max = float(np.max(np.abs(weight)))
    scale = abs_max / 127.0 if abs_max > 0 else 1.0
    q_int8 = np.clip(np.round(weight / scale), -127, 127).astype(np.int8)
    dequant = q_int8.astype(np.float64) * scale
    return q_int8, scale, dequant


def quantize_int4_groupwise(
    weight: np.ndarray, group_size: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Asymmetric unsigned INT4 [0, 15] group-wise quantization (GPTQ / AWQ style):
      For each group g of `group_size` weights:
        scale_g = (max(w_g) - min(w_g)) / 15
        zero_g  = round(-min(w_g) / scale_g)
        q_g     = clip(round(w_g / scale_g) + zero_g, 0, 15)
    """
    orig_shape = weight.shape
    flat = weight.reshape(-1)
    assert len(flat) % group_size == 0, "Weight count must be divisible by group_size"
    groups = flat.reshape(-1, group_size)

    w_min = np.min(groups, axis=1, keepdims=True)
    w_max = np.max(groups, axis=1, keepdims=True)
    scales = np.maximum((w_max - w_min) / 15.0, 1e-8)
    zeros = np.clip(np.round(-w_min / scales), 0, 15)

    q_int4 = np.clip(np.round(groups / scales) + zeros, 0, 15).astype(np.uint8)
    dequant = ((q_int4.astype(np.float64) - zeros) * scales).reshape(orig_shape)
    return q_int4, scales.squeeze(-1), zeros.squeeze(-1), dequant


def run_chapter_demo() -> Dict[str, Any]:
    lora = LoRALinear(in_features=64, out_features=64, r=4, lora_alpha=8.0, seed=42)
    rng = np.random.default_rng(10)
    x = rng.normal(0.0, 1.0, size=(8, 64))

    # At initialization (B=0), LoRA output must exactly equal frozen W_0 output
    init_diff = float(np.max(np.abs(lora.forward(x) - (x @ lora.W_0.T))))

    # Train 2 steps toward a target matrix
    target = rng.normal(0.0, 1.0, size=(8, 64))
    for _ in range(2):
        pred = lora.forward(x)
        grad_out = (pred - target) / x.shape[0]
        lora.backward_and_step(x, grad_out, lr=0.1)

    # Verify merged inference matches unmerged forward pass
    merged_out = x @ lora.merged_weight().T
    unmerged_out = lora.forward(x)
    merge_max_diff = float(np.max(np.abs(merged_out - unmerged_out)))

    # Quantize merged weights to INT8 and INT4
    W_m = lora.merged_weight()
    _, _, w_deq8 = quantize_int8_absmax(W_m)
    _, _, _, w_deq4 = quantize_int4_groupwise(W_m, group_size=8)

    mse_int8 = float(np.mean((W_m - w_deq8) ** 2))
    mse_int4 = float(np.mean((W_m - w_deq4) ** 2))

    return {
        "param_stats": lora.parameter_counts(),
        "init_zero_delta_diff": init_diff,
        "merge_equivalence_max_diff": merge_max_diff,
        "mse_int8": mse_int8,
        "mse_int4": mse_int4,
        "int8_more_accurate_than_int4": mse_int8 < mse_int4,
    }


if __name__ == "__main__":
    res = run_chapter_demo()
    print(f"[Chapter 10 Demo] LoRA Trainable={res['param_stats']['trainable_pct']}%, MergeDiff={res['merge_equivalence_max_diff']:.2e}, INT8_MSE={res['mse_int8']:.2e}, INT4_MSE={res['mse_int4']:.2e}")
