#!/usr/bin/env python3
"""
Chapter 2: Generative Language Models & Decoding Mechanics (生成式语言模型基础)
SJTU Hands-on AI Agent (动手学大模型智能体) — Part 1: Foundations

Implements from scratch (NumPy):
1. Scaled Dot-Product Causal Self-Attention with explicit Key-Value (KV) Cache
2. Incremental Autoregressive Decoding (`prefill` vs `decode_step`) proving exact mathematical
   equivalence between cached and uncached forward passes
3. Decoding & Sampling Strategies:
   - Greedy Decoding
   - Temperature Scaling
   - Top-k Filtering
   - Top-p (Nucleus) Sampling
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=axis, keepdims=True)


@dataclass
class KVCache:
    """Stores past Key and Value tensors across autoregressive decoding steps."""
    k: Optional[np.ndarray] = None  # shape: (seq_len, d_k)
    v: Optional[np.ndarray] = None  # shape: (seq_len, d_v)

    def update(self, new_k: np.ndarray, new_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.k is None or self.v is None:
            self.k = new_k.copy()
            self.v = new_v.copy()
        else:
            self.k = np.concatenate([self.k, new_k], axis=0)
            self.v = np.concatenate([self.v, new_v], axis=0)
        return self.k, self.v

    @property
    def seq_len(self) -> int:
        return 0 if self.k is None else int(self.k.shape[0])


class CausalSelfAttentionBlock:
    """Single-head Causal Transformer Layer with KV-Cache support."""

    def __init__(self, vocab_size: int = 32, d_model: int = 16, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        scale = 1.0 / math.sqrt(d_model)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tok_emb = rng.normal(0.0, scale, size=(vocab_size, d_model))
        self.pos_emb = rng.normal(0.0, scale, size=(128, d_model))
        self.W_q = rng.normal(0.0, scale, size=(d_model, d_model))
        self.W_k = rng.normal(0.0, scale, size=(d_model, d_model))
        self.W_v = rng.normal(0.0, scale, size=(d_model, d_model))
        self.W_o = rng.normal(0.0, scale, size=(d_model, d_model))
        self.lm_head = rng.normal(0.0, scale, size=(d_model, vocab_size))

    def forward_full(self, token_ids: List[int]) -> np.ndarray:
        """
        Uncached full-sequence forward pass with causal lower-triangular mask.
        Returns logits of shape (seq_len, vocab_size).
        """
        seq_len = len(token_ids)
        positions = np.arange(seq_len)
        x = self.tok_emb[token_ids] + self.pos_emb[positions]  # (T, D)

        q = x @ self.W_q
        k = x @ self.W_k
        v = x @ self.W_v

        scores = (q @ k.T) / math.sqrt(self.d_model)  # (T, T)
        causal_mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
        attn_weights = softmax(scores + causal_mask, axis=-1)
        context = (attn_weights @ v) @ self.W_o
        hidden = x + context
        logits = hidden @ self.lm_head
        return logits

    def forward_with_kv_cache(
        self, new_token_ids: List[int], kv_cache: KVCache
    ) -> np.ndarray:
        """
        Incremental forward pass using KVCache.
        During prefill, `new_token_ids` has length T_prompt; during decode, length 1.
        Returns logits for the new positions: (len(new_token_ids), vocab_size).
        """
        start_pos = kv_cache.seq_len
        n_new = len(new_token_ids)
        positions = np.arange(start_pos, start_pos + n_new)
        x_new = self.tok_emb[new_token_ids] + self.pos_emb[positions]

        q_new = x_new @ self.W_q
        k_new = x_new @ self.W_k
        v_new = x_new @ self.W_v

        k_all, v_all = kv_cache.update(k_new, v_new)
        total_len = k_all.shape[0]

        scores = (q_new @ k_all.T) / math.sqrt(self.d_model)  # (n_new, total_len)
        if n_new > 1:
            # Causal mask for prefill chunk
            mask = np.zeros((n_new, total_len))
            for i in range(n_new):
                query_abs_pos = start_pos + i
                if query_abs_pos + 1 < total_len:
                    mask[i, query_abs_pos + 1 :] = -1e9
            scores = scores + mask

        attn_weights = softmax(scores, axis=-1)
        context = (attn_weights @ v_all) @ self.W_o
        hidden = x_new + context
        return hidden @ self.lm_head


def sample_next_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, np.ndarray]:
    """
    Applies temperature scaling, top-k filtering, and top-p (nucleus) filtering
    to a 1D logit vector and returns (sampled_token_id, filtered_probs).
    """
    if temperature <= 0.0:
        best_id = int(np.argmax(logits))
        probs = np.zeros_like(logits, dtype=np.float64)
        probs[best_id] = 1.0
        return best_id, probs

    scaled = logits.astype(np.float64) / temperature

    # Top-k filtering
    if top_k is not None and 0 < top_k < len(scaled):
        top_k_indices = np.argpartition(scaled, -top_k)[-top_k:]
        mask = np.full_like(scaled, -1e9)
        mask[top_k_indices] = scaled[top_k_indices]
        scaled = mask

    probs = softmax(scaled, axis=-1)

    # Top-p (nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        # Keep tokens where cumulative probability before adding the token is < top_p
        cutoff_mask = (cumulative_probs - sorted_probs) >= top_p
        sorted_probs[cutoff_mask] = 0.0
        sorted_probs /= np.sum(sorted_probs)
        probs = np.zeros_like(probs)
        probs[sorted_indices] = sorted_probs

    if rng is None:
        rng = np.random.default_rng(0)
    token_id = int(rng.choice(len(probs), p=probs))
    return token_id, probs


def run_chapter_demo() -> Dict[str, any]:
    model = CausalSelfAttentionBlock(vocab_size=32, d_model=16, seed=42)
    prompt_tokens = [1, 5, 9, 12]

    # Verify KV-cache equivalence during 4-step greedy generation
    cache = KVCache()
    generated_cached = list(prompt_tokens)
    prefill_logits = model.forward_with_kv_cache(prompt_tokens, cache)
    next_tok, _ = sample_next_token(prefill_logits[-1], temperature=0.0)
    generated_cached.append(next_tok)

    max_logit_diff = 0.0
    for _ in range(3):
        step_logits = model.forward_with_kv_cache([generated_cached[-1]], cache)
        full_logits = model.forward_full(generated_cached)
        diff = float(np.max(np.abs(step_logits[-1] - full_logits[-1])))
        max_logit_diff = max(max_logit_diff, diff)
        next_tok, _ = sample_next_token(step_logits[-1], temperature=0.0)
        generated_cached.append(next_tok)

    # Test top-k and top-p probability concentration
    raw_logits = full_logits[-1]
    _, probs_k3 = sample_next_token(raw_logits, temperature=0.8, top_k=3)
    _, probs_p07 = sample_next_token(raw_logits, temperature=0.8, top_p=0.7)

    return {
        "generated_sequence": generated_cached,
        "kv_cache_final_len": cache.seq_len,
        "max_cached_vs_full_logit_diff": max_logit_diff,
        "top_k3_nonzero_count": int(np.sum(probs_k3 > 1e-6)),
        "top_p07_prob_sum": float(np.sum(probs_p07)),
    }


if __name__ == "__main__":
    demo = run_chapter_demo()
    print(f"[Chapter 02 Demo] Generated={demo['generated_sequence']}, Max Logit Diff={demo['max_cached_vs_full_logit_diff']:.2e}")
