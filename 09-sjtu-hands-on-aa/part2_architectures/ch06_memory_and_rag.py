#!/usr/bin/env python3
"""
Chapter 6: Agent Memory Systems & Retrieval-Augmented Generation (智能体记忆与知识检索增强)
SJTU Hands-on AI Agent (动手学大模型智能体) — Part 2: Core Architectures

Implements:
1. Short-Term Sliding-Window & Token-Budget Working Memory (`WorkingMemoryBuffer`)
2. Episodic Memory with Generative Reflection Scoring (Park et al., Generative Agents:
   Recency + Importance + Relevance composite scoring)
3. Hybrid RAG Engine combining Dense Cosine Similarity + Sparse BM25 Lexical Scoring
   via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import numpy as np


@dataclass
class MemoryEntry:
    entry_id: str
    text: str
    timestamp_step: int
    importance: float  # [1.0, 10.0]
    embedding: np.ndarray


class WorkingMemoryBuffer:
    """Short-term conversation buffer with automatic eviction and episodic summarization."""

    def __init__(self, max_turns: int = 4) -> None:
        self.max_turns = max_turns
        self.buffer: List[str] = []
        self.RunningSummary: str = ""

    def add_turn(self, message: str) -> None:
        self.buffer.append(message)
        if len(self.buffer) > self.max_turns:
            evicted = self.buffer.pop(0)
            prefix = f"{self.RunningSummary} | " if self.RunningSummary else ""
            self.RunningSummary = f"{prefix}Summarized({evicted[:32]})"

    def get_context(self) -> Dict[str, Any]:
        return {
            "summary": self.RunningSummary,
            "active_turns": list(self.buffer),
            "active_count": len(self.buffer),
        }


class GenerativeEpisodicMemory:
    """
    Implements the Generative Agents (Park et al., 2023) memory retrieval score:
        S(m, q) = w_rec * Recency(m) + w_imp * Importance(m) + w_rel * CosineSim(e_m, e_q)
    """

    def __init__(self, decay_factor: float = 0.95) -> None:
        self.decay_factor = decay_factor
        self.entries: List[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0

    def retrieve(
        self,
        query_embedding: np.ndarray,
        current_step: int,
        top_k: int = 2,
        weights: Tuple[float, float, float] = (1.0, 1.0, 2.0),
    ) -> List[Tuple[MemoryEntry, float]]:
        w_rec, w_imp, w_rel = weights
        scored: List[Tuple[MemoryEntry, float]] = []
        for m in self.entries:
            dt = max(0, current_step - m.timestamp_step)
            recency = self.decay_factor ** dt
            importance = m.importance / 10.0
            relevance = (self.cosine_similarity(m.embedding, query_embedding) + 1.0) / 2.0
            total_score = w_rec * recency + w_imp * importance + w_rel * relevance
            scored.append((m, round(total_score, 4)))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


class HybridRAGEngine:
    """
    Combines Sparse BM25 lexical matching and Dense Vector similarity using
    Reciprocal Rank Fusion (RRF):
        RRF(d) = 1 / (k + rank_dense(d)) + 1 / (k + rank_sparse(d))
    """

    def __init__(self, vocab: List[str], rrf_k: int = 60) -> None:
        self.vocab = [w.lower() for w in vocab]
        self.rrf_k = rrf_k
        self.docs: List[Dict[str, Any]] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())

    def embed_bag_of_concepts(self, text: str) -> np.ndarray:
        tokens = self.tokenize(text)
        vec = np.zeros(len(self.vocab), dtype=np.float64)
        for i, term in enumerate(self.vocab):
            vec[i] = sum(1.0 for t in tokens if term in t)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def index_document(self, doc_id: str, text: str) -> None:
        self.docs.append(
            {
                "doc_id": doc_id,
                "text": text,
                "tokens": self.tokenize(text),
                "embedding": self.embed_bag_of_concepts(text),
            }
        )

    def _bm25_scores(self, query_tokens: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
        n_docs = len(self.docs)
        if n_docs == 0:
            return []
        avgdl = sum(len(d["tokens"]) for d in self.docs) / n_docs
        scores = [0.0] * n_docs

        for q_term in set(query_tokens):
            df = sum(1 for d in self.docs if q_term in d["tokens"])
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for idx, d in enumerate(self.docs):
                tf = d["tokens"].count(q_term)
                dl = len(d["tokens"])
                denom = tf + k1 * (1.0 - b + b * (dl / max(1.0, avgdl)))
                scores[idx] += idf * ((tf * (k1 + 1.0)) / max(1e-9, denom))
        return scores

    def search_hybrid(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        q_tokens = self.tokenize(query)
        q_emb = self.embed_bag_of_concepts(query)

        bm25 = self._bm25_scores(q_tokens)
        dense = [
            GenerativeEpisodicMemory.cosine_similarity(q_emb, d["embedding"])
            for d in self.docs
        ]

        sparse_order = np.argsort(bm25)[::-1]
        dense_order = np.argsort(dense)[::-1]

        sparse_ranks = {int(doc_i): rank + 1 for rank, doc_i in enumerate(sparse_order)}
        dense_ranks = {int(doc_i): rank + 1 for rank, doc_i in enumerate(dense_order)}

        results: List[Dict[str, Any]] = []
        for idx, d in enumerate(self.docs):
            rrf_score = (1.0 / (self.rrf_k + sparse_ranks[idx])) + (
                1.0 / (self.rrf_k + dense_ranks[idx])
            )
            results.append(
                {
                    "doc_id": d["doc_id"],
                    "text": d["text"],
                    "bm25_score": round(bm25[idx], 4),
                    "dense_score": round(dense[idx], 4),
                    "rrf_score": round(rrf_score, 6),
                }
            )

        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return results[:top_k]


def run_chapter_demo() -> Dict[str, Any]:
    # 1. Working memory demo
    wm = WorkingMemoryBuffer(max_turns=3)
    for i in range(1, 6):
        wm.add_turn(f"Turn {i}: User discussed sub-task #{i}")
    wm_ctx = wm.get_context()

    # 2. Generative episodic memory demo
    vocab = ["cuda", "kernel", "verilog", "rtl", "kv_cache", "radix", "memory"]
    rag = HybridRAGEngine(vocab=vocab)
    rag.index_document("D1", "SGLang uses RadixAttention KV_cache prefix trees for multi-turn memory reuse.")
    rag.index_document("D2", "KernelBench and Kevin optimize CUDA kernel warp tiling on H100 GPUs.")
    rag.index_document("D3", "VerilogEval and RTLCoder synthesize Verilog RTL hardware modules.")

    ep_mem = GenerativeEpisodicMemory(decay_factor=0.9)
    for idx, doc in enumerate(rag.docs):
        ep_mem.add(
            MemoryEntry(
                entry_id=doc["doc_id"],
                text=doc["text"],
                timestamp_step=idx + 1,
                importance=9.0 if doc["doc_id"] == "D1" else 5.0,
                embedding=doc["embedding"],
            )
        )

    q_emb = rag.embed_bag_of_concepts("How does RadixAttention reuse kv_cache memory?")
    ep_top = ep_mem.retrieve(q_emb, current_step=5, top_k=2)
    rag_top = rag.search_hybrid("RadixAttention kv_cache memory reuse", top_k=2)

    return {
        "working_memory": wm_ctx,
        "episodic_top_id": ep_top[0][0].entry_id,
        "episodic_top_score": ep_top[0][1],
        "hybrid_rag_top_id": rag_top[0]["doc_id"],
        "hybrid_rag_results": rag_top,
    }


if __name__ == "__main__":
    res = run_chapter_demo()
    print(f"[Chapter 06 Demo] WorkingMemory Active={res['working_memory']['active_count']}, EpisodicTop={res['episodic_top_id']}, HybridRAGTop={res['hybrid_rag_top_id']}")
