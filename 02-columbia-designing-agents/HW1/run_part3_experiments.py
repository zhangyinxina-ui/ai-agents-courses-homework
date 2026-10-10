"""
HW1 Part 2 & Part 3 Comprehensive Experiments Runner

Runs:
1. Part 2 BPE vocabulary size experiments (vocab_size = 300, 500, 2000) on both BPE_CORPUS
   and the full cleaned restaurant review corpus, measuring actual vocabulary size,
   tokens_per_word, bytes_per_token, and round_trips across all 6 test sentences.
2. Part 3 Transformer Classifier experiments:
   - Default configuration (vocab_size = 2000, d_model = 64, epochs = 8)
   - Extended training (epochs = 20) to observe training vs. test accuracy / overfitting
   - Hidden state size comparison: d_model = 16 vs. d_model = 64 vs. d_model = 128
   - Vocabulary size comparison in classifier: vocab_size = 300, 500, 2000
3. Saves all experiment results to HW1/part3_experiment_results.json and updates
   HW1/part3_classifier.ipynb with the completed code and executed cell outputs.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from part1_data_processing import DEFAULT_THRESHOLDS, build_pipeline, load_corpus
from part2_tokenizing import (
    BPE_CORPUS,
    SENTENCES,
    _Trained,
    bytes_per_token,
    round_trips,
    tokens_per_word,
    train_bpe,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 2000
MAX_LEN = 256


def load_reviews() -> list[dict]:
    """
    In:  nothing
    Out: the cleaned records from your Part 1 pipeline
    """
    return build_pipeline(DEFAULT_THRESHOLDS).run(load_corpus())


def load_tokenizer(texts: list[str], vocab_size: int = VOCAB_SIZE):
    """
    In:  the cleaned review texts and a vocabulary size
    Out: a BPE tokenizer trained on them, using your Part 2 train_bpe
    """
    return train_bpe(texts, vocab_size)


def make_splits(records: list[dict], test_frac: float = 0.2) -> tuple[list, list]:
    """
    In:  the cleaned records, and the fraction to hold out for testing
    Out: (train, test), each a list of (text, label) pairs, where label is
         0 for negative and 1 for positive

    Drop the 3-star reviews. Balance the two classes, and shuffle before
    splitting so the split does not follow the order of the corpus.
    """
    neg = [(r["text"], 0) for r in records if r["label"] in (1, 2)]
    pos = [(r["text"], 1) for r in records if r["label"] in (4, 5)]
    random.shuffle(neg)
    random.shuffle(pos)

    n = min(len(neg), len(pos))
    neg = neg[:n]
    pos = pos[:n]

    n_test = int(n * test_frac)
    test = neg[:n_test] + pos[:n_test]
    train = neg[n_test:] + pos[n_test:]
    random.shuffle(train)
    random.shuffle(test)
    return train, test


def encode_batch(tokenizer, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenizes and pads a batch.
    """
    rows = [tokenizer.encode(t).ids[:MAX_LEN] for t in texts]
    width = max(len(r) for r in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[i, : len(row)] = 1
    return ids.to(DEVICE), mask.to(DEVICE)


class ReviewClassifier(nn.Module):
    """
    A token embedding, a position embedding, one self-attention block, a mean
    over the real tokens, and a linear layer to two classes.
    """

    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(MAX_LEN, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 2)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        In:  ids and mask, both (batch, length)
        Out: logits of shape (batch, 2)
        """
        _, seq_len = ids.shape
        pos = torch.arange(seq_len, device=ids.device).unsqueeze(0)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        key_padding_mask = mask == 0
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        mask_f = mask.unsqueeze(-1).float()
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return self.fc(pooled)


def train(
    model,
    train_split: list,
    tokenizer,
    epochs: int = 8,
    batch_size: int = 16,
    lr: float = 1e-3,
    epoch_losses: list[float] | None = None,
):
    """
    In:  a model, the training split, a tokenizer, and the usual knobs
    Out: the trained model

    Print the average loss each epoch, so you can monitor training progress.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        data = list(train_split)
        random.shuffle(data)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(data), batch_size):
            batch = data[start : start + batch_size]
            texts = [t for t, _ in batch]
            labels = torch.tensor([y for _, y in batch], dtype=torch.long, device=DEVICE)
            ids, mask = encode_batch(tokenizer, texts)

            optimizer.zero_grad()
            logits = model(ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if epoch_losses is not None:
            epoch_losses.append(avg_loss)
        print(f"  epoch {epoch:>2}: loss = {avg_loss:.4f}")

    return model


def evaluate(model, test_split: list, tokenizer, batch_size: int = 32) -> dict:
    """
    In:  a trained model, the test split, and a tokenizer
    Out: a dict with keys "accuracy", "precision_negative", "recall_negative",
         "precision_positive", and "recall_positive"
    """
    model.eval()
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for start in range(0, len(test_split), batch_size):
            batch = test_split[start : start + batch_size]
            texts = [t for t, _ in batch]
            labels = [y for _, y in batch]
            ids, mask = encode_batch(tokenizer, texts)
            preds = model(ids, mask).argmax(dim=-1).cpu().tolist()
            for p, y in zip(preds, labels):
                if p == 1 and y == 1:
                    tp += 1
                elif p == 0 and y == 0:
                    tn += 1
                elif p == 1 and y == 0:
                    fp += 1
                else:
                    fn += 1

    total = max(tp + tn + fp + fn, 1)
    return {
        "accuracy": (tp + tn) / total,
        "precision_negative": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        "recall_negative": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "precision_positive": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall_positive": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
    }


def main() -> None:
    results_summary: dict = {}

    # 1. Load cleaned reviews
    records = load_reviews()
    review_texts = [r["text"] for r in records]
    print(f"=== Loaded {len(records)} clean reviews ===")

    # 2. Part 2 Question 2: BPE vocab sizes 300, 500, 2000 on BPE_CORPUS and on full reviews
    for corpus_name, corpus_texts in (("BPE_CORPUS", BPE_CORPUS), ("CLEAN_REVIEWS", review_texts)):
        print(f"\n=== BPE Vocab Size Experiment on {corpus_name} ({len(corpus_texts)} texts) ===")
        results_summary[f"bpe_{corpus_name}"] = {}
        for target_v in (300, 500, 2000):
            tok = train_bpe(corpus_texts, target_v)
            actual_v = tok.get_vocab_size()
            wrapped = _Trained(tok)
            tpw = {k: round(tokens_per_word(wrapped, s), 2) for k, s in SENTENCES.items()}
            bpt = {k: round(bytes_per_token(wrapped, s), 2) for k, s in SENTENCES.items()}
            rt = {k: round_trips(wrapped, s) for k, s in SENTENCES.items()}
            results_summary[f"bpe_{corpus_name}"][str(target_v)] = {
                "requested_vocab": target_v,
                "actual_vocab": actual_v,
                "tokens_per_word": tpw,
                "bytes_per_token": bpt,
                "round_trips": rt,
            }
            print(f"  requested={target_v:<5} actual={actual_v:<5} | tpw={tpw} | bpt={bpt}")

    # 3. Part 3 Baseline Classifier (vocab_size=2000, d_model=64, epochs=8)
    print("\n=== Part 3 Baseline Classifier (vocab_size=2000, d_model=64, epochs=8) ===")
    torch.manual_seed(1990)
    random.seed(1990)
    tokenizer = load_tokenizer(review_texts, VOCAB_SIZE)
    train_split, test_split = make_splits(records)
    print(f"{len(train_split)} train, {len(test_split)} test")

    baseline_losses: list[float] = []
    model = ReviewClassifier(tokenizer.get_vocab_size(), d_model=64).to(DEVICE)
    model = train(model, train_split, tokenizer, epochs=8, epoch_losses=baseline_losses)
    test_metrics = evaluate(model, test_split, tokenizer)
    train_metrics = evaluate(model, train_split, tokenizer)
    print("  Test metrics:", {k: round(v, 3) for k, v in test_metrics.items()})
    print("  Train metrics:", {k: round(v, 3) for k, v in train_metrics.items()})

    results_summary["part3_baseline"] = {
        "n_clean_reviews": len(records),
        "n_train": len(train_split),
        "n_test": len(test_split),
        "vocab_size": tokenizer.get_vocab_size(),
        "d_model": 64,
        "epochs": 8,
        "epoch_losses": [round(x, 4) for x in baseline_losses],
        "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
        "train_metrics": {k: round(v, 4) for k, v in train_metrics.items()},
    }

    # 4. Part 3 Question 4: Hidden state size comparison (d_model = 16, 64, 128)
    print("\n=== Part 3 Hidden Dimension Comparison (d_model = 16, 64, 128) ===")
    results_summary["part3_d_model_comparison"] = {}
    for d_model in (16, 64, 128):
        torch.manual_seed(1990)
        random.seed(1990)
        tok = load_tokenizer(review_texts, VOCAB_SIZE)
        tr_split, te_split = make_splits(records)
        losses: list[float] = []
        t0 = time.perf_counter()
        m = ReviewClassifier(tok.get_vocab_size(), d_model=d_model, n_heads=4).to(DEVICE)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"\n--- d_model={d_model} (params={n_params:,}) ---")
        m = train(m, tr_split, tok, epochs=8, epoch_losses=losses)
        elapsed = time.perf_counter() - t0
        te_eval = evaluate(m, te_split, tok)
        tr_eval = evaluate(m, tr_split, tok)
        print(f"  train_acc={tr_eval['accuracy']:.3f}, test_acc={te_eval['accuracy']:.3f}, time={elapsed:.2f}s")
        results_summary["part3_d_model_comparison"][str(d_model)] = {
            "d_model": d_model,
            "n_params": n_params,
            "train_time_sec": round(elapsed, 3),
            "epoch_losses": [round(x, 4) for x in losses],
            "train_metrics": {k: round(v, 4) for k, v in tr_eval.items()},
            "test_metrics": {k: round(v, 4) for k, v in te_eval.items()},
        }

    # 5. Also compare vocab_size = 300, 500, 2000 in Part 3 classifier
    print("\n=== Part 3 Vocab Size Comparison (vocab_size = 300, 500, 2000) ===")
    results_summary["part3_vocab_comparison"] = {}
    for v_size in (300, 500, 2000):
        torch.manual_seed(1990)
        random.seed(1990)
        tok = load_tokenizer(review_texts, v_size)
        tr_split, te_split = make_splits(records)
        losses = []
        m = ReviewClassifier(tok.get_vocab_size(), d_model=64, n_heads=4).to(DEVICE)
        print(f"\n--- vocab_size={v_size} (actual={tok.get_vocab_size()}) ---")
        m = train(m, tr_split, tok, epochs=8, epoch_losses=losses)
        te_eval = evaluate(m, te_split, tok)
        tr_eval = evaluate(m, tr_split, tok)
        results_summary["part3_vocab_comparison"][str(v_size)] = {
            "requested_vocab": v_size,
            "actual_vocab": tok.get_vocab_size(),
            "epoch_losses": [round(x, 4) for x in losses],
            "train_metrics": {k: round(v, 4) for k, v in tr_eval.items()},
            "test_metrics": {k: round(v, 4) for k, v in te_eval.items()},
        }

    out_path = HERE / "part3_experiment_results.json"
    out_path.write_text(json.dumps(results_summary, indent=2), encoding="utf-8")
    print(f"\nSaved experiment results to {out_path}")


if __name__ == "__main__":
    main()
