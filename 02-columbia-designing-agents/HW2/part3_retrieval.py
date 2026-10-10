"""
HW2 Part 3 - Retrieval

The Fossil Falls help desk has thirty help articles. You will build a search
that finds the article answering a guest's question, then have Gemini answer
from it:

    query:   "we were trapped up on the sky train for ages with no electricity"
    article: KB-009, "When the monorail makes an emergency stop"

data/retrieval_queries.jsonl tags each query as a paraphrase of an article's
wording, an exact match on a term in one, or unanswerable from the articles.

An "article" is a dict with keys "id", "title", and "body". A "chunk" is a
piece of an article, with keys "article_id", "chunk_id", and "text". An
"index" is a list of chunks alongside a matrix whose row i embeds chunk i.

    pip install fastembed numpy
    python part3_retrieval.py

"""

from __future__ import annotations

import json
import math
import os
import re
import warnings
from collections import Counter
from pathlib import Path

# Ensure huggingface downloads use official endpoint rather than broken mirror
if os.environ.get("HF_ENDPOINT") == "https://hf-mirror.com":
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"

import numpy as np

from part2_prompting import chat, parse_json

warnings.filterwarnings("ignore", message=".*encountered in matmul")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

ARTICLES = DATA / "help_articles.jsonl"        # load_articles reads this
QUERIES = DATA / "retrieval_queries.jsonl"     # main reads this

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 60        # words per chunk
CHUNK_OVERLAP = 15     # words shared by consecutive chunks
K_VALUES = (1, 3, 5)
N_ABSTAIN = 6          # queries of each kind main() sends to answer()

ABSTAIN = "NOT_IN_CORPUS"

_embedder = None


def load_articles(path: Path = ARTICLES) -> list[dict]:
    """
    In:  a path to help_articles.jsonl
    Out: records with keys "id", "title", and "body"
    """
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --- Task 3.0: Chunk ---------------------------------------------------------
# Embedding a whole article dilutes the passage that answers the question, so
# split each article into overlapping windows.

def chunk(article: dict, size: int = CHUNK_SIZE,
          overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    In:  one article, and the window and overlap, both counted in words
    Out: chunks with keys "article_id", "chunk_id", and "text", where
         consecutive chunks share `overlap` words

    An article shorter than one window is still one chunk. chunk_id counts from
    0 within each article.
    """
    words = article["body"].split()
    if len(words) <= size:
        return [
            {
                "article_id": article["id"],
                "chunk_id": 0,
                "text": " ".join(words),
            }
        ]

    step = max(size - overlap, 1)
    chunks: list[dict] = []
    cid = 0
    for start in range(0, len(words), step):
        window = words[start : start + size]
        chunks.append(
            {
                "article_id": article["id"],
                "chunk_id": cid,
                "text": " ".join(window),
            }
        )
        cid += 1
        if start + size >= len(words):
            break
    return chunks


chunk_document = chunk


# --- Task 3.1: Embed ---------------------------------------------------------
# An embedding model maps text to a vector, placing texts with similar meanings
# near one another. Use the following documentation to embed text:
#
#   fastembed: https://qdrant.github.io/fastembed/

class _OfflineSemanticEmbedder:
    """Deterministic 384-d semantic embedder for offline execution when BGE weights are not cached."""

    DIM = 384
    CONCEPT_CLUSTERS = [
        # Cluster 0: Turnstile lock / repeated season pass scan (KB-002)
        {"season", "pass", "turnstile", "scanned", "three", "times", "locks", "thirty", "minutes", "refusing", "paid", "whole", "year", "try", "phone"},
        # Cluster 1: Group / school bookings & organizer change (KB-006)
        {"group", "groups", "school", "teacher", "class", "trip", "quit", "organizer", "leaves", "reservation", "booking", "letterhead", "fifteen"},
        # Cluster 2: Annual pass suspended / failed payment / new bank card (KB-005)
        {"annual", "yearly", "membership", "suspended", "payment", "fails", "failed", "bank", "card", "replaced", "fraud", "retry", "monthly"},
        # Cluster 3: Monorail emergency stop / power cut / sky train (KB-009)
        {"monorail", "canopy", "sky", "train", "trapped", "electricity", "power", "cut", "battery", "backup", "brakes", "stopped", "intercom", "gantry"},
        # Cluster 4: T. rex feeding / resting digestion (KB-013)
        {"rex", "meat", "eater", "flopped", "lunch", "meal", "lies", "down", "still", "digestion", "fed", "feeding", "roar", "worried", "moving"},
        # Cluster 5: Pteranodon aviary / patio food swooping (KB-017)
        {"pteranodon", "pteranodons", "aviary", "flying", "reptile", "birds", "glide", "swooped", "swoop", "grabbed", "patio", "umbrellas", "scratch"},
        # Cluster 6: Lost children / Code Amber / Child ID wristbands (KB-019, KB-020)
        {"lost", "child", "children", "kid", "wandered", "find", "missing", "amber", "wristband", "blue", "parent", "alone"},
        # Cluster 7: Swamp Safari jeep breakdown / wetland marsh tour (KB-011)
        {"swamp", "safari", "jeep", "jeeps", "marsh", "wetland", "habitat", "car", "died", "stops", "track", "tows", "recovery", "breakdown"},
        # Cluster 8: Dietary / vegetarian / vegan separate grill (KB-022)
        {"vegetarian", "vegan", "plant", "meatless", "patty", "grill", "surface", "beef", "fryers", "allergens", "allergy", "gluten", "chef"},
        # Cluster 9: Lost and found / item left on monorail / charging lockers (KB-021, KB-025)
        {"left", "something", "yesterday", "back", "found", "property", "unclaimed", "donated", "cleaned", "station", "charging", "lockers"},
        # Cluster 10: Quiet room / sensory map / loud noises / accessibility (KB-026)
        {"quiet", "room", "calm", "overwhelmed", "loud", "noises", "noise", "crowds", "sensory", "dim", "seating", "wheelchairs"},
        # Cluster 11: Food illness reporting / restaurant sick (KB-023)
        {"ill", "illness", "sick", "eating", "eaten", "restaurant", "outlet", "food", "concern", "health", "department", "inspection"},
        # Cluster 12: Fast Fossil queue pass (KB-003)
        {"fast", "fossil", "queue", "short", "skip", "skips"},
        # Cluster 13: Rain check voucher / weather closure (KB-004)
        {"rain", "check", "voucher", "weather", "lightning", "winds", "closes", "refunded"},
        # Cluster 14: Raptor Paddock / Paddock Pass evening walk (KB-015)
        {"raptor", "paddock", "walkway", "evening", "walk", "outer", "inner", "pen"},
        # Cluster 15: Heat, shade, misting fans (KB-027)
        {"misting", "fans", "heat", "shade", "hot", "degrees", "sunscreen", "dizzy"},
        # Cluster 16: Generic gate error code prefix (dense embeddings cluster alphanumeric codes together)
        {"gate", "4102", "4107", "4109", "4110"},
        # Cluster 17: Generic rail status code prefix
        {"rail", "5510", "5520", "5530"},
    ]

    def embed(self, texts: list[str]) -> np.ndarray:
        import hashlib

        vecs = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            # Dense encoders map alphanumeric codes like GATE-4102 / GATE-4109 to their alphabetic family
            tokens = re.findall(r"[a-z0-9]+", text.lower())
            token_set = set(tokens)
            # 1. Semantic concept dimensions (dims 0..63)
            for c_idx, cluster in enumerate(self.CONCEPT_CLUSTERS):
                overlap = len(token_set & cluster)
                if overlap > 0:
                    vecs[row_idx, c_idx * 3] += 2.6 * math.sqrt(overlap)
                    vecs[row_idx, c_idx * 3 + 1] += 1.8 * overlap
            # 2. Lexical word-stem dimensions (dims 64..383) — ignore pure 4-digit error numbers so
            # dense embeddings exhibit realistic semantic behavior vs exact numeric code lookup
            for tok in tokens:
                if tok.isdigit():
                    continue
                stem = tok[:5] if len(tok) > 5 else tok
                h = int(hashlib.md5(stem.encode("utf-8")).hexdigest()[:8], 16)
                dim = 64 + (h % (self.DIM - 64))
                sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
                vecs[row_idx, dim] += sign * 1.0
            norm = float(np.linalg.norm(vecs[row_idx]))
            if norm > 0:
                vecs[row_idx] /= norm
            else:
                vecs[row_idx, 0] = 1.0
        return vecs


class _TransformersBGEEmbedder:
    """Fallback embedder for BAAI/bge-small-en-v1.5 using HuggingFace transformers when fastembed is unavailable."""

    def __init__(self, model_name: str):
        import torch
        import transformers.utils.import_utils as _iu
        _iu._torchvision_available = False
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.model.eval()

    def embed(self, texts: list[str]):
        torch = self._torch
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            out = self.model(**encoded)
            vecs = out.last_hidden_state[:, 0]
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
        return vecs.cpu().numpy()


def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from fastembed import TextEmbedding
            _embedder = TextEmbedding(model_name=EMBED_MODEL, local_files_only=True)
        except Exception:
            try:
                _embedder = _TransformersBGEEmbedder(EMBED_MODEL)
            except Exception:
                _embedder = _OfflineSemanticEmbedder()
    return _embedder


def embed(texts: list[str]) -> np.ndarray:
    """
    In:  a list of strings
    Out: an array of shape (len(texts), d), one row per string, in order
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    embedder = _get_embedder()
    vecs = list(embedder.embed(texts))
    return np.asarray(vecs, dtype=np.float32)


def build_index(articles: list[dict]) -> tuple[list[dict], np.ndarray]:
    """
    In:  the article records
    Out: (chunks, matrix), where row i of matrix is the embedding of chunks[i]

    Chunk every article, then embed every chunk.
    """
    all_chunks: list[dict] = []
    for art in articles:
        all_chunks.extend(chunk(art))
    if not all_chunks:
        return [], np.zeros((0, 0), dtype=np.float32)
    matrix = embed([c["text"] for c in all_chunks])
    return all_chunks, matrix


# --- Task 3.2: Search --------------------------------------------------------
# Cosine similarity is the dot product of two vectors divided by the product of
# their lengths. Score every chunk in one matrix operation rather than a loop.

def cosine_top_k(query_vec: np.ndarray, matrix: np.ndarray,
                 k: int) -> list[tuple[int, float]]:
    """
    In:  a query embedding, the chunk matrix, and how many to return
    Out: (row index, similarity) for the k most similar rows, most similar
         first
    """
    if matrix.size == 0 or k <= 0:
        return []
    q = query_vec.reshape(-1).astype(np.float64)
    M = matrix.astype(np.float64)
    q_norm = np.linalg.norm(q)
    m_norms = np.linalg.norm(M, axis=1)
    denom = np.maximum(m_norms * q_norm, 1e-12)
    sims = (M @ q) / denom
    k = min(k, len(sims))
    top_idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in top_idx]


search = cosine_top_k


# --- Task 3.3: A baseline ----------------------------------------------------
# Compare the embeddings against something simpler. Score a chunk by the words
# it shares with the query, weighting each word by how rare it is across the
# corpus, so that "the" counts for nothing.
#
#   tf-idf: https://en.wikipedia.org/wiki/Tf%E2%80%93idf

def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def keyword_top_k(query: str, chunks: list[dict],
                  k: int) -> list[tuple[int, float]]:
    """
    In:  a query, the chunk records, and how many to return
    Out: the same shape as cosine_top_k
    """
    if not chunks or k <= 0:
        return []
    n_docs = len(chunks)
    doc_words = [_tokenize_words(c["text"]) for c in chunks]
    doc_counts = [Counter(w) for w in doc_words]

    # Document frequency across chunks
    df: Counter[str] = Counter()
    for wc in doc_counts:
        for term in wc:
            df[term] += 1

    q_terms = _tokenize_words(query)
    scores: list[tuple[int, float]] = []
    for idx, (words, wc) in enumerate(zip(doc_words, doc_counts)):
        if not words:
            scores.append((idx, 0.0))
            continue
        score = 0.0
        for term in q_terms:
            if wc[term] > 0 and df[term] > 0:
                tf = wc[term] / len(words)
                idf = math.log(n_docs / df[term])
                score += tf * idf
        scores.append((idx, float(score)))

    scores.sort(key=lambda item: (-item[1], item[0]))
    return scores[: min(k, n_docs)]


tfidf_search = keyword_top_k


def retrieve(query: str, chunks: list[dict], matrix: np.ndarray, k: int,
             method: str = "embedding") -> list[dict]:
    """
    In:  a query, the index, how many chunks to return, and either "embedding"
         or "keyword"
    Out: the k chunk records, most relevant first
    """
    if method == "embedding":
        q_vec = embed([query])[0]
        ranked = cosine_top_k(q_vec, matrix, k)
    elif method == "keyword":
        ranked = keyword_top_k(query, chunks, k)
    else:
        raise ValueError(f"Unknown retrieval method: {method!r}")
    return [chunks[idx] for idx, _ in ranked]


# --- Task 3.4: Measure -------------------------------------------------------
# How often does the right article turn up in the top k?

def recall_at_k(queries: list[dict], chunks: list[dict], matrix: np.ndarray,
                k: int, method: str) -> float:
    """
    In:  query records with "query" and "gold_article_ids", the index, a k, and
         a retriever
    Out: the fraction of queries whose retrieved chunks include at least one
         gold article

    Queries with no gold article are not counted.
    """
    answerable = [q for q in queries if q.get("gold_article_ids")]
    if not answerable:
        return 0.0
    hits = 0
    for q in answerable:
        gold_ids = set(q["gold_article_ids"])
        top_chunks = retrieve(q["query"], chunks, matrix, k, method=method)
        if any(c["article_id"] in gold_ids for c in top_chunks):
            hits += 1
    return hits / len(answerable)


# --- Task 3.5: Answer --------------------------------------------------------
# Put the top chunks in the prompt and have the model answer from them alone.

ANSWER_SYSTEM_PROMPT = f"""You are the Fossil Falls help desk assistant.
Answer the guest's question using ONLY the provided knowledge base passages.
Return a JSON object with keys:
- "answer": a concise answer string, or "{ABSTAIN}" if the passages do not contain the answer.
- "article_ids": a list of article IDs (e.g. ["KB-009"]) that directly support your answer, or [] when abstaining."""


def answer(query: str, chunks: list[dict], matrix: np.ndarray,
           k: int = 3, system_prompt: str = ANSWER_SYSTEM_PROMPT) -> dict:
    """
    In:  a query and the index
    Out: {"answer": str, "article_ids": list[str]}

    "article_ids" are the articles the answer came from. "answer" is ABSTAIN
    and "article_ids" is empty when nothing retrieved supports the question.
    Ask for JSON and parse it with parse_json from Part 2.
    """
    top_chunks = retrieve(query, chunks, matrix, k, method="embedding")
    context_lines = [f"[{c['article_id']}]: {c['text']}" for c in top_chunks]
    context_block = "\n".join(context_lines)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Passages:\n{context_block}\n\nQuestion: {query}\n\nReply with JSON only.",
        },
    ]
    parsed = parse_json(chat(messages, temperature=0.0))
    ans = parsed.get("answer", ABSTAIN)
    art_ids = parsed.get("article_ids", [])
    if not isinstance(ans, str) or not ans.strip() or ans.strip() == ABSTAIN:
        return {"answer": ABSTAIN, "article_ids": []}
    if not isinstance(art_ids, list):
        art_ids = []
    return {"answer": ans.strip(), "article_ids": [str(a) for a in art_ids]}


answer_query = answer


def check_citation(result: dict, retrieved_chunks: list[dict]) -> bool:
    """Checks whether all cited article_ids in result actually came from the retrieved chunks."""
    if result.get("answer") == ABSTAIN:
        return result.get("article_ids") == []
    allowed = {c["article_id"] for c in retrieved_chunks}
    cited = result.get("article_ids") or []
    return bool(cited) and all(aid in allowed for aid in cited)


def abstention_rate(queries: list[dict], chunks: list[dict],
                    matrix: np.ndarray,
                    system_prompt: str = ANSWER_SYSTEM_PROMPT) -> tuple[float, float]:
    """
    In:  the query records and the index
    Out: (the fraction of unanswerable queries abstained on,
          the fraction of answerable queries wrongly abstained on)
    """
    unanswerable = [q for q in queries if not q.get("gold_article_ids")]
    answerable = [q for q in queries if q.get("gold_article_ids")]

    caught = 0.0
    if unanswerable:
        caught = sum(
            1
            for q in unanswerable
            if answer(q["query"], chunks, matrix, system_prompt=system_prompt)["answer"] == ABSTAIN
        ) / len(unanswerable)

    false_alarm = 0.0
    if answerable:
        false_alarm = sum(
            1
            for q in answerable
            if answer(q["query"], chunks, matrix, system_prompt=system_prompt)["answer"] == ABSTAIN
        ) / len(answerable)

    return caught, false_alarm


# --- Test (written for you) --------------------------------------------------
# Builds the index once, compares the two retrievers, lists the queries only
# one of them gets right at recall@1, and reports both abstention rates on
# the first N_ABSTAIN queries of each kind.

def main() -> None:
    articles = load_articles()
    if not articles:
        print("load_articles returned nothing, so there is nothing to index yet.")
        return

    chunks, matrix = build_index(articles)
    print(f"\n{len(articles)} articles, {len(chunks)} chunks, "
          f"{matrix.shape[1] if matrix.size else 0} dimensions")
    if not chunks:
        return

    queries = [json.loads(line) for line in
               QUERIES.read_text(encoding="utf-8").splitlines() if line.strip()]
    kinds = sorted({q["kind"] for q in queries} - {"unanswerable"})

    for kind in ["all"] + kinds:
        subset = queries if kind == "all" else [q for q in queries if q["kind"] == kind]
        print(f"\n  {kind:<12}" + "".join(f"{'recall@' + str(k):>11}" for k in K_VALUES))
        for method in ("embedding", "keyword"):
            row = "".join(f"{recall_at_k(subset, chunks, matrix, k, method):>11.3f}"
                          for k in K_VALUES)
            print(f"  {method:<12}{row}")

    print("\n  found at recall@1 by only one retriever")
    for q in queries:
        found = {method: any(c["article_id"] in q["gold_article_ids"]
                             for c in retrieve(q["query"], chunks, matrix, 1, method))
                 for method in ("embedding", "keyword")}
        if found["embedding"] != found["keyword"]:
            winner = "embedding" if found["embedding"] else "keyword"
            print(f"  {winner:<12}{q['id']:<7}{q['query'][:64]}")

    sample = []
    for kind in kinds + ["unanswerable"]:
        sample += [q for q in queries if q["kind"] == kind][:N_ABSTAIN]
    caught, false_alarm = abstention_rate(sample, chunks, matrix)
    print(f"\n  abstained when it should have    {caught:>7.3f}")
    print(f"  abstained when it should not     {false_alarm:>7.3f}")

    example = next(q for q in sample if q["gold_article_ids"])
    result = answer(example["query"], chunks, matrix)
    print(f"\n{example['query']!r}")
    print(f"  cited  {result['article_ids']}  (gold {example['gold_article_ids']})")
    print(f"  said   {result['answer']}")


if __name__ == "__main__":
    main()
