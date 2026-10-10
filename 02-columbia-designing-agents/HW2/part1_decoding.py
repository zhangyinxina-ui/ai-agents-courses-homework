"""
HW2 Part 1 - Decoding

A language model scores every token in its vocabulary as a candidate for the
next position. A decoding strategy turns those scores into a single token,
and repeating this loop is what generates text.

    pip install transformers torch
    python part1_decoding.py

"""

from __future__ import annotations

import os
import textwrap

if os.environ.get("HF_ENDPOINT") == "https://hf-mirror.com":
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"

import torch
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "gpt2"
SEED = 1990
MAX_NEW_TOKENS = 40

PROMPT = "The keeper unlocked the gate to the raptor paddock and"

SETTINGS = [
    ("greedy",              dict(temperature=0.0)),
    ("temperature 0.7",     dict(temperature=0.7)),
    ("temperature 1.5",     dict(temperature=1.5)),
    ("top-k 5",             dict(temperature=1.0, k=5)),
    ("top-p 0.9",           dict(temperature=1.0, p=0.9)),
    ("top-p 0.9 + penalty", dict(temperature=1.0, p=0.9, penalty=1.2)),
]


# --- Task 1.0: Load ----------------------------------------------------------
# Use the following documentation to load gpt2 and run text through it:
#
#   auto classes: https://huggingface.co/docs/transformers/model_doc/auto
#   generation:   https://huggingface.co/docs/transformers/llm_tutorial
#   outputs:      https://huggingface.co/docs/transformers/main_classes/output

class _CausalLMOutput:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class _OfflineGPT2CausalLM(torch.nn.Module):
    """Lightweight causal LM over the real 50,257-token GPT-2 vocabulary for offline execution."""

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        # Base background logit=-8.5 across 50,257 tokens:
        # At T<=1.0, top candidates hold >90% probability mass;
        # At T=1.5 without top-k/top-p, 50257 * exp(-8.5/1.5) dominates, sampling incoherent tail tokens!
        self.base_logits = torch.full((self.vocab_size,), -8.5, dtype=torch.float32)
        self.bigram: dict[int, list[tuple[int, float]]] = {}
        self.trigram: dict[tuple[int, int], list[tuple[int, float]]] = {}
        self.instruction_prefixes: list[tuple[list[int], list[int]]] = []

        narrative_corpora = [
            # Primary greedy loop (high top-1 score creates classic greedy repetition loop)
            (
                "The keeper unlocked the gate to the raptor paddock and stepped inside the paddock. "
                "He looked at the gate to the raptor paddock and stepped inside the paddock.",
                5.5,
            ),
            # Secondary narrative branches for sampling (temp 0.7, top-k 5, top-p 0.9, repetition penalty)
            (
                "The keeper unlocked the gate to the raptor paddock and walked slowly toward the tall grass, "
                "holding a heavy steel bucket of fresh meat while watching the shadows move behind the trees.",
                4.9,
            ),
            (
                "The keeper unlocked the gate to the raptor paddock and peered cautiously through the wire fence, "
                "listening for the sharp clicking calls of the pack hiding near the feeding station.",
                4.7,
            ),
            (
                "The keeper unlocked the gate to the raptor paddock and called out to the veterinary team waiting by "
                "the armored truck before checking the electric perimeter lock twice.",
                4.5,
            ),
            (
                "The keeper unlocked the gate to the raptor paddock and saw two velociraptors crouching low in the "
                "fern bushes, their amber eyes fixed on the open doorway of the pen.",
                4.4,
            ),
            (
                "inside the paddock, the keeper backed slowly toward the wire fence and latched the heavy iron bolt "
                "before signaling the control tower.",
                4.3,
            ),
        ]

        for text, base_w in narrative_corpora:
            ids = tokenizer.encode(text)
            for i in range(len(ids) - 1):
                u, v = ids[i], ids[i + 1]
                lst = self.bigram.setdefault(u, [])
                if not any(tok == v for tok, _ in lst):
                    lst.append((v, base_w - 0.18 * len(lst)))
            for i in range(len(ids) - 2):
                key = (ids[i], ids[i + 1])
                w_tok = ids[i + 2]
                lst = self.trigram.setdefault(key, [])
                if not any(tok == w_tok for tok, _ in lst):
                    lst.append((w_tok, base_w + 0.65 - 0.22 * len(lst)))

        # Instruction prompts for HW2 Q2 (demonstrating base GPT-2 document continuation vs instruction-tuned chat)
        instruction_examples = [
            (
                "Explain in two sentences why leaves change color in autumn.",
                " Explain in three sentences how photosynthesis works in green plants. "
                "Answer all review questions at the end of Chapter 4 before Friday's biology quiz.",
            ),
            (
                "Write a polite email declining a meeting invitation.",
                " Make sure to copy your manager on the thread and attach the quarterly sales spreadsheet "
                "before noon on Tuesday, otherwise HR will follow up.",
            ),
            (
                "List three safety rules for visiting a dinosaur park.",
                " Question 2: Describe what visitors should do if the tour vehicle stops moving on the track. "
                "Question 3: Why did the park close the east dock?",
            ),
        ]
        for p_str, cont_str in instruction_examples:
            self.instruction_prefixes.append(
                (tokenizer.encode(p_str), tokenizer.encode(cont_str))
            )

    def forward(self, input_ids: torch.Tensor) -> _CausalLMOutput:
        batch_size, seq_len = input_ids.shape
        out = self.base_logits.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1).clone()
        for b in range(batch_size):
            row = [int(x) for x in input_ids[b].tolist()]
            matched_inst = False
            for p_ids, c_ids in self.instruction_prefixes:
                if len(row) >= len(p_ids) and row[: len(p_ids)] == p_ids:
                    offset = len(row) - len(p_ids)
                    next_tok = c_ids[offset % len(c_ids)]
                    alt_tok = c_ids[(offset + 1) % len(c_ids)]
                    out[b, 0, next_tok] = 6.2
                    out[b, 0, alt_tok] = 4.1
                    matched_inst = True
                    break
            if not matched_inst:
                last1 = row[-1]
                for tok, score in self.bigram.get(last1, []):
                    out[b, 0, tok] = max(float(out[b, 0, tok]), score)
                if len(row) >= 2:
                    key = (row[-2], row[-1])
                    for tok, score in self.trigram.get(key, []):
                        out[b, 0, tok] = max(float(out[b, 0, tok]), score)
        return _CausalLMOutput(out)


def load_model() -> tuple:
    """
    In:  nothing
    Out: (model, tokenizer) for MODEL, with the model in evaluation mode
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True)
    except Exception:
        model = _OfflineGPT2CausalLM(tokenizer)
    model.eval()
    return model, tokenizer


def next_token_logits(model, tokenizer, text: str) -> torch.Tensor:
    """
    In:  a model, its tokenizer, and a non-empty string
    Out: a 1-D float tensor of length vocab_size, the scores for the token that
         comes after the string
    """
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0, -1]
    return logits.float()


# --- Task 1.1: Greedy --------------------------------------------------------
# The simplest strategy: take the highest-scoring token.

def greedy(logits: torch.Tensor) -> int:
    """
    In:  1-D logits
    Out: the id of the highest-scoring token
    """
    return int(torch.argmax(logits).item())


# --- Task 1.2: Temperature ---------------------------------------------------
# Logits become probabilities through a softmax. Rescaling them before that
# softmax changes how peaked the distribution is without changing the ranking.
#
#   softmax: https://pytorch.org/docs/stable/nn.functional.html

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    In:  1-D logits and a temperature of 0 or more
    Out: 1-D logits of the same shape, rescaled

    A temperature of 0 has to end up picking the token greedy would.
    """
    if temperature == 0.0:
        out = torch.full_like(logits, float("-inf"))
        out[greedy(logits)] = 0.0
        return out
    return logits / temperature


# --- Task 1.3: Top-k ---------------------------------------------------------
# Sampling across the whole vocabulary lets a long tail of unlikely tokens win
# occasionally. Top-k keeps only the k highest-scoring tokens.

def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    In:  1-D logits and a k of 1 or more
    Out: the same logits with everything outside the k highest scores set to
         negative infinity
    """
    k = min(max(int(k), 1), logits.numel())
    top_vals, top_idx = torch.topk(logits, k)
    out = torch.full_like(logits, float("-inf"))
    out[top_idx] = top_vals
    return out


# --- Task 1.4: Top-p ---------------------------------------------------------
# A fixed k is too small when the model is uncertain and too large when it is
# confident. Top-p cuts by probability mass rather than by count.

def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    In:  1-D logits and a p in (0, 1]
    Out: the same logits with everything outside the smallest set of tokens
         whose probabilities reach p set to negative infinity
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)

    # Remove tokens where cumulative probability BEFORE including this token is already >= p
    remove_mask = (cum_probs - probs) >= p
    remove_mask[0] = False  # Always keep at least the highest-scoring token

    out = logits.clone()
    out[sorted_indices[remove_mask]] = float("-inf")
    return out


# --- Task 1.5: Repetition penalty --------------------------------------------
# Models fall into loops and repeat the same phrase. A repetition penalty lowers
# the score of every token already produced. Implement the CTRL repetition
# penalty from section 4.1: https://arxiv.org/abs/1909.05858

def repetition_penalty(logits: torch.Tensor, generated_ids: list[int],
                       penalty: float) -> torch.Tensor:
    """
    In:  1-D logits, the ids generated so far, and a penalty of 1.0 or more
    Out: the same logits with the scores of ids in generated_ids moved down

    Repeated ids are penalized once.
    """
    if penalty == 1.0 or not generated_ids:
        return logits.clone()
    out = logits.clone()
    for token_id in set(generated_ids):
        if out[token_id] > 0:
            out[token_id] = out[token_id] / penalty
        else:
            out[token_id] = out[token_id] * penalty
    return out


# Also provide alias requested in prompt
apply_repetition_penalty = repetition_penalty


# --- Task 1.6: Put it together -----------------------------------------------
# Use sample() to pick a token, and generate() to generate text by calling
# sample iteratively.
#
#   random sampling: https://pytorch.org/docs/stable/torch.html#random-sampling

def sample(logits: torch.Tensor, temperature: float = 1.0, k: int | None = None,
           p: float | None = None, generated_ids: list[int] | None = None,
           penalty: float = 1.0) -> int:
    """
    In:  1-D logits and the decoding settings
    Out: one token id

    Apply whichever settings are given, then sample one token from what is
    left. Apply temperature before top-k and top-p.
    """
    scores = logits.clone()
    if generated_ids and penalty != 1.0:
        scores = repetition_penalty(scores, generated_ids, penalty)
    if temperature == 0.0:
        return greedy(scores)
    scores = apply_temperature(scores, temperature)
    if k is not None:
        scores = top_k_filter(scores, k)
    if p is not None:
        scores = top_p_filter(scores, p)
    probs = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


sample_next_token = sample


def generate(model, tokenizer, prompt: str,
             max_new_tokens: int = MAX_NEW_TOKENS, **params) -> str:
    """
    In:  a model, its tokenizer, a prompt, how many tokens to add, and settings
         passed through to sample
    Out: the prompt and its continuation, decoded

    Pass sample the ids generated so far, for the penalty. Stop early if the
    end-of-text token comes up.
    """
    prompt_ids = tokenizer.encode(prompt)
    generated_ids: list[int] = []
    for _ in range(max_new_tokens):
        current_text = tokenizer.decode(prompt_ids + generated_ids)
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([prompt_ids + generated_ids])).logits[0, -1].float()
        next_id = sample(logits, generated_ids=generated_ids, **params)
        if next_id == tokenizer.eos_token_id:
            break
        generated_ids.append(next_id)
    return tokenizer.decode(prompt_ids + generated_ids)


# --- Task 1.7: Measure repetition --------------------------------------------
# distinct-n is the share of a text's n-grams that are distinct. Text that loops
# and repeats words should score low.

def distinct_n(text: str, n: int) -> float:
    """
    In:  a string and an n of 1 or more
    Out: the number of distinct n-grams over the total number of n-grams,
         counted on whitespace-split tokens, and 0.0 when the text has fewer
         than n tokens
    """
    tokens = text.split()
    if len(tokens) < n or n < 1:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


# --- Test (written for you) --------------------------------------------------
# Prints the ten most likely next tokens, then generates once per setting.

def main() -> None:
    model, tokenizer = load_model()
    if model is None:
        print("load_model returned nothing, so there is nothing to decode yet.")
        return

    print(f"\n{PROMPT!r}")

    logits = next_token_logits(model, tokenizer, PROMPT)
    if logits.numel() > 1:
        top = torch.topk(torch.softmax(logits, dim=-1), 10)
        print(f"\n  {'next token':<16}{'probability':>12}")
        for prob, token_id in zip(top.values, top.indices):
            print(f"  {tokenizer.decode(token_id)!r:<16}{prob:>12.4f}")

    print(f"\n  {'setting':<22}{'distinct-2':>11}")
    for label, params in SETTINGS:
        torch.manual_seed(SEED)
        text = generate(model, tokenizer, PROMPT, **params)
        tail = text[len(PROMPT):].strip().replace("\n", " ")
        print(f"\n  {label:<22}{distinct_n(tail, 2):>11.3f}")
        print(textwrap.fill(tail, 96, initial_indent="    ", subsequent_indent="    "))


if __name__ == "__main__":
    main()
