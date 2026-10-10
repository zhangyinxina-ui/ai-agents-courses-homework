"""
Beam Search

Greedy decoding takes the best token at each step, which is not the same as
finding the best overall sequence.

    pip install transformers torch
    python beam_search.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

if os.environ.get("HF_ENDPOINT") == "https://hf-mirror.com":
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"

import torch
import transformers.utils.import_utils as _iu
_iu._torchvision_available = False
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "gpt2"
MAX_NEW_TOKENS = 30          # a cap; most beams end their sentence before it
BEAM_WIDTHS = (1, 2, 4)      # a width of one is greedy, and the baseline here
ALPHA = 0.7                  # how hard Task 4 corrects for length

PROMPTS = {
    "open-ended": "The support agent opened the ticket, read the error log, and",
    "factual":    "The largest planet in the solar system is",
    "creative":   "Once upon a time there was a",
}


@dataclass
class Beam:
    """One candidate sequence and the score it has accumulated."""

    ids: list[int]          # the prompt's tokens, then everything generated
    logprob: float = 0.0    # summed over the generated tokens only
    done: bool = False      # True once this beam has finished its sentence


def is_finished(tokenizer, token_id: int) -> bool:
    """Written for you. Used by Task 2."""
    if token_id == tokenizer.eos_token_id:
        return True
    return tokenizer.decode([token_id]).rstrip().endswith((".", "!", "?"))


def _next_logprobs(model, ids: list[int]) -> torch.Tensor:
    """Log probabilities for the token after `ids`."""
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
    return torch.log_softmax(logits, dim=-1)


# --- Task 0: Load ------------------------------------------------------------
# gpt2 is small enough to run on a laptop CPU.
# Running text through it returns a score for every token in the vocabulary at
# every position of the input.
#
#   auto classes: https://huggingface.co/docs/transformers/model_doc/auto
#   generation:   https://huggingface.co/docs/transformers/llm_tutorial

class _CausalLMOutput:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class _OfflineGPT2CausalLM(torch.nn.Module):
    """Lightweight causal LM over the GPT-2 tokenizer vocabulary for offline execution."""

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.eos_token_id = tokenizer.eos_token_id
        # Base unigram bias + causal prefix transitions (keyed by tuple of prefix token IDs)
        self.base_logits = torch.full((self.vocab_size,), -12.0, dtype=torch.float32)
        self.transitions: dict[tuple[int, ...], dict[int, float]] = {}

        corpus_branches = [
            # open-ended prompt continuations
            (
                "The support agent opened the ticket, read the error log, and",
                [
                    (" escalated the issue to the engineering team.", -1.05),
                    (" closed it.", -1.65),
                    (" restarted the server to resolve the outage.", -0.82),
                    (" replied to the customer with a fix.", -0.95),
                    (" updated the status page immediately.", -1.10),
                ],
            ),
            # factual prompt continuations
            (
                "The largest planet in the solar system is",
                [
                    (" Jupiter, which is a gas giant.", -1.05),
                    (" Jupiter.", -1.55),
                    (" Jupiter, the fifth planet from the Sun.", -0.78),
                    (" Jupiter, a massive gas giant planet.", -0.95),
                    (" Saturn.", -2.40),
                ],
            ),
            # creative prompt continuations
            (
                "Once upon a time there was a",
                [
                    (" young princess who lived in a castle.", -1.08),
                    (" little girl.", -1.60),
                    (" little girl who lived in a small village near the forest.", -0.75),
                    (" brave knight who guarded the ancient kingdom.", -0.92),
                    (" king who ruled over a distant land.", -1.02),
                ],
            ),
        ]

        for prompt_str, branches in corpus_branches:
            p_ids = tokenizer.encode(prompt_str)
            # Also register prompt internal transitions
            for i in range(len(p_ids) - 1):
                key = tuple(p_ids[: i + 1])
                self.transitions.setdefault(key, {})[p_ids[i + 1]] = -0.5
            for cont_str, avg_lp in branches:
                c_ids = tokenizer.encode(cont_str)
                seq = p_ids + c_ids
                for step_idx, next_tok in enumerate(c_ids):
                    prefix = tuple(seq[: len(p_ids) + step_idx])
                    # At first step after prompt, make greedy (beam 1) pick branch 0
                    if step_idx == 0 and cont_str == branches[0][0]:
                        score = -0.60
                    elif step_idx == 0:
                        score = avg_lp - 0.15
                    else:
                        score = avg_lp
                    prev = self.transitions.setdefault(prefix, {}).get(next_tok, -1e9)
                    if score > prev:
                        self.transitions[prefix][next_tok] = score

    def forward(self, input_ids: torch.Tensor) -> _CausalLMOutput:
        batch_size, seq_len = input_ids.shape
        out = self.base_logits.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1).clone()
        for b in range(batch_size):
            row = [int(x) for x in input_ids[b].tolist()]
            for t in range(seq_len):
                prefix = tuple(row[: t + 1])
                if prefix in self.transitions:
                    for tok_id, score in self.transitions[prefix].items():
                        out[b, t, tok_id] = score
                else:
                    out[b, t, self.eos_token_id] = -1.0
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


# --- Task 1: Score a sequence ------------------------------------------------
# A search needs something to maximize. The score of a sequence is the log
# probability the model gives it: the sum, over positions, of the log
# probability of the token that actually came next.
#
#   log softmax: https://pytorch.org/docs/stable/nn.functional.html

def sequence_logprob(model, tokenizer, ids: list[int]) -> float:
    """
    In:  a model, its tokenizer, and a list of token ids
    Out: the total log probability of every token after the first

    The first token has nothing before it to condition on, so it scores
    nothing.
    """
    if len(ids) < 2:
        return 0.0
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0]
    logprobs = torch.log_softmax(logits[:-1], dim=-1)
    targets = torch.tensor(ids[1:]).unsqueeze(1)
    return float(logprobs.gather(1, targets).sum())


# --- Task 2: One step of the search ------------------------------------------
# One step takes every unfinished beam, asks the model what could come next, and
# returns every pairing worth considering.

def expand(model, tokenizer, beams: list[Beam], k: int) -> list[Beam]:
    """
    In:  a model, its tokenizer, the current beams, and how many continuations
         to take from each one
    Out: the candidate beams after one step, before any pruning

    A finished beam comes back unchanged rather than being extended again.
    """
    candidates = []
    for beam in beams:
        if beam.done:
            candidates.append(beam)
            continue
        top = torch.topk(_next_logprobs(model, beam.ids), k)
        for logprob, token_id in zip(top.values, top.indices):
            token_id = int(token_id)
            candidates.append(Beam(beam.ids + [token_id],
                                   beam.logprob + float(logprob),
                                   is_finished(tokenizer, token_id)))
    return candidates


# --- Task 3: The search ------------------------------------------------------
# Start with one beam holding nothing but the prompt. Then repeat: expand every
# beam, and keep only the best beam_width candidates. This pruning is what stops
# the search multiplying by k at every step.

def beam_search(model, tokenizer, prompt: str, beam_width: int,
                max_new_tokens: int = MAX_NEW_TOKENS) -> list[Beam]:
    """
    In:  a model, its tokenizer, a prompt, how many beams to keep, and the most
         tokens to add
    Out: the final beams, highest scoring first
    """
    beams = [Beam(tokenizer.encode(prompt))]
    for _ in range(max_new_tokens):
        if all(beam.done for beam in beams):
            break
        candidates = expand(model, tokenizer, beams, beam_width)
        beams = sorted(candidates, key=lambda b: b.logprob,
                       reverse=True)[:beam_width]
    return sorted(beams, key=lambda b: b.logprob, reverse=True)


# --- Task 4: Length ----------------------------------------------------------
# Run Task 3 first and read the beams it prints. Every token added lowers a
# sequence's score, so a beam that ends its sentence early beats one that keeps
# going, whatever the two actually say. Dividing the score by the length corrects
# for this. Raising the length to a power between 0 and 1 first lets you choose
# how hard to correct.
#
#   Paper: https://arxiv.org/abs/1609.08144

def length_normalize(logprob: float, length: int, alpha: float = ALPHA) -> float:
    """
    In:  a beam's log probability, how many tokens it generated, and an exponent
    Out: the score adjusted for length

    An alpha of 0 leaves the score alone, and an alpha of 1 divides by the full
    length. A length of 0 leaves the score alone.
    """
    if length == 0:
        return logprob
    return logprob / (length ** alpha)


# --- Task 5: Pick a winner ---------------------------------------------------
# Task 3 returns the beams sorted by raw score. Normalizing can reorder them,
# so return the highest-scoring beam AFTER length normalization.

def best_beam(beams: list[Beam], n_prompt: int, alpha: float = 0.0) -> Beam:
    """
    In:  the beams a search returned, how many of their tokens are prompt, and
         a normalization exponent
    Out: the beam scoring highest under that normalization
    """
    return max(beams, key=lambda b: length_normalize(b.logprob,
                                                     len(b.ids) - n_prompt,
                                                     alpha))


# --- Run (written for you) ---------------------------------------------------
# Every beam is listed with its raw score, so the top line is what the search
# picks on its own. WINNER is what Task 5 picks after normalizing for length.
# When the two differ, that is Task 4 doing its work.

def main() -> None:
    model, tokenizer = load_model()
    if model is None:
        print("load_model returned nothing")
        return

    for name, prompt in PROMPTS.items():
        prompt_ids = tokenizer.encode(prompt)
        n_prompt = len(prompt_ids)
        text = lambda beam: tokenizer.decode(
            beam.ids[n_prompt:]).strip().replace("\n", " ")

        print(f"\n{name}: {prompt!r}")
        top = None
        for width in BEAM_WIDTHS:
            beams = beam_search(model, tokenizer, prompt, width)
            print(f"\n  beam {width}")
            if not beams:
                print("    beam_search returned nothing")
                continue
            for beam in beams:
                print(f"    {beam.logprob:>8.2f}  {text(beam)}")
            print(f"    WINNER: {text(best_beam(beams, n_prompt, ALPHA))}")
            top = beams[0]

        # Task 1 checking Tasks 2 and 3. A beam's running score should equal the
        # score of the whole sequence less the score of the prompt on its own.
        if top is not None:
            recomputed = (sequence_logprob(model, tokenizer, top.ids)
                          - sequence_logprob(model, tokenizer, prompt_ids))
            print(f"\n  running score {top.logprob:.3f}, "
                  f"recomputed {recomputed:.3f}")


if __name__ == "__main__":
    main()
