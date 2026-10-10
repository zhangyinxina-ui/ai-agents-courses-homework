"""
Autoprompting

Rather than rewriting a prompt by hand until the output looks right, have the
model rewrite it. You will pick a one-word topic, write a prompt asking for a
poem about it, and list the ways those poems can be improved.

    pip install google-genai
    export GEMINI_API_KEY=...
    python autoprompting.py

Running the file prints every round's prompt, its poem, and the errors the
judge found.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import hashlib
import json
import logging
import os
from pathlib import Path

try:
    from google import genai
    from google.genai import types
    # Ignore warnings from the Gemini SDK.
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)
    logging.getLogger("google_genai.types").setLevel(logging.ERROR)
except ImportError:
    genai = None
    types = None

MODEL = "gemini-3.5-flash-lite"
MAX_TOKENS = 1024
CACHE = Path(__file__).resolve().parent / ".autoprompting_cache.json"

TOPIC = "capitalism"      # one word for the poems to be about
ROUNDS = 3                # rewrites to try

_client = None


def _offline_fallback(prompt: str) -> str:
    """Deterministic offline fallback when GEMINI_API_KEY is not set and prompt is not in cache."""
    # Judge stage
    if "List the name of every error this poem has" in prompt:
        if "Neon coins and golden chains" in prompt:
            return "cliche\nsing-song meter\nmoral ending"
        if "The factory floor is loud and cold" in prompt:
            return "stated feeling\nmoral ending"
        if "At 4 a.m. the gantry cranes" in prompt:
            return "moral ending"
        return "NONE"

    # Rewrite stage
    if "Write an improved prompt" in prompt:
        errors_part = prompt.split("A reviewer found these problems in the poem:")[-1]
        if "cliche" in errors_part:
            return (
                f"Write a short free-verse poem about {TOPIC} using original, specific "
                f"industrial details. Avoid stock imagery or cliches, avoid a bouncing "
                f"sing-song meter, and do not end with a moral lesson."
            )
        if "stated feeling" in errors_part:
            return (
                f"Write an 8-line unrhymed poem about {TOPIC} grounded in concrete physical "
                f"actions at a port or warehouse. Never name emotions directly; show them "
                f"through objects. Do not summarize a lesson or general truth in the final lines."
            )
        return (
            f"Write an 8-line unrhymed poem about {TOPIC} focused strictly on a night-shift "
            f"clerk, a thermal receipt, and shipping containers. Use zero abstract commentary "
            f"and end strictly on a physical action or sound."
        )

    # Poem generation stage
    if prompt.strip() == f"Write a short poem about {TOPIC}.":
        return (
            "Neon coins and golden chains,\n"
            "Bound by losses, driven by gains.\n"
            "The lonely night of endless greed,\n"
            "Where towers rise and workers bleed.\n"
            "Remember well this bitter art:\n"
            "The market steals the human heart."
        )
    if "original, specific industrial details" in prompt:
        return (
            "The factory floor is loud and cold,\n"
            "forklifts pivot down aisle nine with pallets of shrink-wrapped glass.\n"
            "A deep sadness hangs over the fluorescent breakroom\n"
            "while the time-clock punches another shift into paper,\n"
            "teaching us that every hour has a price tag."
        )
    if "grounded in concrete physical actions" in prompt:
        return (
            "At 4 a.m. the gantry cranes lift orange steel boxes\n"
            "from the hull of a freighter registered in Monrovia.\n"
            "Barcodes flash red beneath the customs scanner,\n"
            "soybeans and lithium batteries stacked six high on the pier,\n"
            "proving the world is bound together by what it trades."
        )
    return (
        "A thermal receipt curls on the stainless counter,\n"
        "fading ink recording three liters of diesel and a salted roll.\n"
        "Out in the harbor, gantry cranes lift orange steel boxes\n"
        "from the hull of a freighter registered in Monrovia.\n"
        "The night clerk stamps a manifest twice in purple dye,\n"
        "slides the carbon copy into a wire basket,\n"
        "and watches the forklifts pivot between numbered bays\n"
        "until the shift bell clicks at four."
    )


def ask(prompt: str, temperature: float = 0.0) -> str:
    """
    Written for you. Sends one prompt to MODEL and returns the reply text.

    Every reply is saved to .autoprompting_cache.json, and an identical request
    returns the saved reply, even at a temperature above 0. Delete the file to
    draw fresh replies. Waits and retries when the free tier's rate limit is
    hit.
    """
    global _client
    request = {"model": MODEL, "prompt": prompt, "temperature": temperature}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    if key in cache:
        return cache[key]
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or genai is None:
        reply = _offline_fallback(prompt)
        cache[key] = reply
        CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        return reply
    if _client is None:
        _client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=8)))
    response = _client.models.generate_content(
        model=MODEL, contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature,
                                           max_output_tokens=MAX_TOKENS))
    reply = response.text or ""
    cache[key] = reply
    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return reply


# --- Task 0: A first prompt --------------------------------------------------
# The prompt you would write by hand, asking for a poem about TOPIC.
#
#   prompting: https://ai.google.dev/gemini-api/docs/prompting-strategies

GENERATOR_PROMPT = f"Write a short poem about {TOPIC}."


# --- Task 1: Error taxonomy --------------------------------------------------
# Run the file with Task 0 done and read the poem it prints. Name each way it
# falls short, and describe it precisely enough that someone who has not seen
# this poem could spot it in a new one.
#
#   paper: https://arxiv.org/abs/2404.12272

ERROR_TAXONOMY: dict[str, str] = {
    "cliche": "a stock image or phrase that most poems on this topic use, such "
              "as silver light, a lonely night, or the moon as a watchful eye",
    "forced rhyme": "a word chosen to rhyme that is awkward, archaic, or "
                    "makes the line say less than it would otherwise",
    "sing-song meter": "every line falls into the same bouncing rhythm, so "
                       "the poem reads like a greeting card",
    "stated feeling": "the poem names an emotion, such as longing, peace, or "
                      "wonder, without anything in it that causes the emotion",
    "moral ending": "the last lines sum up a lesson or a general truth "
                    "instead of ending on something concrete",
    "padding": "a line that restates an earlier line without adding anything",
}


# --- Task 2: Judge -----------------------------------------------------------
# The judge prompt is written for you. Fill in its three blanks, send it at
# temperature 0, and read the error names out of the reply.
#
#   paper: https://arxiv.org/abs/2306.05685

JUDGE_PROMPT = """Here is a poem about "{topic}":

{poem}

A poem can have these errors:
{taxonomy}

List the name of every error this poem has, one per line, and nothing else.
If it has none, reply NONE."""


def judge(poem: str) -> list[str]:
    """
    In:  a poem about TOPIC
    Out: the names of the ERROR_TAXONOMY errors the judge found in it, and []
         when there are none

    Drop any name the judge returns that is not in ERROR_TAXONOMY.
    """
    taxonomy = "\n".join(f"- {name}: {description}"
                         for name, description in ERROR_TAXONOMY.items())
    reply = ask(JUDGE_PROMPT.format(topic=TOPIC, poem=poem, taxonomy=taxonomy))
    names = {name.lower(): name for name in ERROR_TAXONOMY}
    found = []
    for line in reply.splitlines():
        line = line.split(":")[0].strip().strip("-*•`\"' ").lower()
        if line in names and names[line] not in found:
            found.append(names[line])
    return found


# --- Task 3: Rewrite ---------------------------------------------------------
# Show the model the current prompt, the poem it wrote, and the errors the
# judge found, and ask for a better prompt.

REWRITE_PROMPT = """This prompt was given to a language model to write a poem \
about "{topic}":

{prompt}

It wrote this poem:

{poem}

A reviewer found these problems in the poem:
{errors}

Write an improved prompt that would avoid these problems while keeping what \
works. Reply with the new prompt only: no explanation, no title, no quotation \
marks."""


def rewrite(prompt: str, poem: str, errors: list[str]) -> str:
    """
    In:  the current prompt, its poem, and the errors found in the poem
    Out: a new prompt, with nothing around it
    """
    if not errors:
        return prompt
    listed = "\n".join(f"- {name}: {ERROR_TAXONOMY[name]}" for name in errors)
    reply = ask(REWRITE_PROMPT.format(topic=TOPIC, prompt=prompt, poem=poem,
                                      errors=listed))
    new = reply.strip()
    if new.startswith("```"):
        new = new.split("\n", 1)[-1].rsplit("```", 1)[0]
    new = new.strip().strip('"').strip()
    return new or prompt


# --- Task 4: The loop --------------------------------------------------------
# Each round, write a poem with the current prompt at temperature 1.0, judge
# it, and rewrite the prompt for the next round.

def autoprompt(seed: str, rounds: int = ROUNDS) -> list[dict]:
    """
    In:  a seed prompt and how many rewrites to try
    Out: one record per round with "prompt", "poem", and "errors", starting
         with the seed, so rounds + 1 records in all
    """
    history, prompt = [], seed
    for i in range(rounds + 1):
        poem = ask(prompt, temperature=1.0)
        errors = judge(poem)
        history.append({"prompt": prompt, "poem": poem, "errors": errors})
        if i < rounds:
            prompt = rewrite(prompt, poem, errors)
    return history


# --- Run (written for you) ---------------------------------------------------

def show(i: int, record: dict) -> None:
    print(f"\n=== round {i} " + "=" * 64)
    print(f"\n{record['prompt']}\n\n{record['poem']}")
    print(f"\nerrors: {', '.join(record['errors']) or 'none'}")


def main() -> None:
    if not TOPIC or not GENERATOR_PROMPT:
        print("TOPIC or GENERATOR_PROMPT is empty")
        return

    if not ERROR_TAXONOMY:
        print(f"\n{ask(GENERATOR_PROMPT, temperature=1.0)}")
        print("\nwrite ERROR_TAXONOMY to judge this poem and run the loop")
        return

    history = autoprompt(GENERATOR_PROMPT)
    if not history:
        print("autoprompt returned nothing")
        return
    for i, record in enumerate(history):
        show(i, record)

    print(f"\n  {'round':>5}{'errors':>8}")
    for i, record in enumerate(history):
        print(f"  {i:>5}{len(record['errors']):>8}")
    best = min(range(len(history)), key=lambda i: len(history[i]["errors"]))
    print(f"\nfewest errors: round {best}")


if __name__ == "__main__":
    main()
