"""
HW2 Part 4 - Memory

data/conversation.json contains 22 turns of a conversation between a guest
planning a visit to Fossil Falls and the park's help desk. Both sides of the
conversation are already written. Early in the conversation, the guest gives
details, such as a booking number, that are needed later. Once a conversation
is too long to fit in the model's context, those details need to be kept some
other way. You will build a memory with three layers:

    1. a buffer that keeps the last few turns word for word
    2. a running summary of all older turns
    3. a store of individual facts that can be searched

main() adds every turn of the conversation to your memory, in order, using
buffer sizes of 2, 4, and 8 turns. No new replies are generated. main() then
prints the context each memory builds for NEXT_MESSAGE, the guest's next
message. By the end of the conversation, a buffer of 2 has dropped all of the
early turns, so any early detail in its context came from the summary or the
fact store.

A "turn" is a dict with keys "role" and "content".

    python part4_memory.py

"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from part2_prompting import chat
from part3_retrieval import cosine_top_k, embed

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

SCRIPT = DATA / "conversation.json"     # load_script reads this

BUFFER_SIZES = (2, 4, 8)
SUMMARY_WORDS = 40     # Task 4.1 keeps the summary under this many words
NEXT_MESSAGE = "Before I go, can you read back the details of my visit?"


def count_tokens(text: str) -> int:
    """Approximate token count (written for you). main() uses it to report
    the size of each context."""
    return int(len(text.split()) * 4 / 3) + 1


# --- Task 4.0: The buffer ----------------------------------------------------
# The buffer keeps the most recent turns of the conversation word for word.

class Buffer:
    """Stores the last n_turns turns word for word."""

    def __init__(self, n_turns: int):
        """
        In:  how many turns to keep
        Out: nothing
        """
        self.n_turns = max(int(n_turns), 0)
        self.turns: list[dict] = []

    def add(self, role: str, content: str) -> list[dict]:
        """
        In:  a role and the turn's content
        Out: the turns removed from the buffer to make room for this turn,
             oldest first, or [] if no turns were removed

        In Task 4.3, these removed turns are passed to the summary.
        """
        self.turns.append({"role": role, "content": content})
        if self.n_turns == 0:
            evicted = list(self.turns)
            self.turns = []
            return evicted
        if len(self.turns) > self.n_turns:
            overflow = len(self.turns) - self.n_turns
            evicted = self.turns[:overflow]
            self.turns = self.turns[overflow:]
            return evicted
        return []

    def render(self) -> str:
        """
        In:  nothing
        Out: the turns currently in the buffer, as text
        """
        return "\n".join(f"{t['role']}: {t['content']}" for t in self.turns)


BufferMemory = Buffer


# --- Task 4.1: The summary ---------------------------------------------------
# The summary is a short paragraph covering every turn the buffer has dropped.

class Summary:
    """Keeps a running summary of the turns dropped from the buffer."""

    def __init__(self, max_words: int | None = None):
        """
        In:  nothing
        Out: nothing
        """
        self.text: str = ""
        self.max_words = max_words

    def update(self, evicted: list[dict]) -> None:
        """
        In:  the turns that were removed from the buffer
        Out: nothing

        Ask the model to write a new summary from the old summary and the
        evicted turns, in fewer than SUMMARY_WORDS words. This keeps the
        summary short no matter how long the conversation runs. If evicted is
        empty, do nothing.
        """
        if not evicted:
            return
        limit = self.max_words if self.max_words is not None else SUMMARY_WORDS
        evicted_text = "\n".join(f"{t['role']}: {t['content']}" for t in evicted)
        prompt = (
            f"Update the running conversation summary with the newly evicted turns in fewer than {limit} words.\n\n"
            f"Current summary:\n{self.text or '(none)'}\n\n"
            f"Evicted turns:\n{evicted_text}\n\n"
            f"Write the updated summary in fewer than {limit} words:"
        )
        reply = chat([{"role": "user", "content": prompt}], temperature=0.0).strip()
        words = reply.split()
        if len(words) >= limit:
            reply = " ".join(words[: max(limit - 1, 1)])
        self.text = reply

    def render(self) -> str:
        """
        In:  nothing
        Out: the current summary, or "" if no turns have been added to it yet
        """
        return self.text


SummaryMemory = Summary


# --- Task 4.2: The fact store ------------------------------------------------
# Old turns are dropped from the buffer, and the summary may leave out details
# when it compresses them. Implement a fact store that keeps individual facts
# and searches them with embed and cosine_top_k from Part 3.
#
# For example, "My booking number is 88431" is a durable fact that should be
# kept. A guest asking "Is the monorail running?" should not be kept.

def extract_facts(turn: dict) -> list[str]:
    """
    In:  one turn
    Out: the durable facts in the turn, each written as a sentence that makes
         sense on its own, or [] if the turn has none

    Ask the model to list the facts, one per line.
    """
    prompt = (
        "Extract any durable facts (such as booking numbers, family members, passes, "
        "dietary needs, or specific park rules explained) from this conversation turn. "
        "Write each durable fact as a standalone sentence, one per line. "
        "If there are no durable facts, reply NONE.\n\n"
        f"{turn['role']}: {turn['content']}"
    )
    reply = chat([{"role": "user", "content": prompt}], temperature=0.0).strip()
    if not reply or reply.upper() == "NONE":
        return []
    facts = []
    for line in reply.splitlines():
        cleaned = line.strip().lstrip("-*•0123456789. ").strip()
        if cleaned and cleaned.upper() != "NONE":
            facts.append(cleaned)
    return facts


class FactStore:
    """Stores facts extracted from the conversation and searches them by
    meaning."""

    def __init__(self):
        """
        In:  nothing
        Out: nothing
        """
        self.facts: list[str] = []
        self.embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def add(self, turn: dict) -> list[str]:
        """
        In:  one turn
        Out: the facts stored from this turn

        Embed each fact once, when it is added.
        """
        new_facts = extract_facts(turn)
        if not new_facts:
            return []
        new_vecs = embed(new_facts)
        self.facts.extend(new_facts)
        if self.embeddings.size == 0:
            self.embeddings = new_vecs
        else:
            self.embeddings = np.vstack([self.embeddings, new_vecs])
        return new_facts

    def search(self, query: str, k: int = 3) -> list[str]:
        """
        In:  a query and how many facts to return
        Out: the k stored facts most similar to the query, most similar first,
             or [] if no facts are stored
        """
        if not self.facts or k <= 0:
            return []
        q_vec = embed([query])[0]
        top = cosine_top_k(q_vec, self.embeddings, min(k, len(self.facts)))
        return [self.facts[idx] for idx, _ in top]


FactMemory = FactStore


# --- Task 4.3: Compose -------------------------------------------------------
# The Memory class keeps the three layers updated as turns are added, and
# combines them into the context the model sees before it answers.

class Memory:
    """Combines a buffer, a summary, and a fact store, and updates all three
    with each turn."""

    def __init__(self, n_turns: int):
        """
        In:  how many turns the buffer keeps
        Out: nothing

        Create one Buffer, one Summary, and one FactStore.
        """
        self.buffer = Buffer(n_turns)
        self.summary = Summary()
        self.facts = FactStore()

    def add(self, role: str, content: str) -> None:
        """
        In:  a role and the turn's content
        Out: nothing

        Send the turn to all three layers, and pass any turns removed from the
        buffer to the summary.
        """
        turn = {"role": role, "content": content}
        evicted = self.buffer.add(role, content)
        if evicted:
            self.summary.update(evicted)
        self.facts.add(turn)

    def build_context(self, query: str, k_facts: int = 5) -> str:
        """
        In:  the question about to be asked
        Out: the context to give the model, built from all three layers

        Search the fact store with the query. Label each part of the context
        with the layer it came from.
        """
        summary_text = self.summary.render() or "(none)"
        retrieved_facts = self.facts.search(query, k=k_facts)
        facts_text = "\n".join(f"- {f}" for f in retrieved_facts) if retrieved_facts else "(none)"
        buffer_text = self.buffer.render() or "(none)"
        return (
            f"[Summary]\n{summary_text}\n\n"
            f"[Facts]\n{facts_text}\n\n"
            f"[Buffer]\n{buffer_text}"
        )


MemoryAgent = Memory


# --- Task 4.4: Replay --------------------------------------------------------

def load_script(path: Path = SCRIPT) -> list[dict]:
    """
    In:  a path to conversation.json
    Out: the list of turns
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "turns" in data:
        return data["turns"]
    return data


def run_conversation(script: list[dict], n_turns: int) -> Memory:
    """
    In:  the scripted turns and a buffer size
    Out: the Memory after every turn has been added

    Add every turn to a new Memory, in order.
    """
    mem = Memory(n_turns)
    for turn in script:
        mem.add(turn["role"], turn["content"])
    return mem


# --- Test (written for you) --------------------------------------------------
# Adds the conversation to a new memory once for each buffer size, and prints
# the context each memory builds for NEXT_MESSAGE.

def main() -> None:
    turns = load_script()
    if not turns:
        print("load_script returned nothing, so there is nothing to replay yet.")
        return

    print(f"\n{len(turns)} turns. Next message: {NEXT_MESSAGE!r}")
    for n_turns in BUFFER_SIZES:
        context = run_conversation(turns, n_turns).build_context(NEXT_MESSAGE)
        print(f"\n=== buffer {n_turns}, {count_tokens(context)} tokens "
              + "=" * 40 + f"\n\n{context}")


if __name__ == "__main__":
    main()
