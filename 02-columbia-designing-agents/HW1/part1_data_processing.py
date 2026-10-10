"""
HW1 Part 1 - Building a Text Processing Pipeline

A "record" below always means a dict with keys "review_id", "label", and
"text". Every pipeline stage takes a list of records and returns a list of
records.

    python part1_data_processing.py

Running the file checks load_corpus, scores the rest of your code against
data/test_cases.json, and prints what failed.
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

CORPUS = DATA / "unprocessed_data.jsonl"   # Task 1.0 reads this
PROCESSED = DATA / "processed_data.jsonl"

# Used for test cases
DEFAULT_THRESHOLDS = {"length": 10, "gibberish": 0.2, "repetition": 0.6}


# --- Task 1.0: Load ---------------------------------------------------------
# The corpus is 1,000 restaurant reviews in data/unprocessed_data.jsonl. A JSONL
# file holds one JSON object per line, so it is read a line at a time rather
# than parsed in one go. Each line has the keys "review_id", "label" and "text".
# Use the following documentation to read the file and parse each line:
#
#   json:    https://docs.python.org/3/library/json.html
#   pathlib: https://docs.python.org/3/library/pathlib.html

def load_corpus() -> list[dict]:
    """
    In:  nothing
    Out: list of 1,000 records, keys "review_id" (int), "label" (int 1-5),
         "text" (str)
    """
    return [
        json.loads(line)
        for line in CORPUS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --- Task 1.1: Normalize ----------------------------------------------------
# Use the following documentation to strip markup and standardize
# character encodings:
#
#   HTML entities: https://docs.python.org/3/library/html.html
#   Unicode:       https://docs.python.org/3/library/unicodedata.html

def normalize_text(text: str) -> str:
    """
    In:  raw review text
    Out: the same text, cleaned in this order:
           1. HTML tags replaced by a single space. Replace, do not delete:
              "a<br/>b" becomes "a b".
           2. HTML entities decoded, so "&amp;" becomes "&".
           3. Unicode normalized to NFKC.
           4. Every run of whitespace collapsed to one space, then leading and
              trailing whitespace stripped.
    """
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Task 1.2: Scrub PII ----------------------------------------------------
# Refer to data/test_cases.json, which carries the correct output for some
# examples. The file also contains elements that look like personal information
# but should be left alone. Use the following documentation to help you:
#
#   regex: https://docs.python.org/3/library/re.html

_URL_RE = re.compile(
    r"(?:https?://\S+|www\.\S+|(?<!\S)[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:com|org|net|edu|gov|io|co\.uk|ly)(?:/\S*)?)",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?:\+\d{1,3}(?:[\s.-]\d{2,4}){2,4}\b|(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)[\s.-]?|\d{3}[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d))"
)
_HANDLE_RE = re.compile(r"(?<!\S)@[A-Za-z_][A-Za-z0-9_]*")
_TRAILING_PUNCT = ".,!?;:)]\"'"


def _replace_preserving_trailing(placeholder: str):
    def _repl(match: re.Match) -> str:
        raw = match.group(0)
        stripped = raw.rstrip(_TRAILING_PUNCT)
        suffix = raw[len(stripped):]
        return placeholder + suffix
    return _repl


def scrub_pii(text: str) -> str:
    """
    In:  normalized text
    Out: the same text with emails, URLs, phone numbers, and @handles replaced
         by [EMAIL], [URL], [PHONE], and [HANDLE]

    Punctuation that ends a sentence stays outside the placeholder:
    "visit example.com." becomes "visit [URL].", not "visit [URL]".
    """
    # Replace URLs first so query parameters containing emails (e.g. ?email=sam@example.com)
    # are absorbed into [URL] as a single link.
    text = _URL_RE.sub(_replace_preserving_trailing("[URL]"), text)
    text = _EMAIL_RE.sub(_replace_preserving_trailing("[EMAIL]"), text)
    text = _PHONE_RE.sub(_replace_preserving_trailing("[PHONE]"), text)
    text = _HANDLE_RE.sub(_replace_preserving_trailing("[HANDLE]"), text)
    return text


# --- Task 1.3: Quality filter -----------------------------------------------
# Statistical checks that drop junk documents. One call runs one check over the
# whole corpus. Add helper functions as needed.
#
#   n-grams:            https://en.wikipedia.org/wiki/N-gram
#   Jaccard similarity: https://en.wikipedia.org/wiki/Jaccard_index

_ORDINARY_PUNCT = set(".,!?;:'\"()-[]")


def _gibberish_ratio(text: str) -> float:
    if not text:
        return 1.0
    unusual = sum(
        1 for ch in text
        if not (ch.isalnum() or ch.isspace() or ch in _ORDINARY_PUNCT)
    )
    return unusual / len(text)


def _char_ngrams(text: str, n: int = 5) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    union = len(a | b)
    if union == 0:
        return 1.0
    return len(a & b) / union


def quality_filter(records: list[dict], check: str, threshold: float) -> list[dict]:
    """
    In:  list of records, the name of one check, and that check's threshold
    Out: the records that pass the check, in their original order

    length      how many words the document has. Very short documents carry no
                signal. Rejects documents with fewer words than the threshold.
    gibberish   share of characters that are not letters, digits, or ordinary
                punctuation. Catches leftover markup, symbol spam, and text
                that is not really language. Rejects documents above the
                threshold.
    repetition  how close a document is to one already kept, which catches
                near-copies. Represent each document as the set of its
                character n-grams with n = 5, then score a pair by intersection
                size over union size: the Jaccard similarity. Walking the
                records in order, reject one when it scores above the threshold
                against any record kept so far.
    """
    if check == "length":
        return [r for r in records if len(r["text"].split()) >= threshold]
    if check == "gibberish":
        return [r for r in records if _gibberish_ratio(r["text"]) <= threshold]
    if check == "repetition":
        kept: list[dict] = []
        kept_ngrams: list[set[str]] = []
        for r in records:
            ng = _char_ngrams(r["text"], 5)
            if any(_jaccard(ng, prev) > threshold for prev in kept_ngrams):
                continue
            kept.append(r)
            kept_ngrams.append(ng)
        return kept
    raise ValueError(f"Unknown quality check: {check!r}")


# --- Task 1.4: Pipeline -----------------------------------------------------
# Wrap the cleaning steps as named stages, and record how many records each
# stage drops.

@dataclass
class Stage:
    name: str
    fn: Callable[[list[dict]], list[dict]]


@dataclass
class Pipeline:
    stages: list[Stage] = field(default_factory=list)
    funnel: list[tuple[str, int, int]] = field(default_factory=list)

    def add(self, name: str, fn: Callable[[list[dict]], list[dict]]) -> "Pipeline":
        """
        In:  a stage name, and a function taking a list of records to a list of
             records
        Out: self, so that calls chain
        """
        self.stages.append(Stage(name=name, fn=fn))
        return self

    def run(self, records: list[dict]) -> list[dict]:
        """
        In:  a list of records
        Out: the records left after every stage has run, in the order the
             stages were added

        Also appends one (name, count_in, count_out) tuple to self.funnel per
        stage.
        """
        self.funnel = []
        current = [dict(r) for r in records]
        for stage in self.stages:
            n_in = len(current)
            current = stage.fn(current)
            n_out = len(current)
            self.funnel.append((stage.name, n_in, n_out))
        return current

    def report(self) -> None:
        """Prints the funnel. Written for you; call it while debugging."""
        if not self.funnel:
            print("  (pipeline has not been run)")
            return
        print(f"\n  {'stage':<24} {'in':>6} {'out':>6} {'dropped':>8}")
        for name, n_in, n_out in self.funnel:
            print(f"  {name:<24} {n_in:>6} {n_out:>6} {n_in - n_out:>8}")


# --- Task 1.5: Assemble -----------------------------------------------------

def build_pipeline(thresholds: dict[str, float]) -> Pipeline:
    """
    In:  a dict mapping each check name to the threshold it should use
    Out: a Pipeline with five stages, added in this order and under these
         names: "normalize", "scrub_pii", "length", "gibberish", "repetition"

    The first two stages should rewrite every record's text. The last three
    should drop records by calling quality_filter with the matching threshold.
    """
    pipeline = Pipeline()
    pipeline.add(
        "normalize",
        lambda recs: [{**r, "text": normalize_text(r["text"])} for r in recs],
    )
    pipeline.add(
        "scrub_pii",
        lambda recs: [{**r, "text": scrub_pii(r["text"])} for r in recs],
    )
    pipeline.add(
        "length",
        lambda recs: quality_filter(recs, "length", thresholds["length"]),
    )
    pipeline.add(
        "gibberish",
        lambda recs: quality_filter(recs, "gibberish", thresholds["gibberish"]),
    )
    pipeline.add(
        "repetition",
        lambda recs: quality_filter(recs, "repetition", thresholds["repetition"]),
    )
    return pipeline


# --- Test (written for you) ------------------------------------------------

def main() -> None:
    """Checks load_corpus, then runs data/test_cases.json through your code."""
    corpus = load_corpus()
    if len(corpus) != 1000:
        print(f"  load_corpus returned {len(corpus)} records, expected 1000")
    elif sorted(corpus[0]) != ["label", "review_id", "text"]:
        print(f"  load_corpus records have keys {sorted(corpus[0])}, "
              f"expected ['label', 'review_id', 'text']")
    else:
        print(f"  load_corpus returned {len(corpus)} records")

    cases = json.loads((DATA / "test_cases.json").read_text(encoding="utf-8"))
    records = [{"review_id": i, "label": 3, "text": c["text"]} for i, c in enumerate(cases)]
    kept = {r["review_id"] for r in build_pipeline(DEFAULT_THRESHOLDS).run(records)}

    failed = 0
    for i, case in enumerate(cases):
        problems = []
        got = scrub_pii(normalize_text(case["text"]))
        if got != case["expected_text"]:
            problems.append(f"got {got[:60]!r}")
        if (i in kept) != case["keep"]:
            problems.append(f"expected {'keep' if case['keep'] else case['reason']}")
        if problems:
            failed += 1
            print(f"  {case['id']}  {case['tests']:<20} {'; '.join(problems)}")
    print(f"{len(cases) - failed} / {len(cases)} test cases passed")

    # Also run the full pipeline on the 1,000-review corpus and save processed_data.jsonl
    corpus_pipeline = build_pipeline(DEFAULT_THRESHOLDS)
    cleaned_corpus = corpus_pipeline.run(corpus)
    corpus_pipeline.report()
    PROCESSED.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in cleaned_corpus) + "\n",
        encoding="utf-8",
    )
    print(f"\n  Saved {len(cleaned_corpus)} cleaned records to {PROCESSED.relative_to(HERE)}")


if __name__ == "__main__":
    main()
