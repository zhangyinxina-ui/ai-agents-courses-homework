"""
HW1 Part 2 - Tokenization

We compare three pretrained tokenizers against one you train yourself. `gpt2`
uses byte-level BPE, `bert` uses WordPiece, and `t5` uses SentencePiece Unigram.

    pip install transformers tokenizers
    python part2_tokenizing.py

"""

from __future__ import annotations

import os
if os.environ.get("HF_ENDPOINT") == "https://hf-mirror.com":
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer

TOKENIZERS = {
    "gpt2": "gpt2",
    "bert": "bert-base-multilingual-cased",
    "t5":   "t5-small",
}

# The strings every tokenizer is measured on.
SENTENCES = {
    "english": "The restaurant on the corner serves an excellent breakfast every single morning.",
    "code":    "def scrub(text):\n    return re.sub(r'\\S+@\\S+', '[EMAIL]', text)",
    "numbers": "Order 8675309 totaled $1,234.56 on 2024-11-03 at 14:27:45 UTC.",
    "spanish": "El restaurante de la esquina sirve un desayuno excelente todas las mañanas.",
    "chinese": "街角的餐厅每天早上都提供非常好的早餐。",
    "emoji":   "the tacos were 🔥🔥🔥 but the service was 😤 and we left 👎",
}

# The text your own tokenizer is trained on in Task 2.3.
BPE_CORPUS = [
    "The pasta was excellent and the service was attentive from start to finish.",
    "We waited forty minutes for a table we had already reserved.",
    "Their handmade dumplings are the reason to come here, and the chili oil is worth it.",
    "The dining room is loud enough that you will be leaning in to talk all night.",
    "I ordered the sea bass and it arrived dry around the edges, which was a shame.",
    "Breakfast is the best value in the neighborhood, especially the fried egg sandwich.",
    "Our server recommended the short rib and she was completely right about it.",
    "The coffee tasted burnt and nobody checked on us after the food came out.",
    "Small room, twelve seats, and a menu that changes every single week.",
    "They comped the appetizer after the delay, which I thought was a nice touch.",
    "The tacos were fine but the salsa had real heat and I would come back for that.",
    "Prices have crept up since last year and the portions have not kept pace.",
    "Everything arrived lukewarm, as though it had been sitting under a lamp.",
    "The patio out back is the best seat in the house when the weather holds.",
    "Dessert was the highlight, a chocolate tart that was not too sweet.",
    "It is a solid neighborhood spot that knows exactly what it is trying to be.",
]


# --- Task 2.0: Load the tokenizers -------------------------------------------
# Each tokenizer is downloaded from Hugging Face by name and cached on disk. Use
# the following documentation to load one from its name:
#
#   tokenizers:   https://huggingface.co/docs/transformers/main_classes/tokenizer
#   auto classes: https://huggingface.co/docs/transformers/model_doc/auto

def load_tokenizers() -> dict:
    """
    In:  nothing
    Out: a dict mapping each name in TOKENIZERS to its loaded tokenizer
    """
    return {
        name: AutoTokenizer.from_pretrained(model_name)
        for name, model_name in TOKENIZERS.items()
    }


# --- Task 2.1: Tokens per word -----------------------------------------------
# A tokenizer turns a string into a list of integer ids, and the number of ids
# it spends on the same text varies from tokenizer to tokenizer. Every tokenizer
# can also add special tokens that are not part of the text, so keep those out
# of every count. Use the following documentation to encode a string:
#
#   tokenizers: https://huggingface.co/docs/transformers/main_classes/tokenizer

def tokens_per_word(tokenizer, text: str) -> float:
    """
    In:  a tokenizer and a non-empty string
    Out: number of tokens divided by number of whitespace-delimited words

    Do not let the tokenizer add its special tokens.
    """
    words = text.split()
    if not words:
        return 0.0
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return len(token_ids) / len(words)


# --- Task 2.2: Bytes per token -----------------------------------------------
# Chinese is written without spaces, so Task 2.1 counts the whole Chinese
# sentence as a single word. A fairer measure should count bytes rather than
# words, since every language has bytes. Use the following documentation to get
# the byte length of a string:
#
#   encoding: https://docs.python.org/3/library/stdtypes.html#str.encode

def bytes_per_token(tokenizer, text: str) -> float:
    """
    In:  a tokenizer and a non-empty string
    Out: length of the text in UTF-8 bytes divided by the number of tokens

    Again, no special tokens.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return 0.0
    return len(text.encode("utf-8")) / len(token_ids)


# --- Task 2.3: Round-trip fidelity -------------------------------------------
# Tokenizers should not lose any information from the input while tokenizing.
# One way to check this is to see whether the input text survives a "round
# trip". Use the following documentation to tokenize the input, then turn the
# token ids back into a string:
#
#   tokenizers: https://huggingface.co/docs/transformers/main_classes/tokenizer

def round_trips(tokenizer, text: str) -> bool:
    """
    In:  a tokenizer and a string
    Out: True if encoding the text and then decoding the result returns the
         original string exactly, and False otherwise

    Again, no special tokens.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(token_ids) == text


# --- Task 2.4: Byte pair encoding --------------------------------------------
# BPE splits each word into characters, counts adjacent pairs, merges the most
# frequent pair everywhere, and repeats. Learning the merges and applying them
# are two separate steps, so you can learn them once and then apply them to any
# word, including words the corpus never contained.
#
# The first two functions below are your own character-level version. The third
# uses a library to train a byte-level one, which can represent any character.
# Use the following documentation:
#
#   the original paper: https://arxiv.org/abs/1508.07909
#   training:           https://huggingface.co/docs/tokenizers/quicktour
#   components:         https://huggingface.co/docs/tokenizers/components

def _merge_pair_in_seq(seq: list[str], pair: tuple[str, str]) -> list[str]:
    a, b = pair
    merged = a + b
    out: list[str] = []
    i = 0
    n = len(seq)
    while i < n:
        if i + 1 < n and seq[i] == a and seq[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def learn_merges(words: list[str], num_merges: int) -> list[tuple[str, str]]:
    """
    In:  a list of words, and how many merges to learn
    Out: the merges in the order they were learned, each a pair of symbols

    Break ties by taking the pair that sorts first. Stop early if no pair
    occurs more than once. `words` is a list, not a set, so a word appearing
    five times counts five times.
    """
    seqs = [list(w) for w in words]
    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        counts: dict[tuple[str, str], int] = {}
        for seq in seqs:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        best_pair, best_count = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if best_count <= 1:
            break
        merges.append(best_pair)
        seqs = [_merge_pair_in_seq(seq, best_pair) for seq in seqs]
    return merges


def apply_merges(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """
    In:  a word, and the merges returned by learn_merges
    Out: the word split into symbols, with the merges replayed in order

    Must also work on words that were never in the training data.
    """
    seq = list(word)
    for pair in merges:
        seq = _merge_pair_in_seq(seq, pair)
    return seq


def train_bpe(texts: list[str], vocab_size: int):
    """
    In:  a list of strings and a target vocabulary size
    Out: a trained byte-level BPE tokenizers.Tokenizer

    Encoding a string and then decoding the result must return the original
    string, for any string.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


# --- Test (written for you) ---------------------------------
# Prints tokens per word, bytes per token, and round-trip fidelity for every
# tokenizer, including the one you trained.

class _Trained:
    """
    Adapts a tokenizers.Tokenizer so it takes the same calls as the pretrained
    ones above. Tasks 2.1 to 2.3 then work on your tokenizer unchanged.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)


def main() -> None:
    tokenizers = load_tokenizers()
    if not tokenizers:
        print("load_tokenizers returned nothing, so there is nothing to compare yet.")
        return

    # Each task is reported as soon as it works, so you can run this after any
    # one of them rather than having to finish all of them first.
    try:
        trained = train_bpe(BPE_CORPUS, 1000)
        tokenizers["bpe"] = _Trained(trained)
        print(f"\ntrained a vocabulary of {trained.get_vocab_size()}, asked for 1000")
    except NotImplementedError:
        print("\ntrain_bpe is not written yet, so the bpe column is missing.")

    names = list(tokenizers)

    for title, measure, cell in (
        ("tokens per word", tokens_per_word, lambda v: f"{v:>9.2f}"),
        ("bytes per token", bytes_per_token, lambda v: f"{v:>9.2f}"),
        ("round trips", round_trips, lambda v: f"{'yes' if v else 'no':>9}"),
    ):
        print(f"\n{title}")
        print("  " + f"{'sentence':<10}" + "".join(f"{n:>9}" for n in names))
        for sentence, text in SENTENCES.items():
            row = "".join(cell(measure(tokenizers[n], text)) for n in names)
            print(f"  {sentence:<10}{row}")

    # Task 2.4's first two functions are not used by the tables above, so they
    # get their own check here.
    words = [w for text in BPE_CORPUS for w in text.split()]
    merges = learn_merges(words, 50)
    print(f"\nlearn_merges found {len(merges)} merges on BPE_CORPUS")
    print(f"  first five: {merges[:5]}")
    print(f"  apply_merges on a word the corpus never had: "
          f"{apply_merges('unremarkable', merges)}")


if __name__ == "__main__":
    main()
