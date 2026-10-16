"""Task (i): dataset acquisition, characterisation, preprocessing and
construction of the input-target sequences for autoregressive modelling."""
from __future__ import annotations

import json
import re
import unicodedata
import urllib.request
from collections import Counter

import numpy as np

from config import (
    CORPUS_FILE,
    DATASET_URL,
    MIN_WORD_FREQ,
    METRICS_DIR,
    PAD_TOKEN,
    RAW_FILE,
    SEED,
    SEQUENCE_LENGTH,
    TRAIN_FRACTION,
    UNK_TOKEN,
    VAL_FRACTION,
)

# Project Gutenberg wraps every book in a licence header and footer. These are
# the official machine-readable markers used to strip them.
START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"

# A word is a run of letters, optionally with an internal apostrophe
# ("don't", "alice's"). Sentence punctuation is kept as its own token so the
# generated text has readable sentence boundaries.
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)*|[.,!?;:]")


# --------------------------------------------------------------------------- #
# 1. Acquisition
# --------------------------------------------------------------------------- #
def download_raw(force: bool = False) -> str:
    """Download the raw UTF-8 eBook from Project Gutenberg (cached on disk)."""
    if RAW_FILE.exists() and not force:
        return RAW_FILE.read_text(encoding="utf-8")

    request = urllib.request.Request(
        DATASET_URL,
        headers={
            "User-Agent": "ICT4403-Assignment2/1.0 (academic coursework; text-generation experiment)"
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw_bytes = response.read()

    text = raw_bytes.decode("utf-8-sig")
    RAW_FILE.write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------- #
# 2. Preprocessing
# --------------------------------------------------------------------------- #
def strip_gutenberg_boilerplate(text: str) -> str:
    """Keep only the body of the book, discarding licence header and footer."""
    start = text.find(START_MARKER)
    if start != -1:
        start = text.find("\n", start) + 1
    else:
        start = 0

    end = text.find(END_MARKER)
    if end == -1:
        end = len(text)
    return text[start:end]


def clean_text(text: str) -> str:
    """Normalise unicode, lower-case, and remove non-narrative artefacts."""
    text = unicodedata.normalize("NFKC", text)
    # Curly quotes / dashes / ellipses -> plain ASCII equivalents.
    replacements = {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "—": " ",
        "–": " ",
        "…": " ",
        " ": " ",
        "_": "",  # Gutenberg marks italics with underscores
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.lower()
    text = re.sub(r"\[illustration[^\]]*\]", " ", text)  # image placeholders
    text = re.sub(r"\s+", " ", text)
    return text.strip()


#: The chapter-one heading, as it appears *after* tokenisation. It occurs twice:
#: once in the table of contents and once where the narrative actually starts.
CHAPTER_ONE_HEADING = ["chapter", "i", ".", "down", "the", "rabbit", "hole"]


def strip_front_and_back_matter(tokens: list) -> list:
    """Drop the title block / table of contents and the closing "the end".

    The table of contents repeats all twelve chapter headings, which would
    otherwise teach the models a list-like token pattern that never occurs in
    the narrative itself. The narrative proper begins at the *last* occurrence
    of the chapter-one heading. This runs on the token list rather than the raw
    string because the source spells the heading "CHAPTER I.\nDown the
    Rabbit-Hole", which no single literal pattern matches reliably.
    """
    width = len(CHAPTER_ONE_HEADING)
    starts = [
        i for i in range(len(tokens) - width)
        if tokens[i : i + width] == CHAPTER_ONE_HEADING
    ]
    if len(starts) > 1:
        tokens = tokens[starts[-1] :]
    if tokens[-2:] == ["the", "end"]:
        tokens = tokens[:-2]
    return tokens


def tokenize(text: str) -> list:
    """Word-level tokenisation; sentence punctuation survives as tokens."""
    return TOKEN_RE.findall(text)


def build_vocabulary(tokens: list, min_freq: int = MIN_WORD_FREQ):
    """Index 0 is <pad>, index 1 is <unk>; the rest are frequency-ordered."""
    counts = Counter(tokens)
    kept = [w for w, c in counts.most_common() if c >= min_freq]
    itos = [PAD_TOKEN, UNK_TOKEN] + kept
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos, counts


def encode(tokens: list, stoi: dict) -> np.ndarray:
    unk = stoi[UNK_TOKEN]
    return np.array([stoi.get(t, unk) for t in tokens], dtype=np.int32)


# --------------------------------------------------------------------------- #
# 3. Input -> target sequences for autoregressive modelling
# --------------------------------------------------------------------------- #
def make_sequences(ids: np.ndarray, seq_len: int = SEQUENCE_LENGTH):
    """Sliding window with stride 1.

    X[i] = ids[i : i + seq_len]      (the context the model may look at)
    y[i] = ids[i + seq_len]          (the next word it must predict)

    This is the standard autoregressive factorisation
    p(w_1..w_T) = prod_t p(w_t | w_{t-seq_len} .. w_{t-1}).
    """
    n = len(ids) - seq_len
    if n <= 0:
        raise ValueError("corpus shorter than the sequence length")
    X = np.lib.stride_tricks.sliding_window_view(ids[:-1], seq_len)[:n].copy()
    y = ids[seq_len : seq_len + n].copy()
    return X, y


def contiguous_split(X: np.ndarray, y: np.ndarray):
    """Split into train/val/test as three contiguous blocks.

    A random split would be wrong here: consecutive windows share 19 of their
    20 tokens, so shuffling before splitting leaks near-duplicate contexts into
    validation and test and inflates every score.
    """
    n = len(X)
    n_train = int(n * TRAIN_FRACTION)
    n_val = int(n * VAL_FRACTION)
    sl_train = slice(0, n_train)
    sl_val = slice(n_train, n_train + n_val)
    sl_test = slice(n_train + n_val, n)
    return (
        (X[sl_train], y[sl_train]),
        (X[sl_val], y[sl_val]),
        (X[sl_test], y[sl_test]),
    )


# --------------------------------------------------------------------------- #
# 4. Dataset characterisation (Task i)
# --------------------------------------------------------------------------- #
def describe_dataset(raw_text, body, tokens, counts, itos, splits) -> dict:
    (Xtr, _), (Xva, _), (Xte, _) = splits
    total = len(tokens)
    unique = len(counts)
    hapax = sum(1 for c in counts.values() if c == 1)
    vocab_set = set(itos)
    covered = sum(c for w, c in counts.items() if w in vocab_set)
    words_only = [t for t in tokens if t[0].isalpha()]
    sentences = sum(counts[p] for p in (".", "!", "?"))

    return {
        "dataset_name": "Alice's Adventures in Wonderland (Project Gutenberg eBook #11)",
        "source_url": DATASET_URL,
        "raw_file_characters": len(raw_text),
        "body_characters_after_boilerplate_removal": len(body),
        "total_tokens": total,
        "total_word_tokens_excluding_punctuation": len(words_only),
        "unique_tokens_before_min_freq_filter": unique,
        "hapax_legomena_count": hapax,
        "hapax_legomena_pct_of_vocab": round(100 * hapax / unique, 2),
        "min_word_frequency": MIN_WORD_FREQ,
        "final_vocabulary_size_incl_pad_unk": len(itos),
        "token_coverage_by_final_vocab_pct": round(100 * covered / total, 2),
        "type_token_ratio": round(unique / total, 5),
        "approx_sentence_count": sentences,
        "mean_tokens_per_sentence": round(total / max(sentences, 1), 2),
        "sequence_length": SEQUENCE_LENGTH,
        "total_sequences": int(len(Xtr) + len(Xva) + len(Xte)),
        "train_sequences": int(len(Xtr)),
        "val_sequences": int(len(Xva)),
        "test_sequences": int(len(Xte)),
        "split_strategy": "contiguous 80/10/10 (no shuffling before split)",
        "top_15_tokens": [[w, int(c)] for w, c in counts.most_common(15)],
    }


# --------------------------------------------------------------------------- #
# Pipeline entry point
# --------------------------------------------------------------------------- #
def prepare(verbose: bool = True):
    np.random.seed(SEED)

    raw_text = download_raw()
    body = strip_gutenberg_boilerplate(raw_text)
    cleaned = clean_text(body)
    tokens = strip_front_and_back_matter(tokenize(cleaned))
    # The cleaned corpus is saved as the whitespace-joined token stream, so the
    # file on disk is exactly what the models consume.
    CORPUS_FILE.write_text(" ".join(tokens), encoding="utf-8")
    stoi, itos, counts = build_vocabulary(tokens)
    ids = encode(tokens, stoi)
    X, y = make_sequences(ids)
    splits = contiguous_split(X, y)

    stats = describe_dataset(raw_text, body, tokens, counts, itos, splits)
    (METRICS_DIR / "dataset_statistics.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    if verbose:
        for key, value in stats.items():
            if key != "top_15_tokens":
                print(f"  {key:<48} {value}")
        print(
            "  top tokens:",
            ", ".join(f"{w}({c})" for w, c in stats["top_15_tokens"][:10]),
        )

    return {
        "tokens": tokens,
        "stoi": stoi,
        "itos": itos,
        "counts": counts,
        "ids": ids,
        "splits": splits,
        "stats": stats,
    }


if __name__ == "__main__":
    prepare()
