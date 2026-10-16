"""Task (iii): autoregressive generation with temperature sampling, plus the
automatic text-quality measures used for the qualitative comparison.

Temperature re-shapes the next-word distribution before a word is drawn:

    p_i(T) = p_i^(1/T) / sum_j p_j^(1/T)

T < 1 sharpens the distribution (greedier, safer, more repetitive); T = 1 samples
from the model's own distribution; T > 1 flattens it (more surprising, more
diverse, less coherent). Computing it in log-space keeps it numerically stable
for small T.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from config import GENERATE_WORDS, PAD_TOKEN, SEQUENCE_LENGTH, UNK_TOKEN

# Tokens that exist only for bookkeeping must never be emitted into the text.
BLOCKED_TOKENS = (PAD_TOKEN, UNK_TOKEN)
PUNCTUATION = {".", ",", "!", "?", ";", ":"}


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Re-scale a probability vector by `temperature` (stable in log-space)."""
    probs = np.asarray(probs, dtype=np.float64)
    if temperature <= 0:
        sharpened = np.zeros_like(probs)
        sharpened[int(np.argmax(probs))] = 1.0
        return sharpened

    logits = np.log(np.clip(probs, 1e-12, None)) / temperature
    logits -= logits.max()
    exponentiated = np.exp(logits)
    return exponentiated / exponentiated.sum()


def generate_text(
    model,
    seed_text: str,
    stoi: dict,
    itos: list,
    n_words: int = GENERATE_WORDS,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple:
    """Generate `n_words` tokens autoregressively from `seed_text`.

    Returns (generated_tokens, full_tokens) where `full_tokens` includes the
    seed. The context window is re-encoded from scratch at every step, which is
    what makes the bidirectional model legitimate here: it only ever reads words
    that have already been generated or supplied.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    blocked = [stoi[t] for t in BLOCKED_TOKENS if t in stoi]
    unk = stoi[UNK_TOKEN]
    pad = stoi[PAD_TOKEN]

    window = [stoi.get(tok, unk) for tok in seed_text.split()]
    # Keep the most recent SEQUENCE_LENGTH tokens; left-pad a short seed.
    window = window[-SEQUENCE_LENGTH:]
    if len(window) < SEQUENCE_LENGTH:
        window = [pad] * (SEQUENCE_LENGTH - len(window)) + window

    generated: list[str] = []
    for _ in range(n_words):
        context = np.asarray([window], dtype=np.int32)
        # np.array(..., copy=True): the tensor Keras returns is read-only.
        probs = np.array(model(context, training=False)[0], dtype=np.float64, copy=True)

        probs[blocked] = 0.0          # never emit <pad>/<unk>
        total = probs.sum()
        probs = probs / total if total > 0 else np.full_like(probs, 1.0 / len(probs))

        scaled = apply_temperature(probs, temperature)
        next_id = int(rng.choice(len(scaled), p=scaled))

        generated.append(itos[next_id])
        window = window[1:] + [next_id]

    seed_tokens = seed_text.split()
    return generated, seed_tokens + generated


def detokenize(tokens: list) -> str:
    """Join tokens into readable prose (punctuation hugs the previous word)."""
    out = ""
    for token in tokens:
        if token in PUNCTUATION:
            out += token
        else:
            out += (" " if out else "") + token
    return re.sub(r"\s+", " ", out).strip()


# --------------------------------------------------------------------------- #
# Automatic text-quality measures (the qualitative half of Task ii / iii)
# --------------------------------------------------------------------------- #
def _ngrams(tokens: list, n: int) -> list:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(tokens: list, n: int) -> float:
    """Distinct-n: unique n-grams / total n-grams. Higher = more diverse."""
    grams = _ngrams(tokens, n)
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def repeated_ngram_rate(tokens: list, n: int = 4) -> float:
    """Fraction of n-grams that occur more than once. Higher = more repetitive."""
    grams = _ngrams(tokens, n)
    if not grams:
        return 0.0
    counts = Counter(grams)
    return sum(c for g, c in counts.items() if c > 1) / len(grams)


def longest_token_run(tokens: list) -> int:
    """Length of the longest run of one token repeated back to back."""
    best = run = 1 if tokens else 0
    for previous, current in zip(tokens, tokens[1:]):
        run = run + 1 if current == previous else 1
        best = max(best, run)
    return best


def corpus_ngram_overlap(tokens: list, corpus_ngram_set: set, n: int = 2) -> float:
    """Fraction of generated n-grams that also occur in the training corpus.

    Read as a fluency / contextual-consistency proxy: a value near 0 means the
    model is stringing together word pairs the author never wrote, while a value
    near 1 means it is largely reciting memorised text.
    """
    grams = _ngrams(tokens, n)
    if not grams:
        return 0.0
    return sum(1 for g in grams if g in corpus_ngram_set) / len(grams)


def sentence_lengths(tokens: list) -> list:
    """Token counts of the sentences delimited by . ! ?"""
    lengths, current = [], 0
    for token in tokens:
        if token in {".", "!", "?"}:
            if current:
                lengths.append(current)
            current = 0
        else:
            current += 1
    if current:
        lengths.append(current)
    return lengths


def text_quality(tokens: list, corpus_bigrams: set, corpus_trigrams: set) -> dict:
    """Bundle every automatic measure for one generated passage."""
    lengths = sentence_lengths(tokens)
    return {
        "tokens_generated": len(tokens),
        "distinct_1": round(distinct_n(tokens, 1), 4),
        "distinct_2": round(distinct_n(tokens, 2), 4),
        "repeated_4gram_rate": round(repeated_ngram_rate(tokens, 4), 4),
        "longest_token_run": longest_token_run(tokens),
        "corpus_bigram_overlap": round(corpus_ngram_overlap(tokens, corpus_bigrams, 2), 4),
        "corpus_trigram_overlap": round(corpus_ngram_overlap(tokens, corpus_trigrams, 3), 4),
        "sentences": len(lengths),
        "mean_sentence_length": round(float(np.mean(lengths)), 2) if lengths else 0.0,
    }


def build_corpus_ngram_sets(tokens: list):
    """Pre-compute the reference n-gram sets once, for the overlap measures."""
    return set(_ngrams(tokens, 2)), set(_ngrams(tokens, 3))
