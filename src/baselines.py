"""Classical baselines on the same test split, for context on the neural scores.

A next-word accuracy of ~19% looks low until it is placed against what is
actually achievable. This module scores four non-neural predictors on exactly
the same held-out positions the neural models are evaluated on, and reports how
ambiguous the task really is.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from config import METRICS_DIR


def evaluate_baselines(prepared: dict) -> pd.DataFrame:
    """Score uniform / majority / bigram / trigram predictors on the test split."""
    itos = prepared["itos"]
    (x_train, y_train), _, (x_test, y_test) = prepared["splits"]
    vocab_size = len(itos)

    rows = []

    # 1. Uniform random guess over the vocabulary.
    rows.append({
        "predictor": "Uniform random guess",
        "test_accuracy": 1.0 / vocab_size,
        "note": f"1 in {vocab_size}",
    })

    # 2. Majority class: always emit the single most frequent training target.
    majority_id, _ = Counter(y_train.tolist()).most_common(1)[0]
    rows.append({
        "predictor": "Always the commonest token",
        "test_accuracy": float((y_test == majority_id).mean()),
        "note": f"always predicts {itos[majority_id]!r}",
    })

    # 3. Bigram: the most likely word given the single preceding word.
    bigram = defaultdict(Counter)
    for context, target in zip(x_train, y_train):
        bigram[int(context[-1])][int(target)] += 1
    best_after_one = {prev: c.most_common(1)[0][0] for prev, c in bigram.items()}
    bigram_pred = np.array(
        [best_after_one.get(int(c[-1]), majority_id) for c in x_test]
    )
    rows.append({
        "predictor": "Bigram (most likely next word)",
        "test_accuracy": float((bigram_pred == y_test).mean()),
        "note": "conditions on the previous word only",
    })

    # 4. Trigram with backoff to the bigram, then to the majority class.
    trigram = defaultdict(Counter)
    for context, target in zip(x_train, y_train):
        trigram[(int(context[-2]), int(context[-1]))][int(target)] += 1
    best_after_two = {ctx: c.most_common(1)[0][0] for ctx, c in trigram.items()}
    trigram_pred = np.array([
        best_after_two.get(
            (int(c[-2]), int(c[-1])),
            best_after_one.get(int(c[-1]), majority_id),
        )
        for c in x_test
    ])
    rows.append({
        "predictor": "Trigram with backoff",
        "test_accuracy": float((trigram_pred == y_test).mean()),
        "note": "sparser contexts; backs off to the bigram",
    })

    frame = pd.DataFrame(rows)
    frame["test_accuracy_pct"] = (frame["test_accuracy"] * 100).round(3)
    return frame[["predictor", "test_accuracy_pct", "note"]]


def task_ambiguity(prepared: dict) -> dict:
    """How many different words legitimately follow a given context?

    This is the reason next-word accuracy cannot approach 90%: the model is
    scored right only on the one word the author happened to write, while many
    alternatives are equally valid English.
    """
    (x_train, y_train), _, (x_test, _) = prepared["splits"]
    followers = defaultdict(set)
    for context, target in zip(x_train, y_train):
        followers[int(context[-1])].add(int(target))

    counts = [len(followers[int(c[-1])]) for c in x_test if int(c[-1]) in followers]
    return {
        "median_distinct_continuations": int(np.median(counts)),
        "mean_distinct_continuations": round(float(np.mean(counts)), 1),
    }


def reference_perplexities(prepared: dict) -> pd.DataFrame:
    """Uniform and unigram perplexity, as reference points for the neural ones."""
    itos = prepared["itos"]
    (_, y_train), _, (_, y_test) = prepared["splits"]
    vocab_size = len(itos)

    counts = Counter(y_train.tolist())
    # Add-one smoothing so unseen test tokens do not give infinite loss.
    probabilities = np.array(
        [counts.get(i, 0) + 1 for i in range(vocab_size)], dtype=np.float64
    )
    probabilities /= probabilities.sum()
    unigram_perplexity = float(np.exp(-np.log(probabilities[y_test]).mean()))

    return pd.DataFrame([
        {"model": "Uniform over the vocabulary", "test_perplexity": float(vocab_size)},
        {"model": "Unigram (word frequency only)", "test_perplexity": round(unigram_perplexity, 2)},
    ])


def run(prepared: dict, verbose: bool = True) -> dict:
    baselines = evaluate_baselines(prepared)
    perplexities = reference_perplexities(prepared)
    ambiguity = task_ambiguity(prepared)

    baselines.to_csv(METRICS_DIR / "baselines.csv", index=False)
    perplexities.to_csv(METRICS_DIR / "reference_perplexities.csv", index=False)

    if verbose:
        print(baselines.to_string(index=False))
        print()
        print(perplexities.to_string(index=False))
        print()
        print(f"median distinct continuations after a context's last word: "
              f"{ambiguity['median_distinct_continuations']} "
              f"(mean {ambiguity['mean_distinct_continuations']})")

    return {"baselines": baselines, "perplexities": perplexities, "ambiguity": ambiguity}


if __name__ == "__main__":
    import data

    run(data.prepare(verbose=False))
