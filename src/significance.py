"""Are the differences between the four models real, or sampling noise?

All four models predict the *same* held-out positions, so their results are
paired. That makes McNemar's test the right instrument: it looks only at the
positions where two models disagree, which is far more powerful than comparing
two independent percentages. Bootstrap intervals are added so every headline
number carries an uncertainty range.
"""
from __future__ import annotations

import itertools
from math import erfc, sqrt

import numpy as np
import pandas as pd

from config import METRICS_DIR, MODELS_DIR, MODEL_LABELS, MODEL_NAMES, SEED

BOOTSTRAP_RESAMPLES = 2000


def _load_predictions(prepared: dict) -> tuple:
    """Per-model correctness mask and per-position loss on the test split."""
    import keras

    (_, _), (_, _), (x_test, y_test) = prepared["splits"]

    correct, losses = {}, {}
    for name in MODEL_NAMES:
        model = keras.models.load_model(MODELS_DIR / f"{name}.keras")
        probabilities = model.predict(x_test, batch_size=256, verbose=0)
        predicted = probabilities.argmax(axis=1)
        correct[name] = (predicted == y_test).astype(np.int8)
        # Cross-entropy of the true word at each position.
        true_word_probability = probabilities[np.arange(len(y_test)), y_test]
        losses[name] = -np.log(np.clip(true_word_probability, 1e-12, None))
    return correct, losses


def _normal_two_sided_p(z: float) -> float:
    """Two-sided p-value from a z statistic, via the complementary error fn."""
    return erfc(abs(z) / sqrt(2.0))


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """McNemar's paired test on two correctness masks.

    b_only = positions only model B gets right; a_only = only model A.
    Under the null they are equally likely, so the statistic uses the
    continuity-corrected normal approximation to the binomial.
    """
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((a == 0) & (b == 1)))
    discordant = a_only + b_only
    if discordant == 0:
        return {"a_only": 0, "b_only": 0, "discordant": 0, "z": 0.0, "p_value": 1.0}

    z = (abs(b_only - a_only) - 1) / sqrt(discordant)
    return {
        "a_only": a_only,
        "b_only": b_only,
        "discordant": discordant,
        "z": round(float(z), 3),
        "p_value": float(_normal_two_sided_p(z)),
    }


def bootstrap_intervals(correct: dict, losses: dict) -> pd.DataFrame:
    """Percentile bootstrap 95% intervals for accuracy and perplexity."""
    rng = np.random.default_rng(SEED)
    n = len(next(iter(correct.values())))
    indices = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))

    rows = []
    for name in MODEL_NAMES:
        accuracies = correct[name][indices].mean(axis=1)
        perplexities = np.exp(losses[name][indices].mean(axis=1))
        rows.append({
            "model": MODEL_LABELS[name],
            "accuracy_pct": round(float(correct[name].mean() * 100), 2),
            "accuracy_ci_low": round(float(np.percentile(accuracies, 2.5) * 100), 2),
            "accuracy_ci_high": round(float(np.percentile(accuracies, 97.5) * 100), 2),
            "perplexity": round(float(np.exp(losses[name].mean())), 2),
            "perplexity_ci_low": round(float(np.percentile(perplexities, 2.5)), 2),
            "perplexity_ci_high": round(float(np.percentile(perplexities, 97.5)), 2),
        })
    return pd.DataFrame(rows)


def pairwise_tests(correct: dict) -> pd.DataFrame:
    rows = []
    for first, second in itertools.combinations(MODEL_NAMES, 2):
        result = mcnemar(correct[first], correct[second])
        better = (
            MODEL_LABELS[second] if result["b_only"] > result["a_only"]
            else MODEL_LABELS[first]
        )
        rows.append({
            "model_a": MODEL_LABELS[first],
            "model_b": MODEL_LABELS[second],
            "only_a_correct": result["a_only"],
            "only_b_correct": result["b_only"],
            "discordant": result["discordant"],
            "better": better,
            "z": result["z"],
            "p_value": round(result["p_value"], 5),
            "significant_at_0.05": bool(result["p_value"] < 0.05),
        })
    return pd.DataFrame(rows)


def paired_loss_tests(losses: dict) -> pd.DataFrame:
    """Paired comparison of per-position cross-entropy between every model pair.

    This is the perplexity counterpart to McNemar, and it is the comparison that
    matters: the two models are scored on identical positions, so the difference
    in loss is measured position by position. Comparing each model's *marginal*
    confidence interval instead would badly understate the evidence, because
    most of the spread in those intervals is shared difficulty of the test text,
    which cancels in the pairing.
    """
    rng = np.random.default_rng(SEED)
    n = len(next(iter(losses.values())))
    indices = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))

    rows = []
    for first, second in itertools.combinations(MODEL_NAMES, 2):
        difference = losses[first] - losses[second]  # >0 means `second` is better
        mean_difference = float(difference.mean())
        resampled = difference[indices].mean(axis=1)

        # Paired t statistic on the per-position differences.
        standard_error = float(difference.std(ddof=1) / sqrt(n))
        z = mean_difference / standard_error if standard_error > 0 else 0.0

        rows.append({
            "model_a": MODEL_LABELS[first],
            "model_b": MODEL_LABELS[second],
            "mean_loss_difference": round(mean_difference, 4),
            "diff_ci_low": round(float(np.percentile(resampled, 2.5)), 4),
            "diff_ci_high": round(float(np.percentile(resampled, 97.5)), 4),
            "perplexity_ratio_a_over_b": round(float(np.exp(mean_difference)), 3),
            "better": MODEL_LABELS[second] if mean_difference > 0 else MODEL_LABELS[first],
            "z": round(z, 2),
            "p_value": round(float(_normal_two_sided_p(z)), 6),
            "significant_at_0.05": bool(_normal_two_sided_p(z) < 0.05),
        })
    return pd.DataFrame(rows)


def run(prepared: dict, verbose: bool = True) -> dict:
    correct, losses = _load_predictions(prepared)

    intervals = bootstrap_intervals(correct, losses)
    tests = pairwise_tests(correct)
    loss_tests = paired_loss_tests(losses)

    intervals.to_csv(METRICS_DIR / "bootstrap_intervals.csv", index=False)
    tests.to_csv(METRICS_DIR / "significance_tests.csv", index=False)
    loss_tests.to_csv(METRICS_DIR / "significance_tests_loss.csv", index=False)

    if verbose:
        print("Bootstrap 95% intervals (2,000 resamples of the test set)")
        print(intervals.to_string(index=False))
        print()
        print("McNemar paired tests on next-word correctness (top-1 accuracy)")
        print(tests.to_string(index=False))
        print()
        print("Paired tests on per-position cross-entropy (perplexity)")
        print(loss_tests.to_string(index=False))

    return {"intervals": intervals, "tests": tests, "loss_tests": loss_tests}


if __name__ == "__main__":
    import data

    run(data.prepare(verbose=False))
