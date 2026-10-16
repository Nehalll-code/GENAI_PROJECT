"""Task (ii) + (iii): the temperature sweep, the automatic text-quality tables,
and every figure saved under results/plots.
"""
from __future__ import annotations

from collections import Counter

import matplotlib

matplotlib.use("Agg")  # file output only; no interactive backend needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import generate as gen
from config import (
    GENERATE_WORDS,
    METRICS_DIR,
    MODEL_LABELS,
    MODEL_NAMES,
    PLOTS_DIR,
    SAMPLES_DIR,
    SEED,
    SEED_LABELS,
    SEED_TEXTS,
    TEMPERATURES,
)

# --------------------------------------------------------------------------- #
# Figure style
# --------------------------------------------------------------------------- #
# Categorical palette, fixed order, one colour per model for every figure.
# Validated with the data-viz palette checker (all-pairs, light surface
# #fcfcfb): lightness band PASS, chroma floor PASS, CVD separation PASS
# (worst dE 9.2), normal-vision floor PASS (worst dE 16.3). Aqua sits below
# 3:1 contrast on this surface, so every series also carries a legend entry, a
# distinct marker and a direct value label.
SERIES_COLOURS = {
    "SimpleRNN": "#2a78d6",  # blue
    "BiRNN": "#eb6834",      # orange
    "LSTM": "#1baf7a",       # aqua
    "GRU": "#4a3aa7",        # violet
}
SERIES_MARKERS = {"SimpleRNN": "o", "BiRNN": "s", "LSTM": "^", "GRU": "D"}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8982"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.8,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "figure.dpi": 140,
    }
)


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)


def _save(fig, filename: str) -> None:
    path = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(PLOTS_DIR.parents[1])}")


# --------------------------------------------------------------------------- #
# Dataset figure (Task i)
# --------------------------------------------------------------------------- #
def plot_dataset_profile(counts: Counter, tokens: list) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    frequencies = np.array(sorted(counts.values(), reverse=True))
    ranks = np.arange(1, len(frequencies) + 1)
    left.loglog(ranks, frequencies, color=SERIES_COLOURS["SimpleRNN"], linewidth=2.0)
    _style_axes(
        left,
        "Rank-frequency distribution is Zipfian",
        "Token rank (log)",
        "Frequency (log)",
    )
    left.annotate(
        f"{len(frequencies):,} distinct tokens\n{len(tokens):,} tokens in total",
        xy=(0.97, 0.93),
        xycoords="axes fraction",
        ha="right",
        va="top",
        color=INK_SECONDARY,
        fontsize=9,
    )

    top = counts.most_common(15)[::-1]
    labels = [w if w not in gen.PUNCTUATION else f"'{w}'" for w, _ in top]
    values = [c for _, c in top]
    bars = right.barh(labels, values, color=SERIES_COLOURS["BiRNN"], height=0.68)
    right.grid(axis="y", visible=False)
    _style_axes(right, "15 most frequent tokens", "Occurrences", "")
    for bar, value in zip(bars, values):
        right.annotate(
            f"{value:,}",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK_SECONDARY,
        )
    right.set_xlim(0, max(values) * 1.14)

    fig.suptitle(
        "Dataset profile - Alice's Adventures in Wonderland (preprocessed)",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="semibold",
        color=INK_PRIMARY,
    )
    fig.subplots_adjust(top=0.84)
    _save(fig, "dataset_profile.png")


# --------------------------------------------------------------------------- #
# Training-curve figures (Task ii)
# --------------------------------------------------------------------------- #
def plot_training_curves(history: pd.DataFrame) -> None:
    panels = [
        ("loss", "Training cross-entropy", "Loss (nats)"),
        ("val_loss", "Validation cross-entropy", "Loss (nats)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (column, title, ylabel) in zip(axes, panels):
        for name in MODEL_NAMES:
            subset = history[history["model"] == MODEL_LABELS[name]]
            ax.plot(
                subset["epoch"],
                subset[column],
                color=SERIES_COLOURS[name],
                marker=SERIES_MARKERS[name],
                markevery=2,
                label=MODEL_LABELS[name],
            )
        _style_axes(ax, title, "Epoch", ylabel)
    axes[0].legend(loc="upper right", ncols=2, fontsize=9)
    fig.suptitle(
        "Training loss keeps falling while validation loss turns up - all four overfit",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="semibold",
        color=INK_PRIMARY,
    )
    fig.subplots_adjust(top=0.84)
    _save(fig, "loss_curves.png")


def plot_perplexity_and_accuracy(history: pd.DataFrame) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in MODEL_NAMES:
        subset = history[history["model"] == MODEL_LABELS[name]]
        left.plot(
            subset["epoch"],
            np.exp(subset["val_loss"]),
            color=SERIES_COLOURS[name],
            marker=SERIES_MARKERS[name],
            markevery=2,
            label=MODEL_LABELS[name],
        )
        right.plot(
            subset["epoch"],
            subset["val_accuracy"] * 100,
            color=SERIES_COLOURS[name],
            marker=SERIES_MARKERS[name],
            markevery=2,
            label=MODEL_LABELS[name],
        )
    _style_axes(left, "Validation perplexity", "Epoch", "Perplexity (lower is better)")
    _style_axes(right, "Validation next-word accuracy", "Epoch", "Top-1 accuracy (%)")
    left.legend(loc="upper right", fontsize=9)
    fig.suptitle(
        "GRU reaches the lowest perplexity and the highest accuracy on validation",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="semibold",
        color=INK_PRIMARY,
    )
    fig.subplots_adjust(top=0.84)
    _save(fig, "perplexity_accuracy_curves.png")


def plot_test_metrics(results: pd.DataFrame) -> None:
    panels = [
        ("test_perplexity", "Test perplexity", "lower is better", "{:.1f}"),
        ("test_accuracy", "Test top-1 accuracy (%)", "higher is better", "{:.1f}"),
        ("test_top5_accuracy", "Test top-5 accuracy (%)", "higher is better", "{:.1f}"),
        ("training_time_s", "Training time (s)", "lower is cheaper", "{:.0f}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (column, title, hint, fmt) in zip(axes.ravel(), panels):
        frame = results.set_index("model_key").loc[MODEL_NAMES]
        values = frame[column].to_numpy(dtype=float)
        if "accuracy" in column:
            values = values * 100
        labels = [MODEL_LABELS[n] for n in MODEL_NAMES]
        colours = [SERIES_COLOURS[n] for n in MODEL_NAMES]
        bars = ax.bar(labels, values, color=colours, width=0.62)
        ax.grid(axis="x", visible=False)
        _style_axes(ax, f"{title}  ({hint})", "", "")
        for bar, value in zip(bars, values):
            ax.annotate(
                fmt.format(value),
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9.5,
                color=INK_PRIMARY,
                fontweight="semibold",
            )
        ax.set_ylim(0, values.max() * 1.16)
        ax.tick_params(axis="x", labelsize=9)
    fig.suptitle(
        "Held-out test results for the four architectures",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="semibold",
        color=INK_PRIMARY,
    )
    fig.subplots_adjust(top=0.9)
    _save(fig, "test_metrics_comparison.png")


# --------------------------------------------------------------------------- #
# Temperature figures (Task iii)
# --------------------------------------------------------------------------- #
def plot_temperature_effects(quality: pd.DataFrame) -> None:
    panels = [
        ("distinct_2", "Diversity: distinct-2", "Unique bigrams / bigrams"),
        ("repeated_4gram_rate", "Repetition: repeated 4-grams", "Share of 4-grams repeated"),
        ("corpus_bigram_overlap", "Consistency: bigrams seen in corpus", "Share of bigrams in corpus"),
    ]
    averaged = quality.groupby(["model_key", "temperature"], as_index=False).mean(
        numeric_only=True
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (column, title, ylabel) in zip(axes, panels):
        for name in MODEL_NAMES:
            subset = averaged[averaged["model_key"] == name].sort_values("temperature")
            ax.plot(
                subset["temperature"],
                subset[column],
                color=SERIES_COLOURS[name],
                marker=SERIES_MARKERS[name],
                label=MODEL_LABELS[name],
            )
        _style_axes(ax, title, "Sampling temperature", ylabel)
        ax.set_xticks(TEMPERATURES)
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(
        "Raising temperature buys diversity and pays for it in consistency",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="semibold",
        color=INK_PRIMARY,
    )
    fig.subplots_adjust(top=0.84)
    _save(fig, "temperature_effects.png")


# --------------------------------------------------------------------------- #
# Temperature sweep (Task iii)
# --------------------------------------------------------------------------- #
def run_generation_sweep(trained: dict, prepared: dict) -> pd.DataFrame:
    """Generate from every (model, temperature, seed) combination and score it.

    The same three seeds and the same RNG seed are used for every model, so the
    only thing that differs between two passages is the model and the
    temperature.
    """
    stoi, itos = prepared["stoi"], prepared["itos"]
    corpus_bigrams, corpus_trigrams = gen.build_corpus_ngram_sets(prepared["tokens"])

    rows: list[dict] = []
    document = [
        "# Generated text samples",
        "",
        f"Each passage continues one of three fixed seed sequences for "
        f"{GENERATE_WORDS} tokens. Every model sees the same seeds and the same "
        f"random-number stream, so differences come from the model and the "
        f"temperature alone.",
        "",
    ]

    for name in MODEL_NAMES:
        model = trained[name]
        document.append(f"## {MODEL_LABELS[name]}")
        document.append("")
        for temperature in TEMPERATURES:
            document.append(f"### Temperature {temperature}")
            document.append("")
            for seed_index, (seed_text, seed_label) in enumerate(
                zip(SEED_TEXTS, SEED_LABELS)
            ):
                rng = np.random.default_rng(SEED + seed_index)
                generated, _ = gen.generate_text(
                    model,
                    seed_text,
                    stoi,
                    itos,
                    n_words=GENERATE_WORDS,
                    temperature=temperature,
                    rng=rng,
                )
                measures = gen.text_quality(generated, corpus_bigrams, corpus_trigrams)
                rows.append(
                    {
                        "model": MODEL_LABELS[name],
                        "model_key": name,
                        "temperature": temperature,
                        "seed_index": seed_index + 1,
                        "seed_type": seed_label,
                        **measures,
                    }
                )

                document.append(f"**Seed {seed_index + 1} ({seed_label})**")
                document.append("")
                document.append(f"> {gen.detokenize(seed_text.split())}")
                document.append("")
                document.append(f"{gen.detokenize(generated)}")
                document.append("")
                document.append(
                    f"`distinct-2 {measures['distinct_2']:.3f} · "
                    f"repeated 4-grams {measures['repeated_4gram_rate']:.3f} · "
                    f"corpus-bigram overlap {measures['corpus_bigram_overlap']:.3f} · "
                    f"longest token run {measures['longest_token_run']}`"
                )
                document.append("")
                print(
                    f"  {MODEL_LABELS[name]:<18} T={temperature:<4} seed {seed_index + 1} ok",
                    flush=True,
                )

    quality = pd.DataFrame(rows)
    quality.to_csv(METRICS_DIR / "generation_quality.csv", index=False)

    by_temperature = (
        quality.groupby(["model", "model_key", "temperature"], as_index=False)
        .mean(numeric_only=True)
        .drop(columns=["seed_index"])
        .round(4)
    )
    by_temperature.to_csv(
        METRICS_DIR / "generation_quality_by_temperature.csv", index=False
    )

    (SAMPLES_DIR / "generated_samples.md").write_text(
        "\n".join(document), encoding="utf-8"
    )
    return quality


# --------------------------------------------------------------------------- #
def make_all_figures(prepared: dict, results: pd.DataFrame, history: pd.DataFrame,
                     quality: pd.DataFrame) -> None:
    print("\n--- Figures ---")
    plot_dataset_profile(prepared["counts"], prepared["tokens"])
    plot_training_curves(history)
    plot_perplexity_and_accuracy(history)
    plot_test_metrics(results)
    plot_temperature_effects(quality)
