"""End-to-end pipeline: data -> four trained models -> metrics -> plots -> samples.

Usage (from the src/ directory):

    python run_all.py

Everything it writes lands under results/ and data/.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieten TensorFlow banners

import pandas as pd

import baselines
import data
import evaluate
import significance
import train
from config import METRICS_DIR


def main() -> None:
    started = time.perf_counter()

    print("=" * 78)
    print("STEP 1/6  Dataset preparation and characterisation")
    print("=" * 78)
    prepared = data.prepare(verbose=True)

    print()
    print("=" * 78)
    print("STEP 2/6  Classical baselines on the same test split")
    print("=" * 78)
    baselines.run(prepared, verbose=True)

    print()
    print("=" * 78)
    print("STEP 3/6  Training the four models under identical conditions")
    print("=" * 78)
    training = train.train_all(prepared)
    results = training["results"]
    history = pd.read_csv(METRICS_DIR / "training_history.csv")

    print()
    print("=" * 78)
    print("STEP 4/6  Are the differences statistically real?")
    print("=" * 78)
    significance.run(prepared, verbose=True)

    print()
    print("=" * 78)
    print("STEP 5/6  Temperature sweep and text-quality scoring")
    print("=" * 78)
    quality = evaluate.run_generation_sweep(training["models"], prepared)

    print()
    print("=" * 78)
    print("STEP 6/6  Figures")
    print("=" * 78)
    evaluate.make_all_figures(prepared, results, history, quality)

    elapsed = time.perf_counter() - started
    print()
    print(f"Pipeline finished in {elapsed / 60:.1f} min "
          f"({training['run_info']['hardware']['device_used']}).")


if __name__ == "__main__":
    main()
