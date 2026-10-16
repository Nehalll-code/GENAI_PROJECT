"""Task (i) + (ii): train all four models under identical conditions and record
quantitative results (loss, perplexity, next-word accuracy, training time).
"""
from __future__ import annotations

import json
import math
import platform
import time

import keras
import numpy as np
import pandas as pd

import models as model_zoo
from config import (
    BATCH_SIZE,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EMBEDDING_DIM,
    LEARNING_RATE,
    LOGS_DIR,
    MAX_EPOCHS,
    METRICS_DIR,
    MODELS_DIR,
    MODEL_LABELS,
    MODEL_NAMES,
    RNN_UNITS,
    SEED,
    SEQUENCE_LENGTH,
)


def perplexity(cross_entropy: float) -> float:
    """Perplexity is the exponential of the mean cross-entropy (in nats)."""
    return float(math.exp(cross_entropy))


def hardware_description() -> dict:
    """Record what the numbers were actually produced on."""
    import tensorflow as tf

    try:
        cpu_count = len(__import__("os").sched_getaffinity(0))  # POSIX
    except AttributeError:
        cpu_count = __import__("os").cpu_count()

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": cpu_count,
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "gpus_visible": [d.name for d in tf.config.list_physical_devices("GPU")],
        "device_used": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
    }


def train_one(name: str, vocab_size: int, splits) -> dict:
    """Build, train and test a single architecture. Returns its result row."""
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = splits

    # Re-seeding before every build gives all four models the same weight
    # initialisation stream and the same dropout/shuffle randomness.
    keras.utils.set_random_seed(SEED)

    model = model_zoo.build_model(name, vocab_size)
    (LOGS_DIR / f"architecture_{name}.txt").write_text(
        model_zoo.summarise(name, vocab_size), encoding="utf-8"
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    csv_logger = keras.callbacks.CSVLogger(str(LOGS_DIR / f"training_log_{name}.csv"))

    print(f"\n=== Training {MODEL_LABELS[name]} ===", flush=True)
    start = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[early_stopping, csv_logger],
        verbose=2,
    )
    training_seconds = time.perf_counter() - start

    # Timed single-batch inference, as a rough generation-cost indicator.
    warmup = model(x_test[:BATCH_SIZE], training=False)
    del warmup
    inference_start = time.perf_counter()
    for _ in range(10):
        model(x_test[:BATCH_SIZE], training=False)
    inference_ms = (time.perf_counter() - inference_start) / 10 * 1000

    test_loss, test_accuracy, test_top5 = model.evaluate(
        x_test, y_test, batch_size=BATCH_SIZE, verbose=0
    )
    val_loss, val_accuracy, val_top5 = model.evaluate(
        x_val, y_val, batch_size=BATCH_SIZE, verbose=0
    )
    train_loss, train_accuracy, train_top5 = model.evaluate(
        x_train, y_train, batch_size=BATCH_SIZE, verbose=0
    )

    model.save(MODELS_DIR / f"{name}.keras")

    epochs_run = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    row = {
        "model": MODEL_LABELS[name],
        "model_key": name,
        "total_parameters": int(model.count_params()),
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "training_time_s": round(training_seconds, 2),
        "time_per_epoch_s": round(training_seconds / epochs_run, 2),
        "train_loss": round(float(train_loss), 4),
        "val_loss": round(float(val_loss), 4),
        "test_loss": round(float(test_loss), 4),
        "train_perplexity": round(perplexity(train_loss), 2),
        "val_perplexity": round(perplexity(val_loss), 2),
        "test_perplexity": round(perplexity(test_loss), 2),
        "train_accuracy": round(float(train_accuracy), 4),
        "val_accuracy": round(float(val_accuracy), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "test_top5_accuracy": round(float(test_top5), 4),
        "inference_ms_per_batch128": round(inference_ms, 2),
    }

    history_frame = pd.DataFrame(history.history)
    history_frame.insert(0, "epoch", range(1, epochs_run + 1))
    history_frame.insert(0, "model", MODEL_LABELS[name])
    return {"row": row, "history": history_frame, "model": model}


def train_all(prepared: dict) -> dict:
    """Train every architecture and write the quantitative result files."""
    vocab_size = len(prepared["itos"])
    splits = prepared["splits"]

    rows, histories, trained = [], [], {}
    for name in MODEL_NAMES:
        result = train_one(name, vocab_size, splits)
        rows.append(result["row"])
        histories.append(result["history"])
        trained[name] = result["model"]

    results = pd.DataFrame(rows)
    results.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(
        METRICS_DIR / "training_history.csv", index=False
    )

    run_info = {
        "hardware": hardware_description(),
        "hyperparameters": {
            "sequence_length": SEQUENCE_LENGTH,
            "embedding_dim": EMBEDDING_DIM,
            "rnn_units": RNN_UNITS,
            "dropout": DROPOUT,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "loss": "sparse_categorical_crossentropy",
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "random_seed": SEED,
            "vocabulary_size": vocab_size,
        },
        "total_training_time_s": round(float(results["training_time_s"].sum()), 2),
    }
    (METRICS_DIR / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )

    print("\n--- Quantitative comparison ---")
    print(
        results[
            ["model", "total_parameters", "epochs_run", "training_time_s",
             "test_loss", "test_perplexity", "test_accuracy", "test_top5_accuracy"]
        ].to_string(index=False)
    )
    return {"results": results, "models": trained, "run_info": run_info}


if __name__ == "__main__":
    import data

    train_all(data.prepare())
