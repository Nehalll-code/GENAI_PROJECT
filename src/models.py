"""The four recurrent architectures compared in this assignment.

Every model is identical apart from its recurrent core:

    Embedding(vocab, 100)  ->  <recurrent core>  ->  Dropout(0.2)  ->  Dense(vocab, softmax)

Keeping the embedding, the regulariser, the output head, the optimiser and the
training schedule fixed is what makes the "comparable experimental conditions"
required by the question actually hold, so any difference in the results is
attributable to the recurrent cell itself.

A note on the Bidirectional model
---------------------------------
A bidirectional layer is unusual in an autoregressive setting, and it is worth
being precise about why it is still valid here. The model is trained to predict
word w[t] from the fixed window w[t-20 .. t-1]. The bidirectional layer runs its
backward pass over *that window only* - it never sees w[t]. So the target is not
leaked, and generation remains strictly autoregressive: each newly sampled word
is appended to the window and the whole window is re-encoded in both directions
before the next word is drawn. What bidirectionality buys is a richer encoding
of the context that is already available; what it costs is that the two
directions give it roughly twice the recurrent parameters of the plain
Simple RNN, which the parameter counts in the results table make explicit.
"""
from __future__ import annotations

import keras
from keras import layers

from config import DROPOUT, EMBEDDING_DIM, LEARNING_RATE, RNN_UNITS, SEQUENCE_LENGTH


def _recurrent_core(name: str):
    """Return the recurrent layer that defines each architecture."""
    if name == "SimpleRNN":
        return layers.SimpleRNN(RNN_UNITS, name="simple_rnn")
    if name == "BiRNN":
        # Bidirectional wrapper around the *same* SimpleRNN cell, so the
        # comparison isolates the effect of reading the context in both
        # directions rather than the effect of a different cell type.
        return layers.Bidirectional(
            layers.SimpleRNN(RNN_UNITS), name="bidirectional_simple_rnn"
        )
    if name == "LSTM":
        return layers.LSTM(RNN_UNITS, name="lstm")
    if name == "GRU":
        return layers.GRU(RNN_UNITS, name="gru")
    raise ValueError(f"unknown model name: {name!r}")


def build_model(name: str, vocab_size: int) -> keras.Model:
    """Assemble and compile one of the four next-word prediction models."""
    model = keras.Sequential(
        [
            keras.Input(shape=(SEQUENCE_LENGTH,), dtype="int32", name="context_window"),
            layers.Embedding(vocab_size, EMBEDDING_DIM, name="embedding"),
            _recurrent_core(name),
            layers.Dropout(DROPOUT, name="dropout"),
            layers.Dense(vocab_size, activation="softmax", name="next_word"),
        ],
        name=name,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )
    return model


def summarise(name: str, vocab_size: int) -> str:
    """Plain-text architecture summary, saved alongside the results."""
    lines: list[str] = []
    model = build_model(name, vocab_size)
    model.summary(print_fn=lines.append, line_length=88)
    return "\n".join(lines)
