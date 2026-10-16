"""Central configuration for the autoregressive text-generation experiment.

Every model is trained under identical conditions; the only thing that changes
between runs is the recurrent core (SimpleRNN / Bidirectional SimpleRNN / LSTM /
GRU). Keeping all hyper-parameters in one place is what makes the comparison in
Task (ii) a fair one.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
METRICS_DIR = RESULTS / "metrics"
PLOTS_DIR = RESULTS / "plots"
SAMPLES_DIR = RESULTS / "samples"
LOGS_DIR = RESULTS / "logs"
MODELS_DIR = RESULTS / "models"

for _d in (DATA_RAW, DATA_PROCESSED, METRICS_DIR, PLOTS_DIR, SAMPLES_DIR, LOGS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Dataset: Alice's Adventures in Wonderland (Project Gutenberg eBook #11)
# --------------------------------------------------------------------------- #
DATASET_NAME = "Alice's Adventures in Wonderland by Lewis Carroll"
DATASET_URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
DATASET_LANDING_PAGE = "https://www.gutenberg.org/ebooks/11"
RAW_FILE = DATA_RAW / "pg11_alice_in_wonderland.txt"
CORPUS_FILE = DATA_PROCESSED / "alice_corpus_clean.txt"

# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
MIN_WORD_FREQ = 2        # words seen fewer times than this are mapped to <unk>
SEQUENCE_LENGTH = 20     # input window: 20 words -> predict word 21
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

# Contiguous (not random) split. Windows overlap with stride 1, so a random
# split would leak almost-identical sequences into validation/test.
TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.10      # test gets the remaining 0.10

# --------------------------------------------------------------------------- #
# Architecture / training - identical for all four models
# --------------------------------------------------------------------------- #
EMBEDDING_DIM = 100
RNN_UNITS = 128
DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 4
SEED = 42

MODEL_NAMES = ["SimpleRNN", "BiRNN", "LSTM", "GRU"]
MODEL_LABELS = {
    "SimpleRNN": "Simple RNN",
    "BiRNN": "Bidirectional RNN",
    "LSTM": "LSTM",
    "GRU": "GRU",
}

# --------------------------------------------------------------------------- #
# Generation (Task iii)
# --------------------------------------------------------------------------- #
TEMPERATURES = [0.5, 1.0, 1.5]
GENERATE_WORDS = 60
#: All four models are seeded with exactly these three sequences so that the
#: comparison in Task (iii) is like-for-like. Each is a verbatim 20-token excerpt
#: of the preprocessed corpus (so every token is in-vocabulary), and the three
#: cover different registers: narration, action and dialogue.
SEED_TEXTS = [
    # Narration - the opening sentence of the book (token offset 7).
    "alice was beginning to get very tired of sitting by her sister on the bank , and of having nothing",
    # Action - the footman scene (token offset 12601).
    "and wondering what to do next , when suddenly a footman in livery came running out of the wood she",
    # Dialogue - ends on a speech tag, so a good model should continue in speech
    # (token offset 10128).
    "i must have been changed several times since then . what do you mean by that ? said the caterpillar",
]
SEED_LABELS = ["narration", "action", "dialogue"]
