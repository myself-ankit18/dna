"""
config.py — All hyperparameters and paths in one place.
Both dna_engine.py and app.py import from here.
"""
import os
import torch

# ── Device ────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── K-mer / Tokenisation ─────────────────────────────────────
K          = 4
KMER_VOCAB = 4 ** K

# ── Transformer architecture (must match training) ───────────
BLOCK_SIZE = 320
EMBED_DIM  = 256
N_HEADS    = 8
N_LAYERS   = 6
FF_DIM     = 1024
DROPOUT    = 0.1

# ── Arithmetic coder constants ────────────────────────────────
PRECISION = 32
FULL      = 1 << PRECISION
HALF      = FULL >> 1
QUARTER   = FULL >> 2
CDF_SCALE = (1 << 31) - KMER_VOCAB

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")

os.makedirs(MODEL_DIR, exist_ok=True)
