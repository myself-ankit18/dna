"""
dna_engine.py — Core DNA compression engine.

Contains:
  • DNA cleaning & FASTA reading
  • K-mer tokenisation / detokenisation
  • Sinusoidal positional encoding
  • CausalTransformer model definition
  • Arithmetic coder (encode / decode)
  • ProbEngine (runs model live on growing context)
  • Helper metrics (loss, accuracy, bits-per-base)

All hyperparameters are imported from config.py so they stay in sync
with the Streamlit frontend.
"""

import math, time, json, struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    DEVICE, K, KMER_VOCAB, BLOCK_SIZE, EMBED_DIM, N_HEADS,
    N_LAYERS, FF_DIM, DROPOUT, PRECISION, FULL, HALF, QUARTER,
    CDF_SCALE, MODEL_PATH,
)


# ╔══════════════════════════════════════════════════════════════╗
# ║                    DNA CLEANING                              ║
# ╚══════════════════════════════════════════════════════════════╝

VALID_BASES = set("ACGT")


def clean_dna(seq: str) -> str:
    """Keep only A, C, G, T (case-insensitive)."""
    return "".join(c for c in seq.upper() if c in VALID_BASES)


def read_fasta(path: str) -> str:
    """Read a FASTA file and return the concatenated sequence."""
    parts: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip().upper()
            if line and not line.startswith(">"):
                parts.append(line)
    return "".join(parts)


def read_fasta_from_string(text: str) -> str:
    """Read FASTA content from a string (for uploaded files)."""
    parts: list[str] = []
    for line in text.splitlines():
        line = line.strip().upper()
        if line and not line.startswith(">"):
            parts.append(line)
    return "".join(parts)


# ╔══════════════════════════════════════════════════════════════╗
# ║                   K-MER TOKENISATION                         ║
# ╚══════════════════════════════════════════════════════════════╝

BASE2IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
IDX2BASE = {0: "A", 1: "C", 2: "G", 3: "T"}


def kmer_to_id(kmer: str) -> int:
    idx = 0
    for c in kmer:
        idx = idx * 4 + BASE2IDX[c]
    return idx


def id_to_kmer(idx: int, k: int = K) -> str:
    chars: list[str] = []
    for _ in range(k):
        chars.append(IDX2BASE[idx % 4])
        idx //= 4
    return "".join(reversed(chars))


def dna_to_tokens(dna: str, k: int = K) -> list[int]:
    return [kmer_to_id(dna[i : i + k]) for i in range(len(dna) - k + 1)]


def tokens_to_dna(tokens: list[int], k: int = K) -> str:
    if not tokens:
        return ""
    dna = id_to_kmer(tokens[0], k)
    for t in tokens[1:]:
        dna += IDX2BASE[t % 4]
    return dna


# ╔══════════════════════════════════════════════════════════════╗
# ║               PROGRESS CALLBACK (pluggable)                  ║
# ╚══════════════════════════════════════════════════════════════╝

# Default progress handler — prints to stdout.
# The Streamlit app replaces this at runtime with a UI callback.

def _default_progress(desc, i, total, t_start, block_size=BLOCK_SIZE):
    """Print progress to the console (default fallback)."""
    if i % block_size != 0 and i != total:
        return
    block_num = i // block_size if i % block_size == 0 else (i // block_size) + 1
    n_blocks  = math.ceil(total / block_size)
    elapsed   = time.time() - t_start
    rate      = i / elapsed if elapsed > 0 else 0
    eta       = (total - i) / rate if rate > 0 else 0
    pct       = 100.0 * i / total
    print(
        f"  {desc} | Block {block_num:,}/{n_blocks:,} | "
        f"{i:,}/{total:,} tokens ({pct:5.1f}%) | "
        f"{rate:,.0f} tok/s | ETA {eta:6.1f}s"
    )


# This module-level variable is what encode/decode call.
# Streamlit will monkey-patch it to drive a progress bar.
_progress = _default_progress


# ╔══════════════════════════════════════════════════════════════╗
# ║              POSITIONAL ENCODING                             ║
# ╚══════════════════════════════════════════════════════════════╝

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# ╔══════════════════════════════════════════════════════════════╗
# ║              CAUSAL TRANSFORMER                              ║
# ╚══════════════════════════════════════════════════════════════╝

class CausalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(KMER_VOCAB, EMBED_DIM)
        self.pe  = SinusoidalPE(EMBED_DIM, max_len=BLOCK_SIZE + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=N_HEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=N_LAYERS
        )
        self.out_proj = nn.Linear(EMBED_DIM, KMER_VOCAB)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, sz: int):
        return torch.triu(
            torch.ones(sz, sz, device=DEVICE), diagonal=1
        ).bool()

    def forward(self, x):
        emb  = self.emb(x)
        emb  = self.pe(emb)
        mask = self._causal_mask(x.size(1))
        out  = self.transformer(emb, mask=mask, is_causal=True)
        return self.out_proj(out)


# ╔══════════════════════════════════════════════════════════════╗
# ║              ARITHMETIC CODER HELPERS                        ║
# ╚══════════════════════════════════════════════════════════════╝

def build_cdf(probs, n_symbols: int = KMER_VOCAB):
    counts = np.maximum(
        1, np.round(probs.astype(np.float64) * CDF_SCALE)
    ).astype(np.int64)
    total = counts.sum()
    diff  = HALF - total
    if diff > 0:
        counts[np.argmax(counts)] += diff
    elif diff < 0:
        diff = -diff
        idx  = np.argsort(counts)[::-1]
        for i in idx:
            take = min(diff, counts[i] - 1)
            counts[i] -= take
            diff -= take
            if diff == 0:
                break
    cdf     = np.zeros(n_symbols + 1, dtype=np.int64)
    cdf[1:] = np.cumsum(counts)
    return cdf


# ╔══════════════════════════════════════════════════════════════╗
# ║              PROBABILITY ENGINE                              ║
# ╚══════════════════════════════════════════════════════════════╝

class ProbEngine:
    """
    Runs the transformer live on a growing context window.
    Encoder and decoder maintain identical deterministic state.
    """

    def __init__(self, model: CausalTransformer):
        self.model   = model
        self.context: list[int] = []

    def reset(self):
        self.context = []

    def get_probs(self) -> np.ndarray:
        self.model.float()
        self.model.eval()
        if len(self.context) == 0:
            return np.ones(KMER_VOCAB, dtype=np.float32) / KMER_VOCAB
        ctx = self.context[-BLOCK_SIZE:]
        t   = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            logits = self.model(t)
        return F.softmax(logits[0, -1], dim=-1).cpu().float().numpy()

    def advance(self, token: int):
        self.context.append(token)


# ╔══════════════════════════════════════════════════════════════╗
# ║              LOSSLESS ENCODE                                 ║
# ╚══════════════════════════════════════════════════════════════╝

def lossless_encode(tokens: list[int], engine: ProbEngine,
                    desc: str = "Encoding") -> bytes:
    engine.reset()
    low     = 0
    high    = FULL
    bits: list[int]  = []
    pending: list     = []
    total   = len(tokens)
    t_start = time.time()

    def emit(bit):
        bits.append(bit)
        for _ in pending:
            bits.append(1 - bit)
        pending.clear()

    for i, sym in enumerate(tokens, 1):
        probs     = engine.get_probs()
        cdf       = build_cdf(probs)
        total_cdf = cdf[-1]
        span      = high - low
        high      = low + (span * int(cdf[sym + 1])) // total_cdf
        low       = low + (span * int(cdf[sym]))     // total_cdf
        engine.advance(sym)

        while True:
            if high <= HALF:
                emit(0); low <<= 1; high <<= 1
            elif low >= HALF:
                emit(1)
                low  = (low  - HALF) << 1
                high = (high - HALF) << 1
            elif low >= QUARTER and high <= 3 * QUARTER:
                pending.append(None)
                low  = (low  - QUARTER) << 1
                high = (high - QUARTER) << 1
            else:
                break

        _progress(desc, i, total, t_start)

    pending.append(None)
    emit(0 if low < QUARTER else 1)

    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j, b in enumerate(bits[i : i + 8]):
            byte |= b << (7 - j)
        out.append(byte)
    return bytes(out)


# ╔══════════════════════════════════════════════════════════════╗
# ║              LOSSLESS DECODE                                 ║
# ╚══════════════════════════════════════════════════════════════╝

def lossless_decode(data: bytes, n_tokens: int, engine: ProbEngine,
                    desc: str = "Decoding") -> list[int]:
    engine.reset()
    bits: list[int] = []
    for byte in data:
        for j in range(7, -1, -1):
            bits.append((byte >> j) & 1)

    pos = 0

    def read_bit() -> int:
        nonlocal pos
        b = bits[pos] if pos < len(bits) else 0
        pos += 1
        return b

    low   = 0
    high  = FULL
    value = 0
    for _ in range(PRECISION):
        value = (value << 1) | read_bit()

    tokens: list[int] = []
    t_start = time.time()

    for i in range(1, n_tokens + 1):
        probs  = engine.get_probs()
        cdf    = build_cdf(probs)
        total  = cdf[-1]
        span   = high - low
        scaled = ((value - low + 1) * total - 1) // span

        lo, hi = 0, KMER_VOCAB - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cdf[mid + 1] <= scaled:
                lo = mid + 1
            else:
                hi = mid
        sym = lo
        tokens.append(sym)
        engine.advance(sym)

        high = low + (span * int(cdf[sym + 1])) // total
        low  = low + (span * int(cdf[sym]))     // total

        while True:
            if high <= HALF:
                low <<= 1; high <<= 1
                value = (value << 1) | read_bit()
            elif low >= HALF:
                low   = (low   - HALF) << 1
                high  = (high  - HALF) << 1
                value = (value - HALF) << 1
                value |= read_bit()
            elif low >= QUARTER and high <= 3 * QUARTER:
                low   = (low   - QUARTER) << 1
                high  = (high  - QUARTER) << 1
                value = (value - QUARTER) << 1
                value |= read_bit()
            else:
                break

        _progress(desc, i, n_tokens, t_start)

    return tokens


# ╔══════════════════════════════════════════════════════════════╗
# ║              BUNDLE HELPERS (pack / unpack .dnacomp)         ║
# ╚══════════════════════════════════════════════════════════════╝

def pack_bundle(compressed: bytes, n_tokens: int, dna_len: int) -> bytes:
    """Create a .dnacomp bundle: [4-byte meta length][JSON meta][compressed bits]."""
    meta       = {"n_tokens": n_tokens, "dna_len": dna_len, "k": K, "version": 1}
    meta_bytes = json.dumps(meta).encode()
    return struct.pack(">I", len(meta_bytes)) + meta_bytes + compressed


def unpack_bundle(raw: bytes) -> tuple[dict, bytes]:
    """Parse a .dnacomp bundle → (metadata dict, compressed bytes)."""
    mlen      = struct.unpack(">I", raw[:4])[0]
    meta      = json.loads(raw[4 : 4 + mlen])
    comp_data = raw[4 + mlen :]
    return meta, comp_data


# ╔══════════════════════════════════════════════════════════════╗
# ║              MODEL LOADER                                    ║
# ╚══════════════════════════════════════════════════════════════╝

def load_model(path: str = MODEL_PATH) -> CausalTransformer:
    """Load trained weights into a CausalTransformer and set to eval mode."""
    model = CausalTransformer().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model
