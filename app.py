"""
app.py — Streamlit frontend for the AI DNA Compressor.

Run with:
    streamlit run app.py
"""

import streamlit as st
import time, json, struct, math, os
import numpy as np
import torch

# ── Page config (must be the first Streamlit call) ────────────
st.set_page_config(
    page_title="🧬 AI DNA Compressor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports from our engine ───────────────────────────────────
from config import DEVICE, K, KMER_VOCAB, BLOCK_SIZE, MODEL_PATH, MODEL_DIR
import dna_engine
from dna_engine import (
    CausalTransformer,
    ProbEngine,
    lossless_encode,
    lossless_decode,
    clean_dna,
    read_fasta_from_string,
    dna_to_tokens,
    tokens_to_dna,
    pack_bundle,
    unpack_bundle,
    load_model,
)


# ╔══════════════════════════════════════════════════════════════╗
# ║                     CUSTOM CSS                               ║
# ╚══════════════════════════════════════════════════════════════╝

st.markdown("""
<style>
    /* Main header */
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #00c6ff, #0072ff, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        color: #888; font-size: 1.1rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.1);
    }
    div[data-testid="stMetric"] label {
        color: #7ec8e3 !important;
    }

    /* File uploader */
    section[data-testid="stFileUploader"] {
        border: 2px dashed #0072ff;
        border-radius: 12px;
        padding: 1rem;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║                     HEADER                                   ║
# ╚══════════════════════════════════════════════════════════════╝

st.markdown("""
<div class="main-header">
    <h1>🧬 AI DNA Compressor</h1>
    <p>Lossless DNA sequence compression using Causal Transformers & Arithmetic Coding</p>
</div>
""", unsafe_allow_html=True)

st.divider()


# ╔══════════════════════════════════════════════════════════════╗
# ║                     SIDEBAR                                  ║
# ╚══════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.image("https://img.icons8.com/color/96/dna-helix.png", width=80)
    st.header("⚙️ Configuration")

    st.markdown(f"**Device:** `{DEVICE.upper()}`")
    st.markdown(f"**K-mer size:** `{K}`")
    st.markdown(f"**Vocab size:** `{KMER_VOCAB:,}`")
    st.markdown(f"**Block size:** `{BLOCK_SIZE}`")

    st.divider()

    # ── Model loading ─────────────────────────────────────────
    st.subheader("🤖 Model")

    # Option 1: Use the default path
    use_default = os.path.isfile(MODEL_PATH)

    # Option 2: Upload a model file
    uploaded_model = st.file_uploader(
        "Upload best_model.pt" if not use_default else "Or upload a different model",
        type=["pt"],
        help="Upload your trained PyTorch model weights (.pt file)",
    )

    if uploaded_model is not None:
        # Save uploaded model to the models/ directory
        save_path = os.path.join(MODEL_DIR, uploaded_model.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_model.read())
        model_path_to_use = save_path
        st.success(f"✅ Model saved to `models/{uploaded_model.name}`")
    elif use_default:
        model_path_to_use = MODEL_PATH
        st.success("✅ Found `models/best_model.pt`")
    else:
        model_path_to_use = None
        st.warning("⚠️ No model found. Place `best_model.pt` in the `models/` folder or upload one above.")

    st.divider()
    st.caption("Built with ❤️ using PyTorch + Streamlit")


# ╔══════════════════════════════════════════════════════════════╗
# ║                   LOAD MODEL (cached)                        ║
# ╚══════════════════════════════════════════════════════════════╝

@st.cache_resource
def cached_load_model(path: str):
    """Load the model once and cache it across reruns."""
    return load_model(path)


if model_path_to_use is None:
    st.info("👈 Please load a model from the sidebar to get started.")
    st.stop()

try:
    model = cached_load_model(model_path_to_use)
    n_params = sum(p.numel() for p in model.parameters())
    with st.sidebar:
        st.markdown(f"**Parameters:** `{n_params:,}`")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()


# ╔══════════════════════════════════════════════════════════════╗
# ║                     MAIN TABS                                ║
# ╚══════════════════════════════════════════════════════════════╝

tab_compress, tab_decompress, tab_about = st.tabs([
    "🗜️ Compress", "🔓 Decompress", "ℹ️ About"
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                       COMPRESS TAB
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_compress:
    st.subheader("Upload a DNA sequence file")
    st.caption("Supported formats: `.fasta`, `.fa`, `.fna`, `.txt`")

    uploaded_file = st.file_uploader(
        "Drag and drop your file here",
        type=["fasta", "fa", "fna", "txt"],
        key="compress_uploader",
    )

    if uploaded_file is not None:
        raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
        orig_bytes = len(raw_text.encode("utf-8"))

        # Parse and clean
        raw_dna = read_fasta_from_string(raw_text)
        dna     = clean_dna(raw_dna)
        tokens  = dna_to_tokens(dna)
        n_tok   = len(tokens)

        # Show file info
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Original Size", f"{orig_bytes / 1024:.2f} KB")
        col_b.metric("DNA Bases", f"{len(dna):,}")
        col_c.metric("K-mer Tokens", f"{n_tok:,}")

        if n_tok == 0:
            st.error("No valid DNA bases found in the file. Make sure it contains A, C, G, T characters.")
            st.stop()

        # Estimate time
        est_seconds = n_tok * 0.004  # ~250 tok/s on CPU
        st.info(
            f"⏱️ Estimated compression time: **~{est_seconds / 60:.1f} min** "
            f"({est_seconds:.0f}s at ~250 tok/s on CPU). "
            f"GPU will be significantly faster."
        )

        if st.button("🚀 Start Compression", type="primary", use_container_width=True):
            engine = ProbEngine(model)

            # ── Progress bar ──────────────────────────────────
            progress_bar  = st.progress(0, text="Preparing...")
            status_area   = st.empty()

            def ui_progress_encode(desc, i, total, t_start, block_size=BLOCK_SIZE):
                if i % block_size == 0 or i == total:
                    pct     = i / total
                    elapsed = time.time() - t_start
                    rate    = i / elapsed if elapsed > 0 else 0
                    eta     = (total - i) / rate if rate > 0 else 0
                    progress_bar.progress(
                        pct,
                        text=f"{desc} — {i:,}/{total:,} tokens ({pct*100:.1f}%) | "
                             f"{rate:,.0f} tok/s | ETA {eta:.0f}s",
                    )

            # Monkey-patch progress callback
            dna_engine._progress = ui_progress_encode

            # ── Compress ──────────────────────────────────────
            t_start    = time.time()
            compressed = lossless_encode(tokens, engine, desc="Compressing")
            comp_time  = time.time() - t_start

            bundle  = pack_bundle(compressed, n_tok, len(dna))
            comp_sz = len(bundle)
            ratio   = orig_bytes / comp_sz if comp_sz > 0 else 0
            bpb     = (comp_sz * 8) / len(dna) if len(dna) > 0 else 0
            savings = max(0, orig_bytes - comp_sz)

            # ── Clear progress, show results ──────────────────
            progress_bar.empty()
            status_area.empty()

            st.success(f"✅ Compression complete in **{comp_time:.2f}s**!")

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Compressed Size", f"{comp_sz / 1024:.2f} KB")
            m2.metric("Compression Ratio", f"{ratio:.3f}x")
            m3.metric("Bits Per Base", f"{bpb:.4f}")
            m4.metric("Saved", f"{savings / 1024:.2f} KB", delta=f"-{savings / orig_bytes * 100:.1f}%")

            # Detailed results
            with st.expander("📊 Detailed Results"):
                st.json({
                    "original_size_bytes": orig_bytes,
                    "compressed_size_bytes": comp_sz,
                    "compression_ratio": round(ratio, 4),
                    "bits_per_base": round(bpb, 4),
                    "dna_length": len(dna),
                    "kmer_tokens": n_tok,
                    "compression_time_seconds": round(comp_time, 2),
                    "tokens_per_second": round(n_tok / comp_time, 1) if comp_time > 0 else 0,
                    "device": DEVICE,
                })

            # Download button
            out_name = os.path.splitext(uploaded_file.name)[0] + ".dnacomp"
            st.download_button(
                label="📥 Download Compressed File (.dnacomp)",
                data=bundle,
                file_name=out_name,
                mime="application/octet-stream",
                type="primary",
                use_container_width=True,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                      DECOMPRESS TAB
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_decompress:
    st.subheader("Upload a compressed `.dnacomp` file")

    uploaded_comp = st.file_uploader(
        "Drag and drop your .dnacomp file here",
        type=["dnacomp"],
        key="decompress_uploader",
    )

    if uploaded_comp is not None:
        raw_bytes = uploaded_comp.read()
        comp_sz   = len(raw_bytes)

        try:
            meta, comp_data = unpack_bundle(raw_bytes)
        except Exception as e:
            st.error(f"❌ Invalid .dnacomp file: {e}")
            st.stop()

        n_tok_back = meta["n_tokens"]
        dna_len    = meta["dna_len"]

        col_x, col_y, col_z = st.columns(3)
        col_x.metric("Compressed Size", f"{comp_sz / 1024:.2f} KB")
        col_y.metric("Original DNA Length", f"{dna_len:,} bases")
        col_z.metric("Tokens to Decode", f"{n_tok_back:,}")

        est_seconds = n_tok_back * 0.004
        st.info(
            f"⏱️ Estimated decompression time: **~{est_seconds / 60:.1f} min** "
            f"({est_seconds:.0f}s at ~250 tok/s on CPU)."
        )

        if st.button("✨ Start Decompression", type="primary", use_container_width=True):
            engine2 = ProbEngine(model)

            progress_bar_dec = st.progress(0, text="Preparing...")

            def ui_progress_decode(desc, i, total, t_start, block_size=BLOCK_SIZE):
                if i % block_size == 0 or i == total:
                    pct     = i / total
                    elapsed = time.time() - t_start
                    rate    = i / elapsed if elapsed > 0 else 0
                    eta     = (total - i) / rate if rate > 0 else 0
                    progress_bar_dec.progress(
                        pct,
                        text=f"{desc} — {i:,}/{total:,} tokens ({pct*100:.1f}%) | "
                             f"{rate:,.0f} tok/s | ETA {eta:.0f}s",
                    )

            dna_engine._progress = ui_progress_decode

            t_dec          = time.time()
            decoded_tokens = lossless_decode(comp_data, n_tok_back, engine2, desc="Decompressing")
            dec_time       = time.time() - t_dec

            recon_dna = tokens_to_dna(decoded_tokens, k=K)[:dna_len]

            progress_bar_dec.empty()

            st.success(f"✅ Decompression complete in **{dec_time:.2f}s**!")

            dm1, dm2 = st.columns(2)
            dm1.metric("Reconstructed DNA Length", f"{len(recon_dna):,} bases")
            dm2.metric("Decompression Time", f"{dec_time:.2f}s")

            # Preview first 500 characters
            with st.expander("🔬 Preview Reconstructed DNA (first 500 bases)"):
                st.code(recon_dna[:500], language=None)

            # Download
            out_name_dec = os.path.splitext(uploaded_comp.name)[0] + "_reconstructed.fasta"
            st.download_button(
                label="📥 Download Reconstructed DNA (.fasta)",
                data=f">reconstructed_sequence\n{recon_dna}\n",
                file_name=out_name_dec,
                mime="text/plain",
                type="primary",
                use_container_width=True,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                        ABOUT TAB
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_about:
    st.subheader("How It Works")

    st.markdown("""
    This application uses a **Causal Transformer** neural network combined with
    **Arithmetic Coding** to achieve lossless compression of DNA sequences.

    #### Pipeline
    1. **Tokenisation** — The raw DNA sequence (A, C, G, T) is converted into
       overlapping **4-mer** tokens, producing a vocabulary of 256 tokens.
    2. **Prediction** — A 6-layer Transformer processes the token sequence
       autoregressively, predicting the probability distribution over the next token.
    3. **Arithmetic Coding** — The predicted probabilities are fed into an
       arithmetic coder, which encodes each token using fewer bits when the model
       is confident. Better predictions → smaller files.
    4. **Decompression** — The decoder runs the *exact same* model to reconstruct
       the identical probability distributions and decodes each token losslessly.

    #### Architecture
    | Component | Value |
    |-----------|-------|
    | K-mer size | 4 |
    | Vocabulary | 256 tokens |
    | Embedding dim | 256 |
    | Attention heads | 8 |
    | Transformer layers | 6 |
    | Feed-forward dim | 1024 |
    | Context window | 320 tokens |

    #### Lossless Guarantee
    The reconstructed DNA is **byte-for-byte identical** to the original.
    No information is lost during compression.
    """)

    st.divider()
    st.caption("© 2026 AI DNA Compressor — Powered by PyTorch & Streamlit")
