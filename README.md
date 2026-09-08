# 🧬 AI DNA Compressor

Lossless DNA sequence compression powered by a Causal Transformer and Arithmetic Coding.

## Features
- **Compress** `.fasta` / `.txt` DNA files using a trained transformer model
- **Decompress** `.dnacomp` files back to the original DNA sequence
- **Lossless** — reconstructed DNA is byte-for-byte identical to the original
- **Live progress** bar and performance metrics dashboard

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your trained model
#    Put best_model.pt in the models/ folder

# 3. Run the app
streamlit run app.py
```

## Project Structure
```
dna_compressor/
├── app.py                 # Streamlit frontend
├── dna_engine.py          # Core compression engine (model, encode, decode)
├── config.py              # All hyperparameters and paths
├── requirements.txt       # Python dependencies
├── packages.txt           # System packages (for cloud hosting)
├── .streamlit/
│   └── config.toml        # Streamlit theme settings
└── models/
    └── (place best_model.pt here)
```

## Hosting on Hugging Face Spaces (Free)

1. Create a free account at https://huggingface.co
2. Create a new Space → select **Streamlit** as the SDK
3. Upload all files from this folder to the Space
4. Upload your `best_model.pt` to the `models/` folder in the Space
5. The app will auto-deploy and give you a public URL

See the detailed hosting guide at the bottom of this README.
