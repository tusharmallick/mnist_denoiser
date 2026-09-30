# MNIST Autoencoder Denoiser (Streamlit)

Deploys the best model from Lab Assignment 04 (Undercomplete AE, 784→64→784).

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Use Python 3.10–3.12 (TensorFlow 2.20 requirement).

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo (keep `model/` — it's only ~1.2 MB).
2. Go to https://share.streamlit.io → New app → pick the repo, branch, main file `app.py`.
3. In *Advanced settings* choose Python 3.12, then Deploy.

## Structure
```
app.py
requirements.txt
.python-version
.streamlit/config.toml
model/BEST_MODEL_Undercomplete_AE.keras
```
