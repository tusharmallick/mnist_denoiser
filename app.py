"""MNIST Image Denoising with an Undercomplete Autoencoder (Lab Assignment 04).

Model: Dense 784 -> 64 (ReLU) -> 784 (sigmoid), trained on noisy (sigma=0.3) -> clean MNIST.
"""
import io
import os
import urllib.request
import zipfile

import h5py
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity as ssim_fn

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "BEST_MODEL_Undercomplete_AE.keras")
TRAIN_NOISE = 0.3
SCALE = 8  # display upscaling (nearest-neighbour keeps pixels crisp)

st.set_page_config(page_title="MNIST Autoencoder Denoiser", page_icon="🧹", layout="wide")


# ----------------------------------------------------------------------------- loaders
class NumpyAutoencoder:
    """Dense 784 -> 64 (ReLU) -> 784 (sigmoid), evaluated with plain NumPy.

    Weights are read straight from the .keras archive (a zip holding model.weights.h5), so the app
    does not depend on TensorFlow/Keras versions matching the ones used for training.
    """

    def __init__(self, path):
        with zipfile.ZipFile(path) as z:
            raw = z.read("model.weights.h5")
        kernels, biases = [], []
        with h5py.File(io.BytesIO(raw), "r") as f:
            def visit(_, obj):
                if isinstance(obj, h5py.Dataset):
                    (kernels if obj.ndim == 2 else biases).append(np.array(obj[()], dtype="float32"))
            f.visititems(visit)
        # identify layers by shape (order in the HDF5 file is alphabetical, not layer order)
        self.W1 = next(k for k in kernels if k.shape == (784, 64))
        self.W2 = next(k for k in kernels if k.shape == (64, 784))
        self.b1 = next(b for b in biases if b.shape == (64,))
        self.b2 = next(b for b in biases if b.shape == (784,))

    def predict(self, x, verbose=0):
        h = np.maximum(x @ self.W1 + self.b1, 0.0)
        return 1.0 / (1.0 + np.exp(-(h @ self.W2 + self.b2)))


@st.cache_resource(show_spinner="Loading autoencoder…")
def load_model():
    return NumpyAutoencoder(MODEL_PATH)


MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


@st.cache_data(show_spinner="Downloading MNIST test set (first run only)…")
def load_mnist_test():
    local = os.path.join(os.path.dirname(__file__), "mnist.npz")
    if not os.path.exists(local):
        local = os.path.join("/tmp", "mnist.npz")
        if not os.path.exists(local):
            urllib.request.urlretrieve(MNIST_URL, local)
    with np.load(local) as d:
        x_test, y_test = d["x_test"], d["y_test"]
    return x_test.reshape(-1, 784).astype("float32") / 255.0, y_test


# ----------------------------------------------------------------------------- helpers
def add_noise(x, sigma, seed):
    rng = np.random.RandomState(seed)
    return np.clip(x + rng.normal(0.0, sigma, size=x.shape), 0.0, 1.0).astype("float32")


def denoise(model, x_flat):
    x_flat = np.atleast_2d(x_flat).astype("float32")
    return model.predict(x_flat, verbose=0)


def metrics(clean, pred):
    clean, pred = clean.reshape(28, 28), pred.reshape(28, 28)
    mse = float(np.mean((clean - pred) ** 2))
    psnr = float(10 * np.log10(1.0 / max(mse, 1e-10)))
    ssim = float(ssim_fn(clean, pred, data_range=1.0))
    return mse, psnr, ssim


def show(img_flat, caption):
    big = np.kron(img_flat.reshape(28, 28), np.ones((SCALE, SCALE)))
    st.image(big, caption=caption, clamp=True, use_container_width=True)


def preprocess_upload(file, invert, autocontrast):
    img = Image.open(file).convert("L")
    if autocontrast:
        img = ImageOps.autocontrast(img)
    if invert:
        img = ImageOps.invert(img)
    img = ImageOps.fit(img, (28, 28), Image.LANCZOS)
    return np.asarray(img, dtype="float32").reshape(784) / 255.0


# ----------------------------------------------------------------------------- UI
st.title("🧹 MNIST Image Denoising with an Autoencoder")
st.caption("Undercomplete AE · Dense 784 → 64 → 784 · trained on noisy (σ = 0.3) → clean MNIST digits")

model = load_model()
tab_demo, tab_grid, tab_results, tab_about = st.tabs(
    ["🔬 Denoise a digit", "🖼️ Batch grid", "📊 Lab results", "ℹ️ About"]
)

# ---- sidebar
with st.sidebar:
    st.header("Settings")
    sigma = st.slider("Gaussian noise σ", 0.0, 0.6, TRAIN_NOISE, 0.05,
                      help="Model was trained at σ = 0.3. It works best around 0.2–0.4.")
    seed = st.number_input("Random seed", 0, 99999, 42, 1)
    if abs(sigma - TRAIN_NOISE) > 0.15:
        st.warning("Far from the training noise level (0.3) – expect weaker results.")

# ---- tab 1: single image
with tab_demo:
    source = st.radio("Image source", ["MNIST test image", "Upload my own image"], horizontal=True)
    clean, label, noisy_given = None, None, False

    if source == "MNIST test image":
        X, y = load_mnist_test()
        if "idx" not in st.session_state:
            st.session_state.idx = 0

        def _rand():
            st.session_state.idx = int(np.random.randint(0, len(X)))

        c1, c2 = st.columns([4, 1])
        c1.slider("Test image index", 0, len(X) - 1, key="idx")
        c2.write("")
        c2.button("🎲 Random", on_click=_rand)
        idx = st.session_state.idx
        clean, label = X[idx], f"Label: {y[idx]}"
    else:
        up = st.file_uploader("Upload a PNG/JPG of a single digit", type=["png", "jpg", "jpeg"])
        o1, o2, o3 = st.columns(3)
        invert = o1.checkbox("Invert colours", True, help="MNIST = white digit on black. Tick for dark ink on white paper.")
        autoc = o2.checkbox("Auto-contrast", True)
        noisy_given = o3.checkbox("Image is already noisy (don't add noise)", False)
        if up is not None:
            clean, label = preprocess_upload(up, invert, autoc), "Uploaded image (28×28)"

    if clean is None:
        st.info("Upload an image to begin.")
    else:
        noisy = clean.copy() if noisy_given else add_noise(clean, sigma, int(seed))
        out = denoise(model, noisy)[0]

        a, b, c = st.columns(3)
        with a:
            show(clean, "Original" if not noisy_given else "Input")
        with b:
            show(noisy, f"Noisy (σ={sigma})" if not noisy_given else "Noisy input")
        with c:
            show(out, "Denoised output")

        if not noisy_given:
            m_noisy, m_out = metrics(clean, noisy), metrics(clean, out)
            st.subheader("Quality vs. the clean original")
            df = pd.DataFrame(
                [m_noisy, m_out], index=["Noisy input", "Autoencoder output"],
                columns=["MSE ↓", "PSNR (dB) ↑", "SSIM ↑"],
            ).round(4)
            st.dataframe(df, use_container_width=True)
        else:
            st.caption("Metrics need a clean reference, so they're hidden for already-noisy uploads.")

# ---- tab 2: grid
with tab_grid:
    st.write("Random MNIST test digits: **original → noisy → denoised**.")
    n = st.slider("Number of digits", 4, 12, 8)
    gseed = st.number_input("Grid seed", 0, 99999, 7, 1)
    if st.button("Generate grid", type="primary"):
        X, y = load_mnist_test()
        ids = np.random.RandomState(int(gseed)).choice(len(X), n, replace=False)
        cl = X[ids]
        ny = add_noise(cl, sigma, int(seed))
        ou = denoise(model, ny)
        for title, arr in [("Original", cl), ("Noisy", ny), ("Denoised", ou)]:
            st.markdown(f"**{title}**")
            cols = st.columns(n)
            for col, img in zip(cols, arr):
                with col:
                    show(img, "")
        mse = float(np.mean((cl - ou) ** 2, axis=1).mean())
        mse_n = float(np.mean((cl - ny) ** 2, axis=1).mean())
        st.success(f"Mean MSE on this grid — noisy: {mse_n:.4f} → denoised: {mse:.4f}")

# ---- tab 3: lab results (from the notebook)
with tab_results:
    st.subheader("Five autoencoders at σ = 0.3 (full 10,000-image test set)")
    res = pd.DataFrame(
        {
            "Parameters": [101200, 402448, 484944, 3217, 101200],
            "Train time (s)": [11.6, 13.3, 14.1, 24.2, 11.3],
            "MSE": [0.01040, 0.08589, 0.01389, 0.11396, 0.04896],
            "MAE": [0.03858, 0.25510, 0.04278, 0.13252, 0.12811],
            "PSNR": [20.23, 10.73, 19.11, 9.73, 13.30],
            "SSIM": [0.8011, 0.1004, 0.8260, 0.2407, 0.3236],
            "Mean rank": [1.25, 4.50, 1.75, 4.50, 3.00],
        },
        index=["Undercomplete AE ⭐", "Sparse AE", "Denoising AE", "Convolutional AE", "Contractive AE"],
    )
    st.dataframe(res, use_container_width=True)
    st.caption("Noisy-input baseline (σ = 0.3): MSE 0.0466 · MAE 0.1279 · PSNR 13.33 · SSIM 0.5281. "
               "Undercomplete AE won on mean rank and is the model deployed here.")

    st.subheader("Undercomplete AE – effect of latent size (σ = 0.3)")
    lat = pd.DataFrame(
        {"Latent dim": [8, 16, 32, 64, 128],
         "MSE": [0.03921, 0.02563, 0.01558, 0.01154, 0.00871],
         "PSNR": [14.34, 16.29, 18.51, 19.80, 20.98],
         "SSIM": [0.4598, 0.6160, 0.7303, 0.7716, 0.8013]}
    ).set_index("Latent dim")
    st.line_chart(lat[["SSIM"]])
    st.dataframe(lat, use_container_width=True)

    st.subheader("SSIM vs. noise level (first 2,000 test images)")
    noise = pd.DataFrame(
        {"Noisy input": [0.6790, 0.5942, 0.5097, 0.4286, 0.3593],
         "Undercomplete AE": [0.5207, 0.6599, 0.7772, 0.7107, 0.5103],
         "Denoising AE": [0.8039, 0.8166, 0.8089, 0.7708, 0.7035]},
        index=pd.Index([0.1, 0.2, 0.3, 0.4, 0.5], name="Noise σ"),
    )
    st.line_chart(noise)
    st.caption("The Undercomplete AE peaks at its training noise (σ = 0.3) and can be *worse* than the raw input at σ = 0.1, "
               "because it was never trained on such clean inputs.")

# ---- tab 4: about
with tab_about:
    st.markdown(
        """
**Pipeline:** clean image → add Gaussian noise (σ) → clip to [0, 1] → encoder (Dense 64, ReLU) → decoder (Dense 784, sigmoid) → cleaned image.

**Training:** MNIST, 10 epochs, Adam, MSE loss, batch size 256, 10 % validation split, noisy input → clean target.

**Metrics:** MSE, PSNR and SSIM are computed between the model output and the *clean* original.

**Tips for uploads:** use a single, centred digit, high contrast, roughly square. The app resizes to 28×28 and (optionally) inverts colours so the digit is white on black, like MNIST.

**Limitations:** a small dense model – it is trained only on MNIST-style digits, so it won't denoise natural photos.
"""
    )
