import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Oil Spill Detection", layout="wide")
st.title("🛢️ Oil Spill Detection – Visualization Dashboard")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/oilspill_unet.h5")

model = load_model()

# ---------------- PREPROCESS ----------------
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("L")
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- OVERLAY ----------------
def create_overlay(rgb_img, mask, alpha=0.4):
    overlay = rgb_img.copy()
    oil_color = np.array([255, 0, 0])
    oil_pixels = mask == 1
    overlay[oil_pixels] = (
        (1 - alpha) * overlay[oil_pixels] + alpha * oil_color
    ).astype(np.uint8)
    return overlay

# ---------------- PDF ----------------
def generate_pdf(original_img, overlay_img, oil_percent, is_oil):
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

    pdf = canvas.Canvas(tmp.name, pagesize=A4)
    w, h = A4

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawCentredString(w / 2, h - 40, "Oil Spill Detection Report")

    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, h - 80, f"Generated on: {timestamp}")
    pdf.drawString(
        50, h - 100,
        f"Result: {'Oil Spill Detected' if is_oil else 'No Oil Spill Detected'}"
    )
    pdf.drawString(50, h - 120, f"Oil Coverage: {oil_percent:.3f}%")

    tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")

    Image.fromarray(original_img).save(tmp1.name)
    Image.fromarray(overlay_img).save(tmp2.name)

    pdf.drawImage(ImageReader(tmp1.name), 50, h - 420, 220, 220)
    pdf.drawImage(ImageReader(tmp2.name), 320, h - 420, 220, 220)

    pdf.setFont("Helvetica-Oblique", 9)
    pdf.drawCentredString(w / 2, 30, "AI-Based Oil Spill Detection System")

    pdf.save()
    return tmp.name

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload SAR Image (JPG / PNG)",
    ["jpg", "jpeg", "png"]
)

if uploaded_file:

    # ---------- MODEL ----------
    img_array = preprocess_image(uploaded_file)
    with st.spinner("Running U-Net segmentation..."):
        prob_map = model.predict(img_array)[0, :, :, 0]

    # ---------- MASK ----------
    binary_mask = (prob_map > 0.7).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # ---------- IMAGES ----------
    original_rgb = Image.open(uploaded_file).convert("RGB")
    original_np = np.array(original_rgb)

    mask_resized = cv2.resize(
        binary_mask,
        (original_np.shape[1], original_np.shape[0]),
        cv2.INTER_NEAREST
    )
    mask_display = (mask_resized * 255).astype(np.uint8)

    prob_resized = cv2.resize(
        prob_map,
        (original_np.shape[1], original_np.shape[0]),
        cv2.INTER_LINEAR
    )

    overlay_img = create_overlay(original_np, mask_resized)

    # ---------- VISUALIZATION ----------
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(original_np); axes[0].set_title("Original Image"); axes[0].axis("off")
    axes[1].imshow(mask_display, cmap="gray"); axes[1].set_title("Predicted Mask"); axes[1].axis("off")
    im = axes[2].imshow(prob_resized, cmap="hot"); axes[2].set_title("Prediction Probability"); axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    axes[3].imshow(overlay_img); axes[3].set_title("Overlay"); axes[3].axis("off")
    st.pyplot(fig)

    # ================= CORRECT DECISION ORDER =================

    mean_prob = np.mean(prob_map)
    std_prob = np.std(prob_map)

    oil_pixels = np.sum(binary_mask == 1)
    total_pixels = binary_mask.size
    oil_percent_raw = (oil_pixels / total_pixels) * 100

    num_labels, labels = cv2.connectedComponents(binary_mask)
    largest_area = max([np.sum(labels == i) for i in range(1, num_labels)] + [0])

    # 🚨 RULE 1: UNIFORM PROBABILITY → NO OIL
    if std_prob < 0.05:
        is_oil = False
        oil_percent = 0.0

    # 🚨 RULE 2: REAL OIL CHARACTERISTICS
    elif (
        (largest_area > 400 and oil_percent_raw > 0.2)
        or
        (std_prob > 0.12 and mean_prob > 0.55)
    ):
        is_oil = True
        oil_percent = oil_percent_raw

    # 🚨 RULE 3: DEFAULT
    else:
        is_oil = False
        oil_percent = 0.0

    # ---------- UI ----------
    if is_oil:
        st.error("⚠️ Oil Spill Detected")
        st.warning(f"Oil Coverage: {oil_percent:.3f}%")
    else:
        st.success("✅ No Oil Spill Detected")
        st.info("Oil Coverage: 0.000%")

    # ---------- METRICS ----------
    with st.expander("🔍 Detection Metrics"):
        st.write(f"Raw Oil %: {oil_percent_raw:.3f}")
        st.write(f"Final Oil %: {oil_percent:.3f}")
        st.write(f"Largest Area: {largest_area}")
        st.write(f"Mean Probability: {mean_prob:.3f}")
        st.write(f"Std Probability: {std_prob:.3f}")

    # ---------- DOWNLOAD ----------
    pdf_path = generate_pdf(original_np, overlay_img, oil_percent, is_oil)
    with open(pdf_path, "rb") as f:
        st.download_button(
            "⬇️ Download Report (PDF)",
            f,
            file_name=f"oil_spill_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )
