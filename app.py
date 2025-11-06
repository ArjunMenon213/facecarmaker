import streamlit as st
from PIL import Image
import numpy as np
import io
from utils import detect_face_bbox, paste_face_on_template

st.set_page_config(page_title="Origami Meme Car Maker", layout="centered")
st.title("Origami Meme Car Maker")
st.write("Upload a portrait and the app will place the face onto an origami car template (printable).")

# Sidebar: template and mask
st.sidebar.header("Template / Mask")
template_file = st.sidebar.file_uploader("Upload car template PNG (optional)", type=["png"])
mask_file = st.sidebar.file_uploader("Upload mask PNG (optional, same size as template)", type=["png"])
use_defaults = st.sidebar.checkbox("Use template.png & mask.png from app folder if available", value=True)

st.sidebar.markdown("---")
st.sidebar.write("Tips: create a mask PNG where the face area is filled white and the rest is transparent/black. The script will auto-detect the white region to place the face.")

# Main uploader
uploaded = st.file_uploader("Upload a portrait (frontal face works best)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a portrait to get started.")
    st.stop()

# Load portrait
try:
    portrait = Image.open(uploaded).convert("RGBA")
except Exception as e:
    st.error(f"Could not open the uploaded image: {e}")
    st.stop()

st.header("Source portrait")
st.image(portrait, use_column_width=True)

# Load template and mask (either uploaded or defaults)
def load_template_and_mask():
    if template_file is not None and mask_file is not None:
        try:
            tpl = Image.open(template_file).convert("RGBA")
            msk = Image.open(mask_file).convert("L")
            return tpl, msk
        except Exception as e:
            st.error(f"Failed to open uploaded template or mask: {e}")
            return None, None

    if use_defaults:
        try:
            tpl = Image.open("template.png").convert("RGBA")
            msk = Image.open("mask.png").convert("L")
            return tpl, msk
        except FileNotFoundError:
            # fall through to None return
            pass

    st.warning("No template/mask provided. The result will use a centered fallback area.")
    return None, None

template, mask = load_template_and_mask()

if template is not None:
    st.subheader("Template preview")
    st.image(template, use_column_width=True)

# Face detection and crop
st.subheader("Face detection")
expansion_pct = st.slider("Face crop expansion (%)", min_value=0, max_value=100, value=20, help="Expand detected face box to include hair/forehead.")
# Detect using Haar cascades (utils.detect_face_bbox)
rgb_np = np.array(portrait.convert("RGB"))
face_bbox = detect_face_bbox(rgb_np, method="haar")

if face_bbox is None:
    st.warning("No face automatically detected. Using the whole uploaded image as the face source.")
    face_crop = portrait.copy()
else:
    x, y, w, h = face_bbox
    ex = int(w * (expansion_pct / 100.0))
    ey = int(h * (expansion_pct / 100.0))
    left = max(0, x - ex)
    top = max(0, y - ey)
    right = min(portrait.width, x + w + ex)
    bottom = min(portrait.height, y + h + ey)
    face_crop = portrait.crop((left, top, right, bottom))

st.image(face_crop, caption="Cropped face source", width=240)

# Adjustment controls
st.subheader("Adjustments")
auto_scale = st.checkbox("Auto scale to mask area (recommended)", value=True)
manual_scale = None
if not auto_scale:
    scale_pct = st.slider("Manual scale (%)", 10, 300, 100)
    manual_scale = scale_pct / 100.0

offset_x = st.slider("Offset X (pixels)", -500, 500, 0)
offset_y = st.slider("Offset Y (pixels)", -500, 500, 0)
blend = st.slider("Blend/opacity (0.0 = transparent, 1.0 = opaque)", 0.0, 1.0, 0.9)

# Compose result
if template is None or mask is None:
    # create a blank white template to preview placement if none supplied
    tw, th = 2480, 3508  # default A4 at 300 DPI-ish for printable preview (large canvas)
    template_preview = Image.new("RGBA", (tw, th), (255, 255, 255, 255))
    mask_preview = Image.new("L", (tw, th), 0)
    # create a centered rectangle as mask target (40% height)
    rect_w = int(tw * 0.4)
    rect_h = int(th * 0.4)
    left = (tw - rect_w) // 2
    top = (th - rect_h) // 2
    mask_arr = mask_preview.load()
    for yy in range(top, top + rect_h):
        for xx in range(left, left + rect_w):
            mask_arr[xx, yy] = 255
    template_to_use, mask_to_use = template_preview, mask_preview
else:
    template_to_use, mask_to_use = template, mask

result = paste_face_on_template(
    template_to_use,
    mask_to_use,
    face_crop,
    manual_scale=manual_scale,
    manual_offset=(offset_x, offset_y),
    blend=blend
)

st.subheader("Result (printable)")
st.image(result, use_column_width=True)

# Download
buf = io.BytesIO()
result.convert("RGBA").save(buf, format="PNG")
buf.seek(0)
st.download_button("Download PNG (printable)", data=buf, file_name="origami_car_with_face.png", mime="image/png")

st.markdown("Instructions: print the PNG at desired size (A4 recommended), cut along lines, and fold as indicated.")
