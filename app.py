import streamlit as st
from PIL import Image, ImageOps
import io
from utils import detect_face_bbox, paste_face_on_template
import numpy as np

st.set_page_config(page_title="Origami Meme Car Maker", layout="centered")

st.title("Origami Meme Car Maker")
st.write("Upload a portrait and automatically place the face onto a printable origami car template.")

# Sidebar: template and mask load
st.sidebar.header("Template settings")
template_file = st.sidebar.file_uploader("Upload car template PNG (optional)", type=["png"])
mask_file = st.sidebar.file_uploader("Upload mask PNG (same size as template, white = face area)", type=["png"])

if template_file is None or mask_file is None:
    st.sidebar.info("If you don't upload template & mask here, the app expects 'template.png' and 'mask.png' in the app folder.")
    use_default = st.sidebar.checkbox("Try default files in working dir", value=True)
else:
    use_default = False

uploaded = st.file_uploader("Upload a portrait (front-facing works best)", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded is None:
    st.info("Upload a portrait to get started. Portrait with face looking roughly forward gives best results.")
else:
    # Load portrait
    image = Image.open(uploaded).convert("RGBA")
    st.header("Source portrait")
    st.image(image, use_column_width=True)

    # Load template & mask
    if not use_default:
        if template_file is None or mask_file is None:
            st.warning("Upload template and mask in the sidebar or enable default files.")
            st.stop()
        template = Image.open(template_file).convert("RGBA")
        mask = Image.open(mask_file).convert("L")
    else:
        try:
            template = Image.open("template.png").convert("RGBA")
            mask = Image.open("mask.png").convert("L")
        except FileNotFoundError:
            st.error("Default template.png / mask.png not found in app folder. Please upload them in the sidebar.")
            st.stop()

    st.subheader("Template preview")
    st.image(template, use_column_width=True)

    # Detect face bbox in uploaded image
    expand = st.slider("Face crop expansion (%)", min_value=0, max_value=100, value=20)
    detection_method = st.selectbox("Face detection engine", ["MediaPipe (default)"])
    face_bbox = detect_face_bbox(np.array(image.convert("RGB")), method="mediapipe")
    if face_bbox is None:
        st.warning("No face automatically detected. The whole image will be used. You can re-upload a different image.")
        face_crop = image
    else:
        x, y, w, h = face_bbox
        # expand box
        ex = int(w * (expand / 100))
        ey = int(h * (expand / 100))
        left = max(0, x - ex)
        top = max(0, y - ey)
        right = min(image.width, x + w + ex)
        bottom = min(image.height, y + h + ey)
        face_crop = image.crop((left, top, right, bottom))

    st.subheader("Detected face (cropped)")
    st.image(face_crop, use_column_width=False, width=240)

    # find mask target box and apply paste
    result = paste_face_on_template(template, mask, face_crop, align="center", blend=0.85)

    st.subheader("Result (printable)")
    st.image(result, use_column_width=True)

    # small manual adjustments: scale and offset
    st.subheader("Manual adjustments (if needed)")
    scale = st.slider("Scale face (percent)", 50, 200, 100)
    offset_x = st.slider("Offset X (pixels)", -300, 300, 0)
    offset_y = st.slider("Offset Y (pixels)", -300, 300, 0)

    result_adjusted = paste_face_on_template(template, mask, face_crop, align="center", blend=0.85,
                                             manual_scale=scale / 100.0, manual_offset=(offset_x, offset_y))

    st.image(result_adjusted, use_column_width=True)

    # Download button
    buf = io.BytesIO()
    result_adjusted.convert("RGBA").save(buf, format="PNG")
    buf.seek(0)
    st.download_button(
        label="Download PNG (printable)",
        data=buf,
        file_name="origami_car_with_face.png",
        mime="image/png"
    )

    st.markdown("Instructions: print the PNG at desired size (A4 recommended), cut along lines, fold where indicated.")
