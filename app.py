import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import io

st.set_page_config(page_title="Simple Origami Meme Car — Manual Crop & Warp", layout="centered")
st.title("Origami Meme Car — Manual rectangular crop + blurred background fill")

st.write(
    "Upload: (1) a portrait/photo, (2) a template image + mask image where the mask uses TWO vivid colors: "
    "GREEN = face rectangular area (will receive the user's crop), WHITE = area to be filled with a blurred/warped version "
    "of the photo. No face recognition — you control the rectangular crop and zoom."
)

# Uploads
col1, col2 = st.columns(2)
with col1:
    uploaded_photo = st.file_uploader("1) Upload portrait/photo", type=["jpg", "jpeg", "png"])
with col2:
    uploaded_template = st.file_uploader("2) Upload template (artwork PNG with mask colors visible)", type=["png"])

uploaded_mask = st.file_uploader("3) Upload mask image (same size as template). Mask must use vivid GREEN and WHITE", type=["png"])

st.markdown("---")

if uploaded_photo is None:
    st.info("Please upload a portrait to begin.")
    st.stop()

# Load photo
photo = Image.open(uploaded_photo).convert("RGBA")
photo_w, photo_h = photo.size

st.subheader("Photo preview")
st.image(photo, use_column_width=False, width=320)

# Load template & mask (optional fallback)
template = None
mask = None
if uploaded_template is not None:
    try:
        template = Image.open(uploaded_template).convert("RGBA")
    except Exception as e:
        st.error(f"Could not open template image: {e}")
if uploaded_mask is not None:
    try:
        mask = Image.open(uploaded_mask).convert("RGBA")
    except Exception as e:
        st.error(f"Could not open mask image: {e}")

# If template/mask mismatch sizes or missing, create a fallback printable canvas
if template is None or mask is None:
    st.warning("Template or mask not provided. Using a fallback A4-like white canvas and generated mask sample.")
    # A reasonable printable canvas (A4-ish) at moderate px
    canvas_w, canvas_h = 1240, 1754
    template = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    # Create a mask with white rectangle (top) and green rectangle (center) for testing/demo
    mask = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    mpx = mask.load()
    # white region (upper-middle)
    white_box = (260, 180, 980, 560)
    for y in range(white_box[1], white_box[3]):
        for x in range(white_box[0], white_box[2]):
            mpx[x, y] = (255, 255, 255, 255)
    # green face rect (below)
    green_box = (360, 640, 880, 1250)
    for y in range(green_box[1], green_box[3]):
        for x in range(green_box[0], green_box[2]):
            mpx[x, y] = (0, 255, 0, 255)

st.subheader("Template preview")
st.image(template, use_column_width=False, width=360)
st.subheader("Mask preview (green = face box, white = blurred fill area)")
st.image(mask, use_column_width=False, width=360)

# Helper: find green and white regions in mask
def find_color_bbox(mask_rgba, color="green"):
    """
    mask_rgba: PIL RGBA image
    color: "green" or "white"
    returns bounding box (left, top, right, bottom) or None
    """
    arr = np.array(mask_rgba.convert("RGBA"))
    r = arr[..., 0].astype(int)
    g = arr[..., 1].astype(int)
    b = arr[..., 2].astype(int)
    a = arr[..., 3].astype(int)

    if color == "green":
        # vivid green detection: G high, R and B low
        mask_bool = (g > 200) & (r < 100) & (b < 100) & (a > 10)
    elif color == "white":
        # white detection: all channels high
        mask_bool = (r > 200) & (g > 200) & (b > 200) & (a > 10)
    else:
        return None

    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    left = int(xs.min())
    right = int(xs.max())
    top = int(ys.min())
    bottom = int(ys.max())
    return (left, top, right, bottom)

green_box = find_color_bbox(mask, "green")
white_box = find_color_bbox(mask, "white")

if green_box is None:
    st.error("Could not detect a GREEN box in the mask. Make sure the mask uses vivid green (R~0,G~255,B~0) for the face rectangle.")
    st.stop()
if white_box is None:
    st.error("Could not detect a WHITE region in the mask. Make sure the mask uses vivid white (R~255,G~255,B~255) for the blurred area.")
    st.stop()

st.write(f"Detected GREEN box (face) at: {green_box}")
st.write(f"Detected WHITE box (blur fill) at: {white_box}")

# Manual rectangular crop control for the face (user-controlled)
st.subheader("Manual rectangular crop (use sliders to position & zoom the crop to fit the green face area)")

# Default crop is centered on the photo
center_x_pct = st.slider("Crop center X (%)", 0, 100, 50)
center_y_pct = st.slider("Crop center Y (%)", 0, 100, 50)
zoom_pct = st.slider("Crop zoom (%) — higher = zoom in (100 = no zoom)", 10, 400, 100)

# Compute crop rectangle in photo coordinates
cx = int(center_x_pct / 100.0 * photo_w)
cy = int(center_y_pct / 100.0 * photo_h)
# base crop size: proportional to green box aspect ratio to help match shape
green_w = green_box[2] - green_box[0]
green_h = green_box[3] - green_box[1]
# Start with photo base dimension proportional to green aspect, but relative to photo size:
# We'll choose base_size = min(photo_w, photo_h) * 0.5 as starting square
base = int(min(photo_w, photo_h) * 0.5)
# Adjust base to match green aspect
if green_w > 0 and green_h > 0:
    target_aspect = green_w / green_h
else:
    target_aspect = 1.0

# determine crop width and height before zoom
crop_w = base
crop_h = int(base / target_aspect) if target_aspect != 0 else base
# apply zoom
crop_w = max(10, int(crop_w * (100.0 / zoom_pct)))
crop_h = max(10, int(crop_h * (100.0 / zoom_pct)))

# ensure within image bounds
left = max(0, cx - crop_w // 2)
top = max(0, cy - crop_h // 2)
right = min(photo_w, left + crop_w)
bottom = min(photo_h, top + crop_h)
# Re-center if truncated
if right - left < crop_w:
    left = max(0, right - crop_w)
if bottom - top < crop_h:
    top = max(0, bottom - crop_h)

crop_box = (left, top, right, bottom)
st.write(f"Crop box on photo (pixels): {crop_box}")

face_crop = photo.crop(crop_box).convert("RGBA")
st.image(face_crop, caption="Face crop preview (this will be placed into the GREEN box)", width=220)

# Provide small offsets for final alignment inside the green box
st.subheader("Fine-tune face placement inside green box")
offset_x = st.slider("Face offset X (pixels)", -500, 500, 0)
offset_y = st.slider("Face offset Y (pixels)", -500, 500, 0)
face_rotation = st.slider("Rotate face (degrees)", -30, 30, 0)

# Prepare blurred + warped fill for white box
st.subheader("Blur & warp settings for WHITE fill area")
blur_radius = st.slider("Blur radius (pixels)", 0, 80, 18)
shear_x = st.slider("Horizontal shear for warp (-0.5..0.5)", -50, 50, 0)  # percent of width
shear_y = st.slider("Vertical shear for warp (-0.5..0.5)", -50, 50, 0)    # percent of height
edge_feather = st.slider("Feather edge of white area (px)", 0, 100, 18)

# Create blurred source for white region: take the full photo, center crop to white box aspect, resize to white box
def create_blurred_warp(photo_img, target_box, blur_r, shear_x_pct, shear_y_pct):
    tw = target_box[2] - target_box[0]
    th = target_box[3] - target_box[1]
    # center-crop photo to match target aspect
    pw, ph = photo_img.size
    target_aspect = tw / th if th != 0 else 1.0
    photo_aspect = pw / ph if ph != 0 else 1.0

    if photo_aspect > target_aspect:
        # photo is wider -> crop width
        new_h = ph
        new_w = int(ph * target_aspect)
    else:
        # crop height
        new_w = pw
        new_h = int(pw / target_aspect) if target_aspect != 0 else ph

    left = max(0, (pw - new_w) // 2)
    top = max(0, (ph - new_h) // 2)
    right = left + new_w
    bottom = top + new_h
    cropped = photo_img.crop((left, top, right, bottom)).convert("RGBA")

    # resize to target box
    resized = cropped.resize((tw, th), Image.LANCZOS)

    # blur
    blurred = resized.filter(ImageFilter.GaussianBlur(radius=blur_r))

    # apply simple affine shear warp
    # Affine transform data: (a, b, c, d, e, f) mapping output x,y -> input x',y' : x' = a*x + b*y + c ; y' = d*x + e*y + f
    # We'll create mild shears based on shear_x_pct and shear_y_pct (percent of width/height)
    shear_x = float(shear_x_pct) / 100.0
    shear_y = float(shear_y_pct) / 100.0
    a = 1.0
    b = -shear_x  # horizontal shear controlled by slider
    c = 0.0
    d = -shear_y
    e = 1.0
    f = 0.0
    try:
        warped = blurred.transform((tw, th), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)
    except Exception:
        warped = blurred
    return warped

warped_blur = create_blurred_warp(photo, white_box, blur_radius, shear_x, shear_y)
st.image(warped_blur, caption="Blurred + warped preview (will fill WHITE area)", width=300)

# Compose final image
st.subheader("Final composition preview")

# Convert mask to binary alpha for white and green
mask_arr = np.array(mask.convert("RGBA"))
r = mask_arr[..., 0]
g = mask_arr[..., 1]
b = mask_arr[..., 2]
a = mask_arr[..., 3]

white_mask_bool = (r > 200) & (g > 200) & (b > 200) & (a > 10)
green_mask_bool = (g > 200) & (r < 100) & (b < 100) & (a > 10)

# Create alpha masks
mask_height, mask_width = mask_arr.shape[0], mask_arr.shape[1]
white_alpha = Image.fromarray((white_mask_bool * 255).astype(np.uint8)).convert("L")
green_alpha = Image.fromarray((green_mask_bool * 255).astype(np.uint8)).convert("L")

# Feather the white mask edges by blurring the alpha
if edge_feather > 0:
    white_alpha_feather = white_alpha.filter(ImageFilter.GaussianBlur(radius=edge_feather))
else:
    white_alpha_feather = white_alpha

# Prepare layer
composed = template.copy().convert("RGBA")
layer = Image.new("RGBA", composed.size, (0, 0, 0, 0))

# Paste warped_blur into white_box with feathered alpha
tw = white_box[2] - white_box[0]
th = white_box[3] - white_box[1]
# Ensure warped_blur is same size (it should be)
wb = warped_blur.resize((tw, th), Image.LANCZOS)
# Create a full-size layer for the warped blur to paste into using alpha
blur_layer = Image.new("RGBA", composed.size, (0, 0, 0, 0))
blur_layer.paste(wb, (white_box[0], white_box[1]))
# Use the feathered white_alpha as mask (full-size)
full_white_alpha = Image.new("L", composed.size, 0)
full_white_alpha.paste(white_alpha_feather, (0, 0))
layer = Image.alpha_composite(layer, Image.composite(blur_layer, Image.new("RGBA", composed.size, (0,0,0,0)), full_white_alpha))

# Prepare face crop: rotate and resize to green box while preserving aspect
gw = green_box[2] - green_box[0]
gh = green_box[3] - green_box[1]
# Rotate around center if requested
if face_rotation != 0:
    face_crop = face_crop.rotate(face_rotation, resample=Image.BICUBIC, expand=True)

# Resize face crop to fit green box while preserving aspect (cover)
fw, fh = face_crop.size
scale_w = gw / fw
scale_h = gh / fh
scale = max(scale_w, scale_h)  # cover approach so face fills box
new_fw = max(1, int(fw * scale))
new_fh = max(1, int(fh * scale))
face_resized = face_crop.resize((new_fw, new_fh), Image.LANCZOS)

# compute paste position inside green box with offsets
paste_x = green_box[0] + (gw - new_fw) // 2 + offset_x
paste_y = green_box[1] + (gh - new_fh) // 2 + offset_y

# Make sure paste coords keep image on canvas (clamp)
paste_x = int(max(-new_fw, min(composed.size[0], paste_x)))
paste_y = int(max(-new_fh, min(composed.size[1], paste_y)))

# Create a mask for the green box only (so face is clipped to green region)
green_clip_mask = Image.new("L", composed.size, 0)
green_clip_mask.paste(green_alpha, (0, 0))

# We'll paste face_resized onto a temporary layer, then composite with green alpha
face_layer = Image.new("RGBA", composed.size, (0, 0, 0, 0))
face_layer.paste(face_resized, (paste_x, paste_y), face_resized)

# Clip face_layer using the green alpha mask so only green-area shows the face
clipped_face = Image.composite(face_layer, Image.new("RGBA", composed.size, (0,0,0,0)), green_clip_mask)

# Put everything together
final = Image.alpha_composite(composed, layer)
final = Image.alpha_composite(final, clipped_face)

st.image(final, use_column_width=True)

# Download button
buf = io.BytesIO()
final.convert("RGBA").save(buf, format="PNG")
buf.seek(0)
st.download_button("Download final PNG (printable)", data=buf, file_name="origami_car_result.png", mime="image/png")

st.markdown(
    "Usage tips:\n"
    "- Move the crop center sliders and adjust zoom so the important part of the face is inside the green preview.\n"
    "- Use rotation and offsets to align the face inside the green rectangle exactly.\n"
    "- Increase blur and feathering to better blend the background into the white area.\n"
    "- If your mask colors are slightly different from pure green/white, edit the mask image to use vivid RGB green (0,255,0) and white (255,255,255)."
)
