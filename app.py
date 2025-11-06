# https://github.com/ArjunMenon213/facecarmaker/blob/main/app.py
"""
Origami Meme Car — Fixed Mask, Interactive Crop

- Place template.png and mask.png in the app folder (mask must use vivid GREEN (0,255,0)
  to mark the rectangular face slot; vivid WHITE (255,255,255) marks the area to be
  filled by a blurred/warped version of the photo).
- Users upload a photo, draw/resize one rectangle on the photo canvas, then the app
  crops that region and places it (cover behavior) into the GREEN region of the template.
- The app uses streamlit-drawable-canvas for the on-canvas rectangle. If the component
  isn't available or the host's Streamlit is older, the app includes a compatibility
  fallback that usually avoids the "image_to_url" AttributeError and also falls back
  to a slider-based crop UI if needed.
"""

import streamlit as st
from PIL import Image, ImageFilter
import numpy as np
import io
import base64
from io import BytesIO

st.set_page_config(page_title="Origami Meme Car — Interactive Crop", layout="centered")
st.title("Origami Meme Car — Place face into fixed GREEN slot")

st.write(
    "Place template.png and mask.png in the app folder. Mask: pure GREEN (0,255,0) = face slot; "
    "pure WHITE (255,255,255) = blurred fill area. Upload a photo and draw a rectangle on it to select the face."
)

# --- Compatibility monkey-patch for streamlit_drawable_canvas when Streamlit missing image_to_url ---
# streamlit-drawable-canvas calls an internal helper to convert background images to URLs.
# Older Streamlit builds might not expose the helper the component expects. We add a safe fallback.
try:
    # Try to import the module that the component expects to call
    from streamlit.elements import image as st_image_module  # newer Streamlit internals
except Exception:
    # If not available, fall back to attaching to the streamlit module itself
    st_image_module = st

if not hasattr(st_image_module, "image_to_url"):
    def _image_to_url_fallback(img):
        # Accept PIL.Image or numpy arrays
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img)
        else:
            pil_img = img
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    st_image_module.image_to_url = _image_to_url_fallback
# --- end monkey-patch ---

# Try to import the drawable canvas component
USE_CANVAS = True
try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    USE_CANVAS = False

# Load fixed template and mask from repo folder
try:
    template = Image.open("template.png").convert("RGBA")
    mask = Image.open("mask.png").convert("RGBA")
except FileNotFoundError:
    st.error("Could not find template.png and/or mask.png in the app folder. Add them and reload.")
    st.stop()
except Exception as e:
    st.error(f"Error loading template/mask: {e}")
    st.stop()

st.subheader("Template (fixed)")
st.image(template, use_column_width=False, width=360)
st.subheader("Mask (fixed) — green = face slot, white = blurred fill area")
st.image(mask, use_column_width=False, width=360)

# Helpers to detect green and white bounding boxes in mask
def find_color_bbox(mask_rgba, color="green"):
    arr = np.array(mask_rgba.convert("RGBA"))
    r = arr[..., 0].astype(int)
    g = arr[..., 1].astype(int)
    b = arr[..., 2].astype(int)
    a = arr[..., 3].astype(int)

    if color == "green":
        mask_bool = (g > 200) & (r < 100) & (b < 100) & (a > 10)
    elif color == "white":
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
    st.error("Mask does not contain a vivid GREEN region (0,255,0). Edit mask.png to use pure green for the face slot.")
    st.stop()

st.write("Detected GREEN box:", green_box)
if white_box is None:
    st.warning("No WHITE fill area detected. The app will skip the blurred fill step.")

# Upload photo
uploaded = st.file_uploader("Upload a portrait/photo (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload a photo to enable the interactive crop tool.")
    st.stop()

photo = Image.open(uploaded).convert("RGBA")
pw, ph = photo.size
st.image(photo, caption="Photo preview", width=360)

# Canvas display width control (user can choose how big the canvas appears)
display_w = st.slider("Canvas display width (px) — affects crop precision", 320, 1200, 700)
scale = display_w / float(pw)
display_h = int(ph * scale)

st.markdown("---")
st.write("Draw a single rectangle on the photo to select the face. If the on-canvas tool isn't available, a slider fallback appears.")

canvas_rect = None  # will hold (left, top, width, height) in display coordinates

if USE_CANVAS:
    st.write("Canvas mode: draw or edit one rectangle, then press 'Apply crop' below.")
    try:
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",  # Transparent fill
            stroke_width=2,
            stroke_color="#00FF00",
            background_image=photo.resize((display_w, display_h)),
            update_streamlit=True,
            height=display_h,
            width=display_w,
            drawing_mode="rect",
            key="canvas",
        )
        # Extract last rect if present
        if canvas_result.json_data and "objects" in canvas_result.json_data:
            objs = canvas_result.json_data["objects"]
            rects = [o for o in objs if o.get("type") == "rect"]
            if rects:
                last = rects[-1]
                left = last.get("left", 0)
                top = last.get("top", 0)
                width = last.get("width", 0) * last.get("scaleX", 1)
                height = last.get("height", 0) * last.get("scaleY", 1)
                canvas_rect = (left, top, width, height)
    except Exception as e:
        st.warning(f"Canvas failed to initialize or run: {e}")
        USE_CANVAS = False
        canvas_rect = None

# Fallback slider crop UI if canvas isn't available or no rectangle drawn
if (not USE_CANVAS) or (canvas_rect is None):
    st.warning("Interactive canvas not available or no rectangle drawn. Using slider-based crop fallback.")
    st.subheader("Slider-based crop fallback (aspect locked to GREEN slot)")
    # compute green aspect
    gw = green_box[2] - green_box[0]
    gh = green_box[3] - green_box[1]
    aspect = None
    if gh != 0:
        aspect = float(gw) / float(gh)
    center_x = st.slider("Crop center X (%)", 0, 100, 50)
    center_y = st.slider("Crop center Y (%)", 0, 100, 50)
    zoom = st.slider("Zoom (%) — higher = zoom in (100 = no zoom)", 20, 400, 120)
    rotation_fallback = st.slider("Rotate crop (degrees)", -30, 30, 0)

    cx = int(center_x / 100.0 * pw)
    cy = int(center_y / 100.0 * ph)
    base = int(min(pw, ph) * 0.6)
    if aspect is not None and aspect > 0:
        crop_w = base
        crop_h = max(10, int(base / aspect))
    else:
        crop_w = base
        crop_h = base
    crop_w = max(10, int(crop_w * (100.0 / zoom)))
    crop_h = max(10, int(crop_h * (100.0 / zoom)))
    left = max(0, cx - crop_w // 2)
    top = max(0, cy - crop_h // 2)
    right = min(pw, left + crop_w)
    bottom = min(ph, top + crop_h)
    if right - left < crop_w:
        left = max(0, right - crop_w)
    if bottom - top < crop_h:
        top = max(0, bottom - crop_h)
    # map to display coords
    d_left = left * scale
    d_top = top * scale
    d_w = (right - left) * scale
    d_h = (bottom - top) * scale
    canvas_rect = (d_left, d_top, d_w, d_h)
    # also create the actual crop now
    crop_box = (left, top, right, bottom)
    face_crop = photo.crop(crop_box).convert("RGBA")
    if rotation_fallback != 0:
        face_crop = face_crop.rotate(rotation_fallback, resample=Image.BICUBIC, expand=True)
    st.image(face_crop, caption="Cropped preview (fallback)", width=240)

# If we do have a canvas_rect from the drawable canvas, compute original image crop coords
if canvas_rect is not None and (USE_CANVAS):
    left_disp, top_disp, width_disp, height_disp = canvas_rect
    # clamp
    left_disp = max(0, left_disp)
    top_disp = max(0, top_disp)
    width_disp = max(1, width_disp)
    height_disp = max(1, height_disp)
    ox = int(left_disp / scale)
    oy = int(top_disp / scale)
    ow = int(width_disp / scale)
    oh = int(height_disp / scale)
    # clamp to image
    ox = max(0, min(pw - 1, ox))
    oy = max(0, min(ph - 1, oy))
    ow = max(1, min(pw - ox, ow))
    oh = max(1, min(ph - oy, oh))
    crop_box = (ox, oy, ox + ow, oy + oh)
    face_crop = photo.crop(crop_box).convert("RGBA")
    st.image(face_crop, caption="Cropped preview", width=240)

# Small placement & transform controls
st.markdown("---")
st.subheader("Fine-tune & white-fill settings")
offset_x = st.slider("Face offset X (px)", -400, 400, 0)
offset_y = st.slider("Face offset Y (px)", -400, 400, 0)
rotation = st.slider("Face rotation (degrees)", -25, 25, 0)

blur_radius = st.slider("White-area blur radius", 0, 64, 18)
feather = st.slider("White-area feather (px)", 0, 64, 18)
shear_x = st.slider("White-area shear X (%)", -30, 30, 0)
shear_y = st.slider("White-area shear Y (%)", -30, 30, 0)

# Apply rotation requested in fine-tune (if not applied earlier)
if rotation != 0:
    face_crop = face_crop.rotate(rotation, resample=Image.BICUBIC, expand=True)

# Create blurred fill for white_box area
def create_blur_fill(photo_img, box, radius, sx_pct, sy_pct):
    tw = max(1, box[2] - box[0])
    th = max(1, box[3] - box[1])
    if tw <= 0 or th <= 0:
        return None
    pw, ph = photo_img.size
    target_aspect = tw / th if th != 0 else 1.0
    photo_aspect = pw / ph if ph != 0 else 1.0

    if photo_aspect > target_aspect:
        new_h = ph
        new_w = int(ph * target_aspect)
    else:
        new_w = pw
        new_h = int(pw / target_aspect) if target_aspect != 0 else ph

    left = max(0, (pw - new_w) // 2)
    top = max(0, (ph - new_h) // 2)
    cropped = photo.crop((left, top, left + new_w, top + new_h)).convert("RGBA")
    resized = cropped.resize((tw, th), Image.LANCZOS)
    blurred = resized.filter(ImageFilter.GaussianBlur(radius=radius))

    sx = float(sx_pct) / 100.0
    sy = float(sy_pct) / 100.0
    a = 1.0
    b = -sx
    c = 0.0
    d = -sy
    e = 1.0
    f = 0.0
    try:
        warped = blurred.transform((tw, th), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)
    except Exception:
        warped = blurred
    return warped

white_fill = None
if white_box is not None:
    white_fill = create_blur_fill(photo, white_box, blur_radius, shear_x, shear_y)

# Compose final
composed = template.copy().convert("RGBA")
layer = Image.new("RGBA", composed.size, (0, 0, 0, 0))

# Paste white fill with feathered alpha (if any)
if white_fill is not None:
    tw = white_box[2] - white_box[0]
    th = white_box[3] - white_box[1]
    wf_resized = white_fill.resize((tw, th), Image.LANCZOS)
    blur_layer = Image.new("RGBA", composed.size, (0, 0, 0, 0))
    blur_layer.paste(wf_resized, (white_box[0], white_box[1]))
    mask_arr = np.array(mask.convert("RGBA"))
    white_mask_bool = (mask_arr[..., 0] > 200) & (mask_arr[..., 1] > 200) & (mask_arr[..., 2] > 200) & (mask_arr[..., 3] > 10)
    white_alpha = Image.fromarray((white_mask_bool * 255).astype(np.uint8)).convert("L")
    if feather > 0:
        white_alpha = white_alpha.filter(ImageFilter.GaussianBlur(radius=feather))
    layer = Image.alpha_composite(layer, Image.composite(blur_layer, Image.new("RGBA", composed.size, (0, 0, 0, 0)), white_alpha))

# Prepare face: resize to cover green box
face = face_crop.convert("RGBA")
gw = green_box[2] - green_box[0]
gh = green_box[3] - green_box[1]
fw, fh = face.size
scale_w = gw / fw
scale_h = gh / fh
scale = max(scale_w, scale_h)  # cover
new_fw = max(1, int(fw * scale))
new_fh = max(1, int(fh * scale))
face_resized = face.resize((new_fw, new_fh), Image.LANCZOS)

paste_x = green_box[0] + (gw - new_fw) // 2 + int(offset_x)
paste_y = green_box[1] + (gh - new_fh) // 2 + int(offset_y)

# Clip face to green mask
mask_arr = np.array(mask.convert("RGBA"))
green_mask_bool = (mask_arr[..., 1] > 200) & (mask_arr[..., 0] < 100) & (mask_arr[..., 2] < 100) & (mask_arr[..., 3] > 10)
green_alpha = Image.fromarray((green_mask_bool * 255).astype(np.uint8)).convert("L")

face_layer = Image.new("RGBA", composed.size, (0, 0, 0, 0))
face_layer.paste(face_resized, (paste_x, paste_y), face_resized)
clipped_face = Image.composite(face_layer, Image.new("RGBA", composed.size, (0, 0, 0, 0)), green_alpha)

final = Image.alpha_composite(composed, layer)
final = Image.alpha_composite(final, clipped_face)

st.subheader("Final result")
st.image(final, use_column_width=True)

# Download
buf = io.BytesIO()
final.save(buf, format="PNG")
buf.seek(0)
st.download_button("Download final PNG", data=buf, file_name="car_with_face.png", mime="image/png")

st.markdown(
    "- Draw or adjust one rectangle on the photo canvas (or use the slider fallback) to select the face crop.\n"
    "- Use the small offset/rotation controls to align the face inside the green slot precisely.\n"
    "- If drawable canvas fails due to host environment, increase the 'Canvas display width' slider or use the fallback sliders.\n"
    "- Ensure mask.png uses pure RGB green (0,255,0) for the face rectangle for reliable detection."
)
