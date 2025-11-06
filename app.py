"""
Origami Meme Car — Single fixed mask, interactive crop, continuous background fill

Usage:
- Put `template.png` and `mask.png` (same size) in the app folder.
  - mask.png must use vivid GREEN (R=0,G=255,B=0) to mark the rectangular face slot.
  - mask.png may use WHITE (R=255,G=255,B=255) to mark the area to be filled with the rest
    of the photo (continuous fill).
- User flow in UI:
  1) Upload a photo.
  2) Draw/resize one rectangle on the canvas to select the face (or use slider fallback).
  3) The app places the selected crop into the GREEN slot and fills the WHITE area with the
     contiguous part of the same photo (the part adjacent to the crop), avoiding duplicate face.
  4) Download the final printable PNG.

This file is intentionally self-contained and robust: it tries to use streamlit-drawable-canvas
for the on-canvas cropper (best UX). If the component or the host Streamlit version is not available,
it falls back to a simple slider-based crop UI so the app still works without extra dependencies.
"""

import streamlit as st
from PIL import Image, ImageFilter
import numpy as np
import io
import base64
from io import BytesIO

st.set_page_config(page_title="Origami Meme Car — Crop to GREEN slot", layout="centered")
st.title("Origami Meme Car — crop face → fill back with the rest of the photo")

st.write(
    "Place template.png and mask.png in the app folder. Mask must use vivid GREEN (0,255,0) "
    "for the face rectangle. Upload a photo, draw one rectangle on the photo canvas (or use the slider fallback). "
    "The app will place your crop into the green slot and fill the white area with the rest of the image (a continuous continuation)."
)

# --- Compatibility helper for streamlit-drawable-canvas (older Streamlit may miss image_to_url) ---
try:
    from streamlit.elements import image as st_image_module  # newer internals
except Exception:
    st_image_module = st  # fallback

if not hasattr(st_image_module, "image_to_url"):
    def _image_to_url_fallback(img):
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img)
        else:
            pil_img = img
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    st_image_module.image_to_url = _image_to_url_fallback

# Try to import drawable-canvas; if missing we'll fallback to sliders
USE_CANVAS = True
try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    USE_CANVAS = False

# Load template & mask (fixed files in repo working directory)
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
st.image(template, width=360)
st.subheader("Mask (fixed) — GREEN = face slot, WHITE = area filled by rest of photo")
st.image(mask, width=360)

# Helper: detect vivid green / white bounding boxes in mask
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
    st.error("Mask does not contain a vivid GREEN region (0,255,0). Edit mask.png to mark the face rectangle with pure green.")
    st.stop()

st.write("Detected GREEN box (face slot):", green_box)
if white_box is None:
    st.info("No white area detected in mask; only the face will be placed into the green slot.")

# Upload photo
uploaded = st.file_uploader("Upload a photo (jpg/png) — then draw a rectangle to crop the face", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload a photo to begin.")
    st.stop()

photo = Image.open(uploaded).convert("RGBA")
pw, ph = photo.size
st.subheader("Photo")
st.image(photo, width=360)

# Canvas display width control (small and unobtrusive)
display_w = st.slider("Canvas display width (px)", 320, 1200, 700, help="Increase to make on-canvas cropping more precise.")
scale = display_w / float(pw)
display_h = int(ph * scale)

# Crop rectangle (in display coords) will be stored in canvas_rect
canvas_rect = None

if USE_CANVAS:
    st.write("Draw one rectangle on the photo canvas to select the face. The app will use the rest of the photo to populate the white area automatically.")
    try:
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=2,
            stroke_color="#00FF00",
            background_image=photo.resize((display_w, display_h)),
            update_streamlit=True,
            height=display_h,
            width=display_w,
            drawing_mode="rect",
            key="canvas",
        )
        if canvas_result.json_data and "objects" in canvas_result.json_data:
            objs = canvas_result.json_data["objects"]
            rects = [o for o in objs if o.get("type") == "rect"]
            if rects:
                last = rects[-1]
                left = float(last.get("left", 0))
                top = float(last.get("top", 0))
                width = float(last.get("width", 0)) * float(last.get("scaleX", 1))
                height = float(last.get("height", 0)) * float(last.get("scaleY", 1))
                canvas_rect = (left, top, width, height)
    except Exception as e:
        st.warning(f"Canvas failed to initialize: {e}. Using slider fallback.")
        USE_CANVAS = False
        canvas_rect = None

# Slider fallback if canvas missing or no rectangle drawn
if (not USE_CANVAS) or (canvas_rect is None):
    st.warning("Interactive canvas not available or no rectangle drawn. Use the fallback sliders below to position the crop.")
    st.subheader("Fallback crop controls")
    center_x = st.slider("Crop center X (%)", 0, 100, 50)
    center_y = st.slider("Crop center Y (%)", 0, 100, 50)
    zoom = st.slider("Zoom (%) — higher = zoom in (100 = no zoom)", 10, 400, 100)

    # aspect locked to green box
    gw = green_box[2] - green_box[0]
    gh = green_box[3] - green_box[1]
    aspect = None
    if gh != 0:
        aspect = float(gw) / float(gh)

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
    # display coords representation
    d_left = left * scale
    d_top = top * scale
    d_w = (right - left) * scale
    d_h = (bottom - top) * scale
    canvas_rect = (d_left, d_top, d_w, d_h)
    crop_box = (left, top, right, bottom)
    face_crop = photo.crop(crop_box).convert("RGBA")
    st.image(face_crop, caption="Cropped preview (fallback)", width=240)

# If canvas_rect came from drawable canvas, convert to original photo coords
if canvas_rect is not None and USE_CANVAS:
    left_disp, top_disp, width_disp, height_disp = canvas_rect
    left_disp = max(0, left_disp)
    top_disp = max(0, top_disp)
    width_disp = max(1, width_disp)
    height_disp = max(1, height_disp)
    ox = int(left_disp / scale)
    oy = int(top_disp / scale)
    ow = int(width_disp / scale)
    oh = int(height_disp / scale)
    ox = max(0, min(pw - 1, ox))
    oy = max(0, min(ph - 1, oy))
    ow = max(1, min(pw - ox, ow))
    oh = max(1, min(ph - oy, oh))
    crop_box = (ox, oy, ox + ow, oy + oh)
    face_crop = photo.crop(crop_box).convert("RGBA")
    st.image(face_crop, caption="Cropped preview", width=240)

# Minimal alignment controls: small offset & rotation (kept optional and small)
st.markdown("---")
st.subheader("Optional small alignment (leave default for automatic placement)")
offset_x = st.number_input("Offset X (px)", value=0, step=1)
offset_y = st.number_input("Offset Y (px)", value=0, step=1)
rotation = st.number_input("Rotation (degrees)", value=0, step=1)

if rotation != 0:
    face_crop = face_crop.rotate(float(rotation), resample=Image.BICUBIC, expand=True)

# --- New continuous fill function: sample contiguous region from photo that continues from the crop ---
from PIL import ImageChops

def create_continuous_fill(photo_img, crop_box, green_box, white_box, blur_overlap_radius=12):
    """
    Sample a region of photo_img that is continuous/adjacent to the user's crop so the white area
    looks like the rest of the same image moving forward from the cropped face.
    - photo_img: PIL RGBA original photo
    - crop_box: (l,t,r,b) in photo pixel coords indicating user's crop
    - green_box: (l,t,r,b) in template coords where crop will be placed
    - white_box: (l,t,r,b) in template coords to be filled
    Returns: PIL RGBA image sized exactly to (white_box.width, white_box.height)
    """
    pw, ph = photo_img.size
    tw = max(1, white_box[2] - white_box[0])
    th = max(1, white_box[3] - white_box[1])

    c_left, c_top, c_right, c_bottom = crop_box
    c_w = max(1, c_right - c_left)
    c_h = max(1, c_bottom - c_top)
    c_cx = c_left + c_w / 2.0
    c_cy = c_top + c_h / 2.0

    g_w = max(1, green_box[2] - green_box[0])
    g_h = max(1, green_box[3] - green_box[1])
    g_cx = green_box[0] + g_w / 2.0
    g_cy = green_box[1] + g_h / 2.0

    w_cx = white_box[0] + tw / 2.0
    w_cy = white_box[1] + th / 2.0

    # template pixels per photo pixel for the crop->green mapping
    scale_template_per_photo = float(g_w) / float(c_w)

    # vector from green center to white center (in template pixels)
    vec_tx = w_cx - g_cx
    vec_ty = w_cy - g_cy

    # convert vector to photo pixels
    vec_px = vec_tx / scale_template_per_photo
    vec_py = vec_ty / scale_template_per_photo

    # source center in photo to sample from (adjacent to crop)
    src_cx = c_cx + vec_px
    src_cy = c_cy + vec_py

    # source region size in photo pixels that corresponds to white target
    src_w = max(1, int(round(tw / scale_template_per_photo)))
    src_h = max(1, int(round(th / scale_template_per_photo)))

    src_left = int(round(src_cx - src_w / 2.0))
    src_top = int(round(src_cy - src_h / 2.0))
    src_right = src_left + src_w
    src_bottom = src_top + src_h

    # Create a transparent canvas to paste the sampled piece (handles out-of-bounds gracefully)
    src_canvas = Image.new("RGBA", (src_w, src_h), (0, 0, 0, 0))

    int_left = max(0, src_left)
    int_top = max(0, src_top)
    int_right = min(pw, src_right)
    int_bottom = min(ph, src_bottom)

    if int_right > int_left and int_bottom > int_top:
        piece = photo_img.crop((int_left, int_top, int_right, int_bottom)).convert("RGBA")
        paste_x = int_left - src_left
        paste_y = int_top - src_top
        src_canvas.paste(piece, (paste_x, paste_y))
    else:
        # fallback: center-crop the photo to the src aspect and use that
        fallback = photo_img.copy().convert("RGBA")
        p_aspect = pw / ph if ph != 0 else 1.0
        t_aspect = src_w / src_h if src_h != 0 else 1.0
        if p_aspect > t_aspect:
            new_h = ph
            new_w = int(ph * t_aspect)
        else:
            new_w = pw
            new_h = int(pw / t_aspect) if t_aspect != 0 else ph
        left_f = max(0, (pw - new_w) // 2)
        top_f = max(0, (ph - new_h) // 2)
        fallback_piece = fallback.crop((left_f, top_f, left_f + new_w, top_f + new_h))
        fallback_resized = fallback_piece.resize((src_w, src_h), Image.LANCZOS)
        src_canvas.paste(fallback_resized, (0, 0))

    # If overlap exists between sample region and crop region, blur the overlap in src_canvas
    overlap_left = max(src_left, c_left)
    overlap_top = max(src_top, c_top)
    overlap_right = min(src_right, c_right)
    overlap_bottom = min(src_bottom, c_bottom)

    if overlap_right > overlap_left and overlap_bottom > overlap_top:
        ov_x = overlap_left - src_left
        ov_y = overlap_top - src_top
        ov_w = overlap_right - overlap_left
        ov_h = overlap_bottom - overlap_top
        blurred_full = src_canvas.filter(ImageFilter.GaussianBlur(radius=blur_overlap_radius))
        blurred_patch = blurred_full.crop((ov_x, ov_y, ov_x + ov_w, ov_y + ov_h))
        src_canvas.paste(blurred_patch, (ov_x, ov_y))

    final_fill = src_canvas.resize((tw, th), Image.LANCZOS)
    return final_fill

# Build white_fill using continuous fill function
white_fill = None
if white_box is not None:
    try:
        white_fill = create_continuous_fill(photo, crop_box, green_box, white_box, blur_overlap_radius=12)
    except Exception as e:
        st.warning(f"Could not create continuous fill automatically: {e}")
        white_fill = None

# Compose final image: paste white_fill then place face into green slot clipped by green mask
composed = template.copy().convert("RGBA")
layer = Image.new("RGBA", composed.size, (0, 0, 0, 0))

# Paste white_fill (if present) using white mask alpha; apply small feather for nicer blend
if white_fill is not None:
    wf_resized = white_fill  # already sized
    blur_layer = Image.new("RGBA", composed.size, (0, 0, 0, 0))
    blur_layer.paste(wf_resized, (white_box[0], white_box[1]))

    mask_arr = np.array(mask.convert("RGBA"))
    white_mask_bool = (mask_arr[..., 0] > 200) & (mask_arr[..., 1] > 200) & (mask_arr[..., 2] > 200) & (mask_arr[..., 3] > 10)
    white_alpha = Image.fromarray((white_mask_bool * 255).astype(np.uint8)).convert("L")
    # slight feather to blend edges
    white_alpha = white_alpha.filter(ImageFilter.GaussianBlur(radius=6))
    layer = Image.alpha_composite(layer, Image.composite(blur_layer, Image.new("RGBA", composed.size, (0, 0, 0, 0)), white_alpha))

# Prepare face crop: resize to cover green box (cover behavior)
face = face_crop.convert("RGBA")
fw, fh = face.size
gw = green_box[2] - green_box[0]
gh = green_box[3] - green_box[1]
scale_w = gw / fw
scale_h = gh / fh
scale = max(scale_w, scale_h)
new_fw = max(1, int(fw * scale))
new_fh = max(1, int(fh * scale))
face_resized = face.resize((new_fw, new_fh), Image.LANCZOS)

paste_x = green_box[0] + (gw - new_fw) // 2 + int(offset_x)
paste_y = green_box[1] + (gh - new_fh) // 2 + int(offset_y)

face_layer = Image.new("RGBA", composed.size, (0, 0, 0, 0))
face_layer.paste(face_resized, (paste_x, paste_y), face_resized)

# Clip face to green mask
mask_arr = np.array(mask.convert("RGBA"))
green_mask_bool = (mask_arr[..., 1] > 200) & (mask_arr[..., 0] < 100) & (mask_arr[..., 2] < 100) & (mask_arr[..., 3] > 10)
green_alpha = Image.fromarray((green_mask_bool * 255).astype(np.uint8)).convert("L")
clipped_face = Image.composite(face_layer, Image.new("RGBA", composed.size, (0, 0, 0, 0)), green_alpha)

final = Image.alpha_composite(composed, layer)
final = Image.alpha_composite(final, clipped_face)

st.subheader("Result")
st.image(final, use_column_width=True)

# Download button
buf = io.BytesIO()
final.save(buf, format="PNG")
buf.seek(0)
st.download_button("Download PNG", data=buf, file_name="car_with_face.png", mime="image/png")

st.markdown(
    "- Draw one rectangle on the canvas to select the face (or use the slider fallback).\n"
    "- The white area is filled automatically from the rest of the photo so it visually continues from the crop.\n"
    "- Use the small optional alignment inputs if you need micro adjustments; otherwise download the result directly."
)
