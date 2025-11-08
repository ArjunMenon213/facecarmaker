"""
Origami Meme Car — Use fixed template+mask, interactive crop, continuous background fill,
with a banner image shown at the top of the app (banner.png).

Place these files in the app folder:
- template.png
- mask.png
- banner.png (500x150)

User flow:
- Upload a photo.
- Draw one rectangle (or use sliders fallback) to select the face area.
- The selected crop is placed into the GREEN slot in the hidden template.
- The WHITE area in the template is filled by sampling the contiguous part of the original photo.
- Download the final PNG.
"""

import streamlit as st
from PIL import Image, ImageFilter
import numpy as np
import io
import base64
from io import BytesIO

st.set_page_config(page_title="Origami Meme Car — Simple UI", layout="centered")

# Display banner.png at the very top if present
try:
    banner = Image.open("banner.png").convert("RGBA")
    # center the banner using columns
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.image(banner, use_column_width=False, width=500)
except Exception:
    # If banner not found or cannot be opened, continue silently
    pass

st.title("Origami Meme Car — upload & crop")

st.write(
    "Upload a photo, draw one rectangle to select the face (or use the slider fallback). "
    "The app places your crop into the template's GREEN slot and fills the WHITE area with the rest of the photo. "
    "Template and mask are used in the background (do not need to be shown)."
)

# Compatibility helper for streamlit-drawable-canvas image_to_url on older Streamlit
try:
    from streamlit.elements import image as st_image_module
except Exception:
    st_image_module = st

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

# Try to import canvas component; fallback to sliders if missing
USE_CANVAS = True
try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    USE_CANVAS = False

# Load fixed template & mask (do not display)
try:
    template = Image.open("template.png").convert("RGBA")
    mask = Image.open("mask.png").convert("RGBA")
except FileNotFoundError:
    st.error("Could not find template.png and/or mask.png in the app folder. Add them and reload.")
    st.stop()
except Exception as e:
    st.error(f"Error loading template/mask: {e}")
    st.stop()

# Helpers to detect vivid green / white bounding boxes in mask
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
    st.error("Mask must contain a vivid GREEN region (0,255,0) marking the face slot.")
    st.stop()

# Short confirmation for user (no templates/masks shown)
st.info("template.png and mask.png loaded (used in background).")

# Upload photo
uploaded = st.file_uploader("Upload a photo (jpg/png) — then draw a rectangle to crop the face", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload a photo to begin.")
    st.stop()

photo = Image.open(uploaded).convert("RGBA")
pw, ph = photo.size
st.subheader("Photo")
st.image(photo, width=360)

# Canvas display width control
display_w = st.slider("Canvas display width (px)", 320, 1200, 700, help="Increase for finer cropping.")
scale = display_w / float(pw)
display_h = int(ph * scale)

# Get crop rectangle via canvas or fallback sliders
canvas_rect = None

if USE_CANVAS:
    st.write("Draw one rectangle on the photo canvas to select the face.")
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
    except Exception:
        USE_CANVAS = False
        canvas_rect = None

# Slider fallback if needed
if (not USE_CANVAS) or (canvas_rect is None):
    st.warning("Interactive canvas not available or no rectangle drawn. Use the fallback sliders below to crop.")
    st.subheader("Fallback crop controls")
    center_x = st.slider("Crop center X (%)", 0, 100, 50)
    center_y = st.slider("Crop center Y (%)", 0, 100, 50)
    zoom = st.slider("Zoom (%) — higher = zoom in (100 = no zoom)", 10, 400, 100)

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
    d_left = left * scale
    d_top = top * scale
    d_w = (right - left) * scale
    d_h = (bottom - top) * scale
    canvas_rect = (d_left, d_top, d_w, d_h)
    crop_box = (left, top, right, bottom)
    face_crop = photo.crop(crop_box).convert("RGBA")
    st.image(face_crop, caption="Cropped preview (fallback)", width=240)

# If rectangle from canvas, convert to original photo coords
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

# Optional small alignment
st.markdown("---")
st.subheader("Optional small alignment")
offset_x = st.number_input("Offset X (px)", value=0, step=1)
offset_y = st.number_input("Offset Y (px)", value=0, step=1)
rotation = st.number_input("Rotation (degrees)", value=0, step=1)
if rotation != 0:
    face_crop = face_crop.rotate(float(rotation), resample=Image.BICUBIC, expand=True)

# --- create_continuous_fill function (as provided earlier) ---
from PIL import Image, ImageFilter

def create_continuous_fill(photo_img, crop_box, green_box, white_box, blur_overlap_radius=12):
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

    scale_template_per_photo = float(g_w) / float(c_w)

    vec_tx = w_cx - g_cx
    vec_ty = w_cy - g_cy

    vec_px = vec_tx / scale_template_per_photo
    vec_py = vec_ty / scale_template_per_photo

    src_cx = c_cx + vec_px
    src_cy = c_cy + vec_py

    src_w = max(1, int(round(tw / scale_template_per_photo)))
    src_h = max(1, int(round(th / scale_template_per_photo)))

    src_left = int(round(src_cx - src_w / 2.0))
    src_top = int(round(src_cy - src_h / 2.0))
    src_right = src_left + src_w
    src_bottom = src_top + src_h

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

# Build white_fill automatically
white_fill = None
if white_box is not None:
    try:
        white_fill = create_continuous_fill(photo, crop_box, green_box, white_box, blur_overlap_radius=12)
    except Exception as e:
        st.warning(f"Could not create continuous fill automatically: {e}")
        white_fill = None

# Compose final image
composed = template.copy().convert("RGBA")
layer = Image.new("RGBA", composed.size, (0,0,0,0))

# Paste white_fill (if present)
if white_fill is not None:
    wf_resized = white_fill
    blur_layer = Image.new("RGBA", composed.size, (0,0,0,0))
    blur_layer.paste(wf_resized, (white_box[0], white_box[1]))
    mask_arr = np.array(mask.convert("RGBA"))
    white_mask_bool = (mask_arr[...,0] > 200) & (mask_arr[...,1] > 200) & (mask_arr[...,2] > 200) & (mask_arr[...,3] > 10)
    white_alpha = Image.fromarray((white_mask_bool * 255).astype(np.uint8)).convert("L")
    white_alpha = white_alpha.filter(ImageFilter.GaussianBlur(radius=6))
    layer = Image.alpha_composite(layer, Image.composite(blur_layer, Image.new("RGBA", composed.size, (0,0,0,0)), white_alpha))

# Place face into green slot (cover) and clip by green mask
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

face_layer = Image.new("RGBA", composed.size, (0,0,0,0))
face_layer.paste(face_resized, (paste_x, paste_y), face_resized)

mask_arr = np.array(mask.convert("RGBA"))
green_mask_bool = (mask_arr[...,1] > 200) & (mask_arr[...,0] < 100) & (mask_arr[...,2] < 100) & (mask_arr[...,3] > 10)
green_alpha = Image.fromarray((green_mask_bool * 255).astype(np.uint8)).convert("L")
clipped_face = Image.composite(face_layer, Image.new("RGBA", composed.size, (0,0,0,0)), green_alpha)

final = Image.alpha_composite(composed, layer)
final = Image.alpha_composite(final, clipped_face)

st.subheader("Result")
st.image(final, use_column_width=True)

# Download final PNG
buf = io.BytesIO()
final.save(buf, format="PNG")
buf.seek(0)
st.download_button("Download PNG", data=buf, file_name="car_with_face.png", mime="image/png")

st.markdown(
    "- Draw one rectangle to select the face (or use the slider fallback). The template and mask run in the background.\n"
    "- The white area is filled automatically from the rest of the photo."
)

