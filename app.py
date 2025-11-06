import streamlit as st
from PIL import Image, ImageFilter
import numpy as np
import io
import base64
from io import BytesIO

st.set_page_config(page_title="Origami Meme Car — Simple crop → full-background fill", layout="centered")
st.title("Place face into the fixed GREEN slot — simple UX")

st.write(
    "Place template.png and mask.png (in the app folder). Mask must use vivid GREEN (0,255,0) for the face slot "
    "and WHITE (255,255,255) for the area filled with the rest of the photo. "
    "Upload a photo, draw/resize a rectangle to crop the face. The app will automatically use the rest of the photo "
    "to fill the white area (face region is blurred out automatically). No extra sliders."
)

# --- Compatibility monkey-patch for drawable-canvas helper used by the component ---
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

# Try to import canvas component; if missing we will fall back to slider-based crop
USE_CANVAS = True
try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    USE_CANVAS = False

# Load fixed template and mask
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
st.subheader("Mask (fixed) — green = face slot, white = rest-fill area")
st.image(mask, width=360)

# helper: detect vivid green and white boxes in mask
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
    st.error("Mask must contain a vivid GREEN (0,255,0) rectangular slot for the face. Edit mask.png and retry.")
    st.stop()

st.write("Detected GREEN box:", green_box)
if white_box is None:
    st.info("No WHITE area detected in mask — the template will only get the face placed into the GREEN slot.")

# upload photo
uploaded = st.file_uploader("Upload a photo (jpg/png) — then draw a rectangle to crop the face", type=["jpg","jpeg","png"])
if uploaded is None:
    st.info("Upload a photo to begin.")
    st.stop()

photo = Image.open(uploaded).convert("RGBA")
pw, ph = photo.size
st.image(photo, caption="Photo preview", width=360)

# display width for canvas mapping
display_w = st.slider("Canvas display width (px) — higher = more precise cropping", 320, 1200, 700)
scale = display_w / float(pw)
display_h = int(ph * scale)

# obtain crop rectangle via canvas (preferred) or fallback sliders
canvas_rect = None

if USE_CANVAS:
    st.write("Draw one rectangle on the photo canvas to select the face. If the canvas doesn't appear, the app will fallback to sliders.")
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
        st.warning(f"Canvas initialization failed: {e}")
        USE_CANVAS = False
        canvas_rect = None

# fallback slider crop if canvas not available or no rectangle drawn
if (not USE_CANVAS) or (canvas_rect is None):
    st.warning("Interactive canvas not available or no rectangle drawn. Use the fallback sliders to approximate the crop.")
    st.subheader("Fallback crop controls")
    center_x = st.slider("Crop center X (%)", 0, 100, 50)
    center_y = st.slider("Crop center Y (%)", 0, 100, 50)
    zoom = st.slider("Zoom (%) — higher = zoom in (100 = no zoom)", 10, 400, 100)
    rotation_fallback = st.slider("Rotate crop (degrees)", -30, 30, 0)

    # lock aspect to green slot
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
    # map to display coords
    d_left = left * scale
    d_top = top * scale
    d_w = (right - left) * scale
    d_h = (bottom - top) * scale
    canvas_rect = (d_left, d_top, d_w, d_h)
    crop_box = (left, top, right, bottom)
    face_crop = photo.crop(crop_box).convert("RGBA")
    if rotation_fallback != 0:
        face_crop = face_crop.rotate(rotation_fallback, resample=Image.BICUBIC, expand=True)
    st.image(face_crop, caption="Cropped preview (fallback)", width=240)

# if canvas rect exists (from canvas) map to original coords and produce face_crop
if canvas_rect is not None and USE_CANVAS:
    left_disp, top_disp, width_disp, height_disp = canvas_rect
    # clamp and convert
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

# Minimal alignment controls (only small offset & rotation)
st.markdown("---")
st.subheader("Small fine-tune (optional)")
offset_x = st.slider("Offset X (px)", -200, 200, 0)
offset_y = st.slider("Offset Y (px)", -200, 200, 0)
rotation = st.slider("Rotate face (degrees)", -20, 20, 0)

if rotation != 0:
    face_crop = face_crop.rotate(rotation, resample=Image.BICUBIC, expand=True)

# Create fill for white area using the rest of the photo (auto)
def create_fill_from_photo(photo_img, crop_box, target_box):
    """
    Use the original photo to create a fill for the white area:
    - Start from the original photo.
    - Obscure the crop_box area by replacing it with a blurred patch sampled from surrounding pixels,
      so the white area fill doesn't prominently show the face.
    - Then center-crop the modified photo to the target aspect and resize to target size.
    """
    pw, ph = photo_img.size
    tw = max(1, target_box[2] - target_box[0])
    th = max(1, target_box[3] - target_box[1])
    if tw <= 0 or th <= 0:
        return None

    fill_img = photo_img.copy().convert("RGBA")

    # Expand crop box slightly to create a region to blur/replace
    left, top, right, bottom = crop_box
    w = right - left
    h = bottom - top
    pad = int(0.25 * max(w, h))  # 25% padding
    ex_left = max(0, left - pad)
    ex_top = max(0, top - pad)
    ex_right = min(pw, right + pad)
    ex_bottom = min(ph, bottom + pad)

    # Create a blurred version of the full image and paste the relevant patch over the face area
    blurred_full = fill_img.filter(ImageFilter.GaussianBlur(radius=18))
    # Paste blurred patch over the expanded crop area to hide the face
    patch = blurred_full.crop((ex_left, ex_top, ex_right, ex_bottom))
    fill_img.paste(patch, (ex_left, ex_top, ex_right, ex_bottom))

    # Now center-crop the modified fill_img to the target aspect and resize to exactly (tw, th)
    target_aspect = tw / th if th != 0 else 1.0
    photo_aspect = pw / ph if ph != 0 else 1.0

    if photo_aspect > target_aspect:
        new_h = ph
        new_w = int(ph * target_aspect)
    else:
        new_w = pw
        new_h = int(pw / target_aspect) if target_aspect != 0 else ph

    cx = (pw - new_w) // 2
    cy = (ph - new_h) // 2
    center_cropped = fill_img.crop((cx, cy, cx + new_w, cy + new_h))
    resized = center_cropped.resize((tw, th), Image.LANCZOS)
    return resized

white_fill = None
if white_box is not None:
    white_fill = create_fill_from_photo(photo, crop_box, white_box)

# Compose final image: place white_fill into white_box, place face into green_box (clipped)
composed = template.copy().convert("RGBA")
layer = Image.new("RGBA", composed.size, (0,0,0,0))

# Paste white fill using white mask (no controls)
if white_fill is not None:
    wf_resized = white_fill  # already the right size
    blur_layer = Image.new("RGBA", composed.size, (0,0,0,0))
    blur_layer.paste(wf_resized, (white_box[0], white_box[1]))

    # create alpha from white mask and softly feather edges a bit for nicer blend
    mask_arr = np.array(mask.convert("RGBA"))
    white_mask_bool = (mask_arr[...,0] > 200) & (mask_arr[...,1] > 200) & (mask_arr[...,2] > 200) & (mask_arr[...,3] > 10)
    white_alpha = Image.fromarray((white_mask_bool * 255).astype(np.uint8)).convert("L")
    white_alpha = white_alpha.filter(ImageFilter.GaussianBlur(radius=6))
    layer = Image.alpha_composite(layer, Image.composite(blur_layer, Image.new("RGBA", composed.size, (0,0,0,0)), white_alpha))

# Prepare face: cover green box and clip exactly to green mask
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

# Clip face_layer to green mask
mask_arr = np.array(mask.convert("RGBA"))
green_mask_bool = (mask_arr[...,1] > 200) & (mask_arr[...,0] < 100) & (mask_arr[...,2] < 100) & (mask_arr[...,3] > 10)
green_alpha = Image.fromarray((green_mask_bool * 255).astype(np.uint8)).convert("L")
clipped_face = Image.composite(face_layer, Image.new("RGBA", composed.size, (0,0,0,0)), green_alpha)

final = Image.alpha_composite(composed, layer)
final = Image.alpha_composite(final, clipped_face)

st.subheader("Final result")
st.image(final, use_column_width=True)

# download
buf = io.BytesIO()
final.save(buf, format="PNG")
buf.seek(0)
st.download_button("Download PNG", data=buf, file_name="car_with_face.png", mime="image/png")

st.markdown(
    "- Draw or adjust one rectangle on the canvas to select the face (or use the slider fallback if canvas unavailable).\n"
    "- The app will automatically use the rest of the photo to fill the WHITE area, and blur the face region out of that fill so the face doesn't show twice.\n"
    "- Only small alignment controls are provided (offset & rotation) to keep the UI simple."
)
