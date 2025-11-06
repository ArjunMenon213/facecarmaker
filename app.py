import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import io
from streamlit_cropper import st_cropper

st.set_page_config(page_title="Simple Face → Green Slot (Origami Car)", layout="centered")
st.title("Place a face into the fixed GREEN slot")

st.write(
    "This app uses a single fixed template+mask (place template.png and mask.png in the app folder). "
    "On the mask: vivid GREEN (0,255,0) marks the rectangular face slot where the user's crop will be placed. "
    "Vivid WHITE marks the area that will be filled by a blurred/warped version of the photo. "
    "UI: user uploads a photo and crops it interactively to fit the green slot."
)

st.markdown("Files required in the app folder: `template.png` (RGBA) and `mask.png` (RGBA).")

# Upload photo only (template & mask are fixed files on the server)
uploaded = st.file_uploader("Upload a portrait/photo (jpg/png)", type=["jpg", "jpeg", "png"])

# Try to load fixed template & mask from working directory
try:
    template = Image.open("template.png").convert("RGBA")
    mask = Image.open("mask.png").convert("RGBA")
except FileNotFoundError:
    st.error("Could not find template.png and/or mask.png in the app folder. Please add them and reload.")
    st.stop()
except Exception as e:
    st.error(f"Error loading template/mask: {e}")
    st.stop()

st.subheader("Template (fixed)")
st.image(template, use_column_width=False, width=360)
st.subheader("Mask (fixed) — green = face slot, white = blur fill area")
st.image(mask, use_column_width=False, width=360)

# Helpers: find green and white bounding boxes in mask
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
    st.error("Mask does not contain a vivid GREEN area (0,255,0). Edit mask.png to set the face rectangle to pure green.")
    st.stop()

st.write(f"Detected GREEN box: {green_box}")
if white_box is None:
    st.warning("No WHITE fill area detected. The app will skip the blurred fill step.")

if uploaded is None:
    st.info("Upload a photo and crop it to fit the green slot. Use the cropper tool that appears after upload.")
    st.stop()

# Load uploaded photo
photo = Image.open(uploaded).convert("RGBA")
st.subheader("Photo preview")
st.image(photo, width=300)

# Prepare aspect ratio for cropper based on green box aspect
gw = green_box[2] - green_box[0]
gh = green_box[3] - green_box[1]
if gh == 0:
    aspect = None
else:
    aspect = gw / gh

st.subheader("Crop the photo to fit the green rectangle")
st.write("Use the interactive cropper. The crop's aspect ratio is locked to the green slot to simplify alignment.")

# Use streamlit-cropper to let user crop with the same aspect ratio as green box
try:
    if aspect is None:
        cropped = st_cropper(photo, realtime_update=True, box_color="#00FF00")
    else:
        cropped = st_cropper(photo, realtime_update=True, box_color="#00FF00", aspect_ratio=aspect)
except Exception as e:
    st.error(f"Cropper failed to initialize: {e}")
    st.stop()

# The cropper returns a PIL.Image or numpy array — normalize to PIL
if isinstance(cropped, np.ndarray):
    cropped = Image.fromarray(cropped)
if cropped is None:
    st.error("No crop returned. Please try cropping again.")
    st.stop()

st.image(cropped, caption="Cropped face (this will be placed in the green slot)", width=250)

# Small placement adjustments
st.subheader("Fine-tune placement")
offset_x = st.slider("Offset X (pixels)", -500, 500, 0)
offset_y = st.slider("Offset Y (pixels)", -500, 500, 0)
rotation = st.slider("Rotation (degrees)", -30, 30, 0)

# Settings for white fill (if present)
st.subheader("White-area fill (blur + simple shear)")
blur_radius = st.slider("Blur radius for white fill", 0, 80, 18)
shear_x = st.slider("Horizontal shear (%)", -40, 40, 0)
shear_y = st.slider("Vertical shear (%)", -40, 40, 0)
feather = st.slider("Feather edges of white area (px)", 0, 80, 20)

# Prepare the white-area blurred image (if white_box exists)
def create_blurred_fill(photo_img, target_box, blur_r, shear_x_pct, shear_y_pct):
    tw = target_box[2] - target_box[0]
    th = target_box[3] - target_box[1]
    if tw <= 0 or th <= 0:
        return None
    pw, ph = photo_img.size
    target_aspect = tw / th if th != 0 else 1.0
    photo_aspect = pw / ph if ph != 0 else 1.0

    # center crop to match aspect
    if photo_aspect > target_aspect:
        new_h = ph
        new_w = int(ph * target_aspect)
    else:
        new_w = pw
        new_h = int(pw / target_aspect) if target_aspect != 0 else ph

    left = max(0, (pw - new_w) // 2)
    top = max(0, (ph - new_h) // 2)
    cropped = photo_img.crop((left, top, left + new_w, top + new_h)).convert("RGBA")
    resized = cropped.resize((tw, th), Image.LANCZOS)
    blurred = resized.filter(ImageFilter.GaussianBlur(radius=blur_r))

    # simple affine shear
    sx = float(shear_x_pct) / 100.0
    sy = float(shear_y_pct) / 100.0
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

if white_box is not None:
    white_fill = create_blurred_fill(photo, white_box, blur_radius, shear_x, shear_y)
else:
    white_fill = None

# Compose final image
composed = template.copy().convert("RGBA")
layer = Image.new("RGBA", composed.size, (0,0,0,0))

# Paste white fill with feathered alpha
if white_fill is not None:
    tw = white_box[2] - white_box[0]
    th = white_box[3] - white_box[1]
    wf_resized = white_fill.resize((tw, th), Image.LANCZOS)
    blur_layer = Image.new("RGBA", composed.size, (0,0,0,0))
    blur_layer.paste(wf_resized, (white_box[0], white_box[1]))
    # create alpha from white mask
    mask_arr = np.array(mask.convert("RGBA"))
    white_mask_bool = (mask_arr[...,0] > 200) & (mask_arr[...,1] > 200) & (mask_arr[...,2] > 200) & (mask_arr[...,3] > 10)
    white_alpha = Image.fromarray((white_mask_bool * 255).astype(np.uint8)).convert("L")
    if feather > 0:
        white_alpha = white_alpha.filter(ImageFilter.GaussianBlur(radius=feather))
    # composite
    layer = Image.alpha_composite(layer, Image.composite(blur_layer, Image.new("RGBA", composed.size, (0,0,0,0)), white_alpha))

# Prepare face crop: rotate, resize to cover green box (cover strategy)
face = cropped.convert("RGBA")
if rotation != 0:
    face = face.rotate(rotation, resample=Image.BICUBIC, expand=True)

fw, fh = face.size
scale_w = (green_box[2] - green_box[0]) / fw
scale_h = (green_box[3] - green_box[1]) / fh
scale = max(scale_w, scale_h)  # cover
new_w = max(1, int(fw * scale))
new_h = max(1, int(fh * scale))
face_resized = face.resize((new_w, new_h), Image.LANCZOS)

paste_x = green_box[0] + (green_box[2] - green_box[0] - new_w)//2 + offset_x
paste_y = green_box[1] + (green_box[3] - green_box[1] - new_h)//2 + offset_y

# Clip face to exactly the green area using green mask
mask_arr = np.array(mask.convert("RGBA"))
green_mask_bool = (mask_arr[...,1] > 200) & (mask_arr[...,0] < 100) & (mask_arr[...,2] < 100) & (mask_arr[...,3] > 10)
green_alpha = Image.fromarray((green_mask_bool * 255).astype(np.uint8)).convert("L")

face_layer = Image.new("RGBA", composed.size, (0,0,0,0))
face_layer.paste(face_resized, (paste_x, paste_y), face_resized)
clipped_face = Image.composite(face_layer, Image.new("RGBA", composed.size, (0,0,0,0)), green_alpha)

final = Image.alpha_composite(composed, layer)
final = Image.alpha_composite(final, clipped_face)

st.subheader("Final result")
st.image(final, use_column_width=True)

# Download button
buf = io.BytesIO()
final.save(buf, format="PNG")
buf.seek(0)
st.download_button("Download PNG", data=buf, file_name="car_with_face.png", mime="image/png")

st.markdown(
    "- Crop so the important facial area sits inside the green preview.\n"
    "- Use offsets/rotation to fine tune.\n"
    "- If green detection fails, edit mask.png so the face region is pure RGB (0,255,0)."
)
