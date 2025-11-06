import streamlit as st
from PIL import Image, ImageFilter
import numpy as np
import io

st.set_page_config(page_title="Fixed-mask Face Slot (no external cropper)", layout="centered")
st.title("Place a face into the fixed GREEN slot (no extra dependencies)")

st.write(
    "Place template.png and mask.png (in the app folder). Mask must use vivid GREEN (0,255,0) for the face rectangle, "
    "and WHITE (255,255,255) for the blurred fill area. Upload a photo and use sliders to crop/zoom/position the face."
)

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
    st.error("Mask does not contain a vivid GREEN box (0,255,0). Edit mask.png to set the face rectangle to pure green.")
    st.stop()

st.write(f"Detected GREEN box: {green_box}")
if white_box is None:
    st.warning("No WHITE fill area detected. The app will skip the blurred fill step.")

# Upload photo
uploaded = st.file_uploader("Upload a portrait/photo (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload a photo to start. Crop controls will appear after upload.")
    st.stop()

photo = Image.open(uploaded).convert("RGBA")
pw, ph = photo.size
st.subheader("Photo preview")
st.image(photo, width=300)

# Compute green aspect for crop aspect locking
gw = green_box[2] - green_box[0]
gh = green_box[3] - green_box[1]
aspect = None
if gh != 0:
    aspect = gw / gh

st.subheader("Crop controls (slider-based)")
st.write("Move center sliders and adjust zoom so the important facial area fits the green slot.")
center_x = st.slider("Crop center X (%)", 0, 100, 50)
center_y = st.slider("Crop center Y (%)", 0, 100, 50)
zoom = st.slider("Zoom (%) — higher = zoom in (100 = no zoom)", 10, 400, 100)
rotation = st.slider("Rotate crop (degrees)", -30, 30, 0)

# Build crop rectangle in photo coords that matches green aspect
cx = int(center_x / 100.0 * pw)
cy = int(center_y / 100.0 * ph)

# Base crop size choose from smaller dimension
base = int(min(pw, ph) * 0.6)
if aspect is not None and aspect > 0:
    # choose crop that respects green aspect: width = base, height = base/aspect
    crop_w = base
    crop_h = max(10, int(base / aspect))
else:
    crop_w = base
    crop_h = base

# apply zoom: zoom >100 means zoom in => smaller crop area
crop_w = max(10, int(crop_w * (100.0 / zoom)))
crop_h = max(10, int(crop_h * (100.0 / zoom)))

left = max(0, cx - crop_w // 2)
top = max(0, cy - crop_h // 2)
right = min(pw, left + crop_w)
bottom = min(ph, top + crop_h)

# re-center if truncated
if right - left < crop_w:
    left = max(0, right - crop_w)
if bottom - top < crop_h:
    top = max(0, bottom - crop_h)

crop_box = (left, top, right, bottom)
st.write(f"Crop box on photo (pixels): {crop_box}")

face_crop = photo.crop(crop_box).convert("RGBA")
if rotation != 0:
    face_crop = face_crop.rotate(rotation, resample=Image.BICUBIC, expand=True)

st.image(face_crop, caption="Cropped face preview", width=240)

# Fine-tune placement
st.subheader("Placement fine-tune")
offset_x = st.slider("Offset X (pixels)", -500, 500, 0)
offset_y = st.slider("Offset Y (pixels)", -500, 500, 0)

# White fill settings
st.subheader("White-area fill settings")
blur_radius = st.slider("Blur radius for white fill", 0, 80, 18)
shear_x = st.slider("Horizontal shear (%)", -40, 40, 0)
shear_y = st.slider("Vertical shear (%)", -40, 40, 0)
feather = st.slider("Feather edges of white area (px)", 0, 80, 20)

# Create blurred fill for white box
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

white_fill = None
if white_box is not None:
    white_fill = create_blurred_fill(photo, white_box, blur_radius, shear_x, shear_y)

# Compose final result
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
    layer = Image.alpha_composite(layer, Image.composite(blur_layer, Image.new("RGBA", composed.size, (0,0,0,0)), white_alpha))

# Resize face_crop to cover green box (cover behavior)
face = face_crop.convert("RGBA")
fw, fh = face.size
scale_w = (green_box[2] - green_box[0]) / fw
scale_h = (green_box[3] - green_box[1]) / fh
scale = max(scale_w, scale_h)
new_w = max(1, int(fw * scale))
new_h = max(1, int(fh * scale))
face_resized = face.resize((new_w, new_h), Image.LANCZOS)

paste_x = green_box[0] + (green_box[2] - green_box[0] - new_w)//2 + offset_x
paste_y = green_box[1] + (green_box[3] - green_box[1] - new_h)//2 + offset_y

# Create green alpha mask
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

# Download
buf = io.BytesIO()
final.save(buf, format="PNG")
buf.seek(0)
st.download_button("Download PNG", data=buf, file_name="car_with_face.png", mime="image/png")

st.markdown(
    "- Crop using the sliders so the important face area sits inside the green slot.\n"
    "- Use rotation and offsets to fine tune alignment.\n"
    "- If green detection fails, edit mask.png to use pure RGB green (0,255,0)."
)
