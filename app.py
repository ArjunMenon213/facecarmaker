import streamlit as st
from PIL import Image, ImageFilter
import numpy as np
import io
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Interactive crop (drawable-canvas)", layout="centered")
st.title("Drag & resize rectangle on your photo → place into GREEN slot")

st.write("Put template.png and mask.png in the app folder (mask: pure green = face slot). Upload a photo, draw/resize the rectangle to crop the face, then apply.")

# load fixed template & mask
try:
    template = Image.open("template.png").convert("RGBA")
    mask = Image.open("mask.png").convert("RGBA")
except FileNotFoundError:
    st.error("Missing template.png and/or mask.png in app folder. Add them then reload.")
    st.stop()

st.image(template, caption="Template (fixed)", width=360)
st.image(mask, caption="Mask (fixed) — green = face slot", width=360)

# detect green bounding box in mask
def find_color_bbox(mask_rgba, color="green"):
    arr = np.array(mask_rgba.convert("RGBA"))
    r = arr[...,0]; g = arr[...,1]; b = arr[...,2]; a = arr[...,3]
    if color == "green":
        mask_bool = (g > 200) & (r < 100) & (b < 100) & (a > 10)
    elif color == "white":
        mask_bool = (r > 200) & (g > 200) & (b > 200) & (a > 10)
    else:
        return None
    ys, xs = np.where(mask_bool)
    if len(xs)==0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

green_box = find_color_bbox(mask, "green")
white_box = find_color_bbox(mask, "white")
if green_box is None:
    st.error("Mask does not contain vivid GREEN region (0,255,0). Edit mask.png to use pure green for the face slot.")
    st.stop()
st.write("Detected GREEN box:", green_box)

# upload photo
uploaded = st.file_uploader("Upload a photo (jpg/png)", type=["jpg","jpeg","png"])
if uploaded is None:
    st.info("Upload a photo to enable the on-canvas crop tool.")
    st.stop()

photo = Image.open(uploaded).convert("RGBA")
pw, ph = photo.size
st.image(photo, caption="Photo preview", width=360)

# Canvas size — show photo at reasonable display size while mapping coords back to original image
display_w = st.slider("Display width for crop canvas (px)", 300, 1200, 600)
scale = display_w / pw
display_h = int(ph * scale)

# Create canvas: allow user to draw one rectangle (by enabling 'rect' tool and locking drawing of multiple shapes)
st.write("Draw a rectangle (or adjust an existing one). When ready, click 'Apply crop' below.")
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

# Get the last rectangle drawn (or existing rect)
rect = None
if canvas_result.json_data and "objects" in canvas_result.json_data:
    objs = canvas_result.json_data["objects"]
    # find latest 'rect' object
    rects = [o for o in objs if o.get("type") == "rect"]
    if rects:
        last = rects[-1]
        left = last.get("left",0)
        top = last.get("top",0)
        width = last.get("width",0) * last.get("scaleX",1)
        height = last.get("height",0) * last.get("scaleY",1)
        rect = (left, top, width, height)

if rect is None:
    st.warning("Draw a rectangle on the image to define the face crop.")
else:
    st.write("Crop rectangle on canvas (display coords):", rect)
    # map to original image coords
    left, top, width, height = rect
    ox = int(left / scale)
    oy = int(top / scale)
    ow = int(width / scale)
    oh = int(height / scale)
    # clamp
    ox = max(0, min(pw-1, ox))
    oy = max(0, min(ph-1, oy))
    ow = max(1, min(pw-ox, ow))
    oh = max(1, min(ph-oy, oh))
    st.write("Crop rectangle in original photo pixels:", (ox, oy, ox+ow, oy+oh))
    face_crop = photo.crop((ox, oy, ox+ow, oy+oh)).convert("RGBA")
    st.image(face_crop, caption="Cropped face preview", width=240)

    # small fine adjust controls
    offset_x = st.slider("Offset X (pixels)", -400, 400, 0)
    offset_y = st.slider("Offset Y (pixels)", -400, 400, 0)
    rotation = st.slider("Rotate crop (degrees)", -30, 30, 0)

    # white fill controls (optional)
    blur_radius = st.slider("Blur radius for white fill", 0, 64, 18)
    feather = st.slider("Feather white area edges (px)", 0, 64, 18)

    # apply rotation
    if rotation != 0:
        face_crop = face_crop.rotate(rotation, resample=Image.BICUBIC, expand=True)

    # prepare white fill if present
    def create_blur_fill(photo_img, box, radius, feather_px):
        tw = box[2] - box[0]; th = box[3] - box[1]
        if tw<=0 or th<=0:
            return None
        # center-crop photo to aspect and resize
        pw, ph = photo_img.size
        target_aspect = tw/th if th!=0 else 1.0
        photo_aspect = pw/ph if ph!=0 else 1.0
        if photo_aspect > target_aspect:
            new_h = ph; new_w = int(ph*target_aspect)
        else:
            new_w = pw; new_h = int(pw/target_aspect) if target_aspect!=0 else ph
        left = max(0,(pw-new_w)//2); top = max(0,(ph-new_h)//2)
        cropped = photo_img.crop((left, top, left+new_w, top+new_h)).convert("RGBA")
        resized = cropped.resize((tw, th), Image.LANCZOS)
        blurred = resized.filter(ImageFilter.GaussianBlur(radius=radius))
        return blurred

    # Compose final result
    composed = template.copy().convert("RGBA")
    layer = Image.new("RGBA", composed.size, (0,0,0,0))

    if white_box is not None:
        wf = create_blur_fill(photo, white_box, blur_radius, feather)
        if wf is not None:
            blur_layer = Image.new("RGBA", composed.size, (0,0,0,0))
            blur_layer.paste(wf, (white_box[0], white_box[1]))
            # build alpha from white mask
            mask_arr = np.array(mask.convert("RGBA"))
            white_mask_bool = (mask_arr[...,0] > 200) & (mask_arr[...,1] > 200) & (mask_arr[...,2] > 200) & (mask_arr[...,3] > 10)
            white_alpha = Image.fromarray((white_mask_bool*255).astype(np.uint8)).convert("L")
            if feather>0:
                white_alpha = white_alpha.filter(ImageFilter.GaussianBlur(radius=feather))
            layer = Image.alpha_composite(layer, Image.composite(blur_layer, Image.new("RGBA", composed.size, (0,0,0,0)), white_alpha))

    # place face crop into green slot (cover)
    face = face_crop
    gw = green_box[2] - green_box[0]; gh = green_box[3] - green_box[1]
    fw, fh = face.size
    scale_factor = max(gw/fw, gh/fh)
    new_w = max(1, int(fw*scale_factor)); new_h = max(1, int(fh*scale_factor))
    face_resized = face.resize((new_w, new_h), Image.LANCZOS)
    paste_x = green_box[0] + (gw - new_w)//2 + offset_x
    paste_y = green_box[1] + (gh - new_h)//2 + offset_y

    face_layer = Image.new("RGBA", composed.size, (0,0,0,0))
    face_layer.paste(face_resized, (paste_x, paste_y), face_resized)
    # clip to green mask
    mask_arr = np.array(mask.convert("RGBA"))
    green_mask_bool = (mask_arr[...,1] > 200) & (mask_arr[...,0] < 100) & (mask_arr[...,2] < 100) & (mask_arr[...,3] > 10)
    green_alpha = Image.fromarray((green_mask_bool*255).astype(np.uint8)).convert("L")
    clipped_face = Image.composite(face_layer, Image.new("RGBA", composed.size, (0,0,0,0)), green_alpha)

    final = Image.alpha_composite(composed, layer)
    final = Image.alpha_composite(final, clipped_face)

    st.subheader("Final result")
    st.image(final, use_column_width=True)

    buf = io.BytesIO()
    final.save(buf, format="PNG")
    buf.seek(0)
    st.download_button("Download final PNG", data=buf, file_name="car_with_face.png", mime="image/png")
