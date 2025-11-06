from PIL import Image, ImageOps
import numpy as np
import mediapipe as mp

def detect_face_bbox(image_np, method="mediapipe"):
    """
    image_np: HxWx3 uint8 numpy array (RGB)
    returns bounding box as (x, y, w, h) in pixel coords or None
    Uses MediaPipe face detection.
    """
    if method != "mediapipe":
        raise ValueError("Only mediapipe method implemented in this helper.")

    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.4) as detector:
        results = detector.process(image_np)
        if not results.detections:
            return None
        # Use the first detection
        det = results.detections[0]
        bboxC = det.location_data.relative_bounding_box
        ih, iw, _ = image_np.shape
        x = int(bboxC.xmin * iw)
        y = int(bboxC.ymin * ih)
        w = int(bboxC.width * iw)
        h = int(bboxC.height * ih)
        # Clamp
        x = max(0, x)
        y = max(0, y)
        w = max(1, min(iw - x, w))
        h = max(1, min(ih - y, h))
        return (x, y, w, h)


def _mask_target_box(mask_img):
    """
    Find the bounding box of the white area in mask_img (PIL Image in 'L' mode).
    Returns (left, top, right, bottom) in pixel coords.
    """
    mask = np.array(mask_img)
    # White threshold
    ys, xs = np.where(mask > 10)
    if len(xs) == 0:
        return None
    left = int(xs.min())
    right = int(xs.max())
    top = int(ys.min())
    bottom = int(ys.max())
    return (left, top, right, bottom)


def paste_face_on_template(template_img, mask_img, face_img, align="center", blend=0.9, manual_scale=None, manual_offset=(0,0)):
    """
    Paste face_img onto template_img guided by mask_img. 
    - template_img: PIL RGBA
    - mask_img: PIL L (white indicates target area)
    - face_img: PIL RGBA (or RGB)
    Returns a new PIL RGBA image.
    Optional:
    - manual_scale multiplier (None to auto fit)
    - manual_offset: (x_pixels, y_pixels) to shift the face placement
    - blend: face opacity or blending amount (0.0-1.0)
    """
    template = template_img.copy().convert("RGBA")
    mask = mask_img.copy().convert("L")
    face = face_img.copy().convert("RGBA")

    target_box = _mask_target_box(mask)
    if target_box is None:
        # fallback: paste centered, scaled to 40% of template width
        tw, th = template.size
        target_w = int(tw * 0.4)
        target_h = int(th * 0.4)
        left = (tw - target_w) // 2
        top = (th - target_h) // 2
        target_box = (left, top, left + target_w, top + target_h)

    left, top, right, bottom = target_box
    target_w = right - left
    target_h = bottom - top

    # compute scale
    fw, fh = face.size
    if manual_scale is None:
        # scale to fit height or width depending on aspect ratio
        scale_w = target_w / fw
        scale_h = target_h / fh
        scale = min(scale_w, scale_h)
    else:
        scale = manual_scale

    new_w = max(1, int(fw * scale))
    new_h = max(1, int(fh * scale))
    face_resized = face.resize((new_w, new_h), resample=Image.LANCZOS)

    # compute paste pos (centered by default)
    paste_x = left + (target_w - new_w) // 2 + manual_offset[0]
    paste_y = top + (target_h - new_h) // 2 + manual_offset[1]

    # Create a layer to composite
    layer = Image.new("RGBA", template.size, (0, 0, 0, 0))
    layer.paste(face_resized, (paste_x, paste_y), face_resized)

    # Blend layer onto template
    combined = Image.alpha_composite(template, Image.new("RGBA", template.size, (255,255,255,0)))
    if blend >= 1.0:
        combined = Image.alpha_composite(combined, layer)
    else:
        # mix using alpha for smoothness
        mask_blend = layer.split()[-1].point(lambda p: int(p * blend))
        layer_with_adjusted_alpha = layer.copy()
        layer_with_adjusted_alpha.putalpha(mask_blend)
        combined = Image.alpha_composite(combined, layer_with_adjusted_alpha)

    return combined
