"""
Image Preprocessing
===================
Converts canvas drawings to 28x28 EMNIST-format images.

Key insight: Canvas strokes are THICK (user draws with finger/mouse).
When resized to 28x28, thick strokes fill gaps and "3" looks like "8".
Solution: Use morphological THINNING (skeletonization) to normalize
stroke width BEFORE placing in 28x28 frame.
"""

import numpy as np
import base64
import io
from PIL import Image, ImageFilter, ImageOps
from scipy.ndimage import center_of_mass, shift, label
from skimage.morphology import skeletonize, dilation, disk


def preprocess_canvas(data_url):
    """
    Convert canvas data URL to a 28x28 EMNIST-format image tensor.

    Proven approach used by top EMNIST/MNIST web apps:
    1. Extract drawn content
    2. Skeletonize to normalize stroke width
    3. Dilate skeleton slightly to get EMNIST-like stroke width
    4. Resize to fit 20x20 box
    5. Center in 28x28 by center of mass
    """
    header, encoded = data_url.split(',', 1)
    img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert('L')
    pixels = np.array(img, dtype=np.float64)

    mask = pixels > 20
    if not mask.any():
        return np.zeros((1, 1, 28, 28), dtype=np.float32)

    # Crop to bounding box with padding
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad = 5
    rmin = max(0, rmin - pad)
    rmax = min(pixels.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(pixels.shape[1] - 1, cmax + pad)
    cropped = pixels[rmin:rmax + 1, cmin:cmax + 1]

    # Skeletonize: reduce any stroke width to 1px skeleton
    binary = cropped > 40
    skeleton = skeletonize(binary)

    # Dilate skeleton to get consistent EMNIST-like stroke width (~2-3px at this scale)
    # Use disk proportional to image size to get consistent results
    img_size = max(cropped.shape)
    radius = max(1, int(img_size / 80))  # ~2-3px for typical 200-400px canvas
    selem = disk(radius)
    thick_skeleton = dilation(skeleton, selem)

    # Convert back to grayscale with smooth edges
    result = thick_skeleton.astype(np.float64) * 255
    # Apply slight blur for anti-aliasing (EMNIST has soft edges)
    result_img = Image.fromarray(result.astype(np.uint8), mode='L')
    result_img = result_img.filter(ImageFilter.GaussianBlur(radius=0.8))
    result = np.array(result_img, dtype=np.float64)

    # Fit into 20x20 box preserving aspect ratio
    h, w = result.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    resized = np.array(Image.fromarray(result.astype(np.uint8)).resize(
        (new_w, new_h), Image.LANCZOS
    ), dtype=np.float64)

    # Place in 28x28 frame
    canvas28 = np.zeros((28, 28), dtype=np.float64)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas28[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    # Shift so center of mass is at (14, 14)
    cy, cx = center_of_mass(canvas28)
    if not (np.isnan(cy) or np.isnan(cx)):
        shift_y = np.clip(14.0 - cy, -4, 4)
        shift_x = np.clip(14.0 - cx, -4, 4)
        canvas28 = shift(canvas28, [shift_y, shift_x], order=1, mode='constant', cval=0)

    canvas28 = np.clip(canvas28, 0, 255) / 255.0
    return canvas28.reshape(1, 1, 28, 28).astype(np.float32)


def segment_characters(data_url):
    """
    Segment a canvas image containing multiple characters into individual characters.
    Returns a list of (preprocessed_image, bbox) tuples sorted left-to-right.
    """
    header, encoded = data_url.split(',', 1)
    img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert('L')
    pixels = np.array(img, dtype=np.float64)

    binary = (pixels > 30).astype(np.int32)
    if not binary.any():
        return []

    labeled, num_features = label(binary)
    if num_features == 0:
        return []

    components = []
    for i in range(1, num_features + 1):
        component_mask = labeled == i
        rows = np.any(component_mask, axis=1)
        cols = np.any(component_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        h = rmax - rmin + 1
        w = cmax - cmin + 1
        if h < 8 or w < 4:
            continue

        components.append({
            'rmin': rmin, 'rmax': rmax,
            'cmin': cmin, 'cmax': cmax,
            'h': h, 'w': w
        })

    if not components:
        return []

    components.sort(key=lambda c: c['cmin'])
    merged = [components[0]]
    for comp in components[1:]:
        prev = merged[-1]
        overlap = min(prev['cmax'], comp['cmax']) - max(prev['cmin'], comp['cmin'])
        min_w = min(prev['w'], comp['w'])
        if overlap > 0 and overlap > min_w * 0.5:
            prev['rmin'] = min(prev['rmin'], comp['rmin'])
            prev['rmax'] = max(prev['rmax'], comp['rmax'])
            prev['cmin'] = min(prev['cmin'], comp['cmin'])
            prev['cmax'] = max(prev['cmax'], comp['cmax'])
            prev['h'] = prev['rmax'] - prev['rmin'] + 1
            prev['w'] = prev['cmax'] - prev['cmin'] + 1
        else:
            merged.append(comp)

    results = []
    for comp in merged:
        pad = 8
        rmin = max(0, comp['rmin'] - pad)
        rmax = min(pixels.shape[0] - 1, comp['rmax'] + pad)
        cmin = max(0, comp['cmin'] - pad)
        cmax = min(pixels.shape[1] - 1, comp['cmax'] + pad)

        char_img = pixels[rmin:rmax + 1, cmin:cmax + 1]

        # Skeletonize + dilate + resize same as single char
        binary_char = char_img > 40
        if not binary_char.any():
            continue
        skeleton = skeletonize(binary_char)
        img_size = max(char_img.shape)
        radius = max(1, int(img_size / 80))
        selem = disk(radius)
        thick = dilation(skeleton, selem).astype(np.float64) * 255
        thick_img = Image.fromarray(thick.astype(np.uint8), mode='L')
        thick_img = thick_img.filter(ImageFilter.GaussianBlur(radius=0.8))
        thick = np.array(thick_img, dtype=np.float64)

        h, w = thick.shape
        scale = 20.0 / max(h, w)
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        resized = np.array(Image.fromarray(thick.astype(np.uint8)).resize(
            (new_w, new_h), Image.LANCZOS
        ), dtype=np.float64)

        canvas28 = np.zeros((28, 28), dtype=np.float64)
        y_off = (28 - new_h) // 2
        x_off = (28 - new_w) // 2
        canvas28[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        cy, cx = center_of_mass(canvas28)
        if not (np.isnan(cy) or np.isnan(cx)):
            shift_y = np.clip(14.0 - cy, -4, 4)
            shift_x = np.clip(14.0 - cx, -4, 4)
            canvas28 = shift(canvas28, [shift_y, shift_x], order=1, mode='constant', cval=0)

        canvas28 = np.clip(canvas28, 0, 255) / 255.0
        tensor = canvas28.reshape(1, 1, 28, 28).astype(np.float32)

        results.append({
            'image': tensor,
            'bbox': [int(cmin), int(rmin), int(cmax), int(rmax)]
        })

    return results
