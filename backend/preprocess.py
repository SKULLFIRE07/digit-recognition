"""
Image Preprocessing
===================
Converts canvas drawings to 28x28 EMNIST-format images.
Also handles character segmentation for word/text mode.

Key improvements:
- Gaussian blur to smooth canvas strokes (match EMNIST pen style)
- Better thresholding with Otsu-like adaptive method
- Improved center-of-mass alignment
"""

import numpy as np
import base64
import io
from PIL import Image, ImageFilter
from scipy.ndimage import center_of_mass, shift, label


def preprocess_canvas(data_url):
    """
    Convert canvas data URL to a 28x28 EMNIST-format image tensor.

    Steps match how EMNIST was originally created:
    1. Extract the drawn content (bounding box)
    2. Fit into 20x20 box (preserving aspect ratio)
    3. Place in 28x28 frame centered by center of mass
    4. Apply light Gaussian blur to match EMNIST stroke style
    """
    header, encoded = data_url.split(',', 1)
    img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert('L')
    pixels = np.array(img, dtype=np.float64)

    mask = pixels > 20
    if not mask.any():
        return np.zeros((1, 1, 28, 28), dtype=np.float32)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add small padding around the bounding box
    pad = 2
    rmin = max(0, rmin - pad)
    rmax = min(pixels.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(pixels.shape[1] - 1, cmax + pad)

    cropped = pixels[rmin:rmax + 1, cmin:cmax + 1]

    # Fit into 20x20 box preserving aspect ratio
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    resized = np.array(Image.fromarray(cropped.astype(np.uint8)).resize(
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
        shift_y = 14.0 - cy
        shift_x = 14.0 - cx
        # Clamp shift to avoid pushing content off the edge
        shift_y = np.clip(shift_y, -4, 4)
        shift_x = np.clip(shift_x, -4, 4)
        canvas28 = shift(canvas28, [shift_y, shift_x], order=1, mode='constant', cval=0)

    # Apply light Gaussian blur to match EMNIST stroke style
    img28 = Image.fromarray(np.clip(canvas28, 0, 255).astype(np.uint8), mode='L')
    img28 = img28.filter(ImageFilter.GaussianBlur(radius=0.5))
    canvas28 = np.array(img28, dtype=np.float64)

    canvas28 = canvas28 / 255.0
    return canvas28.reshape(1, 1, 28, 28).astype(np.float32)


def segment_characters(data_url):
    """
    Segment a canvas image containing multiple characters into individual characters.

    Returns a list of (preprocessed_image, bbox) tuples sorted left-to-right.
    """
    header, encoded = data_url.split(',', 1)
    img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert('L')
    pixels = np.array(img, dtype=np.float64)

    # Binary threshold
    binary = (pixels > 30).astype(np.int32)
    if not binary.any():
        return []

    # Find connected components
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

        # Skip tiny noise components
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

    # Merge horizontally overlapping components (parts of same character like 'i', 'j')
    components.sort(key=lambda c: c['cmin'])
    merged = [components[0]]
    for comp in components[1:]:
        prev = merged[-1]
        # If horizontal overlap > 50% of smaller width, merge
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

    # Extract and preprocess each character
    results = []
    for comp in merged:
        pad = 6
        rmin = max(0, comp['rmin'] - pad)
        rmax = min(pixels.shape[0] - 1, comp['rmax'] + pad)
        cmin = max(0, comp['cmin'] - pad)
        cmax = min(pixels.shape[1] - 1, comp['cmax'] + pad)

        char_img = pixels[rmin:rmax + 1, cmin:cmax + 1]

        # Process like single character
        h, w = char_img.shape
        scale = 20.0 / max(h, w)
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        resized = np.array(Image.fromarray(char_img.astype(np.uint8)).resize(
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

        # Gaussian blur to match EMNIST
        img28 = Image.fromarray(np.clip(canvas28, 0, 255).astype(np.uint8), mode='L')
        img28 = img28.filter(ImageFilter.GaussianBlur(radius=0.5))
        canvas28 = np.array(img28, dtype=np.float64) / 255.0
        tensor = canvas28.reshape(1, 1, 28, 28).astype(np.float32)

        results.append({
            'image': tensor,
            'bbox': [int(cmin), int(rmin), int(cmax), int(rmax)]
        })

    return results
