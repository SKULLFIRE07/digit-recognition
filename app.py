"""
Handwritten Digit Recognition - Web UI
=======================================
Flask backend with MNIST-accurate image preprocessing.

Usage:
    python app.py
    Open http://localhost:5000
"""

import numpy as np
import pandas as pd
import pickle
import base64
import io
import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
from scipy.ndimage import center_of_mass, shift

app = Flask(__name__)

# Load model at startup
with open('models/knn_digit_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load test data for the gallery feature
_test_data = None

def get_test_data():
    global _test_data
    if _test_data is None:
        from sklearn.model_selection import train_test_split
        df = pd.read_csv('data/train.csv')
        X = df.drop('label', axis=1).values / 255.0
        y = df['label'].values
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        _test_data = (X_test, y_test)
    return _test_data


def preprocess_canvas(data_url):
    """
    Convert canvas data URL to a 28x28 MNIST-format image.

    Steps match how MNIST was originally created:
    1. Extract the drawn content (bounding box)
    2. Fit into a 20x20 box (preserving aspect ratio)
    3. Place in 28x28 frame centered by center of mass
    """
    header, encoded = data_url.split(',', 1)
    img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert('L')
    pixels = np.array(img, dtype=np.float64)

    # Find bounding box of drawn content (threshold at 20 to ignore noise)
    mask = pixels > 20
    if not mask.any():
        return np.zeros((1, 784))

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop to bounding box
    cropped = pixels[rmin:rmax + 1, cmin:cmax + 1]

    # Fit into 20x20 box preserving aspect ratio (MNIST standard)
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    resized = np.array(Image.fromarray(cropped.astype(np.uint8)).resize(
        (new_w, new_h), Image.LANCZOS
    ), dtype=np.float64)

    # Place in 28x28 frame — initially centered geometrically
    canvas28 = np.zeros((28, 28), dtype=np.float64)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas28[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    # Shift so center of mass is at (14, 14) — exactly how MNIST does it
    cy, cx = center_of_mass(canvas28)
    shift_y = 14.0 - cy
    shift_x = 14.0 - cx
    canvas28 = shift(canvas28, [shift_y, shift_x], order=1, mode='constant', cval=0)

    # Clip and normalize
    canvas28 = np.clip(canvas28, 0, 255) / 255.0
    return canvas28.reshape(1, -1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    pixels = preprocess_canvas(data['image'])

    # Distance-weighted vote for confidence
    distances, indices = model.kneighbors(pixels)
    neighbor_labels = model._y[indices[0]]

    votes = np.zeros(10)
    for i, label in enumerate(neighbor_labels):
        weight = 1.0 / (distances[0][i] + 1e-8)
        votes[label] += weight

    probs = votes / votes.sum()
    pred = int(np.argmax(probs))

    return jsonify({
        'prediction': pred,
        'confidence': round(float(probs[pred]) * 100, 1),
        'probabilities': [round(float(p) * 100, 1) for p in probs]
    })


@app.route('/random_samples', methods=['GET'])
def random_samples():
    """Return random test samples for the gallery."""
    X_test, y_test = get_test_data()
    indices = np.random.choice(len(X_test), 12, replace=False)
    samples = []
    for i in indices:
        pred = model.predict(X_test[i].reshape(1, -1))[0]
        # Convert to base64 image
        img_arr = (X_test[i].reshape(28, 28) * 255).astype(np.uint8)
        img = Image.fromarray(img_arr, mode='L')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode()
        samples.append({
            'image': f'data:image/png;base64,{b64}',
            'true_label': int(y_test[i]),
            'predicted': int(pred),
            'correct': bool(pred == y_test[i])
        })
    return jsonify(samples)


if __name__ == '__main__':
    print('\n  http://localhost:5000\n')
    app.run(debug=False, port=5000)
