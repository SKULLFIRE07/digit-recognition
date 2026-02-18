"""
Handwritten Digit Recognition - Web UI
=======================================
Flask backend with proper MNIST-style image preprocessing.

Usage:
    python app.py
    Open http://localhost:5000
"""

import numpy as np
import pickle
import base64
import io
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps

app = Flask(__name__)

with open('models/knn_digit_model.pkl', 'rb') as f:
    model = pickle.load(f)


def preprocess_canvas_image(image_data_url):
    """
    Convert a canvas data URL into a 28x28 normalized array
    matching MNIST format: white digit on black background,
    centered in frame with padding.
    """
    # Decode base64
    header, encoded = image_data_url.split(',', 1)
    image_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(image_bytes)).convert('L')

    pixels = np.array(img)

    # Find bounding box of the drawn content (non-black pixels)
    rows = np.any(pixels > 20, axis=1)
    cols = np.any(pixels > 20, axis=0)

    if not rows.any() or not cols.any():
        # Nothing drawn — return zeros
        return np.zeros((1, 784))

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop to bounding box
    cropped = pixels[rmin:rmax + 1, cmin:cmax + 1]

    # Make it square by padding the shorter side
    h, w = cropped.shape
    if h > w:
        pad = (h - w) // 2
        cropped = np.pad(cropped, ((0, 0), (pad, h - w - pad)), mode='constant')
    elif w > h:
        pad = (w - h) // 2
        cropped = np.pad(cropped, ((pad, w - h - pad), (0, 0)), mode='constant')

    # Add ~20% padding around the digit (MNIST has padding)
    size = cropped.shape[0]
    pad_size = max(int(size * 0.2), 4)
    cropped = np.pad(cropped, pad_size, mode='constant')

    # Resize to 28x28 using high-quality resampling
    img_resized = Image.fromarray(cropped.astype(np.uint8))
    img_resized = img_resized.resize((28, 28), Image.LANCZOS)

    # Normalize to [0, 1]
    result = np.array(img_resized).astype(np.float64) / 255.0
    return result.reshape(1, -1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    pixels = preprocess_canvas_image(data['image'])

    # Get prediction with distance-weighted confidence
    distances, indices = model.kneighbors(pixels)
    neighbor_labels = model._y[indices[0]]

    votes = np.zeros(10)
    for i, label in enumerate(neighbor_labels):
        weight = 1.0 / (distances[0][i] + 1e-5)
        votes[label] += weight

    probabilities = votes / votes.sum()
    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction])

    return jsonify({
        'prediction': prediction,
        'confidence': round(confidence * 100, 1),
        'probabilities': [round(float(p) * 100, 1) for p in probabilities]
    })


if __name__ == '__main__':
    print('\n  http://localhost:5000\n')
    app.run(debug=False, port=5000)
