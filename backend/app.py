"""
Character Recognition API
==========================
Flask backend serving the CNN model for character/word recognition.

Usage:
    python backend/app.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import base64
import io
import os
import sys

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from model import CharacterCNN, EMNIST_LABELS, DIGIT_CLASSES, LETTER_CLASSES, ALL_CLASSES
from preprocess import preprocess_canvas, segment_characters

app = Flask(__name__, static_folder='../frontend/dist', static_url_path='')
CORS(app)

# Load model at startup
device = torch.device('cpu')
model = None
model_info = {}


def load_model():
    global model, model_info
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'character_cnn.pth')
    if not os.path.exists(model_path):
        print(f'WARNING: Model not found at {model_path}')
        print('Run: cd backend && python train_cnn.py')
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = CharacterCNN(num_classes=checkpoint.get('num_classes', 47))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model_info = {
        'accuracy': round(checkpoint.get('accuracy', 0), 2),
        'num_classes': checkpoint.get('num_classes', 62),
        'epoch': checkpoint.get('epoch', 0),
    }
    print(f'Model loaded: {model_info["accuracy"]}% accuracy, {model_info["num_classes"]} classes')


def predict_single(image_tensor, mode='all'):
    """Run prediction on a single preprocessed image tensor."""
    if model is None:
        return {'error': 'Model not loaded'}

    with torch.no_grad():
        tensor = torch.from_numpy(image_tensor).to(device)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    # Apply mode filtering
    if mode == 'digit':
        valid_classes = DIGIT_CLASSES
    elif mode == 'letter':
        valid_classes = LETTER_CLASSES
    else:
        valid_classes = ALL_CLASSES

    # Mask invalid classes and renormalize
    masked_probs = torch.zeros_like(probs)
    masked_probs[valid_classes] = probs[valid_classes]
    if masked_probs.sum() > 0:
        masked_probs = masked_probs / masked_probs.sum()

    pred_idx = int(masked_probs.argmax())
    confidence = float(masked_probs[pred_idx]) * 100

    # Build probability list for valid classes
    prob_list = []
    for idx in valid_classes:
        prob_list.append({
            'label': EMNIST_LABELS[idx],
            'value': round(float(masked_probs[idx]) * 100, 1),
            'index': idx,
        })
    prob_list.sort(key=lambda x: x['value'], reverse=True)

    return {
        'prediction': pred_idx,
        'label': EMNIST_LABELS[pred_idx],
        'confidence': round(confidence, 1),
        'probabilities': prob_list,
    }


# ─── ROUTES ───

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_tensor = preprocess_canvas(data['image'])
    mode = data.get('mode', 'all')
    result = predict_single(image_tensor, mode)
    return jsonify(result)


@app.route('/api/predict-word', methods=['POST'])
def predict_word():
    data = request.get_json()
    segments = segment_characters(data['image'])

    if not segments:
        return jsonify({'word': '', 'characters': []})

    characters = []
    word = ''
    for seg in segments:
        result = predict_single(seg['image'], mode='all')
        characters.append({
            'label': result['label'],
            'confidence': result['confidence'],
            'bbox': seg['bbox'],
        })
        word += result['label']

    return jsonify({'word': word, 'characters': characters})


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    return jsonify({
        **model_info,
        'labels': EMNIST_LABELS,
        'model_type': 'CNN (PyTorch)',
        'dataset': 'EMNIST Balanced',
        'input_size': '28x28',
        'total_labels': len(EMNIST_LABELS),
    })


@app.route('/api/random-samples', methods=['GET'])
def random_samples():
    """Return random test samples from EMNIST for the gallery."""
    try:
        from torchvision import datasets, transforms
        test_dataset = datasets.EMNIST(
            root=os.path.join(os.path.dirname(__file__), '..', 'data'),
            split='balanced', train=False, download=False,
            transform=transforms.ToTensor()
        )

        indices = np.random.choice(len(test_dataset), 12, replace=False)
        samples = []

        for i in indices:
            img_tensor, true_label = test_dataset[i]
            # Fix EMNIST orientation
            img_tensor = img_tensor.transpose(1, 2).flip(2)
            # Predict
            with torch.no_grad():
                logits = model(img_tensor.unsqueeze(0).to(device))
                pred_label = int(logits.argmax(1)[0])

            # Convert to base64
            img_arr = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_arr, mode='L')
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode()

            samples.append({
                'image': f'data:image/png;base64,{b64}',
                'true_label': EMNIST_LABELS[int(true_label)],
                'predicted': EMNIST_LABELS[pred_label],
                'correct': pred_label == int(true_label),
            })

        return jsonify(samples)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Serve React static files
@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


load_model()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'\n  http://localhost:{port}\n')
    app.run(debug=False, host='0.0.0.0', port=port)
