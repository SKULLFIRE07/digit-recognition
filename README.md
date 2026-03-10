# CharacterAI — Handwriting Recognition

A deep learning system that recognizes handwritten characters — **digits (0-9)**, **uppercase (A-Z)**, **lowercase (a-z)**, and **entire words** — using a Convolutional Neural Network trained on the EMNIST Balanced dataset (112,800 training samples, 47 classes). Features a modern React UI with real-time predictions.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![React](https://img.shields.io/badge/React-18+-61DAFB)
![Flask](https://img.shields.io/badge/Flask-3.0+-green)

## Features

- **62-class recognition** — digits, uppercase letters, lowercase letters
- **Word mode** — draw multiple characters, auto-segments and recognizes each one
- **Mode filtering** — switch between Digits, Letters, All, or Word mode
- **Modern React UI** — dark/light theme, animations, responsive design
- **Drawing canvas** — adjustable stroke, undo support, touch-friendly
- **Probability distribution** — see top predictions with confidence bars
- **Prediction history** — track your past predictions
- **Test sample gallery** — browse EMNIST samples with live predictions
- **Keyboard shortcuts** — `Enter` predict, `Esc` clear, `Space` undo

## Project Structure

```
├── backend/
│   ├── app.py                  # Flask API server
│   ├── model.py                # CNN architecture (PyTorch)
│   ├── train_cnn.py            # Training script (EMNIST Balanced)
│   ├── preprocess.py           # Image preprocessing & segmentation
│   └── requirements.txt        # Backend dependencies
├── frontend/
│   ├── src/
│   │   ├── App.tsx             # Main React component
│   │   ├── components/         # UI components (Canvas, Header, etc.)
│   │   ├── api/                # API client
│   │   ├── hooks/              # Custom React hooks
│   │   └── types/              # TypeScript interfaces
│   ├── package.json
│   └── vite.config.ts          # Vite + proxy config
├── app.py                      # Original KNN version (legacy)
├── train.py                    # Original KNN training (legacy)
├── predict.py                  # CLI prediction tool
├── knn_digit_recognition.ipynb # Jupyter notebook walkthrough
├── models/                     # Saved model weights (gitignored)
├── data/                       # Datasets (gitignored)
└── outputs/                    # Training plots (gitignored)
```

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/SKULLFIRE07/digit-recognition.git
cd digit-recognition

# Python dependencies
pip install -r backend/requirements.txt

# Frontend dependencies
cd frontend && npm install && cd ..
```

### 2. Train the Model

```bash
cd backend
python train_cnn.py
```

This downloads EMNIST Balanced (~500MB, automatic via torchvision), trains a CNN for 15 epochs, and saves the best model to `models/character_cnn.pth`.

### 3. Build Frontend

```bash
cd frontend
npm run build
```

### 4. Run

```bash
cd backend
python app.py
```

Open **http://localhost:5000** — draw characters and get predictions!

### Development Mode

Run Flask and Vite dev server separately for hot reload:

```bash
# Terminal 1: Backend
cd backend && python app.py

# Terminal 2: Frontend (with API proxy)
cd frontend && npm run dev
```

## How It Works

### CNN Architecture

```
Input: 1×28×28 grayscale image
  ↓
Conv2d(1→32) → BN → ReLU → Conv2d(32→32) → BN → ReLU → MaxPool → Dropout
  ↓
Conv2d(32→64) → BN → ReLU → Conv2d(64→64) → BN → ReLU → MaxPool → Dropout
  ↓
Conv2d(64→128) → BN → ReLU → MaxPool → Dropout
  ↓
Flatten → Linear(1152→256) → ReLU → Dropout → Linear(256→62)
  ↓
Output: 62 class probabilities (softmax)
```

### Preprocessing Pipeline

Canvas drawings are converted to EMNIST format:
1. Crop to bounding box of drawn content
2. Fit into 20×20 box (preserving aspect ratio)
3. Center in 28×28 frame by center of mass
4. Normalize pixel values to [0, 1]

### Word Segmentation

In word mode, connected components are detected, merged for multi-part characters (i, j), sorted left-to-right, and each character is independently recognized.

## Results

| Metric | Value |
|--------|-------|
| **Dataset** | EMNIST Balanced (697,932 samples) |
| **Classes** | 47 (0-9, A-Z, select a-z) |
| **Model** | CNN (3 conv blocks, ~340K parameters) |
| **Input Size** | 28 × 28 px |

## Tech Stack

- **PyTorch** — CNN model training and inference
- **React + TypeScript** — modern frontend UI
- **Tailwind CSS** — utility-first styling
- **Framer Motion** — smooth animations
- **Vite** — fast build tooling
- **Flask** — REST API backend
- **SciPy** — image preprocessing (center of mass)
- **Pillow** — image manipulation

## Legacy KNN Version

The original digit-only KNN version is preserved in the root files (`app.py`, `train.py`, `predict.py`, `templates/`). See the Jupyter notebook for a full walkthrough of the KNN approach.
