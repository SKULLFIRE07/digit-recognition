# Handwritten Digit Recognition

A machine learning system that recognizes handwritten digits (0–9) using the K-Nearest Neighbors algorithm, trained on the full MNIST dataset (70,000 samples). Includes an interactive web UI where you can draw digits and get real-time predictions.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange)
![Flask](https://img.shields.io/badge/Flask-3.0+-green)

## Demo

Draw a digit on the canvas → get an instant prediction with confidence scores and probability distribution for all 10 classes.

## Project Structure

```
├── app.py                          # Flask web app (backend + API)
├── train.py                        # Model training script
├── predict.py                      # CLI prediction script
├── knn_digit_recognition.ipynb     # Jupyter notebook (full walkthrough)
├── templates/
│   └── index.html                  # Web UI (canvas + predictions)
├── data/
│   └── train.csv                   # MNIST dataset (70,000 samples)
├── models/
│   └── knn_digit_model.pkl         # Trained model
├── outputs/                        # Generated plots and figures
├── requirements.txt                # Python dependencies
└── README.md
```

## Setup

```bash
# Clone
git clone https://github.com/SKULLFIRE07/digit-recognition.git
cd digit-recognition

# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Download Dataset

The MNIST dataset is not included in the repo. Run this to download it:

```bash
python -c "
from sklearn.datasets import fetch_openml
import pandas as pd, os
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data, mnist.target.astype(int)
df = pd.DataFrame(X, columns=[f'pixel{i}' for i in range(784)])
df.insert(0, 'label', y)
os.makedirs('data', exist_ok=True)
df.to_csv('data/train.csv', index=False)
print(f'Downloaded {len(df)} samples')
"
```

## Train

```bash
python train.py
```

Trains KNN with hyperparameter tuning (K=1–15), saves the best model to `models/knn_digit_model.pkl`, and generates evaluation plots in `outputs/`.

## Web UI

```bash
python app.py
```

Open **http://localhost:5000** — draw a digit on the canvas and hit Predict.

Features:
- Drawing canvas with adjustable stroke size
- Real-time prediction with confidence percentage
- Probability distribution bars for all 10 digits
- Keyboard shortcuts: `Enter` to predict, `Esc` to clear

## CLI Prediction

```bash
python predict.py                    # Predict on random test samples
python predict.py --image digit.png  # Predict on a custom 28x28 image
```

## Notebook

```bash
jupyter notebook knn_digit_recognition.ipynb
```

Full walkthrough including:
- Data exploration and visualization
- KNN implemented from scratch (pure NumPy)
- KNN with scikit-learn
- Confusion matrix and classification report
- Hyperparameter tuning
- Model persistence

## How It Works

**K-Nearest Neighbors** classifies a digit by:
1. Computing the distance between the input image and all training images (784-dimensional vectors)
2. Selecting the K closest neighbors
3. Taking a majority vote of their labels

The web UI preprocesses canvas drawings to match MNIST format:
- Crops to bounding box of the drawn stroke
- Centers and pads to square
- Resizes to 28x28 pixels with anti-aliasing
- Normalizes pixel values to [0, 1]

## Results

| Metric | Value |
|--------|-------|
| **Dataset** | MNIST (70,000 samples) |
| **Train/Test Split** | 80/20 |
| **Best K** | 3 |
| **Test Accuracy** | ~97% |
| **Training Samples** | 56,000 |
| **Image Size** | 28 x 28 px (784 features) |

## Tech Stack

- **Python** — core language
- **scikit-learn** — KNN classifier
- **NumPy / Pandas** — data processing
- **Matplotlib / Seaborn** — visualization
- **Flask** — web backend
- **HTML / CSS / JS** — web UI (vanilla, no frameworks)
