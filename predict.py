"""
Predict handwritten digits using the trained KNN model.

Usage:
    python predict.py                    # Predict on random test samples
    python predict.py --image path.png   # Predict on a custom image (28x28)
"""

import numpy as np
import pickle
import argparse
import sys


def load_model(filepath='models/knn_digit_model.pkl'):
    """Load the trained KNN model."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


def predict_from_test_data():
    """Load test data and predict on random samples."""
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    model = load_model()

    df = pd.read_csv('data/train.csv')
    X = df.drop('label', axis=1).values / 255.0
    y = df['label'].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    indices = np.random.choice(len(X_test), 10, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('KNN Predictions', fontsize=14)

    correct = 0
    for idx, ax in enumerate(axes.flat):
        i = indices[idx]
        prediction = model.predict(X_test[i].reshape(1, -1))[0]
        if prediction == y_test[i]:
            correct += 1
        ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
        color = 'green' if prediction == y_test[i] else 'red'
        ax.set_title(f'Pred: {prediction} | True: {y_test[i]}', fontsize=9, color=color)

    plt.tight_layout()
    plt.savefig('outputs/predictions.png', dpi=150, bbox_inches='tight')
    print(f'Predictions: {correct}/10 correct')
    print('Saved to outputs/predictions.png')


def predict_from_image(image_path):
    """Predict digit from a custom image file."""
    from PIL import Image

    model = load_model()

    img = Image.open(image_path).convert('L').resize((28, 28))
    pixels = np.array(img).reshape(1, -1) / 255.0

    prediction = model.predict(pixels)[0]
    print(f'Predicted digit: {prediction}')
    return prediction


def main():
    parser = argparse.ArgumentParser(description='Predict handwritten digits')
    parser.add_argument('--image', type=str, help='Path to a 28x28 digit image')
    args = parser.parse_args()

    if args.image:
        predict_from_image(args.image)
    else:
        predict_from_test_data()


if __name__ == '__main__':
    main()
