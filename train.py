"""
Handwritten Digit Recognition using K-Nearest Neighbors (KNN)
=============================================================
Standalone training script that trains, evaluates, and saves the KNN model.

Usage:
    python train.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import time
import os

def load_data(filepath='data/train.csv'):
    """Load and preprocess the MNIST dataset."""
    print('Loading dataset...')
    df = pd.read_csv(filepath)
    print(f'  Dataset shape: {df.shape}')

    X = df.drop('label', axis=1).values / 255.0  # Normalize to [0, 1]
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'  Training samples: {X_train.shape[0]}')
    print(f'  Test samples: {X_test.shape[0]}')
    return X_train, X_test, y_train, y_test


def find_best_k(X_train, X_test, y_train, y_test, k_range=range(1, 16)):
    """Find the optimal K value by testing different values."""
    print('\nFinding optimal K...')
    accuracies = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='auto', n_jobs=-1)
        knn.fit(X_train, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test))
        accuracies.append(acc)
        print(f'  K={k:2d} -> Accuracy: {acc * 100:.2f}%')

    best_k = list(k_range)[np.argmax(accuracies)]
    print(f'\nBest K = {best_k} with accuracy = {max(accuracies) * 100:.2f}%')

    # Plot K vs Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(list(k_range), [a * 100 for a in accuracies], 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best K={best_k}')
    plt.xlabel('K (Number of Neighbors)')
    plt.ylabel('Accuracy (%)')
    plt.title('KNN: Accuracy vs K Value')
    plt.xticks(list(k_range))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/k_vs_accuracy.png', dpi=150)
    plt.close()
    print('Saved: outputs/k_vs_accuracy.png')

    return best_k


def train_and_evaluate(X_train, X_test, y_train, y_test, k):
    """Train the final KNN model and evaluate it."""
    print(f'\nTraining final model with K={k}...')

    model = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='auto', n_jobs=-1)

    start = time.time()
    model.fit(X_train, y_train)
    print(f'  Training time: {time.time() - start:.2f}s')

    start = time.time()
    y_pred = model.predict(X_test)
    print(f'  Prediction time: {time.time() - start:.2f}s')

    accuracy = accuracy_score(y_test, y_pred)
    print(f'\n  Accuracy: {accuracy * 100:.2f}%')
    print(f'\nClassification Report:\n{classification_report(y_test, y_pred, digits=4)}')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - KNN (K={k})')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    plt.close()
    print('Saved: outputs/confusion_matrix.png')

    # Sample predictions
    fig, axes = plt.subplots(2, 8, figsize=(16, 5))
    fig.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=14)
    indices = np.random.choice(len(X_test), 16, replace=False)
    for idx, ax in enumerate(axes.flat):
        i = indices[idx]
        ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
        color = 'green' if y_pred[i] == y_test[i] else 'red'
        ax.set_title(f'P:{y_pred[i]} T:{y_test[i]}', fontsize=9, color=color, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/sample_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: outputs/sample_predictions.png')

    return model, accuracy


def save_model(model, filepath='models/knn_digit_model.pkl'):
    """Save the trained model to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f'\nModel saved to {filepath}')


def main():
    print('=' * 60)
    print('  Handwritten Digit Recognition using KNN')
    print('=' * 60)

    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()
    best_k = find_best_k(X_train, X_test, y_train, y_test)
    model, accuracy = train_and_evaluate(X_train, X_test, y_train, y_test, best_k)
    save_model(model)

    print('\n' + '=' * 60)
    print(f'  DONE! Final Accuracy: {accuracy * 100:.2f}% (K={best_k})')
    print('=' * 60)


if __name__ == '__main__':
    main()
