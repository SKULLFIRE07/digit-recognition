"""
Character Recognition CNN Model
================================
A convolutional neural network for recognizing handwritten characters
trained on EMNIST Balanced (47 classes).
"""

import torch
import torch.nn as nn


class CharacterCNN(nn.Module):
    """CNN for 28x28 grayscale character images."""

    def __init__(self, num_classes=47):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# EMNIST Balanced: 47 classes
# 0-9: digits, 10-35: uppercase A-Z, 36-46: select lowercase
# The lowercase letters included are those visually distinct from uppercase:
# a, b, d, e, f, g, h, n, q, r, t
EMNIST_BALANCED_LABELS = (
    [str(i) for i in range(10)] +                    # 0-9
    [chr(i) for i in range(65, 91)] +                # A-Z (10-35)
    list('abdefghnqrt')                              # select lowercase (36-46)
)

EMNIST_LABELS = EMNIST_BALANCED_LABELS

# Mode-specific class indices
DIGIT_CLASSES = list(range(0, 10))
UPPERCASE_CLASSES = list(range(10, 36))
LOWERCASE_CLASSES = list(range(36, 47))
LETTER_CLASSES = UPPERCASE_CLASSES + LOWERCASE_CLASSES
ALL_CLASSES = list(range(47))
