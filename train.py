# train.py

import os
import pickle
import numpy as np
import cv2
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATASET_DIR = Path("dataset")
MODEL_DIR = Path("model")
IMG_SIZE = 224

MODEL_DIR.mkdir(exist_ok=True)


def load_data():
    X, y = [], []

    for label, cls in enumerate(["anemic", "normal"]):
        folder = DATASET_DIR / cls

        for file in folder.glob("*"):
            img = cv2.imread(str(file))
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img.flatten() / 255.0)
            y.append(label)

    return np.array(X), np.array(y)


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print("Accuracy:", acc)

    with open(MODEL_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved!")


if __name__ == "__main__":
    main()