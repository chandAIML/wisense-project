# save_model.py
#
# GOAL: Train the Random Forest baseline (from Part 3) one more time and
# SAVE it to disk, so the dashboard app can load it instantly instead of
# retraining every time it starts.
#
# Run this with:  python3 save_model.py

import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = "Data/UT_HAR"

def load_csi_file(path):
    with open(path, "rb") as f:
        return np.load(f)

def extract_features(X):
    mean_features = X.mean(axis=1)
    std_features = X.std(axis=1)
    return np.concatenate([mean_features, std_features], axis=1)

def main():
    print("Loading training data...")
    X_train = load_csi_file(os.path.join(DATA_DIR, "data", "X_train.csv"))
    y_train = load_csi_file(os.path.join(DATA_DIR, "label", "y_train.csv")).astype(int)

    print("Extracting features...")
    train_features = extract_features(X_train)

    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(train_features, y_train)

    joblib.dump(model, "wisense_model.pkl")
    print("Saved model to wisense_model.pkl")

if __name__ == "__main__":
    main()
