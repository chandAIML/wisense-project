# train_baseline_model.py
#
# GOAL: Train our FIRST real AI model (Random Forest - a baseline model,
# see Section 13 of the proposal). This is Phase 4 of the roadmap.
#
# STEPS THIS SCRIPT DOES:
#   1. Load train and test data (already split for us by UT-HAR)
#   2. Turn each raw (250, 90) CSI sample into a short FEATURE VECTOR
#      (this is "Feature Extraction" from Figure 6.1 / 13.1)
#   3. Train a Random Forest classifier on the TRAIN features
#   4. Test it on the TEST features it has never seen
#   5. Report Accuracy, Precision, Recall, F1, and a Confusion Matrix
#      (exactly the metrics list in Section 13)
#
# Run this with:  python3 train_baseline_model.py

import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

DATA_DIR = "Data/UT_HAR"

ACTIVITY_NAMES = {
    0: "lie down", 1: "fall", 2: "walk",
    3: "pickup", 4: "run", 5: "sit down", 6: "stand up"
}
CLASS_LABELS = [ACTIVITY_NAMES[i] for i in range(7)]

def load_csi_file(path):
    with open(path, "rb") as f:
        return np.load(f)

def extract_features(X):
    """
    Turn raw CSI of shape (num_samples, 250, 90) into a short feature
    vector per sample.

    WHY: A Random Forest can't directly use a (250, 90) grid per sample -
    it expects one flat row of numbers per sample. So instead of feeding
    it all 250*90 = 22,500 raw numbers (too many, too noisy), we summarize
    each of the 90 CSI subcarriers using two simple statistics over time:
        - mean   (the average signal level - is it high or low overall?)
        - std    (the variability - how much did it move? this is exactly
                  the number we printed in Part 2!)

    Result: 90 means + 90 stds = 180 features per sample.
    This is a standard, simple, and interpretable feature set for a
    baseline model.
    """
    mean_features = X.mean(axis=1)   # shape: (num_samples, 90)
    std_features = X.std(axis=1)     # shape: (num_samples, 90)
    return np.concatenate([mean_features, std_features], axis=1)  # (num_samples, 180)

def main():
    print("Loading data...")
    X_train = load_csi_file(os.path.join(DATA_DIR, "data", "X_train.csv"))
    y_train = load_csi_file(os.path.join(DATA_DIR, "label", "y_train.csv")).astype(int)
    X_test = load_csi_file(os.path.join(DATA_DIR, "data", "X_test.csv"))
    y_test = load_csi_file(os.path.join(DATA_DIR, "label", "y_test.csv")).astype(int)

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    print("\nExtracting features (mean + std per subcarrier)...")
    train_features = extract_features(X_train)
    test_features = extract_features(X_test)
    print(f"Feature vector length per sample: {train_features.shape[1]}")

    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(train_features, y_train)

    print("Testing on unseen test data...")
    y_pred = model.predict(test_features)

    # ------------------------------------------------------------
    # METRICS - exactly the list from Section 13 of the proposal
    # ------------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "=" * 60)
    print(f"ACCURACY: {acc*100:.2f}%")
    print("=" * 60)

    print("\nPrecision / Recall / F1-score per activity:")
    print(classification_report(y_test, y_pred, target_names=CLASS_LABELS,
                                 zero_division=0))

    # Confusion matrix - save as an image
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    plt.title("Confusion Matrix - Baseline Random Forest")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("\nSaved: confusion_matrix.png")

    print("\n" + "=" * 60)
    print("WHAT TO NOTICE (write these down for the report):")
    print("1. Which activities does the model get right most often?")
    print("2. Look at the confusion matrix - which pairs of activities")
    print("   get CONFUSED with each other most? (off-diagonal cells)")
    print("3. Is accuracy higher or lower than you expected?")
    print("=" * 60)

if __name__ == "__main__":
    main()
