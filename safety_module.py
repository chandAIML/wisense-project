# safety_module.py
#
# GOAL: Implement the "Safety & Decision Engine" block from Figure 6.1,
# following the exact reasoning chain from Figure 10.1 of the proposal:
#
#   Sudden Movement Pattern -> Possible Fall-Like Event -> Prolonged
#   Inactivity -> Abnormal-Event Confidence Increases -> Notification
#
# WHY THIS IS SEPARATE FROM THE MODEL:
#   The ML model (Part 3) only answers "what activity is this ONE reading?"
#   It has no concept of TIME or "what happened before/after." The safety
#   module adds that missing piece: it watches a SEQUENCE of predictions
#   and applies simple rules to decide whether to raise an alert.
#   This mirrors a real design principle: keep the "smart pattern
#   recognizer" (ML model) and the "decision maker" (business/safety
#   logic) as separate, understandable pieces.
#
# NOTE ON REALISM:
#   UT-HAR samples are independent short clips, not one continuous
#   recording. Here we CHAIN several test samples together to SIMULATE
#   a monitored session/timeline, purely for demonstrating the logic.
#   This simplification should be stated clearly in your report.
#
# Run this with:  python3 safety_module.py

import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

DATA_DIR = "Data/UT_HAR"
MODEL_PATH = "wisense_model.pkl"

ACTIVITY_NAMES = {
    0: "lie down", 1: "fall", 2: "walk",
    3: "pickup", 4: "run", 5: "sit down", 6: "stand up"
}

# ---------------- Tunable safety-rule parameters ----------------
INACTIVITY_LABELS = {0, 5}          # "lie down" and "sit down" count as inactivity
INACTIVITY_STEPS_TO_ALERT = 3        # how many consecutive inactive steps after
                                      # a possible fall before we actually alert
CONFIDENCE_ALERT_THRESHOLD = 0.35    # minimum model confidence to trust a "fall" reading

def load_csi_file(path):
    with open(path, "rb") as f:
        return np.load(f)

def extract_features(X):
    return np.concatenate([X.mean(axis=1), X.std(axis=1)], axis=1)

class SafetyEngine:
    """A small state machine implementing Figure 10.1's reasoning chain."""

    def __init__(self):
        self.watching_for_fall_aftermath = False
        self.inactivity_count = 0
        self.alerts = []  # list of (step_index, reason)

    def process_reading(self, step_idx, predicted_label, confidence):
        activity = ACTIVITY_NAMES[predicted_label]

        # Step 1: Sudden Movement Pattern -> Possible Fall-Like Event
        if predicted_label == 1 and confidence >= CONFIDENCE_ALERT_THRESHOLD:
            print(f"  [step {step_idx}] Possible fall-like event detected "
                  f"(confidence={confidence:.2f}). Watching for what happens next...")
            self.watching_for_fall_aftermath = True
            self.inactivity_count = 0
            return "possible_fall"

        # Step 2: Prolonged Inactivity (only relevant if we're watching)
        if self.watching_for_fall_aftermath:
            if predicted_label in INACTIVITY_LABELS:
                self.inactivity_count += 1
                print(f"  [step {step_idx}] Inactivity continues "
                      f"({self.inactivity_count}/{INACTIVITY_STEPS_TO_ALERT}) "
                      f"- activity: {activity}")

                # Step 3 & 4: Abnormal-Event Confidence Increases -> Notification
                if self.inactivity_count >= INACTIVITY_STEPS_TO_ALERT:
                    print(f"  [step {step_idx}] *** NOTIFICATION TRIGGERED ***")
                    print(f"      \"Possible abnormal event detected. "
                          f"Please check the monitored room.\"")
                    self.alerts.append((step_idx, "fall_followed_by_inactivity"))
                    self.watching_for_fall_aftermath = False  # reset after alerting
                    return "alert"
            else:
                # Person moved normally again soon after -> likely a false alarm,
                # e.g. they quickly sat down rather than actually falling.
                print(f"  [step {step_idx}] Person resumed normal activity "
                      f"({activity}) - false alarm dismissed, resetting.")
                self.watching_for_fall_aftermath = False
                self.inactivity_count = 0

        return "normal"

def find_indices_by_label(y_test, label, count, exclude=None):
    """Find sample indices whose TRUE label matches, regardless of where
    they sit in the array. We do this instead of taking consecutive
    indices, because the UT-HAR test set is grouped by activity in
    blocks - nearby indices are NOT a real continuous timeline. This is
    itself a useful, honest observation about the dataset, worth noting
    in your report."""
    exclude = exclude or set()
    idxs = [i for i in range(len(y_test)) if y_test[i] == label and i not in exclude]
    return idxs[:count]

def run_scenario(name, session_indices, X_test, model):
    print("\n" + "=" * 70)
    print(f"SCENARIO: {name}  ({len(session_indices)} readings)")
    print("=" * 70)

    engine = SafetyEngine()
    timeline_labels = []
    timeline_events = []

    for step_idx, sample_idx in enumerate(session_indices):
        sample = X_test[sample_idx:sample_idx+1]
        features = extract_features(sample)
        pred = model.predict(features)[0]
        probs = model.predict_proba(features)[0]
        confidence = probs[pred]

        event = engine.process_reading(step_idx, pred, confidence)
        timeline_labels.append(ACTIVITY_NAMES[pred])
        timeline_events.append(event)

    print(f"\nTotal notifications triggered: {len(engine.alerts)}")
    return timeline_labels, timeline_events, engine.alerts

def main():
    model = joblib.load(MODEL_PATH)
    X_test = load_csi_file(os.path.join(DATA_DIR, "data", "X_test.csv"))
    y_test = load_csi_file(os.path.join(DATA_DIR, "label", "y_test.csv")).astype(int)

    # ---------------- Scenario A: a genuine alert should fire ----------------
    # One real "fall"-labeled sample, followed by three real "lie down"-labeled
    # samples (pulled from wherever they exist in the test set - NOT assumed
    # to be adjacent in time, since we now know the data is grouped by class).
    fall_idxs = find_indices_by_label(y_test, 1, 1)
    lie_idxs = find_indices_by_label(y_test, 0, 3)
    if not fall_idxs or len(lie_idxs) < 3:
        print("Not enough fall/lie-down samples in test set to build Scenario A.")
        return
    scenario_a_indices = fall_idxs + lie_idxs

    # ---------------- Scenario B: a false alarm should be dismissed ----------------
    # A DIFFERENT real "fall" sample, followed by real "walk"-labeled samples
    # (person clearly fine and moving normally right after).
    fall_idxs_b = find_indices_by_label(y_test, 1, 1, exclude=set(fall_idxs))
    walk_idxs = find_indices_by_label(y_test, 2, 2)
    if not fall_idxs_b or len(walk_idxs) < 2:
        print("Not enough samples to build Scenario B.")
        scenario_b_indices = None
    else:
        scenario_b_indices = fall_idxs_b + walk_idxs

    results = []
    labels_a, events_a, alerts_a = run_scenario(
        "A - Fall followed by inactivity (should ALERT)", scenario_a_indices, X_test, model)
    results.append(("Scenario A: Fall -> Inactivity", labels_a, events_a))

    if scenario_b_indices:
        labels_b, events_b, alerts_b = run_scenario(
            "B - Fall followed by normal movement (should DISMISS as false alarm)",
            scenario_b_indices, X_test, model)
        results.append(("Scenario B: Fall -> Normal Movement", labels_b, events_b))

    # ---------------- Visualize both scenarios ----------------
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 2.6 * len(results)))
    if len(results) == 1:
        axes = [axes]
    colors = {"normal": "#1B4F72", "possible_fall": "#E8A33D", "alert": "#C0392B"}
    for ax, (title, labels, events) in zip(axes, results):
        for i, (label, event) in enumerate(zip(labels, events)):
            ax.bar(i, 1, color=colors[event], edgecolor="white")
            ax.text(i, 1.05, label, ha="center", va="bottom", fontsize=7, rotation=45)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylim(0, 1.6)
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig("safety_timeline.png", dpi=150)
    print("\nSaved: safety_timeline.png (both scenarios)")

    print("\n" + "=" * 70)
    print("DATASET OBSERVATION FOR YOUR REPORT:")
    print("The UT-HAR test set appears grouped by activity class rather than")
    print("shuffled/temporally continuous - nearby array indices tend to share")
    print("the same true label. Sessions here were built by SEARCHING for")
    print("samples with specific true labels, not by taking consecutive indices.")
    print("=" * 70)

if __name__ == "__main__":
    main()
