# plot_csi.py
#
# GOAL: Visualize raw CSI signals as graphs, and compare two different
# activities side-by-side. This is "Signal Preprocessing" from Section 5/6
# of the proposal - the step BEFORE we feed data into an AI model.
#
# Run this with:  python plot_csi.py

import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = "Data/UT_HAR"

ACTIVITY_NAMES = {
    0: "lie down", 1: "fall", 2: "walk",
    3: "pickup", 4: "run", 5: "sit down", 6: "stand up"
}

def load_csi_file(path):
    with open(path, "rb") as f:
        return np.load(f)

def main():
    X = load_csi_file(os.path.join(DATA_DIR, "data", "X_test.csv"))
    y = load_csi_file(os.path.join(DATA_DIR, "label", "y_test.csv"))

    print(f"Loaded {X.shape[0]} samples, each with shape {X.shape[1:]}")

    # ----------------------------------------------------------------
    # PART A: Plot ONE sample - a single CSI subcarrier over time.
    # This is literally the "Raw CSI" box from Figure 5.1 in our proposal.
    # ----------------------------------------------------------------
    sample_idx = 0
    subcarrier = 0  # just look at the first of the 90 CSI values

    signal = X[sample_idx, :, subcarrier]  # shape: (250,) -> one time-series

    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title(f"Raw CSI signal - one subcarrier over time\n"
              f"(sample #{sample_idx}, activity = {ACTIVITY_NAMES[int(y[sample_idx])]})")
    plt.xlabel("Time step (0 to 250)")
    plt.ylabel("CSI amplitude")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_1_single_signal.png", dpi=150)
    print("Saved: plot_1_single_signal.png")
    plt.close()

    # ----------------------------------------------------------------
    # PART B: Compare 2 DIFFERENT activities side by side.
    # This is the whole point of Section 4 - different activities
    # should visibly change the signal pattern.
    # ----------------------------------------------------------------
    # Find one sample of "walk" (label 2) and one of "sit down" (label 5)
    walk_idx = np.where(y == 2)[0][0]
    sit_idx = np.where(y == 5)[0][0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(X[walk_idx, :, subcarrier], color="tab:orange")
    axes[0].set_title(f"WALK (sample #{walk_idx})")
    axes[0].set_ylabel("CSI amplitude")
    axes[0].grid(alpha=0.3)

    axes[1].plot(X[sit_idx, :, subcarrier], color="tab:blue")
    axes[1].set_title(f"SIT DOWN (sample #{sit_idx})")
    axes[1].set_xlabel("Time step (0 to 250)")
    axes[1].set_ylabel("CSI amplitude")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot_2_walk_vs_sit.png", dpi=150)
    print("Saved: plot_2_walk_vs_sit.png")
    plt.close()

    # ----------------------------------------------------------------
    # PART C: Average signal "energy" per class - a quick numeric check
    # that different activities really do look different, without
    # needing any AI model yet.
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Average signal variability per activity (std deviation):")
    print("(Higher number = more movement / signal change)")
    print("=" * 60)
    for label, name in ACTIVITY_NAMES.items():
        idxs = np.where(y == label)[0]
        if len(idxs) == 0:
            continue
        # standard deviation across time, averaged over all samples of this class
        variability = X[idxs, :, :].std(axis=1).mean()
        print(f"  {name:10s} (n={len(idxs):3d}): {variability:.3f}")

    print("\nOpen the two saved PNG files and compare them visually.")
    print("Then look at the variability numbers above - do 'run' and 'fall'")
    print("have higher variability than 'lie down' or 'sit down'? That's the")
    print("kind of pattern our AI model will later learn to detect automatically.")

if __name__ == "__main__":
    main()
