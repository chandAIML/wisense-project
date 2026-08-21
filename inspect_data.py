# inspect_data.py
#
# GOAL: Load the UT-HAR Wi-Fi CSI dataset and understand its shape.
# This is the FIRST hands-on step of the WiSense project (Phase 1 - Feasibility Study).
#
# WHAT IS THIS DATA?
#   It is Wi-Fi Channel State Information (CSI) collected by a university lab.
#   Every "sample" is a short recording of Wi-Fi signal changes while a person
#   performed one activity: lie down, fall, walk, pick up, run, sit down, or stand up.
#
# Run this with:  python inspect_data.py

import numpy as np
import glob
import os

DATA_DIR = "Data/UT_HAR"

def load_csi_file(path):
    """These files are named .csv but are actually raw NumPy arrays.
    np.load() reads them directly - do NOT use pandas.read_csv() here."""
    with open(path, "rb") as f:
        return np.load(f)

def main():
    data_files = sorted(glob.glob(os.path.join(DATA_DIR, "data", "*.csv")))
    label_files = sorted(glob.glob(os.path.join(DATA_DIR, "label", "*.csv")))

    print("=" * 60)
    print("Found data files: ", [os.path.basename(f) for f in data_files])
    print("Found label files:", [os.path.basename(f) for f in label_files])
    print("=" * 60)

    # Load just the first data file and its matching label file
    if not data_files:
        print("No data files found. Check that Data/UT_HAR/data/ has files in it.")
        return

    sample_data = load_csi_file(data_files[0])
    print(f"\nLoaded: {os.path.basename(data_files[0])}")
    print(f"Raw shape: {sample_data.shape}")
    print("  -> This means:", sample_data.shape[0], "separate CSI recordings (samples)")

    # Reshape exactly like the official SenseFi benchmark code does:
    # each sample becomes (1, 250, 90)
    #   1   -> single "channel" (like a grayscale image has 1 color channel)
    #   250 -> 250 time steps (this IS the time-series part we discussed in Section 5!)
    #   90  -> 90 numbers per time step (30 subcarriers x 3 antennas)
    reshaped = sample_data.reshape(len(sample_data), 1, 250, 90)
    print(f"Reshaped to: {reshaped.shape}  (samples, channel, time_steps, csi_features)")

    print("\nFirst sample, first 5 time steps, first 8 CSI values:")
    print(reshaped[0, 0, :5, :8])

    if label_files:
        sample_labels = load_csi_file(label_files[0])
        print(f"\nLoaded: {os.path.basename(label_files[0])}")
        print(f"Label shape: {sample_labels.shape}")
        print("First 10 labels:", sample_labels[:10])
        print("\nActivity classes are 0-6, meaning:")
        print("  0=lie down, 1=fall, 2=walk, 3=pickup, 4=run, 5=sit down, 6=stand up")

    print("\n" + "=" * 60)
    print("WHAT TO NOTICE (write these answers down for the team):")
    print("1. How many time steps are in one sample? (should be 250)")
    print("2. How many CSI values per time step? (should be 90)")
    print("3. Do consecutive time steps look similar or very different? (scroll up)")
    print("=" * 60)

if __name__ == "__main__":
    main()
