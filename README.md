# WiSense — Privacy-Preserving Human Activity & Safety Monitoring Using Wi-Fi Signals

A major project investigating whether Wi-Fi Channel State Information (CSI) can be
used to infer human presence, activity, and safety-relevant events indoors, as a
privacy-preserving alternative to cameras and wearables.

## Project Status

- **Feasibility Study**: Complete — laptop's Intel 7265 AC Wi-Fi chipset does not
  support CSI extraction, so this project uses the public UT-HAR CSI dataset instead
  of original data collection.
- **ML Models**: Random Forest baseline achieved 93.6% test accuracy; an LSTM model
  was also trained and compared (see Final Report for full analysis, including why
  the LSTM underperformed).
- **Safety Module**: Rule-based fall/inactivity detection, tested and verified.
- **Dashboard**: Live Streamlit prototype integrating the model and safety logic.

Full details, methodology, and results are in `WiSense_Final_Report.docx`.

## ⚠️ Important: Dataset Not Included

The `Data/` folder is intentionally **excluded** from this repository (see
`.gitignore`) because the dataset files are too large for GitHub (100MB+ each).

**To run this project, download the dataset yourself:**

1. Download the `UT_HAR` folder from:
   https://drive.google.com/drive/folders/1R0R8SlVbLI1iUFQCzh_mH90H_4CW2iwt?usp=sharing
2. Place it so the folder structure looks like this:
   ```
   wisense-project/
   └── Data/
       └── UT_HAR/
           ├── data/
           │   ├── X_train.csv
           │   ├── X_test.csv
           │   └── X_val.csv
           └── label/
               ├── y_train.csv
               ├── y_test.csv
               └── y_val.csv
   ```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas matplotlib scikit-learn tensorflow-cpu streamlit joblib
```

## Files

| File | Purpose |
|---|---|
| `inspect_data.py` | Loads and inspects raw UT-HAR CSI data shape |
| `plot_csi.py` | Visualizes raw CSI signals and compares activities |
| `train_baseline_model.py` | Trains and evaluates the Random Forest baseline |
| `train_lstm_model.py` | Trains and evaluates the LSTM model |
| `save_model.py` | Trains and saves the final model for the dashboard |
| `safety_module.py` | Rule-based fall/abnormal-event detection logic |
| `app.py` | Live Streamlit dashboard integrating model + safety engine |

## Running the Dashboard

```bash
python3 save_model.py   # trains and saves wisense_model.pkl (run once)
streamlit run app.py
```

## Dataset Citation

UT-HAR dataset, used via the SenseFi WiFi-CSI-Sensing-Benchmark public repository:

> Yang, J., Chen, X., Zou, H., Wang, D., Xu, Q., and Xie, L. (2023). SenseFi: A
> library and benchmark on deep-learning-empowered WiFi human sensing. *Patterns*,
> Cell Press.
