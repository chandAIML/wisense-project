# app.py (v2)
#
# GOAL: Combine the dashboard (Part 5) and the Safety Engine (Part 7)
# into ONE live, stateful app. Instead of analyzing a single reading in
# isolation, this app now remembers what happened in PREVIOUS readings
# within a session - exactly like a real deployed WiSense system would.
#
# KEY NEW IDEA: st.session_state
#   Streamlit normally re-runs your whole script top-to-bottom on every
#   click. st.session_state is the one thing that SURVIVES between runs -
#   it's how we keep the SafetyEngine's memory (e.g. "we're currently
#   watching for a possible fall's aftermath") alive across button clicks.
#
# Run this with:  streamlit run app.py

import streamlit as st
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from safety_module import (
    SafetyEngine, extract_features, find_indices_by_label,
    ACTIVITY_NAMES, load_csi_file, DATA_DIR, MODEL_PATH
)

TIMELINE_COLORS = {"normal": "#1B4F72", "possible_fall": "#E8A33D", "alert": "#C0392B"}

# ---------------- WiSense brand palette (matches the original proposal mockup) ----------------
NAVY = "#12283F"
TEAL = "#0E7C86"
ACCENT = "#E8A33D"
LIGHT_TEAL = "#DCEFEF"
LIGHT_BLUE = "#E7EEF5"
GREEN = "#1E8449"
RED = "#C0392B"
GREY = "#5D6D7E"

def inject_custom_css():
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: #FFFFFF;
        }}
        div[data-testid="stAppViewContainer"] > .main {{
            padding-top: 0rem;
        }}
        .wisense-header {{
            background-color: {NAVY};
            padding: 28px 32px 22px 32px;
            border-radius: 0 0 14px 14px;
            margin: -1rem -1rem 1.5rem -1rem;
            border-bottom: 4px solid {ACCENT};
        }}
        .wisense-header h1 {{
            color: #FFFFFF;
            font-size: 34px;
            font-weight: 800;
            margin: 0;
            padding: 0;
        }}
        .wisense-header p {{
            color: {LIGHT_TEAL};
            font-size: 14px;
            margin: 4px 0 0 0;
        }}
        .wisense-card {{
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 12px;
            border: 1px solid #C9D6E3;
        }}
        .wisense-card .label {{
            font-size: 12px;
            color: {GREY};
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            margin: 0 0 4px 0;
        }}
        .wisense-card .value {{
            font-size: 20px;
            font-weight: 800;
            margin: 0;
        }}
        div.stButton > button[kind="primary"] {{
            background-color: {TEAL};
            border-color: {TEAL};
        }}
        div.stButton > button[kind="primary"]:hover {{
            background-color: {NAVY};
            border-color: {NAVY};
        }}
        h2, h3 {{
            color: {NAVY} !important;
        }}
        </style>
    """, unsafe_allow_html=True)

def status_card(label, value, tone="teal"):
    tone_map = {
        "teal": (LIGHT_TEAL, GREEN),
        "blue": (LIGHT_BLUE, NAVY),
        "red": ("#FBE1E1", RED),
        "amber": ("#FCF0DC", "#8A5A00"),
    }
    bg, fg = tone_map.get(tone, tone_map["blue"])
    st.markdown(f"""
        <div class="wisense-card" style="background-color:{bg};">
            <p class="label">{label}</p>
            <p class="value" style="color:{fg};">{value}</p>
        </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_test_data():
    X_test = load_csi_file(os.path.join(DATA_DIR, "data", "X_test.csv"))
    y_test = load_csi_file(os.path.join(DATA_DIR, "label", "y_test.csv")).astype(int)
    return X_test, y_test

def init_session():
    if "engine" not in st.session_state:
        st.session_state.engine = SafetyEngine()
        st.session_state.history_labels = []
        st.session_state.history_events = []
        st.session_state.demo_step = 0
        st.session_state.demo_sequence = []
        st.session_state.last_alert_step = None

def reset_session():
    st.session_state.engine = SafetyEngine()
    st.session_state.history_labels = []
    st.session_state.history_events = []
    st.session_state.demo_step = 0
    st.session_state.demo_sequence = []
    st.session_state.last_alert_step = None

def build_demo_sequence(mode, y_test):
    if mode == "Demo: Fall -> Inactivity (should alert)":
        fall = find_indices_by_label(y_test, 1, 1)
        lie = find_indices_by_label(y_test, 0, 3)
        return fall + lie
    elif mode == "Demo: Fall -> Recovers (false alarm)":
        fall = find_indices_by_label(y_test, 1, 1)
        walk = find_indices_by_label(y_test, 2, 2)
        return fall + walk
    return []

def process_one_reading(sample_idx, X_test, model):
    sample = X_test[sample_idx:sample_idx+1]
    features = extract_features(sample)
    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    confidence = probs[pred]

    step_idx = len(st.session_state.history_labels)
    event = st.session_state.engine.process_reading(step_idx, pred, confidence)

    st.session_state.history_labels.append(ACTIVITY_NAMES[pred])
    st.session_state.history_events.append(event)
    if event == "alert":
        st.session_state.last_alert_step = step_idx

    return pred, confidence, event

def draw_timeline():
    labels = st.session_state.history_labels
    events = st.session_state.history_events
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(9, 2.2))
    for i, (label, event) in enumerate(zip(labels, events)):
        ax.bar(i, 1, color=TIMELINE_COLORS[event], edgecolor="white")
        ax.text(i, 1.05, label, ha="center", va="bottom", fontsize=7, rotation=45)
    ax.set_xlim(-0.5, max(len(labels) - 0.5, 0.5))
    ax.set_ylim(0, 1.6)
    ax.set_yticks([])
    ax.set_xlabel("Reading number in this session")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def main():
    st.set_page_config(page_title="WiSense Dashboard", page_icon="\U0001F4F6", layout="centered")
    inject_custom_css()
    init_session()

    st.markdown(
        '<div class="wisense-header">'
        '<h1>WiSense</h1>'
        '<p>Live Home Monitoring Dashboard (Demo)</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Run `python3 save_model.py` first.")
        return

    model = load_model()
    X_test, y_test = load_test_data()

    st.subheader("Session Controls")
    mode = st.selectbox(
        "Choose how to feed readings into this session:",
        ["Manual (pick any sample)",
         "Demo: Fall -> Inactivity (should alert)",
         "Demo: Fall -> Recovers (false alarm)"],
    )

    col_a, col_b = st.columns([3, 1])
    with col_a:
        if mode == "Manual (pick any sample)":
            sample_idx = st.slider("Sample index to feed as the next reading", 0, len(X_test) - 1, 0)
            advance_clicked = st.button("Feed This Reading \u2192", type="primary", key="feed_manual_btn")
            next_sample_idx = sample_idx if advance_clicked else None
        else:
            if not st.session_state.demo_sequence:
                st.session_state.demo_sequence = build_demo_sequence(mode, y_test)
            seq = st.session_state.demo_sequence
            remaining = len(seq) - st.session_state.demo_step
            advance_clicked = st.button(
                f"Feed Next Reading \u2192 ({remaining} left in this demo)"
                if remaining > 0 else "Demo complete - press Reset to run again",
                type="primary", disabled=(remaining <= 0), key="feed_demo_btn",
            )
            next_sample_idx = None
            if advance_clicked and remaining > 0:
                next_sample_idx = seq[st.session_state.demo_step]
                st.session_state.demo_step += 1
    with col_b:
        if st.button("\u21BA Reset Session", key="reset_btn"):
            reset_session()
            st.rerun()

    if next_sample_idx is not None:
        pred, confidence, event = process_one_reading(next_sample_idx, X_test, model)

    st.markdown("---")

    # ---------------- Live status card (matches Figure 20.1 mockup field-for-field) ----------------
    total_alerts = len(st.session_state.engine.alerts)

    if st.session_state.history_labels:
        latest_label = st.session_state.history_labels[-1]
        latest_event = st.session_state.history_events[-1]
        is_alert_now = (latest_event == "alert")
        was_recent_alert = (st.session_state.last_alert_step is not None)

        home_status = "ALERT" if was_recent_alert else "Normal"
        presence = "Person Detected"
        current_activity = latest_label.title()
        safety = "Possible Abnormal Event" if was_recent_alert else "No Abnormal Event"
        sleep_value = "Not tracked in this demo (Phase 7 - future research)"
        alerts_value = str(total_alerts) if total_alerts else "None"
    else:
        home_status = "Normal"
        presence = "No Reading Yet"
        current_activity = "\u2014"
        safety = "No Abnormal Event"
        sleep_value = "Not tracked in this demo (Phase 7 - future research)"
        alerts_value = "None"

    status_card("Home Status", home_status, tone="red" if home_status == "ALERT" else "teal")
    status_card("Presence", presence, tone="blue")
    status_card("Current Activity", current_activity, tone="blue")
    status_card("Safety", safety, tone="red" if safety != "No Abnormal Event" else "teal")
    status_card("Sleep", sleep_value, tone="amber")
    status_card("Alerts", alerts_value, tone="red" if total_alerts else "teal")

    if st.session_state.history_labels:
        if is_alert_now:
            st.error("\u26A0\uFE0F Possible abnormal event detected. Please check the monitored room.")
        elif latest_event == "possible_fall":
            st.warning("\U0001F440 Possible fall-like movement detected - watching next readings closely.")
        else:
            st.success("\u2705 No abnormal event detected.")
    else:
        st.info("No readings yet - feed a reading to begin monitoring this session.")

    st.markdown("---")
    st.subheader("Session Timeline")
    st.caption("Blue = normal, Orange = possible fall (watching), Red = notification triggered")
    draw_timeline()

    st.markdown("---")
    st.caption(
        "This demo uses the UT-HAR public dataset in place of a live Wi-Fi receiver. "
        "The Safety Engine remembers previous readings within a session (via "
        "st.session_state) to implement the fall -> inactivity -> notification "
        "reasoning chain from Section 10 of the project proposal."
    )

if __name__ == "__main__":
    main()
