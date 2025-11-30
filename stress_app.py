import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# ---------------------------------------------------
# CONFIG – CLOUD-SAFE PATHS
# ---------------------------------------------------

BASE_DIR = Path(__file__).parent  # <-- FIXED (file_ correct)
DATA_PATH = BASE_DIR / "Stress_Dataset.csv"
MODEL_PATH = BASE_DIR / "best_stress_model.pkl"

TARGET_COL = "Which type of stress do you primarily experience?"

st.set_page_config(
    page_title="PsyTrack – Student Stress Analyzer",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------
# DARK THEME CSS
# ---------------------------------------------------
CUSTOM_CSS = """
<style>
    .stApp {
        background: radial-gradient(circle at top left, #15161f 0%, #05060a 45%, #010109 100%);
        font-family: "Segoe UI", system-ui, sans-serif;
        color: #e6f1ff;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .hero-title {
        font-size: 26px !important;
        font-weight: 800;
        margin: 28px 0 6px 0 !important;
        color: #f8fafc;
        animation: fadeInUp 0.4s ease-out;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .hero-subtitle {
        font-size: 14px;
        color: #9ca3af;
        margin-bottom: 16px;
    }

    .block-container {
        padding-top: 2.4rem !important;
    }

    .card {
        background: rgba(17, 24, 39, 0.95);
        border-radius: 18px;
        padding: 18px;
        border: 1px solid rgba(55, 65, 81, 0.9);
        box-shadow: 0 18px 38px rgba(0,0,0,0.55);
    }

    .section-title {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
        color: #e5e7eb;
    }

    .metric-card {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        border-radius: 16px;
        padding: 14px;
        color: #022c22 !important;
        font-weight: 600;
    }

    .metric-label {
        font-size: 10px;
        text-transform: uppercase;
        opacity: 0.86;
        letter-spacing: 0.12em;
    }

    .metric-value {
        font-size: 20px;
        margin-top: 4px;
    }

    label, .stMarkdown, .stText {
        color: #e5e7eb !important;
    }

    .stSlider label, .stSlider span {
        color: #e5e7eb !important;
    }

    .stNumberInput input, .stTextInput input {
        background: #020617 !important;
        color: #e5e7eb !important;
        border-radius: 10px;
        border: 1px solid #4b5563;
    }

    .stSelectbox div[data-baseweb=select] > div {
        background: #020617 !important;
        color: #e5e7eb !important;
        border-radius: 10px;
        border: 1px solid #4b5563;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL + DATA
# ---------------------------------------------------
@st.cache_resource
def load_model_and_data():

    if not DATA_PATH.exists():
        st.error("❌ ERROR: Stress_Dataset.csv missing in repo.")
        st.stop()

    if not MODEL_PATH.exists():
        st.error("❌ ERROR: best_stress_model.pkl missing in repo.")
        st.stop()

    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        st.error(f"❌ Target column '{TARGET_COL}' not found in dataset.")
        st.stop()

    X = df.drop(columns=[TARGET_COL])
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    stats = X[numeric_cols].describe()

    model = joblib.load(MODEL_PATH)

    return model, df, numeric_cols, stats


model, full_df, feature_cols, stats = load_model_and_data()


def feature_range(col):
    mn = float(stats.loc["min", col])
    mx = float(stats.loc["max", col])
    mean_val = float(stats.loc["mean", col])
    return mn, mx, mean_val


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 PsyTrack Dashboard")
    st.write("AI-based Student Stress Analyzer")
    st.write("---")

    st.write("*Feature Ranges (dataset-based):*")
    for col in feature_cols:
        mn, mx, _ = feature_range(col)
        st.write(f"- *{col}* → {mn:.1f} - {mx:.1f}")

    st.caption("Created by Suraj • AIML Mini Project")


# ---------------------------------------------------
# HERO TITLE
# ---------------------------------------------------
st.markdown('<div class="hero-title">PsyTrack – Student Stress Analyzer</div>',
            unsafe_allow_html=True)

st.markdown(
    '<div class="hero-subtitle">Adjust responses below and click Analyze</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------------------------------
# MAIN PAGE LAYOUT
# ---------------------------------------------------
left, right = st.columns([2.2, 1])

# ---------------- LEFT SIDE ------------------------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Questionnaire Responses</div>',
                unsafe_allow_html=True)

    colA, colB = st.columns(2)
    inputs = {}

    for i, col in enumerate(feature_cols):
        mn, mx, mean_val = feature_range(col)

        # choose slider type
        is_int = mn.is_integer() and mx.is_integer() and (mx - mn <= 10)

        if is_int:
            step = 1.0
            default = int(round(mean_val))
        else:
            step = max((mx - mn) / 40, 0.5)
            default = float(mean_val)

        box = colA if i % 2 == 0 else colB
        with box:
            val = st.slider(
                col,
                min_value=float(mn),
                max_value=float(mx),
                value=default,
                step=step,
            )
            inputs[col] = int(val) if is_int else float(val)

    analyze = st.button("Analyze Stress Type", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- RIGHT SIDE ------------------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analysis Result</div>',
                unsafe_allow_html=True)

    if analyze:
        df_input = pd.DataFrame([inputs], columns=feature_cols)
        prediction = model.predict(df_input)[0]

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Stress Type</div>
                <div class="metric-value">{prediction}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df_input)[0]

            prob_df = pd.DataFrame({
                "Stress Type": model.classes_,
                "Probability": np.round(proba, 3)
            }).sort_values("Probability", ascending=False)

            st.markdown("#### Prediction Confidence")
            st.table(prob_df.reset_index(drop=True))

        st.markdown("#### Interpretation")
        st.write(
            "The prediction represents the most likely stress type experienced by the student."
        )
    else:
        st.write("Adjust values on the left and click *Analyze Stress Type*.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("PsyTrack – AI Student Stress Analyzer • Model A")