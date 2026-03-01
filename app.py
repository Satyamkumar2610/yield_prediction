"""
🌾 Crop Yield Prediction
========================
Accept crop, soil, and weather data → preprocess → predict → display.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Page Config ──
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    #MainMenu, footer, header { visibility: hidden; }

    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1a1a2e 50%, #16213e 100%);
    }

    .hero {
        background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(59,130,246,0.08));
        border: 1px solid rgba(16,185,129,0.25);
        border-radius: 20px;
        padding: 36px;
        text-align: center;
        margin-bottom: 28px;
    }
    .hero h1 {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, #10B981, #3B82F6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero p { color: #94A3B8; font-size: 1rem; line-height: 1.6; }

    .result-box {
        background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05));
        border: 2px solid #10B981;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin-top: 24px;
    }
    .result-box .val {
        font-size: 3rem; font-weight: 800; color: #10B981;
    }
    .result-box .lbl {
        color: #94A3B8; font-size: 1rem; margin-top: 4px;
    }
    .result-box .cat {
        display: inline-block; padding: 6px 22px;
        border-radius: 999px; font-weight: 700;
        font-size: 0.9rem; margin-top: 14px;
    }
    .cat-high   { background: rgba(16,185,129,0.2);  color: #10B981; border: 1px solid #10B981; }
    .cat-medium { background: rgba(245,158,11,0.2);  color: #F59E0B; border: 1px solid #F59E0B; }
    .cat-low    { background: rgba(239,68,68,0.2);   color: #EF4444; border: 1px solid #EF4444; }

    .divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #10B981, transparent);
        border: none; margin: 28px 0; border-radius: 2px;
    }

    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #F1F5F9 !important;
        border-radius: 10px !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #10B981, #059669) !important;
        color: white !important; border: none !important;
        border-radius: 12px !important;
        padding: 14px 32px !important;
        font-weight: 700 !important; font-size: 1.05rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(16,185,129,0.3) !important;
    }

    .step-box {
        background: rgba(30,41,59,0.8);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 18px 22px;
        margin-top: 16px;
    }
    .step-box h4 { color: #10B981; margin-bottom: 8px; font-size: 1rem; }
    .step-box p  { color: #CBD5E1; font-size: 0.88rem; line-height: 1.6; margin: 0; }
    .step-box code { color: #F59E0B; }
</style>
""", unsafe_allow_html=True)


# ── Load Model Artifacts ──
@st.cache_resource
def load_artifacts():
    meta_path = os.path.join(MODELS_DIR, "pipeline_metadata.json")
    if not os.path.exists(meta_path):
        return None, None, None, None
    model = joblib.load(os.path.join(MODELS_DIR, "crop_yield_rf_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    with open(meta_path) as f:
        meta = json.load(f)
    return model, scaler, encoders, meta


model, scaler, label_encoders, meta = load_artifacts()

if model is None:
    st.error("⚠️ Model artifacts not found. Run `python3 ml_pipeline.py` first.")
    st.stop()


# ── Yield Category Helper ──
def yield_category(value: float):
    if value >= 50000:
        return "High Yield", "cat-high"
    elif value >= 20000:
        return "Medium Yield", "cat-medium"
    else:
        return "Low Yield", "cat-low"


# ══════════════════════════════════════════
# HERO
# ══════════════════════════════════════════
st.markdown("""
<div class="hero">
    <h1>🌾 Crop Yield Predictor</h1>
    <p>Enter crop, soil &amp; weather data → preprocessing → yield prediction</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
# 1. ACCEPT INPUT  (Crop, Soil, Weather)
# ══════════════════════════════════════════
st.markdown("### 📥 Input — Crop, Soil & Weather Data")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**🌱 Crop**")
    area = st.selectbox("Country / Region", meta["unique_areas"])
    item = st.selectbox("Crop Type", meta["unique_items"])

with c2:
    st.markdown("**🌦️ Weather**")
    rainfall = st.number_input("Avg Rainfall (mm/yr)", min_value=0.0, value=1100.0, step=10.0)
    temp = st.number_input("Avg Temperature (°C)", min_value=-10.0, max_value=50.0, value=22.0, step=0.1)

with c3:
    st.markdown("**🧪 Soil / Agricultural**")
    pesticides = st.number_input("Pesticides (tonnes)", min_value=0.0, value=500.0, step=10.0)
    year = st.number_input("Year", min_value=1990, max_value=2050, value=2020, step=1)

st.markdown("")
predict_clicked = st.button("🚀  Predict Yield", use_container_width=True)


# ══════════════════════════════════════════
# 2–4. PREPROCESS → PREDICT → DISPLAY
# ══════════════════════════════════════════
if predict_clicked:

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── STEP 2: Preprocessing (Encoding + Scaling) ──
    # Encode categorical inputs
    area_encoded = label_encoders["Area"].transform([area])[0]
    item_encoded = label_encoders["Item"].transform([item])[0]

    # Build feature dataframe
    input_df = pd.DataFrame({
        "Area": [area_encoded],
        "Item": [item_encoded],
        "Year": [year],
        "average_rain_fall_mm_per_year": [rainfall],
        "pesticides_tonnes": [pesticides],
        "avg_temp": [temp],
    })

    # Scale numerical features
    num_cols = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "Year"]
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Show preprocessing steps to user
    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f"""
        <div class="step-box">
            <h4>🏷️ Step 1 — Label Encoding</h4>
            <p>
                <code>{area}</code> → <code>{area_encoded}</code><br>
                <code>{item}</code> → <code>{item_encoded}</code>
            </p>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown(f"""
        <div class="step-box">
            <h4>📏 Step 2 — Feature Scaling (StandardScaler)</h4>
            <p>
                Rainfall, Pesticides, Temperature, Year<br>
                scaled to zero-mean, unit-variance.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── STEP 3: Predict ──
    predicted_yield = model.predict(input_df)[0]
    cat_label, cat_class = yield_category(predicted_yield)

    # ── STEP 4: Display Result ──
    st.markdown(f"""
    <div class="result-box">
        <div class="lbl">Predicted Crop Yield</div>
        <div class="val">{predicted_yield:,.0f} <span style="font-size:1.2rem; color:#64748B;">hg/ha</span></div>
        <div class="cat {cat_class}">{cat_label}</div>
    </div>
    """, unsafe_allow_html=True)
