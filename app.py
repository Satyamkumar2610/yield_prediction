import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    /* Reset & base */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        color: #1e293b;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .stApp { background: #f1f5f2; }
    /* ── Navbar ── */
    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #ffffff;
        padding: 16px 36px;
        border-bottom: 1px solid #e2e8f0;
        margin: -1rem -1rem 0 -1rem;
    }
    .navbar .brand {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .navbar .brand-icon {
        font-size: 1.5rem;
    }
    .navbar .brand-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #166534;
    }
    .navbar .nav-tag {
        font-size: 0.72rem;
        font-weight: 600;
        color: #166534;
        background: #dcfce7;
        padding: 3px 10px;
        border-radius: 12px;
        letter-spacing: 0.3px;
    }
    /* ── Page container ── */
    .page-wrap {
        max-width: 1100px;
        margin: 0 auto;
        padding: 28px 0 40px 0;
    }
    /* ── Page title ── */
    .page-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .page-desc {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 24px;
        line-height: 1.5;
    }
    /* ── Card ── */
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 22px 24px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .card-label {
        font-size: 1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #334155;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-label .dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-green  { background: #22c55e; }
    .dot-blue   { background: #3b82f6; }
    .dot-amber  { background: #f59e0b; }
    /* ── Horizontal Rule ── */
    .hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 28px 0;
    }
    /* ── Preprocessing chips ── */
    .prep-row {
        display: flex;
        gap: 16px;
        margin-bottom: 0;
    }
    .prep-chip {
        flex: 1;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .prep-chip .chip-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #166534;
        margin-bottom: 10px;
    }
    .prep-chip .chip-body {
        font-size: 0.85rem;
        color: #334155;
        line-height: 1.65;
    }
    .prep-chip code {
        background: #f0fdf4;
        color: #166534;
        padding: 1px 6px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-family: 'SF Mono', 'Fira Code', monospace;
    }
    /* ── Result Section ── */
    .result-section {
        text-align: center;
    }
    .result-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 40px 32px 36px 32px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        display: inline-block;
        min-width: 380px;
    }
    .result-box .r-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748b;
        margin-bottom: 10px;
    }
    .result-box .r-value {
        font-size: 2.8rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.1;
    }
    .result-box .r-unit {
        font-size: 1rem;
        font-weight: 400;
        color: #94a3b8;
    }
    .result-box .r-badge {
        display: inline-block;
        padding: 5px 16px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        margin-top: 14px;
    }
    .b-high   { background: #dcfce7; color: #166534; }
    .b-medium { background: #fef9c3; color: #854d0e; }
    .b-low    { background: #fee2e2; color: #991b1b; }
    /* ── Summary row ── */
    .summary-row {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 20px;
        flex-wrap: wrap;
    }
    .summary-item {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 24px;
        text-align: center;
        min-width: 160px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .summary-item .s-label {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .summary-item .s-value {
        font-size: 1rem;
        font-weight: 600;
        color: #334155;
    }
    /* ── Button override ── */
    .stButton > button {
        background: #166534 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.2px !important;
        transition: background 0.15s !important;
    }
    .stButton > button:hover {
        background: #15803d !important;
    }
    /* ── Input labels ── */
    .stSelectbox label, .stNumberInput label {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: #475569 !important;
    }
    /* ── Section heading ── */
    .sec-heading {
        font-size: 1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #475569;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_artifacts():
    meta_path = os.path.join(MODELS_DIR, "pipeline_metadata.json")
    if not os.path.exists(meta_path):
        return None, None, None, None
    rf_path = os.path.join(MODELS_DIR, "crop_yield_rf_model.pkl")
    dt_path = os.path.join(MODELS_DIR, "crop_yield_dt_model.pkl")
    if os.path.exists(rf_path):
        model = joblib.load(rf_path)
    else:
        model = joblib.load(dt_path)
    sc = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    with open(meta_path) as f:
        meta = json.load(f)
    return model, sc, le, meta
model, scaler, label_encoders, meta = load_artifacts()
if model is None:
    st.error("Model artifacts not found. Run `python3 ml_pipeline.py` first to train models.")
    st.stop()
def yield_category(v):
    if v >= 50000:
        return "HIGH YIELD", "b-high"
    elif v >= 20000:
        return "MEDIUM YIELD", "b-medium"
    else:
        return "LOW YIELD", "b-low"
st.markdown("""
<div class="navbar">
    <div class="brand">
        <span class="brand-icon">🌾</span>
        <span class="brand-name">CropYield Predictor</span>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("")
st.markdown('<div class="page-title">Predict Crop Yield</div>', unsafe_allow_html=True)
st.markdown('<div class="page-desc">Provide crop, weather, and agricultural details below. The system will encode categorical inputs, scale numerical features, and predict yield using a Random Forest model.</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-heading">Input Parameters</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    st.markdown("""
    <div class="card">
        <div class="card-label"><span class="dot dot-green"></span> CROP DETAILS</div>
    </div>
    """, unsafe_allow_html=True)
    area = st.selectbox("Country / Region", meta["unique_areas"])
    item = st.selectbox("Crop Type", meta["unique_items"])
with c2:
    st.markdown("""
    <div class="card">
        <div class="card-label"><span class="dot dot-blue"></span> WEATHER DATA</div>
    </div>
    """, unsafe_allow_html=True)
    rainfall = st.number_input("Average Rainfall (mm/year)", min_value=0.0, value=1100.0, step=10.0)
    temp = st.number_input("Average Temperature (°C)", min_value=-10.0, max_value=50.0, value=22.0, step=0.1)
with c3:
    st.markdown("""
    <div class="card">
        <div class="card-label"><span class="dot dot-amber"></span> AGRICULTURAL INPUTS</div>
    </div>
    """, unsafe_allow_html=True)
    pesticides = st.number_input("Pesticides Used (tonnes)", min_value=0.0, value=500.0, step=10.0)
    year = st.number_input("Year", min_value=1990, max_value=2050, value=2020, step=1)
st.markdown("")
predict_clicked = st.button("Run Prediction", use_container_width=True)
if predict_clicked:
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    area_encoded = label_encoders["Area"].transform([area])[0]
    item_encoded = label_encoders["Item"].transform([item])[0]
    input_df = pd.DataFrame({
        "Area": [area_encoded],
        "Item": [item_encoded],
        "Year": [year],
        "average_rain_fall_mm_per_year": [rainfall],
        "pesticides_tonnes": [pesticides],
        "avg_temp": [temp],
    })
    num_cols = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "Year"]
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    st.markdown('<div class="sec-heading">Preprocessing Applied</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="prep-row">
        <div class="prep-chip">
            <div class="chip-title">1 — Label Encoding</div>
            <div class="chip-body">
                Area: <code>{area}</code> → <code>{area_encoded}</code><br>
                Crop: <code>{item}</code> → <code>{item_encoded}</code>
            </div>
        </div>
        <div class="prep-chip">
            <div class="chip-title">2 — Standard Scaling</div>
            <div class="chip-body">
                Rainfall, Pesticides, Temperature, and Year normalised to zero mean and unit variance using a fitted StandardScaler.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    predicted_yield = model.predict(input_df)[0]
    cat_label, cat_class = yield_category(predicted_yield)
    st.markdown('<div class="sec-heading">Prediction Output</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-section">
        <div class="result-box">
            <div class="r-label">Predicted Crop Yield</div>
            <div class="r-value">{predicted_yield:,.0f} <span class="r-unit">hg/ha</span></div>
            <div><span class="r-badge {cat_class}">{cat_label}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="summary-row">
        <div class="summary-item">
            <div class="s-label">Region</div>
            <div class="s-value">{area}</div>
        </div>
        <div class="summary-item">
            <div class="s-label">Crop</div>
            <div class="s-value">{item}</div>
        </div>
        <div class="summary-item">
            <div class="s-label">Year</div>
            <div class="s-value">{year}</div>
        </div>
        <div class="summary-item">
            <div class="s-label">Rainfall</div>
            <div class="s-value">{rainfall:,.0f} mm/yr</div>
        </div>
        <div class="summary-item">
            <div class="s-label">Pesticides</div>
            <div class="s-value">{pesticides:,.0f} t</div>
        </div>
        <div class="summary-item">
            <div class="s-label">Avg Temp</div>
            <div class="s-value">{temp}°C</div>
        </div>
    </div>
    """, unsafe_allow_html=True)