# 🌾 Crop Yield Prediction

> A machine learning pipeline for agricultural productivity forecasting using ensemble learning methods.

**Authors:** Satyam Kumar · Daksh Yadav · Adarsh · Himank  
**Repository:** [github.com/Satyamkumar2610/yield_prediction](https://github.com/Satyamkumar2610/yield_prediction)

---

## 📌 Overview

Agriculture is the backbone of food security for billions of people — yet accurately predicting crop yields before or during a growing season remains a persistent challenge. Yield outcomes depend on a complex interplay of variables: regional climate, rainfall, temperature fluctuations, pesticide usage, and crop variety, many of which interact in non-linear ways.

This project applies **machine learning** to a curated multi-source dataset to demonstrate that high-accuracy yield prediction is achievable with well-engineered data pipelines and classical ensemble methods. The pipeline covers everything from raw data ingestion to model serialization, and is designed to plug directly into a future web-based prediction interface.

---

## 🗂️ Project Structure

```
yield_prediction/
│
├── raw_data/                       # Original datasets: weather, pesticide usage, yield records
├── cleandata/                      # Processed & merged datasets from the preprocessing pipeline
│
├── crop_yield_prediction.ipynb     # Main notebook: EDA, preprocessing, training, evaluation
│
├── rf_model.pkl                    # Trained Random Forest Regressor
├── dt_model.pkl                    # Trained Decision Tree Regressor
├── scaler.pkl                      # Fitted StandardScaler for numerical normalization
└── label_encoders.pkl              # Fitted LabelEncoders for Area and Item features
```

---

## 📊 Data Pipeline

The raw datasets originated from multiple sources and weren't natively compatible in schema or format. A structured preprocessing pipeline was built to standardize and merge them into a single clean training corpus.

**Data Sources**
- Historical crop yield records (quantity per area, by crop type and region)
- Weather / climate data (temperature, rainfall, seasonal indicators)
- Pesticide usage data (application rates by crop and region)

**Preprocessing Steps**
- Standardized column names and aligned schemas across all raw datasets
- Removed irrelevant administrative rows and noisy/redundant columns
- Handled missing numerical values using **median imputation**
- Encoded categorical variables (`Area`, `Item`) using **Label Encoding** — kept simple since tree-based models don't require one-hot encoding

The focus was on simplicity and reproducibility. Nothing overly complex at this stage — just a clean, merged dataset that can serve as a reliable baseline and is easy to audit or extend.

---

## 🔍 Exploratory Data Analysis

EDA was conducted inside the main notebook and covered:

- Distribution analysis of numerical features (yield quantities, weather variables, pesticide usage)
- Correlation heatmaps to identify multicollinearity and feature relevance
- Group-level aggregations by crop type (`Item`) and region (`Area`)
- Outlier detection and treatment in yield and weather variables
- Integrity validation of the merged dataset post-preprocessing

---

## 🤖 Models

Two models were trained and evaluated as part of this project:

| Model | R² Score | Strengths | Limitations |
|-------|----------|-----------|-------------|
| **Random Forest Regressor** | **~0.98** | High accuracy, handles non-linearity, robust to overfitting | Less interpretable, higher compute cost |
| Decision Tree Regressor | < 0.98 | Highly interpretable, fast to train | Prone to overfitting, lower generalization |

**Random Forest** was selected as the primary model due to its superior ability to generalize across diverse feature interactions and its inherent resistance to overfitting through ensemble averaging.

> ⚠️ **Important:** The R² of 0.98 is based on a single train-test split without cross-validation. This result may be optimistic. Rigorous k-fold cross-validation is planned before drawing conclusions about real-world generalization.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Run the Notebook

```bash
git clone https://github.com/Satyamkumar2610/yield_prediction.git
cd yield_prediction
jupyter notebook crop_yield_prediction.ipynb
```

### Making Predictions with Saved Artifacts

The inference workflow mirrors the exact preprocessing applied at training time — ensuring prediction consistency.

```python
import pickle
import numpy as np

# Step 1 — Encode categorical inputs
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

area_encoded = label_encoders["Area"].transform(["India"])
item_encoded = label_encoders["Item"].transform(["Wheat"])

# Step 2 — Normalize numerical features
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

numerical_features = [[avg_temp, rainfall, pesticide_usage]]
scaled_features = scaler.transform(numerical_features)

# Step 3 — Load model and predict
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

features = np.concatenate([area_encoded, item_encoded, scaled_features[0]])
prediction = model.predict([features])
print(f"Predicted Yield: {prediction[0]:.2f} hg/ha")
```

---

## 🔭 Next Steps

- [ ] Implement k-fold cross-validation (k=5 or k=10) for robust performance estimates
- [ ] Hyperparameter tuning with `GridSearchCV` / `RandomizedSearchCV`
- [ ] Feature importance analysis to identify the most influential predictors
- [ ] SHAP (SHapley Additive exPlanations) integration for model explainability
- [ ] Web interface using Flask/FastAPI + React or Streamlit, powered by the serialized `.pkl` artifacts
- [ ] Explore Gradient Boosting models (XGBoost, LightGBM) for further accuracy improvements
- [ ] Expand dataset with more recent years and additional regions to improve generalization

---

## 💡 Why This Matters

- **Food Security** — Better forecasts help governments and NGOs anticipate shortfalls and pre-position aid before crises occur.
- **Farmer Empowerment** — Smallholder farmers benefit from advance yield knowledge for seed investment, irrigation planning, and market timing decisions.
- **Supply Chain Efficiency** — Accurate forecasts reduce post-harvest losses and inventory mismatches across the food supply chain.
- **Climate Adaptation** — As climate change introduces greater variability, data-driven models become an essential tool for adaptive agricultural policy.
- **Scalability** — Unlike expert surveys, a trained ML model can generate predictions for hundreds of crop-region combinations in seconds.

---

## 👥 Team

| Name | Enrollment No. |
|------|---------------|
| Satyam Kumar | 2401010429 |
| Daksh Yadav | 2401010139 |
| Adarsh | 2401010026 |
| Himank | 2401010187 |

---

## 📄 License

This project is for academic purposes only.
