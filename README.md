# 🌾 Crop Yield Prediction

A machine learning pipeline for forecasting agricultural crop yields based on historical weather patterns, pesticide usage, and regional crop records.

> **Authors:** Daksh Yadav · Satyam Kumar · Adarsh · Himank  
> **Institution:** B.Tech — System Design Project (2026)

---

## Project Structure

**`raw_data/`**  
Contains the original datasets collected from different sources — weather data, pesticide usage, historical crop yield records, etc.

**`cleandata/`**  
Includes the processed and merged datasets generated after running the cleaning and preprocessing pipeline.

**`crop_yield_prediction.ipynb`**  
The main notebook where everything happens:
- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature preparation
- Training baseline models (Decision Tree & Random Forest)
- Model evaluation

**`Model Artifacts` (`*_model.pkl`, `scaler.pkl`, `label_encoders.pkl`)**  
These are the saved models and preprocessing objects. They're exported so they can be loaded directly into a future web app without retraining.

---

## Data Pipeline Overview

The raw datasets weren't perfectly aligned, so some preprocessing was necessary before training:

- Standardized column names and aligned schemas across datasets
- Removed irrelevant administrative rows and noisy columns
- Handled missing numerical values using median imputation
- Encoded categorical variables (`Area` and `Item`) using label encoding (kept simple since tree-based models don't require one-hot encoding)

The focus here was simplicity and reproducibility — nothing overly complex at this stage.

---

## Getting Started

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

```python
import pickle
import numpy as np

# 1. Load encoders and encode categorical inputs
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

area_encoded = label_encoders["Area"].transform(["India"])
item_encoded = label_encoders["Item"].transform(["Wheat"])

# 2. Load scaler and normalize numerical features
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

scaled_features = scaler.transform([[avg_temp, rainfall, pesticide_usage]])

# 3. Load model and predict
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

features = np.concatenate([area_encoded, item_encoded, scaled_features[0]])
prediction = model.predict([features])
print(f"Predicted Yield: {prediction[0]:.2f} hg/ha")
```

---

## Model Performance

A Random Forest Regressor was used as the primary baseline model. It performed surprisingly well straight away:

- **R² ≈ 0.98**

A Decision Tree model was also trained for comparison and showed strong performance, though Random Forest generalized better overall.

> ⚠️ **Note:** This result is based on a single train-test split. Cross-validation is planned to confirm how well the model generalizes to unseen data.

---

## Next Steps

- [ ] Hyperparameter tuning
- [ ] Cross-validation for more robust evaluation
- [ ] Feature importance analysis
- [ ] Model explainability (SHAP or similar tools)
- [ ] Integration into a web interface

---

## 👥 Team

| Name | Enrollment No. |
|------|---------------|
| Satyam Kumar | 2401010429 |
| Daksh Yadav | 2401010139 |
| Adarsh | 2401010026 |
| Himank | 2401010187 |
