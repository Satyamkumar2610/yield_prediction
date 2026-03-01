# Crop Yield Prediction

## Project Structure

**raw_data/**  
Contains the original datasets collected from different sources — weather data, pesticide usage, historical crop yield records, etc.

**cleandata/**  
Includes the processed and merged datasets generated after running the cleaning and preprocessing pipeline.

**crop_yield_prediction.ipynb**  
The main notebook where everything happens:
- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature preparation
- Training baseline models (Decision Tree & Random Forest)
- Model evaluation

**Model Artifacts (*_model.pkl, scaler.pkl, label_encoders.pkl)**  
These are the saved models and preprocessing objects. They’re exported so they can be loaded directly into a future web app without retraining.

## Data Pipeline Overview

The raw datasets weren’t perfectly aligned, so some preprocessing was necessary before training:
- Standardized column names and aligned schemas across datasets
- Removed irrelevant administrative rows and noisy columns
- Handled missing numerical values using median imputation
- Encoded categorical variables (Area and Item) using label encoding (kept simple since tree-based models don’t require one-hot encoding)

The focus here was simplicity and reproducibility — nothing overly complex at this stage.

## Model Performance

A Random Forest Regressor was used as the primary baseline model. It performed surprisingly well straight away:
- R² ≈ 0.98

A Decision Tree model was also trained for comparison and showed strong performance, though Random Forest generalized better overall.

## Next Steps

- Hyperparameter tuning
- Cross-validation for more robust evaluation
- Feature importance analysis
- Model explainability (SHAP or similar tools)
- Integration into a web interface
