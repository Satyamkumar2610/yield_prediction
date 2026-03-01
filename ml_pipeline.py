import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw_data")
CLEAN_DIR = os.path.join(BASE_DIR, "cleandata")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.astype(str).str.strip()
    return df


def evaluate_model(model_name: str, y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"model": model_name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}


def run_pipeline():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CLEAN_DIR, exist_ok=True)

    yield_df = standardize_columns(pd.read_csv(os.path.join(RAW_DIR, "yield.csv")))
    pesticides_df = standardize_columns(pd.read_csv(os.path.join(RAW_DIR, "pesticides.csv")))
    rainfall_df = standardize_columns(pd.read_csv(os.path.join(RAW_DIR, "rainfall.csv")))
    temp_df = standardize_columns(pd.read_csv(os.path.join(RAW_DIR, "temp.csv")))

    raw_shapes = {
        "yield": yield_df.shape,
        "pesticides": pesticides_df.shape,
        "rainfall": rainfall_df.shape,
        "temperature": temp_df.shape,
    }

    yield_df = yield_df[["Area", "Item", "Year", "Value"]].copy()
    yield_df.rename(columns={"Value": "hg/ha_yield"}, inplace=True)

    pesticides_df = pesticides_df[["Area", "Year", "Value"]].copy()
    pesticides_df.rename(columns={"Value": "pesticides_tonnes"}, inplace=True)

    rainfall_df["average_rain_fall_mm_per_year"] = pd.to_numeric(
        rainfall_df["average_rain_fall_mm_per_year"], errors="coerce"
    )

    temp_df.rename(columns={"country": "Area", "year": "Year"}, inplace=True)
    temp_df = temp_df[["Area", "Year", "avg_temp"]].copy()

    merged_df = yield_df.merge(rainfall_df, on=["Area", "Year"], how="inner")
    merged_df = merged_df.merge(pesticides_df, on=["Area", "Year"], how="inner")
    merged_df = merged_df.merge(temp_df, on=["Area", "Year"], how="inner")
    shape_after_merge = merged_df.shape

    duplicates_count = int(merged_df.duplicated().sum())
    merged_df.drop_duplicates(inplace=True)
    shape_after_dedup = merged_df.shape

    numeric_cols = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
    missing_before = merged_df.isnull().sum().to_dict()
    for col in numeric_cols:
        median_val = merged_df[col].median()
        merged_df[col] = merged_df[col].fillna(median_val)
    missing_after = merged_df.isnull().sum().to_dict()

    merged_df.to_csv(os.path.join(CLEAN_DIR, "cleaned_yield_data.csv"), index=False)

    Q1 = merged_df["hg/ha_yield"].quantile(0.25)
    Q3 = merged_df["hg/ha_yield"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_removed = int(((merged_df["hg/ha_yield"] < lower_bound) | (merged_df["hg/ha_yield"] > upper_bound)).sum())
    merged_df = merged_df[(merged_df["hg/ha_yield"] >= lower_bound) & (merged_df["hg/ha_yield"] <= upper_bound)]
    shape_after_outlier = merged_df.shape

    unique_areas = sorted(merged_df["Area"].unique().tolist())
    unique_items = sorted(merged_df["Item"].unique().tolist())

    label_encoders = {}
    for col in ["Area", "Item"]:
        le = LabelEncoder()
        merged_df[col] = le.fit_transform(merged_df[col])
        label_encoders[col] = le

    merged_df.to_csv(os.path.join(CLEAN_DIR, "encoded_yield_data.csv"), index=False)

    X = merged_df.drop(columns=["hg/ha_yield"])
    y = merged_df["hg/ha_yield"]

    scaler = StandardScaler()
    scale_cols = numeric_cols + ["Year"]
    X[scale_cols] = scaler.fit_transform(X[scale_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    lr_pipeline = Pipeline([
        ("preprocessor", ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Area", "Item"])
            ],
            remainder="passthrough"
        )),
        ("regressor", LinearRegression())
    ])

    models = {
        "Linear Regression": lr_pipeline,
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    }

    results = []
    trained_models = {}
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(name, y_test, y_pred)
        results.append(metrics)
        trained_models[name] = model
        predictions[name] = y_pred

    rf = trained_models["Random Forest"]
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": X.columns.tolist(),
        "Importance": importances,
    }).sort_values(by="Importance", ascending=False)

    dt = trained_models["Decision Tree"]
    dt_importances = dt.feature_importances_
    dt_feature_importance_df = pd.DataFrame({
        "Feature": X.columns.tolist(),
        "Importance": dt_importances,
    }).sort_values(by="Importance", ascending=False)

    joblib.dump(rf, os.path.join(MODELS_DIR, "crop_yield_rf_model.pkl"), compress=3)
    joblib.dump(trained_models["Decision Tree"], os.path.join(MODELS_DIR, "crop_yield_dt_model.pkl"), compress=3)
    joblib.dump(trained_models["Linear Regression"], os.path.join(MODELS_DIR, "crop_yield_lr_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(label_encoders, os.path.join(MODELS_DIR, "label_encoders.pkl"))

    yield_stats = {
        "mean": round(float(y.mean()), 2),
        "median": round(float(y.median()), 2),
        "std": round(float(y.std()), 2),
        "min": round(float(y.min()), 2),
        "max": round(float(y.max()), 2),
        "q1": round(float(y.quantile(0.25)), 2),
        "q3": round(float(y.quantile(0.75)), 2),
    }

    metadata = {
        "raw_shapes": {k: list(v) for k, v in raw_shapes.items()},
        "shape_after_merge": list(shape_after_merge),
        "duplicates_removed": duplicates_count,
        "shape_after_dedup": list(shape_after_dedup),
        "missing_before": {k: int(v) for k, v in missing_before.items()},
        "missing_after": {k: int(v) for k, v in missing_after.items()},
        "outliers_removed": outliers_removed,
        "shape_after_outlier": list(shape_after_outlier),
        "model_results": results,
        "feature_importance_rf": feature_importance_df.to_dict(orient="records"),
        "feature_importance_dt": dt_feature_importance_df.to_dict(orient="records"),
        "feature_names": X.columns.tolist(),
        "unique_areas": unique_areas,
        "unique_items": unique_items,
        "yield_stats": yield_stats,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    with open(os.path.join(MODELS_DIR, "pipeline_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Pipeline completed successfully!")
    print(f"   Models saved to: {MODELS_DIR}")
    for r in results:
        print(f"   {r['model']:>20s}  —  MAE: {r['MAE']:.2f}  |  RMSE: {r['RMSE']:.2f}  |  R²: {r['R2']:.4f}")

    return metadata


if __name__ == "__main__":
    run_pipeline()
