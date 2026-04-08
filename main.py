"""
DataFlow Pro — main.py (Fixed & Complete)
Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score,
)
import joblib
import io
import os

app = FastAPI(title="DataFlow Pro API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.abspath("latest_model.joblib")


def json_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message})


# ── /train ────────────────────────────────────────────────────────────────────

@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    model_type: str = Form(...),
    features: str = Form(...),
    complexity: int = Form(100),
):
    try:
        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            return json_error(f"Failed to parse CSV: {e}")

        if df.empty:
            return json_error("Dataset is empty.")

        if target_col not in df.columns:
            return json_error(f"Target column '{target_col}' not found in dataset. Available columns: {list(df.columns)}")

        # Drop rows with missing target
        df = df.dropna(subset=[target_col])
        if df.empty:
            return json_error("Dataset is empty after dropping rows with missing target values.")

        # Parse & validate features
        selected_features = [f.strip() for f in features.split(",") if f.strip()]
        selected_features = [f for f in selected_features if f in df.columns and f != target_col]

        if not selected_features:
            return json_error("No valid features found. Check that the selected columns exist and differ from the target.")

        # Drop high-cardinality columns (likely identifiers)
        clean_features: list[str] = []
        dropped_features: list[str] = []

        for f in selected_features:
            series = df[f]
            if series.dtype == "object" and series.nunique(dropna=True) > 0.8 * len(df):
                dropped_features.append(f)
            else:
                clean_features.append(f)

        if not clean_features:
            return json_error("All selected features were identified as high-cardinality identifiers and dropped. Please select different features.")

        X = df[clean_features].copy()
        y = df[target_col].copy()

        # ── Auto-detect task type ──────────────────────────────────────────────
        is_classification = (
            y.dtype == "object"
            or y.dtype == "bool"
            or str(y.dtype) == "category"
            or y.nunique() < 15
        )

        label_encoder: LabelEncoder | None = None
        if is_classification and (y.dtype == "object" or str(y.dtype) == "category" or y.dtype == "bool"):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y.astype(str))
        elif is_classification:
            # Numeric with few unique values — cast to int
            y = y.astype(int)

        # ── Column type separation ─────────────────────────────────────────────
        num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32", "int16", "float16"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # ── Preprocessor ──────────────────────────────────────────────────────
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])

        try:
            cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            # Older sklearn (< 1.2) uses `sparse` instead of `sparse_output`
            cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  cat_encoder),
        ])

        transformers = []
        if num_cols:
            transformers.append(("num", num_transformer, num_cols))
        if cat_cols:
            transformers.append(("cat", cat_transformer, cat_cols))

        if not transformers:
            return json_error("No usable columns remain after preprocessing.")

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        # ── Train / test split ─────────────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if is_classification and len(np.unique(y)) > 1 else None,
        )

        # ── Model selection ────────────────────────────────────────────────────
        n = max(10, min(int(complexity), 500))   # clamp estimator count

        if model_type == "random_forest":
            model = (
                RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
                if is_classification
                else RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1)
            )
        elif model_type == "gradient_boost":
            model = (
                GradientBoostingClassifier(n_estimators=n, random_state=42)
                if is_classification
                else GradientBoostingRegressor(n_estimators=n, random_state=42)
            )
        else:  # linear_reg
            model = (
                LogisticRegression(max_iter=1000, random_state=42)
                if is_classification
                else Ridge(alpha=max(0.01, 100.0 / max(n, 1)))
            )

        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model",        model),
        ])

        full_pipeline.fit(X_train, y_train)

        # ── Evaluate ───────────────────────────────────────────────────────────
        preds = full_pipeline.predict(X_test)

        # ── Persist ────────────────────────────────────────────────────────────
        joblib.dump(
            {
                "pipeline":          full_pipeline,
                "target":            target_col,
                "is_classification": is_classification,
                "features":          clean_features,
                "label_encoder":     label_encoder,
            },
            MODEL_PATH,
        )

        # ── Build response ─────────────────────────────────────────────────────
        y_test_list  = y_test.tolist() if hasattr(y_test, "tolist") else list(y_test)
        preds_list   = preds.tolist()  if hasattr(preds, "tolist")  else list(preds)

        results: dict = {
            "task_type":   "classification" if is_classification else "regression",
            "model_name":  type(model).__name__,
            "test_size":   int(len(y_test)),
            "actuals":     y_test_list[:100],
            "predictions": preds_list[:100],
            "dropped_cols": dropped_features,
            "features":    clean_features,
        }

        if is_classification:
            results["accuracy"] = float(accuracy_score(y_test, preds))
            results["f1"]       = float(f1_score(y_test, preds, average="weighted", zero_division=0))
        else:
            results["r2"]   = float(r2_score(y_test, preds))
            results["mae"]  = float(mean_absolute_error(y_test, preds))
            results["rmse"] = float(np.sqrt(mean_squared_error(y_test, preds)))

        return results

    except Exception as e:
        import traceback
        traceback.print_exc()
        return json_error(str(e), 500)


# ── /download_model ───────────────────────────────────────────────────────────

@app.get("/download_model")
async def download_model():
    if os.path.exists(MODEL_PATH):
        return FileResponse(
            MODEL_PATH,
            media_type="application/octet-stream",
            filename="dataflow_model.joblib",
        )
    return json_error("Model not found. Train a model first.", 404)


# ── /predict_new ──────────────────────────────────────────────────────────────

@app.post("/predict_new")
async def predict_new(file: UploadFile = File(...)):
    try:
        if not os.path.exists(MODEL_PATH):
            return json_error("No trained model found. Please train a model first.", 404)

        saved       = joblib.load(MODEL_PATH)
        pipeline    = saved["pipeline"]
        target_col  = saved["target"]
        features    = saved.get("features", [])
        le          = saved.get("label_encoder")

        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            return json_error(f"Failed to parse prediction CSV: {e}")

        if df.empty:
            return json_error("Prediction file is empty.")

        # Add missing columns as NaN so the pipeline doesn't crash
        if features:
            for f in features:
                if f not in df.columns:
                    df[f] = np.nan
            X_new = df[features].copy()
        else:
            X_new = df.copy()

        preds = pipeline.predict(X_new)

        if le is not None:
            try:
                preds = le.inverse_transform(np.asarray(preds).astype(int))
            except Exception:
                pass  # keep numeric predictions if inverse_transform fails

        df[f"Predicted_{target_col}"] = preds

        stream = io.StringIO()
        df.to_csv(stream, index=False)
        stream.seek(0)

        response = StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
        )
        response.headers["Content-Disposition"] = "attachment; filename=predictions_output.csv"
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return json_error(str(e), 500)
