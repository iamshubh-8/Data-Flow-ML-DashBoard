# =====================================================
# Carbon Emission Predictor — FastAPI Backend
#
# Run with:  uvicorn main:app --reload
# Frontend expects this at http://localhost:8000
#
# Task type: AUTO-DETECTED per dataset
#   → Numeric target  = Regression  (predict a number)
#   → Categorical target = Classification (predict a category)
# =====================================================

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
import pandas as pd
import numpy as np
import io, os, joblib

# Sklearn pipeline pieces
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score,
)

app = FastAPI(title="Carbon Emission Predictor API")

# Allow the frontend (any origin) to talk to us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path where we save the trained model so /predict_new can reload it
MODEL_PATH = "carbon_model.joblib"


# =====================================================
# HELPERS
# =====================================================

def json_error(msg: str, code: int = 400):
    """Return a JSON error response — consistent format for the frontend."""
    return JSONResponse(status_code=code, content={"error": msg})


def parse_csv(contents: bytes) -> pd.DataFrame:
    """
    Read raw CSV bytes into a DataFrame.
    We replace '?' with NaN because some datasets use '?' for missing values.
    """
    try:
        df = pd.read_csv(io.BytesIO(contents))
        df.replace("?", np.nan, inplace=True)
        return df
    except Exception as e:
        raise ValueError(f"CSV parse failed: {e}")


def build_preprocessor(X: pd.DataFrame):
    """
    Build a ColumnTransformer that handles numeric and categorical columns differently.

    For numeric columns:
      - Impute missing values with the column median (robust to outliers)
      - Scale using RobustScaler (uses IQR instead of std, so outliers don't dominate)

    For categorical columns:
      - Impute missing values with the most frequent value (mode)
      - One-Hot Encode: convert each category into a binary 0/1 column
        e.g. 'Diet' → 'Diet_vegan', 'Diet_omnivore', 'Diet_vegetarian', ...
    """
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if num_cols: transformers.append(("num", num_pipe, num_cols))
    if cat_cols: transformers.append(("cat", cat_pipe, cat_cols))

    if not transformers:
        raise ValueError("No usable columns found after preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def detect_task(y: pd.Series) -> bool:
    """
    Auto-detect whether this is a classification or regression task.

    Rules:
    - If target is text/bool → classification
    - If target is float → regression
    - If target is integer with fewer than 20 unique values → classification
    - Otherwise → regression

    Returns True if classification, False if regression.
    """
    if y.dtype == "object" or y.dtype == "bool":
        return True   # definitely classification
    if y.dtype in ["float64", "float32"]:
        return False  # definitely regression
    # Integer column — use unique value count as heuristic
    return y.nunique() < 20


def pick_model(model_type: str, is_clf: bool):
    """
    Return the right sklearn model based on user choice + task type.
    Each algorithm has both a Regressor and Classifier variant.
    """
    if model_type == "random_forest":
        return (RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
                if is_clf else
                RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))

    if model_type == "gradient_boost":
        return (GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
                if is_clf else
                GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42))

    # Linear fallback: Logistic for classification, Ridge for regression
    return (LogisticRegression(max_iter=1000, random_state=42)
            if is_clf else Ridge(alpha=1.0))


def get_feature_importance(pipeline, X: pd.DataFrame):
    """
    Extract which input features the model relied on most.
    Works for both tree models (feature_importances_) and linear models (coef_).
    Returns top 10 features sorted by importance descending.
    """
    try:
        model = pipeline.named_steps["model"]
        pre   = pipeline.named_steps["pre"]

        # Reconstruct feature names after OHE expansion
        feature_names = []
        for name, trans, cols in pre.transformers_:
            if name == "num":
                feature_names.extend(cols)
            elif name == "cat":
                ohe = trans.named_steps["encoder"]
                feature_names.extend(ohe.get_feature_names_out(cols).tolist())

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coefs = model.coef_
            # Multiclass classifiers have a 2D coef_ — take mean absolute value
            importances = np.abs(coefs[0] if coefs.ndim > 1 else coefs)
        else:
            return []

        pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
        return [{"feature": k, "importance": float(v)} for k, v in pairs]

    except Exception:
        return []


# =====================================================
# ROUTE: POST /train
# =====================================================

@app.post("/train")
async def train_model(
    file:       UploadFile = File(...),
    target_col: str        = Form(...),   # chosen by user in the dropdown
    model_type: str        = Form(...),   # 'random_forest' | 'gradient_boost' | 'linear_reg'
    features:   str        = Form(...),   # comma-separated feature column names
    test_size:  float      = Form(0.2),   # fraction of data held out for evaluation
):
    try:
        contents = await file.read()
        df       = parse_csv(contents)

        if df.empty:
            return json_error("Dataset is empty.")
        if target_col not in df.columns:
            return json_error(f"Target column '{target_col}' not found in dataset.")

        test_size = max(0.1, min(0.4, test_size))

        # Drop rows where the target is missing — can't train on those
        df = df.dropna(subset=[target_col])
        if df.empty:
            return json_error("No rows left after dropping missing target values.")

        # Parse the feature list from the comma-separated string
        selected = [f.strip() for f in features.split(",")
                    if f.strip() and f.strip() in df.columns and f.strip() != target_col]
        if not selected:
            return json_error("No valid feature columns found.")

        # Drop high-cardinality text columns (likely IDs or free text)
        # Rule: if a text column has >80% unique values, it's not useful for ML
        dropped, clean = [], []
        for col in selected:
            if df[col].dtype == "object" and df[col].nunique() > 0.8 * len(df):
                dropped.append(col)
            else:
                clean.append(col)

        if not clean:
            return json_error("All features appear to be high-cardinality identifiers.")

        X = df[clean].copy()
        y = df[target_col].copy()

        # Auto-detect task type from the target column
        is_clf = detect_task(y)

        # If classification with text labels, encode them to integers
        le = None
        if is_clf and y.dtype == "object":
            le = LabelEncoder()
            y  = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
        elif is_clf:
            y = y.astype(int)
        else:
            y = y.astype(float)

        # Build the full ML pipeline: preprocessor → model
        preprocessor = build_preprocessor(X)
        model        = pick_model(model_type, is_clf)
        pipeline     = Pipeline([("pre", preprocessor), ("model", model)])

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if is_clf and y.nunique() > 1 else None,
        )

        # Train the model
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        # Learning curve — how performance changes with more training data
        scoring = "accuracy" if is_clf else "r2"
        lc_data = None
        try:
            cv_folds = 5
            if is_clf:
                class_counts = y_train.value_counts()
                if not class_counts.empty:
                    cv_folds = min(cv_folds, int(class_counts.min()))
            else:
                cv_folds = min(cv_folds, len(y_train))

            if cv_folds >= 2:
                sizes, lc_train, lc_val = learning_curve(
                    pipeline, X_train, y_train,
                    train_sizes=np.linspace(0.1, 1.0, 8),
                    cv=cv_folds, scoring=scoring, n_jobs=-1,
                )
                lc_data = {
                    "sizes":      [int(s) for s in sizes],
                    "train_mean": [round(float(v), 4) for v in np.mean(lc_train, axis=1)],
                    "val_mean":   [round(float(v), 4) for v in np.mean(lc_val,   axis=1)],
                }
        except Exception:
            pass

        # Save the pipeline so /predict_new can load it later
        joblib.dump({
            "pipeline":      pipeline,
            "target":        target_col,
            "features":      clean,
            "is_clf":        is_clf,
            "label_encoder": le,
        }, MODEL_PATH)

        # Build the response object with the right metrics for the task type
        result = {
            "task_type":          "classification" if is_clf else "regression",
            "model_name":         type(pipeline.named_steps["model"]).__name__,
            "test_size":          int(len(y_test)),
            "dropped_cols":       dropped,
            "features":           clean,
            "feature_importance": get_feature_importance(pipeline, X_train),
            "learning_curve":     lc_data,
            "actuals":            y_test.tolist()[:150],
            "predictions":        preds.tolist()[:150],
        }

        if is_clf:
            result["accuracy"] = float(accuracy_score(y_test, preds))
            result["f1"]       = float(f1_score(y_test, preds, average="weighted", zero_division=0))
        else:
            result["r2"]   = float(r2_score(y_test, preds))
            result["mae"]  = float(mean_absolute_error(y_test, preds))
            result["rmse"] = float(np.sqrt(mean_squared_error(y_test, preds)))

        return result

    except ValueError as e:
        return json_error(str(e))
    except Exception as e:
        import traceback; traceback.print_exc()
        return json_error(str(e), 500)


# =====================================================
# ROUTE: POST /predict_new
# =====================================================

@app.post("/predict_new")
async def predict_new(file: UploadFile = File(...)):
    try:
        if not os.path.exists(MODEL_PATH):
            return json_error("No trained model found. Train a model first.", 404)

        saved    = joblib.load(MODEL_PATH)
        pipeline = saved["pipeline"]
        features = saved.get("features", [])
        target   = saved.get("target", "Target")
        le       = saved.get("label_encoder")

        contents = await file.read()
        df = parse_csv(contents)
        if df.empty:
            return json_error("Uploaded file is empty.")

        for col in features:
            if col not in df.columns:
                df[col] = np.nan

        df = df[features]

        X_new = df.copy()
        preds = pipeline.predict(X_new)

        # If we encoded labels during training, decode them back to original strings
        if le is not None:
            try:
                preds = le.inverse_transform(np.asarray(preds).astype(int))
            except Exception:
                pass

        df[f"Predicted_{target}"] = preds

        stream = io.StringIO()
        df.to_csv(stream, index=False)
        stream.seek(0)

        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv; charset=utf-8")
        response.headers["Content-Disposition"] = f"attachment; filename=predictions_{target}.csv"
        return response

    except Exception as e:
        import traceback; traceback.print_exc()
        return json_error(str(e), 500)


# =====================================================
# ROUTE: GET /download_model
# =====================================================

@app.get("/download_model")
async def download_model():
    if not os.path.exists(MODEL_PATH):
        return json_error("No trained model found.", 404)
    return FileResponse(MODEL_PATH, media_type="application/octet-stream",
                        filename="carbon_model.joblib")