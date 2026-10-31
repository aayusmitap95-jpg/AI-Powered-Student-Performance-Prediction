"""Train and evaluate a leakage-safe student score regression model."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_performance_synthetic.csv"
MODEL_DIR = ROOT / "models"; OUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
before = len(df)
df = df.drop_duplicates()
df = df.dropna(subset=["final_score"])
print(f"Rows before cleanup: {before}; after duplicate/target cleanup: {len(df)}")

target = "final_score"
X = df.drop(columns=[target])
y = df[target]
numeric = X.select_dtypes(include=np.number).columns.tolist()
categorical = X.select_dtypes(exclude=np.number).columns.tolist()

# Split BEFORE fitting imputers/encoders. Pipeline learns preprocessing only from train data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
preprocess = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])
baseline = Pipeline([("preprocess", preprocess), ("model", DummyRegressor(strategy="mean"))])
model = Pipeline([("preprocess", preprocess), ("model",
    RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42))])
baseline.fit(X_train, y_train)
model.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
pred = model.predict(X_test)
metrics = {
    "rows_after_cleanup": int(len(df)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "baseline_mae": float(mean_absolute_error(y_test, baseline_pred)),
    "random_forest_mae": float(mean_absolute_error(y_test, pred)),
    "random_forest_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    "random_forest_r2": float(r2_score(y_test, pred))
}
joblib.dump(model, MODEL_DIR / "student_score_pipeline.joblib")
(OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
print(f"Saved model to {MODEL_DIR / 'student_score_pipeline.joblib'}")
