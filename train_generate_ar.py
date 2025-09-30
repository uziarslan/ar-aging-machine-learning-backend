import os
import re
import json
import math
import pickle
import random
import string
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ------------------------------
# Utility data classes
# ------------------------------


@dataclass
class ModelArtifacts:
    version: str
    classifier_path: str
    regressor_path: str
    registry_path: str


# ------------------------------
# Helpers
# ------------------------------


MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

# Common typo corrections found in historical data
DESCRIPTION_CORRECTIONS = {
    "sloutions": "solutions",
    "solutons": "solutions", 
    "soltuions": "solutions",
    "servies": "services",
    "servces": "services",
    "serivces": "services",
    "managment": "management",
    "mangement": "management",
    "compnay": "company",
    "compny": "company",
    "coporation": "corporation",
    "coropration": "corporation",
    "corpration": "corporation",
    "incoporated": "incorporated",
    "incorprated": "incorporated",
    "enterprizes": "enterprises",
    "enterprses": "enterprises",
    "techology": "technology",
    "tecnology": "technology",
    "technolgy": "technology",
    "logisitcs": "logistics",
    "logistcs": "logistics",
    "trasport": "transport",
    "transprot": "transport",
    "internatinal": "international",
    "internation": "international",
    "internatioanl": "international",
    "constructon": "construction",
    "constuction": "construction",
    "contruction": "construction",
    "developement": "development",
    "developent": "development",
    "devlopment": "development",
    "distibution": "distribution",
    "distribtion": "distribution",
    "distrubution": "distribution",
    "manufacutring": "manufacturing",
    "manufaturing": "manufacturing",
    "manfuacturing": "manufacturing",
}


def _safe_float(x: Any) -> float:
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x)
    s = s.replace(",", "").replace("$", "").strip()
    # Handle (1,234) accounting negative format
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except Exception:
        return 0.0


def _clean_description(desc: str) -> str:
    """Clean and standardize description names using historical corrections."""
    if not desc or not isinstance(desc, str):
        return ""
    
    # Basic cleaning
    cleaned = desc.strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single
    
    # Apply typo corrections (case-insensitive)
    words = cleaned.split()
    corrected_words = []
    
    for word in words:
        word_lower = word.lower().rstrip('.,!?;:')  # Remove punctuation for matching
        
        # Check for typo corrections
        if word_lower in DESCRIPTION_CORRECTIONS:
            # Preserve original case pattern
            corrected = DESCRIPTION_CORRECTIONS[word_lower]
            if word.isupper():
                corrected = corrected.upper()
            elif word.istitle():
                corrected = corrected.title()
            
            # Add back punctuation if it was there
            punct = word[len(word.rstrip('.,!?;:')):] if word != word.rstrip('.,!?;:') else ""
            corrected_words.append(corrected + punct)
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)


def _infer_month_from_sheet(sheet_name: str) -> Optional[str]:
    """Infer YYYY-MM from sheet name like 'AR-CAD Sept-2021' or 'Sep-2021'."""
    s = sheet_name.lower()
    # Find tokens like 'sept-2021' or 'sep 2021'
    m = re.search(
        r"(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)[ -_]?([0-9]{4})",
        s,
    )
    if not m:
        return None
    mon_token = m.group(1)
    year = int(m.group(2))
    month = MONTH_MAP.get(mon_token, None)
    if not month:
        return None
    return f"{year:04d}-{month:02d}"


def _month_features(month_str: str) -> Tuple[int, int, float, float]:
    year, month = [int(p) for p in month_str.split("-")]
    # Seasonality encoding
    angle = 2.0 * math.pi * (month - 1) / 12.0
    return year, month, math.sin(angle), math.cos(angle)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Deprecated: No-op placeholder retained for backward compatibility."""

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


# ------------------------------
# Serialization-safe helpers
# ------------------------------


def flatten_to_1d_strings(X):
    try:
        arr = X.to_numpy().ravel()
    except AttributeError:
        arr = np.asarray(X).ravel()
    return pd.Series(arr).astype(str).values


# ------------------------------
# Parsing
# ------------------------------


def parse_data(excel_path: str, xml_document: Optional[str] = None) -> pd.DataFrame:
    """Parse historical SBS AR aging data from Excel and optional XML document.

    Returns a normalized DataFrame with columns:
    ['description','month','current','0_30','31_60','61_90','90_plus','total']
    """
    all_rows: List[Dict[str, Any]] = []

    if os.path.exists(excel_path):
        xls = pd.ExcelFile(excel_path)
        for sheet in xls.sheet_names:
            month_tag = _infer_month_from_sheet(sheet)
            if not month_tag:
                continue
            raw = pd.read_excel(excel_path, sheet_name=sheet, header=None, dtype=object)
            header_idx = None
            for i in range(min(15, len(raw))):
                row_vals = [str(x).strip().lower() for x in list(raw.iloc[i].values)]
                if any("description" == v for v in row_vals) and any("current" == v for v in row_vals):
                    header_idx = i
                    break
            if header_idx is None:
                # Fallback: first non-empty row containing 'description'
                for i in range(min(20, len(raw))):
                    row_vals = [str(x).strip().lower() for x in list(raw.iloc[i].values)]
                    if any("description" in v for v in row_vals):
                        header_idx = i
                        break
            if header_idx is None:
                continue

            df = pd.read_excel(excel_path, sheet_name=sheet, header=header_idx, dtype=object)
            df.columns = [str(c).strip() for c in df.columns]
            cols = [c for c in df.columns]

            def find_col(patterns: List[str]) -> Optional[str]:
                for c in cols:
                    s = str(c).strip().lower()
                    for p in patterns:
                        if p in s:
                            return c
                return None

            desc_col = find_col(["description", "customer", "name"])
            cur_col = find_col(["current"])
            b0_col = find_col(["1-30", "0-30", "1 – 30", "0 – 30", "1 –30", "0 –30", "1 – 30 days", "1 - 30", "0 - 30"])
            b31_col = find_col(["31-60", "31 – 60", "31 - 60"])
            b61_col = find_col(["61-90", "61 – 90", "61 - 90"])
            b90_col = find_col(["91 and over", "90+", "90 plus", "90 and over", "over 90", "91 and over"])
            tot_col = find_col(["total"])  # optional

            if not desc_col or not cur_col:
                continue

            for _, row in df.iterrows():
                desc = str(row.get(desc_col, "")).strip()
                if not desc or desc.lower().startswith("description"):
                    continue
                if "total" in desc.lower():
                    continue

                current = _safe_float(row.get(cur_col, 0))
                b0 = _safe_float(row.get(b0_col, 0)) if b0_col else 0.0
                b31 = _safe_float(row.get(b31_col, 0)) if b31_col else 0.0
                b61 = _safe_float(row.get(b61_col, 0)) if b61_col else 0.0
                b90 = _safe_float(row.get(b90_col, 0)) if b90_col else 0.0
                total = _safe_float(row.get(tot_col, 0)) if tot_col else current + b0 + b31 + b61 + b90

                if current == 0 and b0 == 0 and b31 == 0 and b61 == 0 and b90 == 0:
                    continue

                all_rows.append(
                    {
                        "description": _clean_description(desc),  # Clean description
                        "month": month_tag,
                        "current": current,
                        "0_30": b0,
                        "31_60": b31,
                        "61_90": b61,
                        "90_plus": b90,
                        "total": total if total > 0 else (current + b0 + b31 + b61 + b90),
                    }
                )

    # Optional: parse XML if provided; expected structure includes entries similar to Mongo JSON
    if xml_document and isinstance(xml_document, str) and "<document" in xml_document:
        # Very lightweight XML parsing without external libs: find lines that look like JSON
        # Users can plug in their own strict XML parser if needed.
        json_blocks = re.findall(r"\{[\s\S]*?\}", xml_document)
        for jb in json_blocks:
            try:
                rec = json.loads(jb)
                aging = rec.get("aging", {})
                all_rows.append(
                    {
                            "description": _clean_description(rec.get("description", "")),  # Clean description
                        "month": rec.get("month", ""),
                        "current": _safe_float(aging.get("current", 0)),
                        "0_30": _safe_float(aging.get("0_30", 0)),
                        "31_60": _safe_float(aging.get("31_60", 0)),
                        "61_90": _safe_float(aging.get("61_90", 0)),
                        "90_plus": _safe_float(aging.get("90_plus", 0)),
                        "total": _safe_float(rec.get("total", 0)),
                    }
                )
            except Exception:
                continue

    if not all_rows:
        raise ValueError("No data parsed. Ensure the Excel file path is correct and sheets are named with months.")

    df = pd.DataFrame(all_rows)
    # Normalize description whitespace
    df["description"] = df["description"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    # Ensure totals are consistent
    summed = df[["current", "0_30", "31_60", "61_90", "90_plus"]].sum(axis=1)
    df["total"] = np.where(df["total"] > 0, df["total"], summed)
    return df


# ------------------------------
# Feature assembly
# ------------------------------


def _build_time_index(df: pd.DataFrame) -> List[str]:
    months = sorted(df["month"].unique())
    return months


def _assemble_carry_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Create dataset where each row is (description, month_t) and label indicates
    whether it appears in month_{t+1}. Features from month_t.
    """
    months = _build_time_index(df)
    feature_rows: List[Dict[str, Any]] = []
    labels: List[int] = []

    df = df.copy()
    for mon in ["current", "0_30", "31_60", "61_90", "90_plus", "total"]:
        df[mon] = pd.to_numeric(df[mon], errors="coerce").fillna(0.0)

    for i in range(len(months) - 1):
        m_this = months[i]
        m_next = months[i + 1]
        prev_df = df[df["month"] == m_this].copy()
        next_df = df[df["month"] == m_next].copy()

        next_desc = set(next_df["description"].unique())
        prev_df["year"], prev_df["month_num"], prev_df["month_sin"], prev_df["month_cos"] = zip(*prev_df["month"].map(_month_features))

        for _, row in prev_df.iterrows():
            feature_rows.append(
                {
                    "description": row["description"],
                    "month": row["month"],
                    "current": row["current"],
                    "0_30": row["0_30"],
                    "31_60": row["31_60"],
                    "61_90": row["61_90"],
                    "90_plus": row["90_plus"],
                    "total": row["total"],
                    "year": row["year"],
                    "month_num": row["month_num"],
                    "month_sin": row["month_sin"],
                    "month_cos": row["month_cos"],
                }
            )
            labels.append(1 if row["description"] in next_desc else 0)

    X = pd.DataFrame(feature_rows)
    # Add per-description aggregates for better generalization
    if not X.empty:
        agg = (
            df.groupby("description")[
                ["total", "current", "0_30", "31_60", "61_90", "90_plus"]
            ]
            .agg(["mean", "std", "count"]).reset_index()
        )
        # Flatten columns
        agg.columns = [
            "description"
            if col[0] == "description"
            else f"desc_{col[0]}_{col[1]}" for col in agg.columns.to_list()
        ]
        X = X.merge(agg, on="description", how="left")
        # Bucket percentages historical means
        df_pct = df.copy()
        denom = df_pct[["current", "0_30", "31_60", "61_90", "90_plus"]].sum(axis=1)
        for b in ["current", "0_30", "31_60", "61_90", "90_plus"]:
            df_pct[f"pct_{b}"] = np.where(denom > 0, df_pct[b] / denom, 0.0)
        pct_agg = df_pct.groupby("description")[[f"pct_{b}" for b in ["current", "0_30", "31_60", "61_90", "90_plus"]]].mean().reset_index().rename(columns={f"pct_{b}": f"desc_mean_pct_{b}" for b in ["current", "0_30", "31_60", "61_90", "90_plus"]})
        X = X.merge(pct_agg, on="description", how="left")
    y = pd.Series(labels, name="carry")
    return X, y


def _assemble_bucket_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rows only for (description, month_t) where description appears in month_{t+1}.
    Targets are next month's aged buckets (0_30, 31_60, 61_90, 90_plus).
    """
    months = _build_time_index(df)
    feature_rows: List[Dict[str, Any]] = []
    targets: List[Dict[str, float]] = []

    df = df.copy()
    for mon in ["current", "0_30", "31_60", "61_90", "90_plus", "total"]:
        df[mon] = pd.to_numeric(df[mon], errors="coerce").fillna(0.0)

    for i in range(len(months) - 1):
        m_this = months[i]
        m_next = months[i + 1]
        prev_df = df[df["month"] == m_this].copy()
        next_df = df[df["month"] == m_next].copy()
        next_df = next_df.set_index("description")

        prev_df["year"], prev_df["month_num"], prev_df["month_sin"], prev_df["month_cos"] = zip(*prev_df["month"].map(_month_features))

        for _, row in prev_df.iterrows():
            desc = row["description"]
            if desc not in next_df.index:
                continue
            nrow = next_df.loc[desc]
            feature_rows.append(
                {
                    "description": desc,
                    "month": row["month"],
                    "current": row["current"],
                    "0_30": row["0_30"],
                    "31_60": row["31_60"],
                    "61_90": row["61_90"],
                    "90_plus": row["90_plus"],
                    "total": row["total"],
                    "year": row["year"],
                    "month_num": row["month_num"],
                    "month_sin": row["month_sin"],
                    "month_cos": row["month_cos"],
                }
            )
            targets.append(
                {
                    "0_30": float(nrow["0_30"]),
                    "31_60": float(nrow["31_60"]),
                    "61_90": float(nrow["61_90"]),
                    "90_plus": float(nrow["90_plus"]),
                }
            )

    X = pd.DataFrame(feature_rows)
    if not X.empty:
        agg = (
            df.groupby("description")[
                ["total", "current", "0_30", "31_60", "61_90", "90_plus"]
            ]
            .agg(["mean", "std", "count"]).reset_index()
        )
        agg.columns = [
            "description"
            if col[0] == "description"
            else f"desc_{col[0]}_{col[1]}" for col in agg.columns.to_list()
        ]
        X = X.merge(agg, on="description", how="left")
        df_pct = df.copy()
        denom = df_pct[["current", "0_30", "31_60", "61_90", "90_plus"]].sum(axis=1)
        for b in ["current", "0_30", "31_60", "61_90", "90_plus"]:
            df_pct[f"pct_{b}"] = np.where(denom > 0, df_pct[b] / denom, 0.0)
        pct_agg = df_pct.groupby("description")[[f"pct_{b}" for b in ["current", "0_30", "31_60", "61_90", "90_plus"]]].mean().reset_index().rename(columns={f"pct_{b}": f"desc_mean_pct_{b}" for b in ["current", "0_30", "31_60", "61_90", "90_plus"]})
        X = X.merge(pct_agg, on="description", how="left")
    Y = pd.DataFrame(targets)[["0_30", "31_60", "61_90", "90_plus"]]
    return X, Y


def _make_preprocessor(numeric_cols: List[str]) -> ColumnTransformer:
    text_col = "description"
    text_pipeline = Pipeline(
        steps=[
            ("flatten", FunctionTransformer(flatten_to_1d_strings, validate=False)),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ]
    )
    num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    transformers = [
        ("num", num_pipeline, numeric_cols),
        ("txt", text_pipeline, [text_col]),
    ]
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def _positive_class_proba(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    """Return probability of positive class robustly even if only one class was seen in training.

    Works for Pipeline with a 'clf' step or direct estimators exposing classes_.
    """
    proba = estimator.predict_proba(X)
    if isinstance(proba, list):
        proba = np.asarray(proba)
    proba = np.asarray(proba)
    if proba.ndim == 1:
        return proba
    if proba.shape[1] == 1:
        # Get the only class label
        clf = getattr(estimator, "named_steps", {}).get("clf", estimator)
        classes = getattr(clf, "classes_", np.array([0]))
        only = int(classes[0]) if len(classes) else 0
        return np.full(proba.shape[0], 1.0 if only == 1 else 0.0)
    # Otherwise pick column for class == 1 if present, else take last column
    clf = getattr(estimator, "named_steps", {}).get("clf", estimator)
    classes = getattr(clf, "classes_", np.arange(proba.shape[1]))
    idx = None
    try:
        idxs = np.where(classes == 1)[0]
        idx = int(idxs[0]) if len(idxs) else None
    except Exception:
        idx = None
    if idx is None:
        idx = min(1, proba.shape[1] - 1)
    return proba[:, idx]


# ------------------------------
# Training
# ------------------------------


def train_model(df: pd.DataFrame, client_name: str = "SBS", models_dir: Optional[str] = None) -> Tuple[ModelArtifacts, Dict[str, Any]]:
    """Train carry classifier and bucket regressor. Persist artifacts and registry.

    Returns paths and version information.
    """
    models_dir = models_dir or os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    # Assemble datasets
    Xc, yc = _assemble_carry_dataset(df)
    Xr, Yr = _assemble_bucket_dataset(df)

    # Chronological split 80/20 by month
    months = sorted(Xc["month"].unique())
    split_idx = int(len(months) * 0.8)
    carry_train_months = set(months[:split_idx])
    carry_test_months = set(months[split_idx:])

    c_numeric = ["current", "0_30", "31_60", "61_90", "90_plus", "total", "year", "month_num", "month_sin", "month_cos"]
    c_pre = _make_preprocessor(c_numeric)
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    c_pipe = Pipeline(steps=[("prep", c_pre), ("clf", clf)])

    # TimeSeries CV
    cv = TimeSeriesSplit(n_splits=min(5, max(2, len(months) - 2)))
    param_grid = {
        "clf__n_estimators": [300, 500],
        "clf__max_depth": [10, 14],
        "clf__min_samples_leaf": [1, 2],
    }

    # Prepare split indices aligned by month to avoid leakage
    Xc_train = Xc[Xc["month"].isin(carry_train_months)].reset_index(drop=True)
    yc_train = yc.loc[Xc_train.index]
    Xc_test = Xc[Xc["month"].isin(carry_test_months)].reset_index(drop=True)
    yc_test = yc.loc[Xc_test.index]

    # Grid search on training data only
    gs = GridSearchCV(c_pipe, param_grid=param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
    gs.fit(Xc_train, yc_train)
    best_clf_pipe = gs.best_estimator_

    # Evaluate
    carry_proba = _positive_class_proba(best_clf_pipe, Xc_test)
    carry_pred = (carry_proba >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(yc_test, carry_pred, average="binary", zero_division=0)
    auc = roc_auc_score(yc_test, carry_proba) if len(np.unique(yc_test)) > 1 else 1.0

    # Bucket regressor split by month as well
    months_r = sorted(Xr["month"].unique())
    split_idx_r = int(len(months_r) * 0.8)
    reg_train_months = set(months_r[:split_idx_r])
    reg_test_months = set(months_r[split_idx_r:])

    r_numeric = ["current", "0_30", "31_60", "61_90", "90_plus", "total", "year", "month_num", "month_sin", "month_cos"]
    r_pre = _make_preprocessor(r_numeric)
    base_regr = RandomForestRegressor(
        n_estimators=600,
        max_depth=18,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    regr = MultiOutputRegressor(base_regr)
    r_pipe = Pipeline(steps=[("prep", r_pre), ("regr", regr)])

    Xr_train = Xr[Xr["month"].isin(reg_train_months)].reset_index(drop=True)
    Yr_train = Yr.loc[Xr_train.index]
    Xr_test = Xr[Xr["month"].isin(reg_test_months)].reset_index(drop=True)
    Yr_test = Yr.loc[Xr_test.index]

    r_pipe.fit(Xr_train, Yr_train)
    Yr_pred = pd.DataFrame(r_pipe.predict(Xr_test), columns=["0_30", "31_60", "61_90", "90_plus"])  # type: ignore

    metrics_bucket = {}
    for col in ["0_30", "31_60", "61_90", "90_plus"]:
        metrics_bucket[f"b{col}_mape"] = safe_mape(Yr_test[col].values, Yr_pred[col].values, epsilon=1.0)
        metrics_bucket[f"b{col}_rmse"] = rmse(Yr_test[col].values, Yr_pred[col].values)

    metrics_carry = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }

    # Persist artifacts
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_client = re.sub(r"[^a-z0-9]+", "_", client_name.lower()).strip("_")
    version = f"ar_aging_{ts}"
    clf_path = os.path.join(models_dir, f"{safe_client}_carry_clf_{version}.pkl")
    regr_path = os.path.join(models_dir, f"{safe_client}_bucket_regr_{version}.pkl")
    with open(clf_path, "wb") as f:
        pickle.dump(best_clf_pipe, f)
    with open(regr_path, "wb") as f:
        pickle.dump(r_pipe, f)

    registry_path = os.path.join(models_dir, f"{safe_client}_model_registry.json")
    record = {
        "client": client_name,
        "version": version,
        "metrics": {
            "carry_classifier": metrics_carry,
            "bucket_regressor": metrics_bucket,
            "training_samples": {
                "carry_train": int(len(Xc_train)),
                "bucket_train": int(len(Xr_train)),
            },
        },
        "created_at": datetime.utcnow().isoformat() + "Z",
        "status": "trained",
    }
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, list):
            obj = [obj]
    else:
        obj = []
    obj.append(record)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

    return ModelArtifacts(version=version, classifier_path=clf_path, regressor_path=regr_path, registry_path=registry_path), {
        "carry_classifier": metrics_carry,
        "bucket_regressor": metrics_bucket,
        "training_samples": {
            "carry_train": int(len(Xc_train)),
            "bucket_train": int(len(Xr_train)),
        },
    }


# ------------------------------
# Generation logic
# ------------------------------


def _load_models(classifier_path: str, regressor_path: str):
    with open(classifier_path, "rb") as f:
        clf = pickle.load(f)
    with open(regressor_path, "rb") as f:
        regr = pickle.load(f)
    return clf, regr


def _prepare_last_month_df(last_month_records: List[Dict[str, Any]], month_str: str) -> pd.DataFrame:
    rows = []
    for rec in last_month_records:
        desc = rec.get("description") or rec.get("Description") or ""
        aging = rec.get("aging", rec)
        rows.append(
            {
                "description": _clean_description(str(desc).strip()),  # Clean descriptions
                "month": month_str,
                "current": _safe_float(aging.get("current", aging.get("Current", 0))),
                "0_30": _safe_float(aging.get("0_30", aging.get("1-30", 0))),
                "31_60": _safe_float(aging.get("31_60", aging.get("31-60", 0))),
                "61_90": _safe_float(aging.get("61_90", aging.get("61-90", 0))),
                "90_plus": _safe_float(aging.get("90_plus", aging.get("91 and over", 0))),
                "total": _safe_float(rec.get("total", 0)),
            }
        )
    df = pd.DataFrame(rows)
    if "total" not in df or df["total"].isna().all():
        df["total"] = df[["current", "0_30", "31_60", "61_90", "90_plus"]].sum(axis=1)
    df["year"], df["month_num"], df["month_sin"], df["month_cos"] = zip(*df["month"].map(_month_features))
    return df


def _historical_description_stats(df_hist: pd.DataFrame) -> pd.DataFrame:
    # Frequency and last seen month index for prioritization
    months = sorted(df_hist["month"].unique())
    month_to_idx = {m: i for i, m in enumerate(months)}
    grp = df_hist.groupby("description")
    stats = grp["total"].agg(["count", "mean", "sum"]).rename(columns={"count": "freq"}).reset_index()
    last_seen = grp["month"].max().map(lambda m: month_to_idx.get(m, -1)).rename("last_idx")
    stats = stats.merge(last_seen, left_on="description", right_index=True)
    stats["recency_weight"] = stats["last_idx"] / (len(months) - 1 + 1e-9)
    return stats.sort_values(["recency_weight", "freq", "sum"], ascending=False)


def _get_historical_bucket_stats(history_df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    """Extract historical bucket statistics for realistic aging patterns."""
    if history_df is None or history_df.empty:
        return {}
    
    df = history_df.copy()
    stats = {}
    
    # Overall bucket percentages and variations
    totals = df[["current", "0_30", "31_60", "61_90", "90_plus"]].sum(axis=1)
    valid_mask = totals > 0
    
    if not valid_mask.any():
        return stats
        
    valid_df = df[valid_mask]
    valid_totals = totals[valid_mask]
    
    for bucket in ["current", "0_30", "31_60", "61_90", "90_plus"]:
        pcts = valid_df[bucket] / valid_totals
        stats[f"{bucket}_mean"] = float(pcts.mean())
        stats[f"{bucket}_std"] = float(pcts.std())
    
    return stats


def _consolidate_small_rows(rows: List[Dict[str, Any]], max_small: int = 25) -> List[Dict[str, Any]]:
    """Consolidate small rows (<1k) to reduce count and add variety."""
    large_rows = []
    small_rows = []
    
    # Separate large and small rows
    for row in rows:
        # Never consolidate override rows or the Adjustment row
        if row.get("is_override") or row.get("description") == "Adjustment":
            large_rows.append(row)
            continue
        if row["total"] >= 1000:
            large_rows.append(row)
        else:
            small_rows.append(row)
    
    # If we have too many small rows, consolidate them
    if len(small_rows) > max_small:
        # Sort small rows by total (largest first)
        small_rows.sort(key=lambda x: x["total"], reverse=True)
        
        # Keep the largest small rows
        kept_small = small_rows[:max_small//2]
        to_consolidate = small_rows[max_small//2:]
        
        # Group remaining small rows for consolidation
        consolidated = []
        while to_consolidate:
            # Take 3-5 rows to merge
            group_size = min(5, max(3, len(to_consolidate)))
            group = to_consolidate[:group_size]
            to_consolidate = to_consolidate[group_size:]
            
            # Create consolidated row
            total_amount = sum(r["total"] for r in group)
            if total_amount > 0:
                # Use the first description as base, modify to indicate consolidation
                base_desc = group[0]["description"]
                if len(group) > 1:
                    # Add indicator that this is consolidated
                    base_desc = f"{base_desc} & Others"
                
                # Sum the buckets from all consolidated rows
                consolidated_buckets = {
                    "current": 0, "0_30": 0, "31_60": 0, "61_90": 0, "90_plus": 0
                }
                any_user_added = any(bool(r.get("user_added")) for r in group)
                any_auto_generated = any(bool(r.get("auto_generated")) for r in group)
                any_protected = any(bool(r.get("protected_0_30")) for r in group)
                
                for bucket in consolidated_buckets:
                    bucket_sum = sum(r["aging"][bucket] for r in group)
                    consolidated_buckets[bucket] = int(bucket_sum)
                
                # If this consolidated row includes any user-added or auto-generated source rows,
                # enforce 0-30-only: zero older buckets and make current=total
                if any_user_added or any_auto_generated:
                    consolidated_buckets["31_60"] = 0
                    consolidated_buckets["61_90"] = 0
                    consolidated_buckets["90_plus"] = 0
                    consolidated_buckets["current"] = int(consolidated_buckets["0_30"])
                
                # CRITICAL: Calculate total as sum of buckets
                calculated_total = sum(consolidated_buckets.values())
                
                consolidated.append({
                    "description": base_desc,
                    "month": group[0]["month"],
                    "aging": consolidated_buckets,
                    "predicted": True,
                    "total": float(calculated_total),  # Total = sum of buckets
                    "user_added": bool(any_user_added),
                    "auto_generated": bool(any_auto_generated),
                    "protected_0_30": bool(any_protected)
                })
        
        # Combine all rows
        result = large_rows + kept_small + consolidated
    else:
        result = large_rows + small_rows

    return result


def _apply_noise_and_round(predicted_buckets: Dict[str, float], noise_factor: float = 0.075, is_small_row: bool = False) -> Dict[str, int]:
    """Apply noise to buckets and round with strict decreasing constraints.

    - Aged buckets (0_30, 31_60, 61_90, 90_plus): negative-biased noise only (-10% to 0%)
    - Current bucket: small ±5% noise allowed
    """
    rng = np.random.default_rng()
    result = {}
    
    for bucket, value in predicted_buckets.items():
        if value > 0:
            # Apply noise
            if bucket in ["0_30", "31_60", "61_90", "90_plus"]:
                # Negative-only noise to guarantee decreases
                noise = rng.uniform(-0.10 * value, 0.0)
                noisy_value = max(0.0, value + noise)
                # Floor to further avoid accidental increases from rounding
                rounded_value = int(noisy_value)
            elif bucket == "current":
                # Current can vary slightly up/down
                noise = rng.uniform(-0.05 * value, 0.05 * value)
                noisy_value = max(0.0, value + noise)
                rounded_value = int(round(noisy_value))
            else:
                # Default conservative behavior
                noise_range = noise_factor * value
                noise = rng.uniform(-noise_range, 0.0)
                noisy_value = max(0.0, value + noise)
                rounded_value = int(noisy_value)
            
            # Enhanced variety boost for small rows' older buckets
            if is_small_row and bucket in ["31_60", "61_90", "90_plus"] and rounded_value > 0:
                # If value is around 100, vary it 80-120 with ±10-20 noise
                if 80 <= rounded_value <= 120:
                    # Add extra ±10-20 noise from historical std
                    extra_noise = rng.uniform(-20, 20)
                    rounded_value = max(10, int(rounded_value + extra_noise))
                else:
                    # Standard minimum for other older buckets
                    min_val = rng.integers(10, 51)
                    rounded_value = max(rounded_value, min_val)
            elif bucket in ["31_60", "61_90", "90_plus"] and rounded_value > 0:
                # Non-zero older buckets should be at least 10-50 for realism
                min_val = rng.integers(10, 51)
                rounded_value = max(rounded_value, min_val)
            elif bucket in ["current", "0_30"] and rounded_value > 0:
                # Current/0-30 buckets should be at least 50 for realism
                rounded_value = max(rounded_value, 50)
            
            result[bucket] = rounded_value
        else:
            result[bucket] = 0
    
    # SAFETY CHECK: Ensure aging constraints (older buckets <= newer buckets)
    # This prevents unrealistic increases during aging simulation
    if "current" in result and "0_30" in result:
        # 0-30 should not exceed what could have come from current
        if result["0_30"] > result["current"] * 0.9:  # Allow up to 90% shift
            result["0_30"] = max(0, int(result["current"] * rng.uniform(0.6, 0.9)))
    
    if "0_30" in result and "31_60" in result:
        # 31-60 should not exceed what could have come from 0-30
        if result["31_60"] > result["0_30"] * 0.9:
            result["31_60"] = max(0, int(result["0_30"] * rng.uniform(0.6, 0.9)))
    
    if "31_60" in result and "61_90" in result:
        # 61-90 should not exceed what could have come from 31-60
        if result["61_90"] > result["31_60"] * 0.9:
            result["61_90"] = max(0, int(result["31_60"] * rng.uniform(0.6, 0.9)))
    
    if "61_90" in result and "90_plus" in result:
        # 90+ should be reasonable compared to 61-90
        if result["90_plus"] > result["61_90"] * 1.2:  # Allow some accumulation in 90+
            result["90_plus"] = max(0, int(result["61_90"] * rng.uniform(0.8, 1.2)))
    
    return result


def _ensure_current_dominance(buckets: Dict[str, int], is_new: bool = False) -> Dict[str, int]:
    """Ensure realistic current bucket proportions with historical aging patterns."""
    total = sum(buckets.values())
    if total == 0:
        return buckets
    
    current_and_030 = buckets["current"] + buckets["0_30"]
    
    if is_new:
        # New descriptions should have ~40-60% in current for large customers
        # ~65% current -> 0-30, ~20% to 31-60 for historical aging patterns
        target_current_ratio = 0.5  # 40-60% range
        target_030_ratio = 0.35     # From current aging
    else:
        # Carried descriptions: maintain historical shifts
        # ~65% of previous current moves to 0-30, ~20% to 31-60
        target_current_ratio = 0.4   # 40-60% for large customers
        target_030_ratio = 0.4       # Historical aging pattern
    
    # Adjust if current+0-30 is too low (should be >70% for most)
    if current_and_030 < total * 0.7:
        # Redistribute from older buckets
        older_total = buckets["31_60"] + buckets["61_90"] + buckets["90_plus"]
        if older_total > 0:
            reduction_needed = int(total * 0.7) - current_and_030
            reduction_factor = max(0, (older_total - reduction_needed) / older_total)
            
            buckets["31_60"] = int(buckets["31_60"] * reduction_factor)
            buckets["61_90"] = int(buckets["61_90"] * reduction_factor)
            buckets["90_plus"] = int(buckets["90_plus"] * reduction_factor)
            
            # Add recovered amount to current and 0-30
            recovered = older_total - (buckets["31_60"] + buckets["61_90"] + buckets["90_plus"])
            buckets["current"] += int(recovered * 0.6)
            buckets["0_30"] += int(recovered * 0.4)
    
    return buckets


def _force_exact_row_totals(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Force Total = sum of aged buckets only, and Current = Total - CRITICAL enforcement."""
    for row in rows:
        # Calculate sum of aged buckets only (excluding current)
        aged_sum = (
            row["aging"]["0_30"] + 
            row["aging"]["31_60"] + 
            row["aging"]["61_90"] + 
            row["aging"]["90_plus"]
        )
        # FORCE total to equal aged buckets sum only
        row["total"] = float(int(aged_sum))
        # CRITICAL: Set Current = Total as per requirements
        row["aging"]["current"] = float(int(aged_sum))
    return rows


def _scale_to_exact_target(
    rows: List[Dict[str, Any]],
    target_total: float,
    column_targets: Optional[Dict[str, float]] = None,
    max_iterations: int = 5
) -> List[Dict[str, Any]]:
    """Scale to exact target while enforcing specific column targets.

    If column_targets is provided, it should be a mapping with keys:
    {"0_30", "31_60", "61_90", "90_plus"} and numeric values.
    """
    
    # CRITICAL: Use provided column targets when available; otherwise fall back to defaults
    COLUMN_TARGETS = column_targets or {
        "0_30": 1537823,
        "31_60": 1100000,
        "61_90": 800000,
        "90_plus": 500000
    }
    
    # CRITICAL: Force exact row totals first
    rows = _force_exact_row_totals(rows)
    
    # Calculate current column totals
    current_col_totals = {
        col: sum(r["aging"][col] for r in rows)
        for col in ["0_30", "31_60", "61_90", "90_plus"]
    }
    
    print(f"Current column totals before scaling: {current_col_totals}")
    print(f"Target column totals: {COLUMN_TARGETS}")
    
    # Step 1: Scale each column independently to hit exact targets
    for col, target in COLUMN_TARGETS.items():
        current_total = current_col_totals[col]
        
        if current_total == 0 and target > 0:
            # Need to create this column from scratch - distribute to all rows
            if rows:
                amount_per_row = target / len(rows)
                for row in rows:
                    if not row.get("is_override") or col != "0_30":
                        row["aging"][col] = amount_per_row
        elif current_total > 0:
            # Scale this column to exact target
            scale_factor = target / current_total
            
            for row in rows:
                if not row.get("is_override") or col != "0_30":
                    row["aging"][col] *= scale_factor
    
    # Convert to integers
    for row in rows:
        for col in ["0_30", "31_60", "61_90", "90_plus"]:
            row["aging"][col] = max(0, int(round(row["aging"][col])))
    
    # CRITICAL: Recalculate current as sum of aged buckets
    rows = _force_exact_row_totals(rows)
    
    # Step 2: Fine-tune to hit exact column targets with precision
    current_col_totals_after = {
        col: sum(r["aging"][col] for r in rows)
        for col in ["0_30", "31_60", "61_90", "90_plus"]
    }
    
    # Calculate discrepancies for each column
    discrepancies = {
        col: target - current_col_totals_after[col]
        for col, target in COLUMN_TARGETS.items()
    }
    
    print(f"Column discrepancies after scaling: {discrepancies}")
    
    # Distribute discrepancies intelligently
    for col, diff in discrepancies.items():
        if abs(diff) > 0:
            # Sort rows by appropriate criteria for distribution
            if diff > 0:
                # Need to add - prefer larger rows and non-override rows
                candidate_rows = [r for r in rows if not r.get("is_override") or col != "0_30"]
                candidate_rows.sort(key=lambda x: x["total"], reverse=True)
            else:
                # Need to remove - prefer larger rows and non-override rows  
                candidate_rows = [r for r in rows if not r.get("is_override") or col != "0_30"]
                candidate_rows.sort(key=lambda x: x["total"], reverse=True)
            
            if candidate_rows:
                remaining_diff = diff
                max_per_row = max(1, abs(remaining_diff) // max(1, len(candidate_rows)))
                
                for row in candidate_rows:
                    if remaining_diff == 0:
                        break
                    
                    if diff > 0:
                        # Add to this row
                        add_amount = min(remaining_diff, max_per_row)
                        row["aging"][col] += add_amount
                        remaining_diff -= add_amount
                    else:
                        # Remove from this row (but not below zero)
                        remove_amount = min(abs(remaining_diff), max_per_row, row["aging"][col])
                        row["aging"][col] -= remove_amount
                        remaining_diff += remove_amount
    
    # Final integer conversion
    for row in rows:
        for col in ["0_30", "31_60", "61_90", "90_plus"]:
            row["aging"][col] = max(0, int(round(row["aging"][col])))
    
    # CRITICAL: Final enforcement of exact row totals
    rows = _force_exact_row_totals(rows)
    
    # Step 3: Add adjustment rows for any remaining discrepancies
    final_col_totals = {
        col: sum(r["aging"][col] for r in rows)
        for col in ["0_30", "31_60", "61_90", "90_plus"]
    }
    
    final_discrepancies = {
        col: COLUMN_TARGETS[col] - final_col_totals[col]
        for col in COLUMN_TARGETS.keys()
    }
    
    # If 0-30 is over target (negative discrepancy), reduce 0-30 across eligible rows
    over_0_30 = -final_discrepancies.get("0_30", 0)
    if over_0_30 > 0:
        # Prefer non-override and non-protected rows with largest 0-30
        candidates = [r for r in rows if not r.get("is_override") and not r.get("protected_0_30")]
        candidates.sort(key=lambda r: r["aging"].get("0_30", 0), reverse=True)
        remaining = int(round(over_0_30))
        # First pass: proportional reduction
        for r in candidates:
            if remaining <= 0:
                break
            avail = int(r["aging"].get("0_30", 0))
            if avail <= 0:
                continue
            take = min(avail, max(1, remaining // max(1, len(candidates))))
            r["aging"]["0_30"] = max(0, avail - take)
            remaining -= take
        # Second pass: precise per-unit reduction for small leftover
        if remaining > 0 and candidates:
            idx = 0
            # Limit to reasonable iterations to avoid long loops if huge; remaining is typically small now
            max_steps = remaining * 2
            steps = 0
            while remaining > 0 and steps < max_steps:
                r = candidates[idx % len(candidates)]
                avail = int(r["aging"].get("0_30", 0))
                if avail > 0:
                    r["aging"]["0_30"] = avail - 1
                    remaining -= 1
                idx += 1
                steps += 1
        # Re-enforce after subtraction
        rows = _force_exact_row_totals(rows)
        # Recompute discrepancies
        final_col_totals = { col: sum(r["aging"][col] for r in rows) for col in ["0_30", "31_60", "61_90", "90_plus"] }
        final_discrepancies = { col: COLUMN_TARGETS[col] - final_col_totals[col] for col in COLUMN_TARGETS.keys() }

    # Add adjustment rows ONLY as positive additions in 0-30. No negatives, no older buckets.
    # If a column is over target (negative diff), we will not create a negative adjustment.
    # Such cases should have been handled in earlier fine-tuning; clamp here for safety.
    adjustments_needed = False
    adjustment_rows = []
    remaining_0_30 = final_discrepancies.get("0_30", 0)
    if remaining_0_30 > 0:
        adjustments_needed = True
        adjustment_rows.append({
            "description": "Adjustment_0_30",
            "month": "",
                    "aging": {
                        "current": 0.0,
                "0_30": float(max(0, int(round(remaining_0_30)))),
                        "31_60": 0.0,
                        "61_90": 0.0,
                "90_plus": 0.0,
                    },
                    "predicted": True,
            "total": float(max(0, int(round(remaining_0_30)))),
            "is_adjustment": True
        })
        print(f"Adding positive 0-30 adjustment: {remaining_0_30}")
    # Ignore negative diffs (overages) to avoid negative rows. Earlier scaling should have minimized them.
    
    if adjustments_needed:
        rows.extend(adjustment_rows)
    
    # Clamp any accidental negatives to zero and final enforcement
    for r in rows:
        for b in ["current", "0_30", "31_60", "61_90", "90_plus"]:
            if isinstance(r.get("aging", {}).get(b, 0), (int, float)):
                r["aging"][b] = float(max(0, int(round(r["aging"][b]))))
    rows = _force_exact_row_totals(rows)

    # Enforce per-row caps for older buckets based on previous month, then rebalance to meet column targets
    def _get_row_key(desc: str) -> str:
        try:
            return _clean_description(str(desc)).lower()
        except Exception:
            return str(desc).strip().lower()

    if column_targets is not None:
        # First clamp to caps for older buckets and collect removed amounts per column
        for col in ["31_60", "61_90", "90_plus"]:
            removed = 0
            for r in rows:
                cap = float('inf')
                # row_caps expected keys match column names; if not provided, no cap
                # NOTE: caps are passed via column_targets? No—row caps are bound to last month; use optional attribute injected in rows? Fallback: no cap
                # We attach caps via a hidden field on row if present
                r_caps = r.get("_caps") or {}
                if col in r_caps:
                    cap = float(max(0, int(round(r_caps[col]))))
                cur = float(r["aging"][col])
                # Never change user-added older buckets (already zero) but keep logic uniform
                if cur > cap:
                    removed += (cur - cap)
                    r["aging"][col] = cap
            # Redistribute any removed to rows with headroom in same column
            if removed > 0:
                remaining = int(round(removed))
                # Build candidates with headroom
                candidates = []
                for r in rows:
                    r_caps = r.get("_caps") or {}
                    cap_val = r_caps.get(col, None)
                    cur = int(round(float(r["aging"][col])))
                    # Do not use user-added rows as recipients for older buckets redistribution
                    if col in ["31_60", "61_90", "90_plus"] and (r.get("user_added") or r.get("auto_generated")):
                        continue
                    if cap_val is None:
                        # No cap: treat as very large headroom but finite
                        headroom = 10**9
                    else:
                        cap = int(round(max(0, float(cap_val))))
                        headroom = max(0, cap - cur)
                    if headroom > 0:
                        candidates.append((int(headroom), r))
                # Sort by headroom descending
                candidates.sort(key=lambda x: x[0], reverse=True)
                idx = 0
                while remaining > 0 and candidates:
                    hr, r = candidates[idx % len(candidates)]
                    if hr <= 0:
                        idx += 1
                        continue
                    add = 1 if remaining < 10 else min(hr, max(1, remaining // len(candidates)))
                    r["aging"][col] = float(int(round(r["aging"][col])) + add)
                    hr -= add
                    remaining -= add
                    candidates[idx % len(candidates)] = (hr, r)
                    idx += 1
        rows = _force_exact_row_totals(rows)

        # Now ensure each column meets target again within caps and per-row total caps
        for col in ["31_60", "61_90", "90_plus", "0_30"]:
            target = int(round(column_targets.get(col, 0)))
            cur_total = int(round(sum(int(round(r["aging"][col])) for r in rows)))
            if cur_total < target:
                need = target - cur_total
                # Grow within caps (no caps for 0_30 unless protected_0_30)
                candidates = []
                for r in rows:
                    if col == "0_30" and r.get("protected_0_30"):
                        continue
                    # If user_added or auto_generated, they cannot have older buckets (>0_30)
                    if col in ["31_60", "61_90", "90_plus"] and (r.get("user_added") or r.get("auto_generated")):
                        continue
                    r_caps = r.get("_caps") or {}
                    total_cap = int(round(float(r_caps.get("total", 10**9))))
                    current_aged_sum = int(round(
                        float(r["aging"]["0_30"]) + float(r["aging"]["31_60"]) + float(r["aging"]["61_90"]) + float(r["aging"]["90_plus"])
                    ))
                    total_headroom = max(0, total_cap - current_aged_sum)
                    cap_val = r_caps.get(col, None)
                    cur_val = int(round(float(r["aging"][col])))
                    if cap_val is None:
                        headroom = min(10**9, total_headroom)
                    else:
                        cap = int(round(max(0, float(cap_val))))
                        headroom = max(0, min(cap - cur_val, total_headroom))
                    if headroom > 0:
                        candidates.append((int(headroom), r))
                candidates.sort(key=lambda x: x[0], reverse=True)
                idx = 0
                while need > 0 and candidates:
                    hr, cand = candidates[idx % len(candidates)]
                    if hr <= 0:
                        idx += 1
                        continue
                    add_amt = 1 if need < 10 else min(hr, max(1, need // len(candidates)))
                    cand["aging"][col] = float(int(round(cand["aging"][col])) + add_amt)
                    hr -= add_amt
                    need -= add_amt
                    candidates[idx % len(candidates)] = (hr, cand)
                    idx += 1
            elif cur_total > target:
                excess = cur_total - target
                # Reduce from rows with largest values in this column (but never reduce user_added/auto_generated older buckets)
                candidates = []
                for r in rows:
                    cur_val = int(round(float(r["aging"][col])))
                    if cur_val > 0 and not (col == "0_30" and r.get("protected_0_30")) and not (col in ["31_60", "61_90", "90_plus"] and (r.get("user_added") or r.get("auto_generated"))):
                        candidates.append((cur_val, r))
                candidates.sort(key=lambda x: x[0], reverse=True)
                idx = 0
                while excess > 0 and candidates:
                    cur_val, cand = candidates[idx % len(candidates)]
                    if cur_val <= 0:
                        idx += 1
                        continue
                    take = 1 if excess < 10 else min(cur_val, max(1, excess // len(candidates)))
                    cand["aging"][col] = float(cur_val - take)
                    cur_val -= take
                    excess -= take
                    candidates[idx % len(candidates)] = (cur_val, cand)
        rows = _force_exact_row_totals(rows)

        # Final sweep: ensure auto_generated and user_added have zero in older buckets
        # and then rebalance columns to maintain targets (excluding those rows)
        if column_targets is not None:
            # Zero older buckets for auto/user rows and recompute totals
            for r in rows:
                if r.get("auto_generated") or r.get("user_added"):
                    r["aging"]["31_60"] = 0.0
                    r["aging"]["61_90"] = 0.0
                    r["aging"]["90_plus"] = 0.0
                    # Recompute total/current from aged buckets
                    aged_sum_local = (
                        float(r["aging"]["0_30"]) + float(r["aging"]["31_60"]) + float(r["aging"]["61_90"]) + float(r["aging"]["90_plus"])
                    )
                    r["aging"]["current"] = float(int(round(aged_sum_local)))
                    r["total"] = float(int(round(aged_sum_local)))
            rows = _force_exact_row_totals(rows)

            # Rebalance each older column if now short vs target
            for col in ["31_60", "61_90", "90_plus"]:
                target_val = int(round(column_targets.get(col, 0)))
                cur_sum = int(round(sum(int(round(rr["aging"][col])) for rr in rows)))
                if cur_sum < target_val:
                    need = target_val - cur_sum
                    # Eligible candidates: not user_added/auto_generated, respect caps and per-row total cap
                    cands = []
                    for rr in rows:
                        if rr.get("user_added") or rr.get("auto_generated"):
                            continue
                        caps = rr.get("_caps") or {}
                        total_cap = int(round(float(caps.get("total", 10**9))))
                        aged_total_rr = int(round(
                            float(rr["aging"]["0_30"]) + float(rr["aging"]["31_60"]) + float(rr["aging"]["61_90"]) + float(rr["aging"]["90_plus"]) 
                        ))
                        total_headroom_rr = max(0, total_cap - aged_total_rr)
                        cap_val = caps.get(col, None)
                        cur_val_rr = int(round(float(rr["aging"][col])))
                        if cap_val is None:
                            headroom_rr = min(10**9, total_headroom_rr)
                        else:
                            col_cap = int(round(max(0, float(cap_val))))
                            headroom_rr = max(0, min(col_cap - cur_val_rr, total_headroom_rr))
                        if headroom_rr > 0:
                            cands.append((int(headroom_rr), rr))
                    cands.sort(key=lambda x: x[0], reverse=True)
                    idx = 0
                    while need > 0 and cands:
                        hr, rr = cands[idx % len(cands)]
                        if hr <= 0:
                            idx += 1
                            continue
                        add_amt = 1 if need < 10 else min(hr, max(1, need // len(cands)))
                        rr["aging"][col] = float(int(round(rr["aging"][col])) + add_amt)
                        # update the tuple in place
                        hr -= add_amt
                        need -= add_amt
                        cands[idx % len(cands)] = (hr, rr)
                        idx += 1
                    rows = _force_exact_row_totals(rows)
                elif cur_sum > target_val:
                    excess = cur_sum - target_val
                    # Reduce from largest non user-added/auto_generated rows
                    red_cands = []
                    for rr in rows:
                        if rr.get("user_added") or rr.get("auto_generated"):
                            continue
                        cur_val_rr = int(round(float(rr["aging"][col])))
                        if cur_val_rr > 0:
                            red_cands.append((cur_val_rr, rr))
                    red_cands.sort(key=lambda x: x[0], reverse=True)
                    idx = 0
                    while excess > 0 and red_cands:
                        cur_val_rr, rr = red_cands[idx % len(red_cands)]
                        if cur_val_rr <= 0:
                            idx += 1
                            continue
                        take = 1 if excess < 10 else min(cur_val_rr, max(1, excess // len(red_cands)))
                        rr["aging"][col] = float(cur_val_rr - take)
                        cur_val_rr -= take
                        excess -= take
                        red_cands[idx % len(red_cands)] = (cur_val_rr, rr)
                        idx += 1
                    rows = _force_exact_row_totals(rows)

    # Final verification
    final_col_totals_verified = {
        col: sum(r["aging"][col] for r in rows)
        for col in ["0_30", "31_60", "61_90", "90_plus"]
    }
    
    final_total = sum(r["total"] for r in rows)
    
    print(f"Final column totals: {final_col_totals_verified}")
    print(f"Final grand total: {final_total}")
    print(f"Target grand total: {target_total}")
    
    # Verify all targets are met
    all_targets_met = True
    for col, target in COLUMN_TARGETS.items():
        actual = final_col_totals_verified[col]
        if abs(actual - target) > 1:
            print(f"✗ Column {col} target not met: {actual} vs {target}")
            all_targets_met = False
        else:
            print(f"✓ Column {col} target met: {actual} = {target}")
    
    if abs(final_total - target_total) > 1:
        print(f"✗ Grand total target not met: {final_total} vs {target_total}")
        all_targets_met = False
    else:
        print(f"✓ Grand total target met: {final_total} = {target_total}")
    
    if not all_targets_met:
        print("WARNING: Not all targets were met exactly")
    
    return rows


def _get_column_target_ratios(history_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate realistic column ratios from historical data."""
    if history_df is None or history_df.empty:
        # Default aging pattern
        return {
            "current": 0.45,  # 45% current
            "0_30": 0.35,     # 35% 0-30  
            "31_60": 0.12,    # 12% 31-60
            "61_90": 0.05,    # 5% 61-90
            "90_plus": 0.03   # 3% 90+
        }
    
    # Calculate average ratios from historical data
    totals = history_df[["current", "0_30", "31_60", "61_90", "90_plus"]].sum(axis=1)
    valid_mask = totals > 0
    
    if not valid_mask.any():
        return {
            "current": 0.45, "0_30": 0.35, "31_60": 0.12, 
            "61_90": 0.05, "90_plus": 0.03
        }
    
    valid_df = history_df[valid_mask]
    valid_totals = totals[valid_mask]
    
    ratios = {}
    for col in ["current", "0_30", "31_60", "61_90", "90_plus"]:
        pcts = valid_df[col] / valid_totals
        ratios[col] = float(pcts.mean())
    
    # Normalize to ensure they sum to 1
    total_ratio = sum(ratios.values())
    ratios = {k: v/total_ratio for k, v in ratios.items()}
    
    return ratios


def _simulate_realistic_aging(
    prev_buckets: Dict[str, float], 
    rng: np.random.Generator,
    is_small_row: bool = False
) -> Dict[str, float]:
    """
    Simulate realistic aging with GUARANTEED decreases - no bucket can increase.
    Enforces 90%+ decrease compliance with strict mathematical limits.
    
    Args:
        prev_buckets: Previous month's bucket values
        rng: Random number generator
        is_small_row: Whether this is a small row (affects clearing rates)
    
    Returns:
        New month's bucket values with GUARANTEED decreasing aging
    """
    # CONSERVATIVE shift rates to ensure 90%+ decreases
    if is_small_row:
        # Small rows: higher clearing rates (more likely to be paid/written off)
        shift_rates = {
            "current_to_0_30": rng.uniform(0.6, 0.8),    # 60-80% shifts
            "0_30_to_31_60": rng.uniform(0.6, 0.8),      # 60-80% shifts (reduced from 0.7-0.9)
            "31_60_to_61_90": rng.uniform(0.6, 0.8),     # 60-80% shifts (reduced from 0.7-0.9)
            "61_90_to_90_plus": rng.uniform(0.5, 0.7),   # 50-70% shifts
            "90_plus_retention": rng.uniform(0.7, 0.9)   # 70-90% retained
        }
    else:
        # Large rows: still conservative to prevent increases
        shift_rates = {
            "current_to_0_30": rng.uniform(0.6, 0.8),    # 60-80% shifts
            "0_30_to_31_60": rng.uniform(0.6, 0.8),      # 60-80% shifts (reduced from 0.7-0.9)
            "31_60_to_61_90": rng.uniform(0.6, 0.8),     # 60-80% shifts (reduced from 0.7-0.9)
            "61_90_to_90_plus": rng.uniform(0.5, 0.7),   # 50-70% shifts
            "90_plus_retention": rng.uniform(0.8, 0.95)  # 80-95% retained
        }
    
    # STRICT AGING SIMULATION - amounts can only decrease
    new_buckets = {
        "current": 0.0,  # Start with 0, will add new business later
        "0_30": prev_buckets.get("current", 0) * shift_rates["current_to_0_30"],
        "31_60": prev_buckets.get("0_30", 0) * shift_rates["0_30_to_31_60"],
        "61_90": prev_buckets.get("31_60", 0) * shift_rates["31_60_to_61_90"],
        "90_plus": (prev_buckets.get("61_90", 0) * shift_rates["61_90_to_90_plus"] + 
                   prev_buckets.get("90_plus", 0) * shift_rates["90_plus_retention"])
    }
    
    # ABSOLUTE MAXIMUM LIMITS: No bucket can exceed 90% of its source
    new_buckets["0_30"] = min(new_buckets["0_30"], prev_buckets.get("current", 0) * 0.9)
    new_buckets["31_60"] = min(new_buckets["31_60"], prev_buckets.get("0_30", 0) * 0.9)
    new_buckets["61_90"] = min(new_buckets["61_90"], prev_buckets.get("31_60", 0) * 0.9)
    
    # Apply NEGATIVE-BIASED noise (-10% to -5%) to guarantee further decreases
    for bucket in ["0_30", "31_60", "61_90", "90_plus"]:
        if new_buckets[bucket] > 0:
            # Negative-biased noise: -10% to -5% reduction
            noise_factor = rng.uniform(0.05, 0.10)  # 5-10% reduction
            new_buckets[bucket] *= (1.0 - noise_factor)
            
    # FINAL ENFORCEMENT: Ensure no bucket exceeds 80% of its source (stricter)
    new_buckets["0_30"] = min(new_buckets["0_30"], prev_buckets.get("current", 0) * 0.8)
    new_buckets["31_60"] = min(new_buckets["31_60"], prev_buckets.get("0_30", 0) * 0.8)
    new_buckets["61_90"] = min(new_buckets["61_90"], prev_buckets.get("31_60", 0) * 0.8)
    
    # MATHEMATICAL GUARANTEE: Round down to ensure no floating point increases
    for bucket in new_buckets:
        if new_buckets[bucket] > 0:
            new_buckets[bucket] = float(int(new_buckets[bucket]))  # Floor to integer
    
    return new_buckets


def _generate_next_month_core(
    clf: Any,
    regr: Any,  # Still needed for compatibility but won't be used for bucket prediction
    last_month_records: List[Dict[str, Any]],
    last_month_str: str,
    next_month_str: str,
    target_total: float,
    history_df: Optional[pd.DataFrame] = None,
    overrides: Optional[Dict[str, float]] = None,
    carry_threshold: float = 0.2,
) -> List[Dict[str, Any]]:
    """Generate next month's AR aging rows using realistic aging simulation."""
    
    # Get historical column ratios for realistic distribution
    column_ratios = _get_column_target_ratios(history_df)
    
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    # Normalize overrides: clean description keys and coerce to positive floats
    if overrides:
        try:
            overrides = {
                _clean_description(str(k)).lower(): float(v)
                for k, v in overrides.items()
                if v is not None and float(v) > 0
            }
        except Exception:
            # If any issue occurs, fall back to empty overrides to avoid breaking generation
            overrides = {}
    
    last_df = _prepare_last_month_df(last_month_records, last_month_str)
    # Preserve last month's visual order for sorting carried rows later
    last_order_map: Dict[str, int] = {}
    for idx, rec in enumerate(last_month_records):
        d = _clean_description(str((rec.get("description") or rec.get("Description") or "").strip()))
        if d and d not in last_order_map:
            last_order_map[d] = idx
    
    # Predict carry-over using classifier
    carry_proba = _positive_class_proba(clf, last_df)
    last_df = last_df.copy()
    last_df["carry_proba"] = carry_proba
    # Use provided threshold from UI/backend (default 0.2)
    try:
        thr = float(carry_threshold)
    except Exception:
        thr = 0.2
    thr = max(0.0, min(1.0, thr))
    carried_df = last_df[last_df["carry_proba"] >= thr].copy()
    
    result_rows = []
    
    # Process carried descriptions with STRICT aging simulation (NO REGRESSOR)
    if not carried_df.empty:
        for _, row in carried_df.iterrows():
            desc = row["description"]
            
            # Get previous month's buckets
            prev_buckets = {
                "current": row.get("current", 0),
                "0_30": row.get("0_30", 0),
                "31_60": row.get("31_60", 0),
                "61_90": row.get("61_90", 0),
                "90_plus": row.get("90_plus", 0)
            }
            
            prev_total = row["total"]
            is_small = prev_total < 1000
            
            # STRICT aging simulation - amounts ONLY DECREASE
            aged_buckets = _simulate_realistic_aging(prev_buckets, rng, is_small)
            
            # Calculate aged total (excluding current)
            aged_total = aged_buckets["0_30"] + aged_buckets["31_60"] + aged_buckets["61_90"] + aged_buckets["90_plus"]
            
            # REALISTIC approach: Maintain aging ratios by limiting growth based on customer size
            # Small customers grow less, large customers can grow more
            
            # Calculate realistic growth based on historical customer size
            if prev_total < 10000:  # Very small customer
                # Minimal growth to maintain realistic ratios
                growth_factor = rng.uniform(0.8, 1.1)  # -20% to +10%
            elif prev_total < 50000:  # Small customer  
                # Moderate growth
                growth_factor = rng.uniform(0.9, 1.3)  # -10% to +30%
            else:  # Large customer
                # Can have more significant growth
                growth_factor = rng.uniform(1.0, 1.5)  # 0% to +50%
            
            # Calculate target size for this customer
            target_customer_size = prev_total * growth_factor
            
            # Determine how much new business is needed
            remaining_for_new = max(0, target_customer_size - aged_total)
            
            if remaining_for_new > 0:
                # ALL new business goes to current only (most realistic)
                aged_buckets["current"] = remaining_for_new
            else:
                aged_buckets["current"] = 0

            # Round to integers with minimal noise (only negative for aged buckets)
            final_buckets = {}
            for bucket, value in aged_buckets.items():
                if bucket == "current":
                    # Current can have small positive noise (new business variation)
                    noise = rng.uniform(-0.05, 0.05) * value if value > 0 else 0
                    final_buckets[bucket] = max(0, int(value + noise))
                else:
                    # Aged buckets: only negative noise to ensure decreases
                    noise = rng.uniform(-0.10, 0.0) * value if value > 0 else 0
                    final_buckets[bucket] = max(0, int(value + noise))
            
            # STRICT CAPS after noise: enforce <= 90% of source buckets
            final_buckets["0_30"] = min(
                final_buckets["0_30"],
                int(prev_buckets.get("current", 0) * 0.9)
            )
            final_buckets["31_60"] = min(
                final_buckets["31_60"],
                int(prev_buckets.get("0_30", 0) * 0.9)
            )
            final_buckets["61_90"] = min(
                final_buckets["61_90"],
                int(prev_buckets.get("31_60", 0) * 0.9)
            )
            # For 90+, cap relative to prior (61-90 + 90+) at 90%
            cap_90 = int((prev_buckets.get("61_90", 0) + prev_buckets.get("90_plus", 0)) * 0.9)
            final_buckets["90_plus"] = min(final_buckets["90_plus"], cap_90)
            
            # Apply override for 0-30 if provided for this carried description
            desc_clean = _clean_description(desc).lower()
            if overrides and desc_clean in overrides:
                override_val = max(0, int(round(float(overrides[desc_clean]))))
                # Never exceed previous month's CURRENT (same or less rule)
                cap_prev_current = int(prev_buckets.get("current", 0))
                final_buckets["0_30"] = min(override_val, cap_prev_current)
                # Re-apply non-increase caps for older buckets relative to their sources
                final_buckets["31_60"] = min(final_buckets["31_60"], int(prev_buckets.get("0_30", 0) * 0.9))
                final_buckets["61_90"] = min(final_buckets["61_90"], int(prev_buckets.get("31_60", 0) * 0.9))
                cap_90_recalc = int((prev_buckets.get("61_90", 0) + prev_buckets.get("90_plus", 0)) * 0.9)
                final_buckets["90_plus"] = min(final_buckets["90_plus"], cap_90_recalc)

            # GLOBAL CAP for carried rows: total of aged buckets must not exceed previous month's total
            prev_total_cap = int(round(prev_total)) if isinstance(prev_total, (int, float)) else 0
            aged_sum_pre = final_buckets["0_30"] + final_buckets["31_60"] + final_buckets["61_90"] + final_buckets["90_plus"]
            if prev_total_cap > 0 and aged_sum_pre > prev_total_cap:
                scale_factor = prev_total_cap / float(aged_sum_pre)
                # Proportionally scale down aged buckets
                for k in ["0_30", "31_60", "61_90", "90_plus"]:
                    final_buckets[k] = int(max(0, int(round(final_buckets[k] * scale_factor))))

            # CRITICAL: Calculate total as sum of aged buckets only
            aged_sum = final_buckets["0_30"] + final_buckets["31_60"] + final_buckets["61_90"] + final_buckets["90_plus"]
            final_buckets["current"] = aged_sum  # ENFORCE: Current = Total = aged sum
            
            if aged_sum > 0:
                result_rows.append({
                    "description": desc,
                    "month": next_month_str,
                    "aging": {k: float(v) for k, v in final_buckets.items()},
                    "predicted": True,
                    "total": float(aged_sum),  # Total = sum of aged buckets only
                    # Flags
                    "protected_0_30": bool(overrides and desc_clean in overrides),
                    "is_override": bool(overrides and desc_clean in overrides),
                    # Per-row caps to be respected during later scaling passes
                    "_caps": {
                        "31_60": float(prev_buckets.get("0_30", 0)),
                        "61_90": float(prev_buckets.get("31_60", 0)),
                        "90_plus": float(prev_buckets.get("61_90", 0) + prev_buckets.get("90_plus", 0)),
                        "total": float(prev_total),
                    },
                })
    
    # Add rows for any override descriptions that classifier did NOT carry over
    if overrides:
        carried_descs = set(carried_df["description"].tolist()) if not carried_df.empty else set()
        already_emitted = {r["description"] for r in result_rows}
        # Convert already_emitted to cleaned lowercase for comparison
        already_emitted_clean = {_clean_description(d).lower() for d in already_emitted}
        missing_override_descs = [d for d in overrides.keys() if d not in already_emitted_clean]
        if missing_override_descs:
            for desc in missing_override_descs:
                # Find previous row for this description (need to match against cleaned descriptions)
                last_df_clean = last_df.copy()
                last_df_clean["description_clean"] = last_df_clean["description"].apply(lambda x: _clean_description(x).lower())
                prev_rows = last_df_clean[last_df_clean["description_clean"] == desc]
                if prev_rows.empty:
                    continue
                row = prev_rows.iloc[0]
                prev_buckets = {
                    "current": row.get("current", 0),
                    "0_30": row.get("0_30", 0),
                    "31_60": row.get("31_60", 0),
                    "61_90": row.get("61_90", 0),
                    "90_plus": row.get("90_plus", 0)
                }
                prev_total = float(row.get("total", 0)) if "total" in row else (
                    float(prev_buckets["0_30"] + prev_buckets["31_60"] + prev_buckets["61_90"] + prev_buckets["90_plus"]) )
                is_small = prev_total < 1000
                # Use the same aging simulation as carried rows to follow learned trend
                aged_buckets = _simulate_realistic_aging(prev_buckets, rng, is_small)
                # Inject override for 0-30, capped at previous current (same or less rule)
                override_val = max(0, int(round(float(overrides[desc]))))
                cap_prev_current = int(prev_buckets.get("current", 0))
                aged_buckets["0_30"] = min(override_val, cap_prev_current)
                # Recalculate aged total and set current accordingly
                aged_sum = aged_buckets["0_30"] + aged_buckets["31_60"] + aged_buckets["61_90"] + aged_buckets["90_plus"]
                final_buckets = {
                    "0_30": int(max(0, aged_buckets["0_30"])),
                    "31_60": int(max(0, min(aged_buckets["31_60"], int(prev_buckets.get("0_30", 0) * 0.9)))),
                    "61_90": int(max(0, min(aged_buckets["61_90"], int(prev_buckets.get("31_60", 0) * 0.9)))),
                    "90_plus": int(max(0, min(aged_buckets["90_plus"], int((prev_buckets.get("61_90", 0) + prev_buckets.get("90_plus", 0)) * 0.9))))
                }
                aged_sum = final_buckets["0_30"] + final_buckets["31_60"] + final_buckets["61_90"] + final_buckets["90_plus"]
                final_buckets["current"] = aged_sum
                if aged_sum > 0:
                    result_rows.append({
                        "description": desc,
                        "month": next_month_str,
                        "aging": {k: float(v) for k, v in final_buckets.items()},
                        "predicted": True,
                        "total": float(aged_sum),
                        # Flags
                        "protected_0_30": True,
                        "is_override": True,
                        "_caps": {
                            "31_60": float(prev_buckets.get("0_30", 0)),
                            "61_90": float(prev_buckets.get("31_60", 0)),
                            "90_plus": float(prev_buckets.get("61_90", 0) + prev_buckets.get("90_plus", 0)),
                            "total": float(prev_total),
                        },
                    })

    # Add new descriptions if needed to reach target
    current_total = sum(r["total"] for r in result_rows)
    remaining = target_total - current_total
    
    if remaining > 500:  # Add more rows if needed
        # Get candidate descriptions from history
        existing_descs = {r["description"] for r in result_rows}
        candidates = []
        
        if history_df is not None:
            hist_stats = _historical_description_stats(history_df)
            for desc in hist_stats["description"].head(100):
                cleaned_desc = _clean_description(desc)
                if cleaned_desc not in existing_descs:
                    candidates.append(cleaned_desc)
                if len(candidates) >= 80:
                    break
        
        # Limit new descriptions but allow more to reach target row count
        max_new = min(50, len(candidates))  # Increased from 20 to 50
        if max_new > 0:
            selected_candidates = candidates[:max_new]
            
            # Generate varied amounts for new descriptions
            base_amounts = []
            for i in range(len(selected_candidates)):
                # Vary amounts: some large (2k-10k), some medium (500-2k)
                if i < len(selected_candidates) // 3:
                    amount = rng.uniform(2000, 10000)
                else:
                    amount = rng.uniform(500, 2000)
                base_amounts.append(amount)
            
            # Scale to use remaining amount
            if sum(base_amounts) > 0:
                scale = remaining / sum(base_amounts)
                amounts = [max(500, int(amt * scale)) for amt in base_amounts]
                
                # Adjust for exact total
                diff = int(remaining - sum(amounts))
                if diff != 0 and amounts:
                    amounts[0] += diff
                    amounts[0] = max(500, amounts[0])
                
                # Create records for new descriptions with 0-30 only allocation
                for desc, amount in zip(selected_candidates, amounts):
                    if amount > 0:
                        is_small = amount < 1000
                        
                        # New descriptions: allocate ONLY to 0-30; older buckets remain 0
                        aged_buckets = {
                            "0_30": amount,
                            "31_60": 0,
                            "61_90": 0,
                            "90_plus": 0,
                        }
                        
                        # Apply ±5-10% noise to 0-30 only
                        noise_mag = rng.uniform(0.05, 0.10)
                        noise_sign = -1 if rng.random() < 0.5 else 1
                        noisy_0_30 = int(round(aged_buckets["0_30"] * (1 + noise_sign * noise_mag)))
                        noisy_0_30 = max(0, noisy_0_30)
                        
                        # Ensure minimum 100 in 0-30 for small new rows (< 1k)
                        if is_small:
                            noisy_0_30 = max(100, noisy_0_30)
                        
                        aged_buckets["0_30"] = noisy_0_30
                        
                        # Aged sum equals 0-30; set Current = Total = aged sum
                        aged_sum = aged_buckets["0_30"]
                        aged_buckets["current"] = aged_sum
                        
                        if aged_sum > 0:
                            result_rows.append({
                                "description": desc,
                                "month": next_month_str,
                                "aging": {k: float(v) for k, v in aged_buckets.items()},
                                "predicted": True,
                                "total": float(aged_sum),
                        # New descriptions added by model to reach target are auto_generated and 0-30-only
                        "protected_0_30": True,  # protect from scaling, but not an override
                        "is_override": False,
                        "auto_generated": True,
                        # Hard cap zero for older buckets to be explicit for downstream checks
                        "_caps": {
                            "31_60": 0.0,
                            "61_90": 0.0,
                            "90_plus": 0.0,
                            "total": float(aged_sum),
                        },
                            })
    
    # Consolidate small rows to reduce count and add variety
    result_rows = _consolidate_small_rows(result_rows, max_small=40)  # Increased from 25 to 40
    
    # Scale to exact target with column-aware scaling
    result_rows = _scale_to_exact_target(result_rows, target_total, max_iterations=3)
    
    # Final validation and sorting
    for row in result_rows:
        # Ensure all values are non-negative integers
        for bucket in ["current", "0_30", "31_60", "61_90", "90_plus"]:
            row["aging"][bucket] = float(max(0, int(round(row["aging"][bucket]))))
    
    # CRITICAL: Final enforcement of exact row totals
    result_rows = _force_exact_row_totals(result_rows)
    
    # Sort to match previous month's order; append new descriptions at bottom
    # Preserve insertion order among new descriptions
    insertion_pos = {r["description"]: i for i, r in enumerate(result_rows)}
    def _order_key(r: Dict[str, Any]):
        desc = r["description"]
        if desc in last_order_map:
            return (0, last_order_map[desc])
        return (1, insertion_pos[desc])
    result_rows.sort(key=_order_key)
    
    # Don't add grand totals row here - let frontend handle totals display
    # Grand totals will be added only when saving to CSV
    return result_rows


def generate_next_month(
    classifier_path: str,
    regressor_path: str,
    last_month_records: List[Dict[str, Any]],
    last_month_str: str,
    next_month_str: str,
    target_total: float,
    history_df: Optional[pd.DataFrame] = None,
    overrides: Optional[Dict[str, float]] = None,
    carry_threshold: float = 0.2,
) -> List[Dict[str, Any]]:
    clf, regr = _load_models(classifier_path, regressor_path)
    return _generate_next_month_core(
        clf,
        regr,
        last_month_records,
        last_month_str,
        next_month_str,
        target_total,
        history_df,
        overrides,
        carry_threshold,
    )


def generate_next_month_with_models(
    clf: Any,
    regr: Any,
    last_month_records: List[Dict[str, Any]],
    last_month_str: str,
    next_month_str: str,
    target_total: float,
    history_df: Optional[pd.DataFrame] = None,
    overrides: Optional[Dict[str, float]] = None,
    carry_threshold: float = 0.2,
) -> List[Dict[str, Any]]:
    return _generate_next_month_core(
        clf,
        regr,
        last_month_records,
        last_month_str,
        next_month_str,
        target_total,
        history_df,
        overrides,
        carry_threshold,
    )


def save_predictions_to_csv(predictions: List[Dict[str, Any]], output_path: str):
    """Save predictions to CSV file in the required format with grand totals row."""
    # Convert customer rows to DataFrame
    rows = []
    for pred in predictions:
        aging = pred['aging']
        rows.append({
            'Description': pred['description'],
            'Current': int(aging['current']),
            '0-30 Days': int(aging['0_30']),
            '31-60 Days': int(aging['31_60']),
            '61-90 Days': int(aging['61_90']),
            '90+ Days': int(aging['90_plus']),
            'Total': int(pred['total'])
        })
    
    df = pd.DataFrame(rows)
    
    # Add grand totals row for CSV only
    if len(df) > 0:
        grand_totals = {
            'Description': 'Total',
            'Current': df['Current'].sum(),
            '0-30 Days': df['0-30 Days'].sum(),
            '31-60 Days': df['31-60 Days'].sum(),
            '61-90 Days': df['61-90 Days'].sum(),
            '90+ Days': df['90+ Days'].sum(),
            'Total': df['Total'].sum()
        }
        
        # Add grand totals as the last row
        df = pd.concat([df, pd.DataFrame([grand_totals])], ignore_index=True)
    
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


# ------------------------------
# Main / Example usage
# ------------------------------


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    excel_path = os.path.join(base_dir, "Data_Cleaning", "SBS for Abdur reports till april 2025.xlsx")

    print(f"Parsing data from: {excel_path}")
    df = parse_data(excel_path)
    print(f"Parsed rows: {len(df)}; months: {sorted(df['month'].unique())[:3]} ... {sorted(df['month'].unique())[-3:]}")

    artifacts, metrics = train_model(df, client_name="SBS")
    print(f"Model trained. Version: {artifacts.version}")

    # Prepare generation for next month
    months = sorted(df["month"].unique())
    if not months:
        raise RuntimeError("No months parsed")
    last_month = months[-1]
    last_df = df[df["month"] == last_month]

    # EXACT TARGETS FROM YOUR REQUIREMENTS
    target_total = 3937823  # $3,937,823
    
    # Build last month records in the expected JSON-like format
    last_records = []
    for _, row in last_df.iterrows():
        last_records.append(
            {
                "description": row["description"],
                "month": last_month,
                "aging": {
                    "current": float(row["current"]),
                    "0_30": float(row["0_30"]),
                    "31_60": float(row["31_60"]),
                    "61_90": float(row["61_90"]),
                    "90_plus": float(row["90_plus"]),
                },
                "total": float(row["total"]),
                "predicted": False,
            }
        )

    # Compute next month string (September 2025 as requested)
    y, m = [int(p) for p in last_month.split("-")]
    next_month = "2025-09"

    generated = generate_next_month(
        artifacts.classifier_path,
        artifacts.regressor_path,
        last_month_records=last_records,
        last_month_str=last_month,
        next_month_str=next_month,
        target_total=target_total,
        history_df=df,
    )

    print(f"Generated {len(generated)} rows for {next_month}.")
    
    # Calculate final column totals for verification
    final_col_totals = {
        "0_30": sum(r["aging"]["0_30"] for r in generated),
        "31_60": sum(r["aging"]["31_60"] for r in generated),
        "61_90": sum(r["aging"]["61_90"] for r in generated),
        "90_plus": sum(r["aging"]["90_plus"] for r in generated)
    }
    
    final_total = sum(r['total'] for r in generated)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"0-30 Days: ${final_col_totals['0_30']:,.0f} (Target: $1,537,823)")
    print(f"31-60 Days: ${final_col_totals['31_60']:,.0f} (Target: $1,100,000)")
    print(f"61-90 Days: ${final_col_totals['61_90']:,.0f} (Target: $800,000)")
    print(f"90+ Days: ${final_col_totals['90_plus']:,.0f} (Target: $500,000)")
    print(f"Grand Total: ${final_total:,.0f} (Target: $3,937,823)")
    
    # Verify all targets
    targets_met = True
    if abs(final_col_totals['0_30'] - 1537823) > 1:
        print(f"✗ 0-30 Days target not met: difference = {final_col_totals['0_30'] - 1537823:,.0f}")
        targets_met = False
    if abs(final_col_totals['31_60'] - 1100000) > 1:
        print(f"✗ 31-60 Days target not met: difference = {final_col_totals['31_60'] - 1100000:,.0f}")
        targets_met = False
    if abs(final_col_totals['61_90'] - 800000) > 1:
        print(f"✗ 61-90 Days target not met: difference = {final_col_totals['61_90'] - 800000:,.0f}")
        targets_met = False
    if abs(final_col_totals['90_plus'] - 500000) > 1:
        print(f"✗ 90+ Days target not met: difference = {final_col_totals['90_plus'] - 500000:,.0f}")
        targets_met = False
    if abs(final_total - 3937823) > 1:
        print(f"✗ Grand Total target not met: difference = {final_total - 3937823:,.0f}")
        targets_met = False
    
    if targets_met:
        print("✓ ALL TARGETS MET EXACTLY!")
    else:
        print("✗ SOME TARGETS NOT MET")

    # Save to CSV
    csv_path = os.path.join(os.path.dirname(__file__), "ar_predictions_SBS_2025-09-12.csv")
    save_predictions_to_csv(generated, csv_path)
    print(f"Predictions saved to: {csv_path}")

    # Save to JSON for inspection
    out_path = os.path.join(os.path.dirname(__file__), f"generated_{next_month}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(generated, f, indent=2)
    print(f"Saved generated report to: {out_path}")


if __name__ == "__main__":
    main()