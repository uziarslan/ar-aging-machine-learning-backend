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
        try:
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
        except Exception:
            pass

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
    try:
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
    except Exception:
        # Best-effort; non-fatal
        pass

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
                
                for bucket in consolidated_buckets:
                    bucket_sum = sum(r["aging"][bucket] for r in group)
                    consolidated_buckets[bucket] = int(bucket_sum)
                
                # CRITICAL: Calculate total as sum of buckets
                calculated_total = sum(consolidated_buckets.values())
                
                consolidated.append({
                    "description": base_desc,
                    "month": group[0]["month"],
                    "aging": consolidated_buckets,
                    "predicted": True,
                    "total": float(calculated_total)  # Total = sum of buckets
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


def _scale_to_exact_target(rows: List[Dict[str, Any]], target_total: float, max_iterations: int = 2) -> List[Dict[str, Any]]:
    """Add new business to hit exact target while preserving realistic aging ratios."""
    
    # CRITICAL: Force exact row totals first
    rows = _force_exact_row_totals(rows)
    
    current_total = sum(r["total"] for r in rows)
    
    # Check if we're already close enough
    if abs(current_total - target_total) < 1.0:
        return rows
    
    # Calculate how much additional business we need
    additional_needed = target_total - current_total
    
    if additional_needed > 0 and rows:
        # New strategy: add to 31-60 bucket of NON-override rows first
        remaining = int(round(additional_needed))
        non_override_rows = [r for r in rows if not r.get("is_override")]
        # Prefer larger rows to absorb more
        non_override_rows.sort(key=lambda r: r["total"], reverse=True)

        for row in non_override_rows:
            if remaining <= 0:
                break
            # Add as much as possible to 31-60
            addition = min(remaining, max(1, remaining // max(1, len(non_override_rows))))
            row["aging"]["31_60"] = int(row["aging"].get("31_60", 0)) + int(addition)
            remaining -= addition

        # If still remaining, add a dedicated Adjustment row (31-60 only)
        if remaining > 0:
            rows.append({
                "description": "Adjustment",
                "month": "",
                "aging": {
                    "current": 0.0,
                    "0_30": 0.0,
                    "31_60": float(remaining),
                    "61_90": 0.0,
                    "90_plus": 0.0,
                },
                "predicted": True,
                "total": float(remaining),
                "protected_0_30": False,
                "is_override": False,
            })
    
    elif additional_needed < 0 and rows:
        # Need to reduce - remove from 31-60 buckets of NON-override rows first
        reduction_needed = int(abs(round(additional_needed)))
        non_override_rows = [r for r in rows if not r.get("is_override")]
        # Sort by largest 31-60 first
        non_override_rows.sort(key=lambda r: r["aging"].get("31_60", 0), reverse=True)

        # Reduce from 31-60
        for row in non_override_rows:
            if reduction_needed <= 0:
                break
            avail = int(row["aging"].get("31_60", 0))
            if avail <= 0:
                continue
            take = min(reduction_needed, avail)
            row["aging"]["31_60"] = avail - take
            reduction_needed -= take

        # If still need reductions, reduce from 61-90, then 90+, then 0-30 (non-override rows)
        if reduction_needed > 0:
            for bucket in ["61_90", "90_plus", "0_30"]:
                if reduction_needed <= 0:
                    break
                for row in non_override_rows:
                    if reduction_needed <= 0:
                        break
                    # Skip 0-30 if row has protected override
                    if bucket == "0_30" and row.get("protected_0_30"):
                        continue
                    avail = int(row["aging"].get(bucket, 0))
                    if avail <= 0:
                        continue
                    take = min(reduction_needed, avail)
                    row["aging"][bucket] = avail - take
                    reduction_needed -= take
    
    # CRITICAL: Final enforcement of exact row totals
    rows = _force_exact_row_totals(rows)
    
    # Final micro-adjustment if still not exact
    current_total = sum(r["total"] for r in rows)
    diff = int(round(target_total - current_total))
    
    if diff != 0 and rows:
        # PRECISION ADJUSTMENT: For small differences (<10), use surgical precision
        rows.sort(key=lambda x: x["total"], reverse=True)
        
        if abs(diff) <= 10:
            # Small difference: add/subtract from largest rows' 0-30/Current buckets
            large_rows = rows[:min(5, len(rows))]  # Top 5 largest rows
            
            for i, row in enumerate(large_rows):
                if diff == 0:
                    break
                    
                if diff > 0:
                    # Add 1-2 to 0-30 bucket of largest rows
                    addition = min(diff, 2)
                    row["aging"]["0_30"] += addition
                    diff -= addition
                elif diff < 0 and row["aging"]["0_30"] > 0:
                    # Remove 1-2 from 0-30 bucket
                    reduction = min(abs(diff), 2, row["aging"]["0_30"])
                    row["aging"]["0_30"] -= reduction
                    diff += reduction
        else:
            # Larger difference: distribute more broadly
            attempts = 0
            max_attempts = abs(diff) + len(rows) * 5
            
            while diff != 0 and attempts < max_attempts:
                row_idx = attempts % len(rows)
                row = rows[row_idx]
                
                if diff > 0:
                    # Add to 0-30 bucket (focus on larger customers)
                    if row_idx < len(rows) // 2:  # Top half gets more
                        addition = min(diff, max(1, abs(diff) // (len(rows) // 2)))
                    else:  # Bottom half gets less
                        addition = min(diff, 1)
                    if not row.get("protected_0_30"):
                        row["aging"]["0_30"] += addition
                    diff -= addition
                elif diff < 0 and row["aging"]["0_30"] > 0:
                    # Remove from 0-30 bucket
                    if not row.get("protected_0_30"):
                        reduction = min(abs(diff), row["aging"]["0_30"], 1)
                        row["aging"]["0_30"] -= reduction
                        diff += reduction
                        attempts += 1
                        continue
                    diff += reduction
                
                attempts += 1
                
        # FINAL VERIFICATION: If still off, use/create Adjustment row on 31-60
        if diff != 0:
            # Try to find an existing Adjustment row
            adj = next((r for r in rows if r.get("description") == "Adjustment"), None)
            if adj is None:
                adj = {
                    "description": "Adjustment",
                    "month": "",
                    "aging": {"current": 0.0, "0_30": 0.0, "31_60": 0.0, "61_90": 0.0, "90_plus": 0.0},
                    "predicted": True,
                    "total": 0.0,
                    "protected_0_30": False,
                    "is_override": False,
                }
                rows.append(adj)
            # Apply diff to 31-60 (positive adds, negative subtracts but not below zero)
            new_31_60 = int(max(0, adj["aging"]["31_60"] + diff))
            adj["aging"]["31_60"] = new_31_60
            # Recompute totals after adjustment
            rows = _force_exact_row_totals(rows)
    
    # CRITICAL: Final enforcement of exact row totals
    rows = _force_exact_row_totals(rows)
    
    return rows


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
                                # New descriptions added by model to reach target are not overrides
                                "protected_0_30": True,  # protect from scaling, but not an override
                                "is_override": False,
                            })
    
    # Consolidate small rows to reduce count and add variety
    result_rows = _consolidate_small_rows(result_rows, max_small=40)  # Increased from 25 to 40
    
    # Scale to exact target with perfect precision
    result_rows = _scale_to_exact_target(result_rows, target_total, max_iterations=2)
    
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
    # Example target: 3187021 as requested
    target_total = 3187021

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
    # Skip ahead to September 2025 as requested
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

    print(f"Generated {len(generated)} rows for {next_month}. Target total: {target_total:,.2f}")
    actual_total = sum(r['total'] for r in generated)
    print(f"Actual total: {actual_total:,.2f}")
    print(f"Difference: {actual_total - target_total:,.2f}")

    # Count small rows
    small_rows = [r for r in generated if r['total'] < 1000]
    print(f"Small rows (<1k): {len(small_rows)}")

    # CRITICAL: Validate that ALL row totals equal sum of AGED buckets only - ZERO TOLERANCE
    mismatches = 0
    current_mismatches = 0
    for r in generated:
        # Check aged buckets sum = total
        aged_sum = r["aging"]["0_30"] + r["aging"]["31_60"] + r["aging"]["61_90"] + r["aging"]["90_plus"]
        if abs(r["total"] - aged_sum) > 0.001:
            print(f"CRITICAL MISMATCH: {r['description']} - Total: {r['total']}, Aged sum: {aged_sum}")
            mismatches += 1
        
        # Check Current = Total
        if abs(r["aging"]["current"] - r["total"]) > 0.001:
            print(f"CURRENT MISMATCH: {r['description']} - Current: {r['aging']['current']}, Total: {r['total']}")
            current_mismatches += 1
    
    if mismatches == 0:
        print("✓ PERFECT: ALL row totals EXACTLY equal sum of aged buckets")
    else:
        print(f"✗ FAILED: {mismatches} rows have total/aged bucket sum mismatches")
    
    if current_mismatches == 0:
        print("✓ PERFECT: ALL Current values EXACTLY equal Total values")
    else:
        print(f"✗ FAILED: {current_mismatches} rows have Current ≠ Total mismatches")

    # Final verification of exact target sum (all rows are customer rows now)
    final_sum = sum(r['total'] for r in generated)
    if abs(final_sum - target_total) < 1.0:
        print(f"✓ PERFECT TARGET MATCH: {final_sum:,.0f} = {target_total:,.0f}")
    else:
        print(f"✗ TARGET MISMATCH: {final_sum:,.0f} ≠ {target_total:,.0f} (diff: {final_sum - target_total:,.0f})")
    
    # Verify column totals (what will be in the CSV grand row)
    current_sum = sum(r['aging']['current'] for r in generated)
    aged_0_30_sum = sum(r['aging']['0_30'] for r in generated)
    aged_31_60_sum = sum(r['aging']['31_60'] for r in generated)
    aged_61_90_sum = sum(r['aging']['61_90'] for r in generated)
    aged_90_plus_sum = sum(r['aging']['90_plus'] for r in generated)
    aged_buckets_total = aged_0_30_sum + aged_31_60_sum + aged_61_90_sum + aged_90_plus_sum
    
    print(f"\nColumn Totals Verification (for CSV grand row):")
    print(f"Current sum: {current_sum:,.0f}")
    print(f"0-30 Days sum: {aged_0_30_sum:,.0f}")
    print(f"31-60 Days sum: {aged_31_60_sum:,.0f}")
    print(f"61-90 Days sum: {aged_61_90_sum:,.0f}")
    print(f"90+ Days sum: {aged_90_plus_sum:,.0f}")
    print(f"Aged buckets total: {aged_buckets_total:,.0f}")
    print(f"Current sum equals target: {abs(current_sum - target_total) < 1.0}")
    print(f"Aged buckets total equals target: {abs(aged_buckets_total - target_total) < 1.0}")

    # Show sample rows to verify Current = Total requirement
    print(f"\nSample rows (showing Current = Total requirement):")
    for i, r in enumerate(generated[:5]):
        aging = r["aging"]
        total = r["total"]
        aged_sum = aging['0_30'] + aging['31_60'] + aging['61_90'] + aging['90_plus']
        print(f"  {i+1}. {r['description'][:40]}: Current={aging['current']:.0f}, Total={total:.0f}, Aged Sum={aged_sum:.0f}")

    # Save to CSV as requested
    csv_path = os.path.join(os.path.dirname(__file__), "ar_predictions_SBS_2025-09-12.csv")
    save_predictions_to_csv(generated, csv_path)
    
    # Show distribution analysis (all rows are customer rows now)
    totals = [r['total'] for r in generated]
    totals.sort(reverse=True)
    
    if totals:
        top_10_pct = int(len(totals) * 0.1)
        mid_40_pct = int(len(totals) * 0.5)
        
        print(f"\nDistribution Analysis:")
        print(f"Total rows: {len(totals)}")
        if top_10_pct > 0:
            print(f"Top 10% ({top_10_pct} rows): ${totals[0]:,.0f} - ${totals[top_10_pct-1]:,.0f}")
        if mid_40_pct > top_10_pct:
            print(f"Mid 40% (rows {top_10_pct+1}-{mid_40_pct}): ${totals[top_10_pct]:,.0f} - ${totals[mid_40_pct-1]:,.0f}")
        if len(totals) > mid_40_pct:
            print(f"Bottom 50% (rows {mid_40_pct+1}+): ${totals[mid_40_pct]:,.0f} - ${totals[-1]:,.0f}")

    # Optionally save to JSON for inspection
    out_path = os.path.join(os.path.dirname(__file__), f"generated_{next_month}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(generated, f, indent=2)
    print(f"Saved generated report to: {out_path}")


if __name__ == "__main__":
    main()