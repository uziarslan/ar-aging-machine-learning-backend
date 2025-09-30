"""
Integrated FastAPI Backend with Embedded ML Pipeline

This backend integrates the ML pipeline directly into the FastAPI application,
supporting multi-client models, CSV uploads, and auto-retraining.
"""

import os
import sys
import json
import logging
import asyncio
import pandas as pd
import numpy as np
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4
from io import StringIO, BytesIO

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from pymongo import MongoClient, UpdateOne
from bson import ObjectId
from bson.binary import Binary
from gridfs import GridFS
from scipy.optimize import minimize
from train_generate_ar import train_model as new_train_model
from train_generate_ar import generate_next_month as new_generate_next_month
from train_generate_ar import generate_next_month_with_models
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AR Aging ML API",
    description="Integrated API for AR aging predictions with embedded ML pipeline",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
mongo_client = None
mongo_db = None
client_models = {}  # Deprecated cache; kept for backward compatibility but unused

# ------------------------------- Data Cleaning Functions ------------------------------- #

def _normalize_header(text: str) -> str:
    """Normalize a column header for robust matching."""
    if text is None:
        return ""
    normalized = str(text).strip().lower()
    # Replace special dashes and spaces with single hyphen/space for matching
    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized

def _identify_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Identify likely columns for description, buckets, and total."""
    header_map = {col: _normalize_header(col) for col in df.columns}

    # Potential synonyms
    description_candidates = [
        "description", "desc", "customer", "customer name", "account", "account name",
        "name", "invoice", "detail", "department", "project", "category",
    ]

    bucket_patterns = {
        "b_current": [r"^current$", r"^curr$"],
        "b_0_30": [r"^(0\s*[-to]+\s*30)$", r"^0[-_]?30$", r"^0\s*to\s*30$", r"^1\s*[-to]+\s*30$", r"^1[-_]?30$"],
        "b_31_60": [r"^(31\s*[-to]+\s*60)$", r"^31[-_]?60$", r"^31\s*to\s*60$"],
        "b_61_90": [r"^(61\s*[-to]+\s*90)$", r"^61[-_]?90$", r"^61\s*to\s*90$"],
        "b_90_plus": [r"^90\+?$", r"^90\s*\+$", r"^90\s*plus$", r"^over\s*90$", r"^>\s*90$", r"^greater\s*than\s*90$", r"^91\+?$", r"^91\s*and\s*over$"],
    }

    total_synonyms = [
        "total", "grand total", "balance", "total balance", "overall total", "amount total",
    ]

    mapping: Dict[str, Optional[str]] = {
        "description": None,
        "b_current": None,
        "b_0_30": None,
        "b_31_60": None,
        "b_61_90": None,
        "b_90_plus": None,
        "total": None,
    }

    # Identify description
    for orig, norm in header_map.items():
        if norm in description_candidates:
            mapping["description"] = orig
            break

    # If not found, choose the first non-numeric-heavy column as description
    if mapping["description"] is None:
        for orig, norm in header_map.items():
            # Heuristic: headers that don't look like numbers or bucket labels
            if not re.search(r"\d", norm) and norm not in ["total", "balance", "amount"]:
                mapping["description"] = orig
                break

    # Identify bucket columns
    for orig, norm in header_map.items():
        for key, patterns in bucket_patterns.items():
            if mapping[key] is not None:
                continue
            if any(re.match(p, norm) for p in patterns):
                mapping[key] = orig
                break

    # Identify total
    for orig, norm in header_map.items():
        if norm in total_synonyms:
            mapping["total"] = orig
            break

    return mapping

def _to_number(value) -> Optional[float]:
    """Convert common AR cell formats to float."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s == "" or s.lower() in {"na", "n/a", "nan", "none"}:
        return None
    # Treat plain dash as zero
    if s in {"-", "—", "–"}:
        return 0.0
    # Remove currency symbols and thousands separators
    s = s.replace("$", "").replace(",", "")
    # Parentheses indicate negatives
    s = s.replace("(", "-").replace(")", "")
    # Remove stray spaces
    s = re.sub(r"\s+", "", s)
    try:
        return float(s)
    except ValueError:
        # Last attempt: extract the first numeric-like pattern
        match = re.search(r"-?\d+(?:\.\d+)?", s)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
        return None

def _clean_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame into canonical columns."""
    # Drop fully empty columns/rows to minimize noise
    df = df.copy()
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        return pd.DataFrame(columns=[
            "description", "bucket_current", "bucket_0_30", "bucket_31_60", "bucket_61_90", "bucket_90_plus", "total"
        ])

    mapping = _identify_columns(df)

    # If we still do not have a description, attempt to use the first column
    description_col = mapping.get("description") or df.columns[0]
    c_current = mapping.get("b_current")
    c_0_30 = mapping.get("b_0_30")
    c_31_60 = mapping.get("b_31_60")
    c_61_90 = mapping.get("b_61_90")
    c_90_plus = mapping.get("b_90_plus")
    c_total = mapping.get("total")

    standardized = pd.DataFrame({
        "description": df[description_col].astype(str).str.strip()
    })

    # Numeric conversions with fallbacks to 0
    def to_numeric_series(series_like) -> pd.Series:
        if series_like is None or series_like not in df.columns:
            return pd.Series([0.0] * len(df))
        return df[series_like].apply(_to_number).fillna(0.0).astype(float).round().astype(int)

    standardized["bucket_current"] = to_numeric_series(c_current)
    standardized["bucket_0_30"] = to_numeric_series(c_0_30)
    standardized["bucket_31_60"] = to_numeric_series(c_31_60)
    standardized["bucket_61_90"] = to_numeric_series(c_61_90)
    standardized["bucket_90_plus"] = to_numeric_series(c_90_plus)

    if c_total is not None and c_total in df.columns:
        total_series = df[c_total].apply(_to_number).fillna(0.0).astype(float).round().astype(int)
    else:
        total_series = (
            standardized["bucket_current"]
            + standardized["bucket_0_30"]
            + standardized["bucket_31_60"]
            + standardized["bucket_61_90"]
            + standardized["bucket_90_plus"]
        ).round().astype(int)
    standardized["total"] = total_series

    # Clean description, remove rows where description is missing/blank
    standardized["description"] = standardized["description"].fillna("").astype(str).str.strip()
    before_desc_drop = len(standardized)
    standardized = standardized[standardized["description"] != ""]
    dropped_blank_desc = before_desc_drop - len(standardized)

    # Remove obvious total/subtotal summary rows
    lower_desc = standardized["description"].str.lower()
    drop_values = {"total", "totals", "subtotal", "sub total", "grand total"}
    before_totals_drop = len(standardized)
    standardized = standardized[~lower_desc.isin(drop_values)]
    dropped_totals = before_totals_drop - len(standardized)

    return standardized.reset_index(drop=True)

def _parse_month_from_sheet_name(sheet_name: str) -> str:
    """Parse month from sheet name. Returns 'YYYY-MM' string."""
    if sheet_name is None or str(sheet_name).strip() == "":
        now = datetime.now(timezone.utc)
        return f"{now.year:04d}-{now.month:02d}"

    text = str(sheet_name).strip()
    cleaned = text.replace("_", " ").replace("/", " ")

    # Month mapping
    month_map = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    lower = cleaned.lower()
    year_match = re.search(r"(20\d{2}|19\d{2})", lower)
    month_num = None
    for key, idx in month_map.items():
        if re.search(rf"\b{key}\b", lower):
            month_num = idx
            break
    if year_match and month_num:
        return f"{int(year_match.group(0)):04d}-{month_num:02d}"

    # Try simple YYYY-MM in name
    m = re.search(r"(20\d{2})[- ](\d{1,2})", lower)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"

    # Last resort: pandas loose parse
    try:
        ts = pd.to_datetime(cleaned, errors="raise")
        return f"{ts.year:04d}-{ts.month:02d}"
    except Exception:
        pass

    # Final fallback: current month
    now = datetime.now(timezone.utc)
    return f"{now.year:04d}-{now.month:02d}"

def _normalize_cell_value(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return _normalize_header(str(value))

def _make_unique_columns(columns: List[str]) -> List[str]:
    """Ensure DataFrame columns are unique by appending suffixes to duplicates."""
    seen: Dict[str, int] = {}
    unique_cols: List[str] = []
    for col in columns:
        base = col or ""
        count = seen.get(base, 0)
        if count == 0 and base not in unique_cols:
            unique_cols.append(base)
            seen[base] = 1
        else:
            new_col = f"{base}.{count}"
            while new_col in seen:
                count += 1
                new_col = f"{base}.{count}"
            unique_cols.append(new_col)
            seen[base] = count + 1
    return unique_cols

def _detect_header_and_reframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[int]]:
    """Scan the first N rows to find a header row containing description and bucket labels."""
    if df.empty:
        return df, None

    max_scan_rows = min(len(df), 40)

    description_candidates = {
        "description", "desc", "customer", "customer name", "account", "account name",
        "name", "invoice", "detail", "department", "project", "category",
    }
    bucket_regexes = [
        re.compile(r"^(0\s*[-to]+\s*30)$"),
        re.compile(r"^0[-_]?30$"),
        re.compile(r"^current$"),
        re.compile(r"^1\s*[-to]+\s*30$"),
        re.compile(r"^1[-_]?30$"),
        re.compile(r"^(31\s*[-to]+\s*60)$"),
        re.compile(r"^31[-_]?60$"),
        re.compile(r"^(61\s*[-to]+\s*90)$"),
        re.compile(r"^61[-_]?90$"),
        re.compile(r"^90\+?$"),
        re.compile(r"^90\s*\+$"),
        re.compile(r"^90\s*plus$"),
        re.compile(r"^over\s*90$"),
        re.compile(r"^>\s*90$"),
        re.compile(r"^greater\s*than\s*90$"),
        re.compile(r"^91\+?$"),
        re.compile(r"^total$"),
        re.compile(r"^grand\s*total$"),
    ]

    header_idx: Optional[int] = None
    
    for idx in range(max_scan_rows):
        row = df.iloc[idx].tolist()
        norm_cells = [_normalize_cell_value(v) for v in row]
        if not any(norm_cells):
            continue
        has_description = any(cell in description_candidates for cell in norm_cells)
        bucket_hits = sum(1 for cell in norm_cells if any(rx.match(cell) for rx in bucket_regexes))
        # Be permissive: description + at least one bucket-like label
        if has_description and bucket_hits >= 1:
            header_idx = idx
            break

    if header_idx is None:
        # Fallback: if a row contains 'description' anywhere, use it
        for idx in range(max_scan_rows):
            cells = [_normalize_cell_value(v) for v in df.iloc[idx].tolist()]
            if any(c == "description" for c in cells):
                header_idx = idx
                break
        # Final fallback: first non-empty row
        if header_idx is None:
            for idx in range(max_scan_rows):
                if any(pd.notna(x) and str(x).strip() != "" for x in df.iloc[idx].tolist()):
                    header_idx = idx
                    break

    if header_idx is None:
        return df, None

    # Set columns and trim rows above header
    header_values = [str(v).strip() if pd.notna(v) else "" for v in df.iloc[header_idx].tolist()]
    trimmed = df.iloc[header_idx + 1 :].copy()
    # Ensure 2D shape
    if trimmed.ndim == 1:
        trimmed = trimmed.to_frame()
    # Ensure column length matches
    num_cols = int(trimmed.shape[1]) if hasattr(trimmed, "shape") else len(header_values)
    if num_cols <= 0:
        return pd.DataFrame(columns=_make_unique_columns(header_values)), header_idx
    if len(header_values) < num_cols:
        header_values.extend([f"col_{i}" for i in range(len(header_values), num_cols)])
    cols = header_values[: num_cols]
    trimmed.columns = _make_unique_columns(cols)
    return trimmed, header_idx

def _read_all_sheets(file_contents: bytes) -> Dict[str, pd.DataFrame]:
    """Read all sheets from the given Excel file contents."""
    # Read raw with header=None so we can detect header rows within each sheet robustly.
    sheets: Dict[str, pd.DataFrame] = pd.read_excel(BytesIO(file_contents), sheet_name=None, header=None, dtype=object)
    return sheets

def _prepare_documents(df: pd.DataFrame, client_id: ObjectId, month_str: str) -> List[dict]:
    """Prepare documents for MongoDB insertion."""
    now = datetime.now(timezone.utc)
    docs: List[dict] = []
    for _, row in df.iterrows():
        doc = {
            "client_id": client_id,
            "month": month_str,  # 'YYYY-MM'
            "description": str(row["description"]).strip(),
            "aging": {
                "current": float(row.get("bucket_current", 0.0)),
                "0_30": float(row["bucket_0_30"]),
                "31_60": float(row["bucket_31_60"]),
                "61_90": float(row["bucket_61_90"]),
                "90_plus": float(row["bucket_90_plus"]),
            },
            "total": float(row["total"]),
            "predicted": False,
            "created_at": now,
        }
        docs.append(doc)
    return docs

def _bulk_upsert_ar_data(ar_data_collection, documents: List[dict]) -> Tuple[int, int]:
    """Upsert each document in the ar_data collection."""
    if not documents:
        return 0, 0
    operations: List[UpdateOne] = []
    for doc in documents:
        filter_key = {
            "client_id": doc["client_id"],
            "month": doc["month"],
            "description": doc["description"],
        }
        update_doc = {
            "$set": {
                "aging": doc["aging"],
                "total": doc["total"],
                "predicted": doc["predicted"],
            },
            "$setOnInsert": {
                "client_id": doc["client_id"],
                "month": doc["month"],
                "description": doc["description"],
                "created_at": doc["created_at"],
            },
        }
        operations.append(UpdateOne(filter_key, update_doc, upsert=True))

    result = ar_data_collection.bulk_write(operations, ordered=False)
    matched_or_upserted = (result.matched_count or 0) + (len(result.upserted_ids) if result.upserted_ids else 0)
    upserted = len(result.upserted_ids) if result.upserted_ids else 0
    return matched_or_upserted, upserted

def _get_or_create_client(clients_collection, client_name: str) -> ObjectId:
    """Find or create a client document."""
    existing = clients_collection.find_one({"name": {"$regex": f"^{re.escape(client_name)}$", "$options": "i"}})
    if existing:
        return existing["_id"]

    now = datetime.now(timezone.utc)
    inserted = clients_collection.insert_one({
        "name": client_name,
        "created_at": now,
    })
    return inserted.inserted_id

# Pydantic models
class PredictionRequest(BaseModel):
    client_id: str = Field(..., description="Client ID to predict for")
    target_month: str = Field(..., description="Target month in YYYY-MM format")
    target_total: float = Field(..., gt=0, description="Target total amount")
    carry_threshold: float = Field(0.2, ge=0, le=1, description="Carry-forward threshold")
    # New: optional additional entries that should be deducted from target_total
    additional_entries: Optional[List[Dict[str, Any]]] = Field(None, description="List of additional entries: {description, amount}")
    # New: column-specific targets from frontend (optional)
    column_targets: Optional[Dict[str, float]] = Field(None, description="Column targets: {b0_30, b31_60, b61_90, b90_plus}")

    @validator("additional_entries", pre=True, always=True)
    def _normalize_additional_entries(cls, v):
        if not v:
            return []
        # Accept list of dicts; filter invalid
        normalized: List[Dict[str, Any]] = []
        for item in v:
            try:
                desc = str(item.get("description", "")).strip()
                amt = float(item.get("amount", 0))
                if desc and amt > 0:
                    normalized.append({"description": desc, "amount": amt})
            except Exception:
                continue
        return normalized

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    model_meta: Dict[str, Any]
    grand_total: float
    warnings: List[str]
    audit_snapshot_id: str

class ApproveRequest(BaseModel):
    client_id: str
    target_month: str
    predictions: List[Dict[str, Any]]
    model_version: str
    retrain: bool = False
    comment: Optional[str] = None

class ApproveResponse(BaseModel):
    success: bool
    job_id: Optional[str] = None
    message: str

class ClientInfo(BaseModel):
    id: str
    name: str
    last_prediction: Optional[str] = None
    model_version: Optional[str] = None
    has_model: bool = False

class TrainJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    metrics: Optional[Dict[str, Any]] = None

class UploadResponse(BaseModel):
    success: bool
    client_id: str
    records_processed: int
    message: str
    clients: List[ClientInfo]  # Include updated client list

# Dependency for API key validation
async def verify_api_key(x_api_key: str = Header(None)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=500, detail="Admin API key not configured")
    
    if x_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return x_api_key

"""
Old embedded ML classes removed. We now rely on train_generate_ar.train_model and generate_next_month.
"""

# Environment configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/ar_aging")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "demo-api-key-123")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Startup event
@app.on_event("startup")
async def startup_event():
    global mongo_client, mongo_db
    
    # Connect to MongoDB
    mongo_client = MongoClient(MONGO_URI)
    mongo_db = mongo_client.get_default_database()
    
    logger.info(f"Integrated backend initialized successfully")
    logger.info(f"MongoDB URI: {MONGO_URI}")
    logger.info(f"Host: {HOST}, Port: {PORT}")
    logger.info(f"Debug mode: {DEBUG}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    if mongo_client:
        mongo_client.close()

# API Endpoints

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/clients", response_model=List[ClientInfo])
async def get_clients():
    """Get list of clients with their information"""
    try:
        clients_collection = mongo_db["clients"]
        ar_data_collection = mongo_db["ar_data"]
        models_collection = mongo_db["models"]
        
        # Get all clients
        clients = list(clients_collection.find({}, {"_id": 1, "name": 1}))
        
        client_info = []
        for client in clients:
            client_id = str(client["_id"])
            
            # Find last prediction for this client
            last_prediction = ar_data_collection.find_one(
                {"client_id": ObjectId(client_id), "predicted": True},
                sort=[("month", -1)]
            )
            
            # Check if client has a trained model
            model_doc = models_collection.find_one({"client_id": ObjectId(client_id)})
            has_model = model_doc is not None
            model_version = model_doc.get("version") if model_doc else None
            
            client_info.append(ClientInfo(
                id=client_id,
                name=client["name"],
                last_prediction=last_prediction["month"] if last_prediction else None,
                model_version=model_version,
                has_model=has_model
            ))
        
        return client_info
        
    except Exception as e:
        logger.error(f"Error fetching clients: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch clients")

@app.post("/api/upload", response_model=UploadResponse, dependencies=[Depends(verify_api_key)])
async def upload_client_data(
    file: UploadFile = File(...),
    client_name: str = Form(...),
    description: str = Form("")
):
    """Upload Excel data for a new client and train models using data cleaning pipeline"""
    try:
        # Read file contents
        contents = await file.read()
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension not in ['xlsx', 'xls']:
            raise HTTPException(status_code=400, detail="Only Excel files (.xlsx/.xls) are supported. Please upload an Excel file with AR aging data.")
        
        # Use data cleaning pipeline to process all sheets
        sheets = _read_all_sheets(contents)
        all_documents = []
        
        # Create or get client
        clients_collection = mongo_db["clients"]
        client_id = _get_or_create_client(clients_collection, client_name)
        
        # Process each sheet
        total_raw_rows = 0
        total_standardized_rows = 0
        processed_sheets = 0
        
        for sheet_name, sheet_df in sheets.items():
            # Detect header row and set proper columns first
            reframed_df, header_idx = _detect_header_and_reframe(sheet_df)
            # Parse month from sheet name
            month_str = _parse_month_from_sheet_name(sheet_name)
            standardized = _clean_and_standardize(reframed_df)
            
            if standardized.empty:
                continue
                
            processed_sheets += 1
            total_raw_rows += len(reframed_df)
            total_standardized_rows += len(standardized)
            
            # Prepare documents for this sheet
            docs = _prepare_documents(standardized, client_id, month_str)
            all_documents.extend(docs)
        
        if not all_documents:
            raise HTTPException(status_code=400, detail="No valid data found in any sheet. Please check that your Excel file contains AR aging data with proper headers.")
        
        # Insert all data using bulk upsert
        ar_data_collection = mongo_db["ar_data"]
        matched_or_upserted, upserted = _bulk_upsert_ar_data(ar_data_collection, all_documents)
        
        # Train models using the new pipeline from train_generate_ar
        # Build DataFrame of historical data for this upload
        df_training = pd.DataFrame([
            {
                'month': d['month'],
                'description': d['description'],
                'current': d['aging'].get('current', 0.0),
                '0_30': d['aging'].get('0_30', 0.0),
                '31_60': d['aging'].get('31_60', 0.0),
                '61_90': d['aging'].get('61_90', 0.0),
                '90_plus': d['aging'].get('90_plus', 0.0),
                'total': d['total'],
            } for d in all_documents
        ])

        # Remove any previous model records and artifacts for a clean test
        models_collection = mongo_db["models"]
        models_collection.delete_many({"client_id": client_id})
        # Best-effort: remove prior GridFS files for this client
        try:
            from gridfs import GridFSBucket
            bucket = GridFSBucket(mongo_db)
            existing_files = list(bucket.find({"metadata.client_id": str(client_id)}))
            for file_doc in existing_files:
                try:
                    bucket.delete(file_doc._id)
                except Exception:
                    pass
        except Exception:
            pass

        # Remove old artifact files for this client (cache cleanup)
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(models_dir, exist_ok=True)
        safe_client = re.sub(r"[^a-z0-9]+", "_", client_name.lower()).strip("_")
        try:
            for fname in os.listdir(models_dir):
                if fname.startswith(f"{safe_client}_"):
                    try:
                        os.remove(os.path.join(models_dir, fname))
                    except Exception:
                        pass
        except Exception:
            pass

        artifacts, metrics = new_train_model(df_training, client_name=client_name, models_dir=models_dir)
        # Save model artifacts to GridFS using GridFSBucket for consistency
        from gridfs import GridFSBucket
        bucket = GridFSBucket(mongo_db)
        
        with open(artifacts.classifier_path, "rb") as f:
            classifier_file_id = bucket.upload_from_stream(
                f"classifier_{client_id}_{artifacts.version}",
                f,
                metadata={"client_id": str(client_id), "model_type": "classifier", "version": artifacts.version}
            )
        with open(artifacts.regressor_path, "rb") as f:
            regressor_file_id = bucket.upload_from_stream(
                f"regressor_{client_id}_{artifacts.version}",
                f,
                metadata={"client_id": str(client_id), "model_type": "regressor", "version": artifacts.version}
            )

        model_doc = {
            "client_id": client_id,
            "version": artifacts.version,
            "metrics": metrics,
            "created_at": datetime.now(),
            "status": "trained",
            "gridfs_files": {
                "carry_classifier": classifier_file_id,
                "bucket_regressor": regressor_file_id,
            },
        }
        models_collection.insert_one(model_doc)
        
        # Remove artifact files from disk
        try:
            for p in [artifacts.classifier_path, artifacts.regressor_path, artifacts.registry_path]:
                if p and os.path.exists(p):
                    os.remove(p)
        except Exception:
            pass
        
        logger.info(f"Uploaded and trained model for client {client_name} ({client_id}) - {processed_sheets} sheets, {total_standardized_rows} records")
        
        # Get updated client list to return to frontend
        updated_clients = []
        clients = list(clients_collection.find({}, {"_id": 1, "name": 1}))
        ar_data_collection = mongo_db["ar_data"]
        
        for client in clients:
            client_obj_id = client["_id"]
            
            # Find last prediction for this client
            last_prediction = ar_data_collection.find_one(
                {"client_id": client_obj_id, "predicted": True},
                sort=[("month", -1)]
            )
            
            # Check if client has a trained model
            model_doc = models_collection.find_one({"client_id": client_obj_id})
            has_model = model_doc is not None
            model_version = model_doc.get("version") if model_doc else None
            
            updated_clients.append(ClientInfo(
                id=str(client_obj_id),
                name=client["name"],
                last_prediction=last_prediction["month"] if last_prediction else None,
                model_version=model_version,
                has_model=has_model
            ))
        
        return UploadResponse(
            success=True,
            client_id=str(client_id),
            records_processed=total_standardized_rows,
            message=f"Successfully processed {processed_sheets} sheets with {total_standardized_rows} records and trained model",
            clients=updated_clients
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_aging(request: PredictionRequest):
    """Generate AR aging predictions for a client"""
    try:
        # Validate month format
        try:
            datetime.strptime(request.target_month, "%Y-%m")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid month format. Use YYYY-MM")
        
        # Load latest model doc and GridFS ids
        models_collection = mongo_db["models"]
        model_doc = models_collection.find_one({"client_id": ObjectId(request.client_id)}, sort=[("created_at", -1)])
        if not model_doc or ("gridfs" not in model_doc and "gridfs_files" not in model_doc):
            raise HTTPException(status_code=404, detail="No trained model found for this client")
        
        # Get ALL data for the client (historical + approved predictions)
        ar_data_collection = mongo_db["ar_data"]
        historical_data = list(ar_data_collection.find(
            {"client_id": ObjectId(request.client_id)},
            {"description": 1, "month": 1, "aging": 1, "total": 1}
        ))
        
        if not historical_data:
            raise HTTPException(status_code=404, detail="No data found for this client")
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        df['b0_30'] = df['aging'].apply(lambda x: x['0_30'])
        df['b31_60'] = df['aging'].apply(lambda x: x['31_60'])
        df['b61_90'] = df['aging'].apply(lambda x: x['61_90'])
        df['b90p'] = df['aging'].apply(lambda x: x['90_plus'])
        
        # Use the new generator: find last month and build last_month_records
        tgt_year, tgt_month = [int(x) for x in request.target_month.split('-')]
        prev_year = tgt_year if tgt_month > 1 else tgt_year - 1
        prev_month = tgt_month - 1 if tgt_month > 1 else 12
        prev_month_str = f"{prev_year:04d}-{prev_month:02d}"

        ar_data_collection = mongo_db["ar_data"]
        # Include both historical and approved predictions for previous month
        prev_docs = list(ar_data_collection.find(
            {"client_id": ObjectId(request.client_id), "month": prev_month_str},
            {"description": 1, "aging": 1, "total": 1}
        ))
        if not prev_docs:
            raise HTTPException(status_code=404, detail=f"No data found for previous month {prev_month_str}")

        # Convert docs to last_month_records format
        last_records = []
        for d in prev_docs:
            last_records.append({
                "description": d["description"],
                "month": prev_month_str,
                "aging": {
                    "current": float(d["aging"].get("current", 0.0)),
                    "0_30": float(d["aging"].get("0_30", 0.0)),
                    "31_60": float(d["aging"].get("31_60", 0.0)),
                    "61_90": float(d["aging"].get("61_90", 0.0)),
                    "90_plus": float(d["aging"].get("90_plus", 0.0)),
                },
                "total": float(d.get("total", 0.0)),
                "predicted": False,
            })

        # Build history df (all historical rows for this client, non-predicted)
        historical_data = list(ar_data_collection.find(
            {"client_id": ObjectId(request.client_id), "predicted": False},
            {"description": 1, "month": 1, "aging": 1, "total": 1}
        ))
        df_hist = pd.DataFrame([
            {
                "description": r["description"],
                "month": r["month"],
                "current": float(r["aging"].get("current", 0.0)),
                "0_30": float(r["aging"].get("0_30", 0.0)),
                "31_60": float(r["aging"].get("31_60", 0.0)),
                "61_90": float(r["aging"].get("61_90", 0.0)),
                "90_plus": float(r["aging"].get("90_plus", 0.0)),
                "total": float(r.get("total", 0.0)),
            } for r in historical_data
        ])

        # Load models from GridFS (support both old and new format)
        import pickle
        try:
            # Handle both old and new GridFS formats
            if "gridfs_files" in model_doc:
                # New format using GridFSBucket
                from gridfs import GridFSBucket
                bucket = GridFSBucket(mongo_db)
                clf_file_id = model_doc["gridfs_files"]["carry_classifier"]
                regr_file_id = model_doc["gridfs_files"]["bucket_regressor"]
                clf_bytes = bucket.open_download_stream(clf_file_id).read()
                regr_bytes = bucket.open_download_stream(regr_file_id).read()
            else:
                # Old format using GridFS
                fs = GridFS(mongo_db)
                clf_bytes = fs.get(model_doc["gridfs"]["classifier_file_id"]).read()
                regr_bytes = fs.get(model_doc["gridfs"]["regressor_file_id"]).read()
            
            clf = pickle.loads(clf_bytes)
            regr = pickle.loads(regr_bytes)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load models from database: {e}")

        # Partition additional entries into those matching last month's clients vs new clients
        def _norm_desc(s: str) -> str:
            return (s or "").strip().lower()

        # Use the same cleaning function as the generator for consistent matching
        from train_generate_ar import _clean_description
        # Normalize case for matching by converting to lowercase after cleaning
        prev_desc_set = {_clean_description(d.get("description", "")).lower() for d in prev_docs}
        
        add_entries_all = request.additional_entries or []
        matched_add_entries: List[Dict[str, Any]] = []
        new_add_entries: List[Dict[str, Any]] = []
        for e in add_entries_all:
            try:
                amt = float(e.get("amount", 0) or 0)
            except Exception:
                amt = 0.0
            if amt <= 0:
                continue
            desc_clean = _clean_description(str(e.get("description", ""))).lower()
            if desc_clean in prev_desc_set:
                matched_add_entries.append({"description": e.get("description", ""), "amount": amt})
            else:
                new_add_entries.append({"description": e.get("description", ""), "amount": amt})


        # Don't deduct user overrides from target - let generator produce full target
        # We'll override specific rows after generation
        # Prefer column_targets sum if provided
        effective_target_total = float(request.target_total)
        column_targets_payload = request.column_targets or {}
        if column_targets_payload:
            ct_sum = float(column_targets_payload.get("b0_30", 0) + column_targets_payload.get("b31_60", 0) + column_targets_payload.get("b61_90", 0) + column_targets_payload.get("b90_plus", 0))
            if ct_sum > 0:
                effective_target_total = ct_sum
        warnings: List[str] = []

        # Generate predictions without overrides - we'll apply them manually after
        predictions_list = generate_next_month_with_models(
            clf=clf,
            regr=regr,
            last_month_records=last_records,
            last_month_str=prev_month_str,
            next_month_str=request.target_month,
            target_total=effective_target_total,
            history_df=df_hist,
            overrides=None,  # No overrides to generator
            carry_threshold=float(request.carry_threshold or 0.2),
        )

        # Flatten to legacy schema expected by frontend: top-level bucket keys
        flat_predictions: List[Dict[str, Any]] = []
        for p in predictions_list:
            aging = p.get("aging", {})
            flat_predictions.append({
                "description": p.get("description", ""),
                "current": float(aging.get("current", 0.0)),
                "0_30": float(aging.get("0_30", 0.0)),
                "31_60": float(aging.get("31_60", 0.0)),
                "61_90": float(aging.get("61_90", 0.0)),
                "90_plus": float(aging.get("90_plus", 0.0)),
                "total": float(p.get("total", 0.0)),
                # Expose override flag for frontend logic
                "is_override": bool(p.get("is_override", False)),
                # Preserve auto_generated marker from generator for downstream logic
                "auto_generated": bool(p.get("auto_generated", False)),
            })

        # Fallback: if predictions are empty or all-zero buckets, build a DB-only roll-forward
        if not flat_predictions or all(
            (p.get("0_30", 0) == 0 and p.get("31_60", 0) == 0 and p.get("61_90", 0) == 0 and p.get("90_plus", 0) == 0)
            for p in flat_predictions
        ):
            # Build from previous month docs deterministically
            legacy_rows: List[Dict[str, Any]] = []
            for d in prev_docs:
                prev_0_30 = float(d["aging"].get("0_30", 0.0))
                prev_31_60 = float(d["aging"].get("31_60", 0.0))
                prev_61_90 = float(d["aging"].get("61_90", 0.0))
                prev_90p = float(d["aging"].get("90_plus", 0.0))
                row_total = prev_31_60 + prev_61_90 + prev_90p
                legacy_rows.append({
                    "description": d["description"],
                    "current": 0.0,
                    "0_30": 0.0,  # allocated later
                    "31_60": prev_0_30,
                    "61_90": prev_31_60,
                    "90_plus": prev_61_90 + prev_90p,
                    "total": row_total,
                })
            baseline_total = float(sum(r["total"] for r in legacy_rows))
            residual = max(0.0, float(request.target_total) - baseline_total)
            # Weight by recent totals from history df
            recent_weights: Dict[str, float] = {}
            if not df_hist.empty:
                df_hist["month_date"] = pd.to_datetime(df_hist["month"] + "-01")
                for desc, g in df_hist.sort_values(["description", "month_date"]).groupby("description"):
                    recent_weights[desc] = float(g.tail(3)["total"].sum()) or 1.0
            total_w = sum(recent_weights.get(r["description"], 1.0) for r in legacy_rows) or 1.0
            for r in legacy_rows:
                w = recent_weights.get(r["description"], 1.0) / total_w
                add0 = residual * w
                r["0_30"] = r.get("0_30", 0.0) + add0
                r["current"] = r["0_30"]
                r["total"] = r["0_30"] + r["31_60"] + r["61_90"] + r["90_plus"]
            flat_predictions = legacy_rows

        # As a final safety, if buckets are zero but total > 0, push total into 0_30/current
        for p in flat_predictions:
            if p.get("total", 0) > 0 and (p.get("0_30", 0) + p.get("31_60", 0) + p.get("61_90", 0) + p.get("90_plus", 0) == 0):
                p["0_30"] = float(p["total"]) 
                p["current"] = float(p["total"])

        # Build previous month map for override calculations
        prev_map: Dict[str, Dict[str, float]] = {}
        for d in prev_docs:
            prev_map[_norm_desc(d.get("description", ""))] = {
                "0_30": float(d["aging"].get("0_30", 0.0)),
                "31_60": float(d["aging"].get("31_60", 0.0)),
                "61_90": float(d["aging"].get("61_90", 0.0)),
                "90_plus": float(d["aging"].get("90_plus", 0.0)),
            }

        # Apply manual overrides for matched entries
        for e in matched_add_entries:
            amt = float(e["amount"]) if e.get("amount") is not None else 0.0
            if amt <= 0:
                continue
            desc_raw = str(e.get("description", "")).strip()
            desc_clean = _clean_description(desc_raw).lower()
            
            # Find matching row in predictions
            found = False
            for p in flat_predictions:
                pred_desc_clean = _clean_description(p["description"]).lower()
                if pred_desc_clean == desc_clean:
                    # Apply override: set 0-30 to exact amount (no capping for user overrides)
                    prev_buckets = prev_map.get(_norm_desc(desc_raw), {})
                    override_0_30 = amt  # Use exact user amount, no capping
                    
                    # Apply aging trend to other buckets (decreasing)
                    prev_0_30 = prev_buckets.get("0_30", 0)
                    prev_31_60 = prev_buckets.get("31_60", 0)
                    prev_61_90 = prev_buckets.get("61_90", 0)
                    prev_90_plus = prev_buckets.get("90_plus", 0)
                    
                    p["0_30"] = float(override_0_30)
                    p["31_60"] = float(min(p.get("31_60", 0), prev_0_30 * 0.9))
                    p["61_90"] = float(min(p.get("61_90", 0), prev_31_60 * 0.9))
                    p["90_plus"] = float(min(p.get("90_plus", 0), (prev_61_90 + prev_90_plus) * 0.9))
                    
                    # Recalculate total and current
                    p["total"] = p["0_30"] + p["31_60"] + p["61_90"] + p["90_plus"]
                    p["current"] = p["total"]
                    p["is_override"] = True
                    found = True
                    break 

        # Merge additional entries into predictions
        # Build lookup maps
        pred_map: Dict[str, Dict[str, Any]] = { _norm_desc(p["description"]): p for p in flat_predictions }
        # Secondary map using stricter cleaning to align with generator cleaning
        def _clean_key(s: str) -> str:
            s = (s or "").lower().strip()
            # normalize whitespace and strip punctuation-like chars
            import re
            s = re.sub(r"[\-_,.;:/\\]+", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s
        pred_map_clean: Dict[str, Dict[str, Any]] = {}
        for p in flat_predictions:
            pred_map_clean[_clean_key(p["description"])] = p

        # 1) Matched entries already handled via overrides in the generator
        # 2) Append brand-new entries (not in last month) as standalone rows with only 0-30
        # NOTE: Matched entries are processed by generator as overrides, so we only add truly new entries here
        for e in new_add_entries:
            amt = float(e["amount"]) if e.get("amount") is not None else 0.0
            if amt <= 0:
                continue
            desc_raw = str(e.get("description", "")).strip() or "New Entry"
            desc_key = _norm_desc(desc_raw)
            desc_clean_key = _clean_key(desc_raw)
            # Avoid duplicate if somehow predicted created same desc
            if desc_key in pred_map or desc_clean_key in pred_map_clean:
                # If generator already created this description, OVERRIDE it to EXACT user 0-30 (others 0)
                pred_row = pred_map.get(desc_key) or pred_map_clean.get(desc_clean_key)
                amt_int = int(round(amt))
                pred_row["0_30"] = float(amt_int)
                pred_row["31_60"] = 0.0
                pred_row["61_90"] = 0.0
                pred_row["90_plus"] = 0.0
                pred_row["total"] = float(amt_int)
                pred_row["current"] = float(amt_int)
                pred_row["is_override"] = True
            else:
                amt_int = int(round(amt))
                row = {
                    "description": desc_raw,
                    "current": float(amt_int),
                    "0_30": float(amt_int),
                    "31_60": 0.0,
                    "61_90": 0.0,
                    "90_plus": 0.0,
                    "total": float(amt_int),
                    "is_override": True,
                    "user_added": True,
                }
                flat_predictions.append(row)
                pred_map[desc_key] = row
                pred_map_clean[desc_clean_key] = row

        combined_predictions = flat_predictions
        # Make sure all numbers are integers and totals are consistent per row
        for rp in combined_predictions:
            rp["0_30"] = float(int(round(rp.get("0_30", 0.0))))
            rp["31_60"] = float(int(round(rp.get("31_60", 0.0))))
            rp["61_90"] = float(int(round(rp.get("61_90", 0.0))))
            rp["90_plus"] = float(int(round(rp.get("90_plus", 0.0))))
            row_total = int(rp["0_30"] + rp["31_60"] + rp["61_90"] + rp["90_plus"]) 
            rp["total"] = float(row_total)
            rp["current"] = float(row_total)

        # Final small ±1 reconciliation if needed
        def _sum_col(rows, key):
            return int(sum(int(round(x.get(key, 0.0) or 0.0)) for x in rows))

        col_sums = {
            "0_30": _sum_col(combined_predictions, "0_30"),
            "31_60": _sum_col(combined_predictions, "31_60"),
            "61_90": _sum_col(combined_predictions, "61_90"),
            "90_plus": _sum_col(combined_predictions, "90_plus"),
        }
        final_total = float(_sum_col(combined_predictions, "total"))

        provided_targets = None
        if column_targets_payload:
            provided_targets = {
                "0_30": int(round(float(column_targets_payload.get("b0_30", 0)))),
                "31_60": int(round(float(column_targets_payload.get("b31_60", 0)))),
                "61_90": int(round(float(column_targets_payload.get("b61_90", 0)))),
                "90_plus": int(round(float(column_targets_payload.get("b90_plus", 0)))),
            }

        # First, reconcile any single-bucket ±1 drift to match provided targets
        if provided_targets is not None:
            # Recompute column sums in case they were rounded
            col_sums = {
                "0_30": _sum_col(combined_predictions, "0_30"),
                "31_60": _sum_col(combined_predictions, "31_60"),
                "61_90": _sum_col(combined_predictions, "61_90"),
                "90_plus": _sum_col(combined_predictions, "90_plus"),
            }
            for bucket in ["31_60", "61_90", "90_plus", "0_30"]:
                diff = provided_targets[bucket] - col_sums[bucket]
                if abs(diff) == 1:
                    sign = 1 if diff > 0 else -1
                    for rp in combined_predictions:
                        if bucket in ["31_60", "61_90", "90_plus"] and (rp.get("user_added") or rp.get("auto_generated")):
                            continue
                        rp[bucket] = float(int(round(rp[bucket])) + sign)
                        new_total = int(rp["0_30"] + rp["31_60"] + rp["61_90"] + rp["90_plus"]) 
                        rp["total"] = float(new_total)
                        rp["current"] = float(new_total)
                        break
                    # update sums post-adjustment
                    col_sums[bucket] += sign

        # Then, reconcile grand total if off by ±1
        total_target = sum(provided_targets.values()) if provided_targets is not None else int(round(effective_target_total))
        final_total = float(_sum_col(combined_predictions, "total"))
        grand_diff = total_target - int(final_total)
        if abs(grand_diff) == 1:
            sign = 1 if grand_diff > 0 else -1
            # Try to adjust a bucket that won't violate user_added/auto_generated constraints
            for candidate_bucket in ["31_60", "61_90", "90_plus", "0_30"]:
                for rp in combined_predictions:
                    if candidate_bucket in ["31_60", "61_90", "90_plus"] and (rp.get("user_added") or rp.get("auto_generated")):
                        continue
                    rp[candidate_bucket] = float(int(round(rp[candidate_bucket])) + sign)
                    new_total = int(rp["0_30"] + rp["31_60"] + rp["61_90"] + rp["90_plus"]) 
                    rp["total"] = float(new_total)
                    rp["current"] = float(new_total)
                    break
                else:
                    continue
                break
            final_total = float(_sum_col(combined_predictions, "total"))

        
        # CRITICAL: Re-apply exact column target scaling after user entries are processed
        # The ML model calculates perfect targets, but user entries can break them
        from train_generate_ar import _scale_to_exact_target
        
        # Build per-row caps based on previous month: caps map (use same cleaning as generator)
        caps_map: Dict[str, Dict[str, float]] = {}
        for d in prev_docs:
            from train_generate_ar import _clean_description as _gen_clean_desc
            key = _gen_clean_desc(str(d.get("description", ""))).lower()
            prev_0_30 = float(d.get("aging", {}).get("0_30", 0.0))
            prev_31_60 = float(d.get("aging", {}).get("31_60", 0.0))
            prev_61_90 = float(d.get("aging", {}).get("61_90", 0.0))
            prev_90_plus = float(d.get("aging", {}).get("90_plus", 0.0))
            caps_map[key] = {
                "31_60": prev_0_30,
                "61_90": prev_31_60,
                "90_plus": prev_61_90 + prev_90_plus,
            }

        # Convert flattened format back to 'aging' structure for _scale_to_exact_target
        converted_predictions = []
        for p in combined_predictions:
            converted_predictions.append({
                "description": p["description"],
                "month": p.get("month", ""),
                "aging": {
                    "current": float(p.get("current", 0.0)),
                    "0_30": float(p.get("0_30", 0.0)),
                    "31_60": float(p.get("31_60", 0.0)),
                    "61_90": float(p.get("61_90", 0.0)),
                    "90_plus": float(p.get("90_plus", 0.0)),
                },
                "total": float(p.get("total", 0.0)),
                "predicted": p.get("predicted", True),
                "is_override": p.get("is_override", False),
                # Ensure scaler knows this is a user-added row (0-30 only)
                "user_added": bool(p.get("user_added", False)),
                # Preserve auto_generated flag through scaling so we can enforce 0-30-only after
                "auto_generated": bool(p.get("auto_generated", False)),
                # Attach per-row caps for older buckets to guide the scaler rebalancing
                "_caps": caps_map.get(_gen_clean_desc(p["description"]).lower(), {}),
            })
        
        # Map frontend keys to scaler's keys if provided
        if column_targets_payload:
            column_targets_for_scaler = {
                "0_30": float(column_targets_payload.get("b0_30", 0)),
                "31_60": float(column_targets_payload.get("b31_60", 0)),
                "61_90": float(column_targets_payload.get("b61_90", 0)),
                "90_plus": float(column_targets_payload.get("b90_plus", 0)),
            }
        else:
            column_targets_for_scaler = None
        
        # Apply scaling to converted format with dynamic column targets
        scaled_predictions = _scale_to_exact_target(converted_predictions, effective_target_total, column_targets_for_scaler)
        
        # Convert back to flattened format for API response (no post-processing of buckets here)
        combined_predictions = []
        for p in scaled_predictions:
            desc = p["description"]
            is_adjustment = bool(p.get("is_adjustment"))
            # Optional: Rename adjustment description only; do not change numbers
            if is_adjustment:
                used_descs = { _norm_desc(x["description"]) for x in flat_predictions }
                hist_candidates = [
                    d for d in df_hist["description"].astype(str).unique().tolist()
                    if _norm_desc(d) not in prev_desc_set and _norm_desc(d) not in used_descs
                ]
                if hist_candidates:
                    desc = hist_candidates[0]
                else:
                    desc = "Adjustment Client"
            # Determine if this row existed in previous month (definition of carried vs generated)
            from train_generate_ar import _clean_description as _gen_clean_desc
            is_new_this_month = (_gen_clean_desc(desc).lower() not in prev_desc_set)
            must_zero_older = bool(p.get("user_added") or p.get("auto_generated") or is_new_this_month)

            combined_predictions.append({
                "description": desc,
                # Enforce 0-30-only for user_added and auto_generated at the boundary (double safety)
                "current": float(p["aging"].get("0_30", 0.0)) if must_zero_older else float(p["aging"].get("current", 0.0)),
                "0_30": float(p["aging"].get("0_30", 0.0)),
                "31_60": 0.0 if must_zero_older else float(p["aging"].get("31_60", 0.0)),
                "61_90": 0.0 if must_zero_older else float(p["aging"].get("61_90", 0.0)),
                "90_plus": 0.0 if must_zero_older else float(p["aging"].get("90_plus", 0.0)),
                "total": float(p["aging"].get("0_30", 0.0)) if must_zero_older else float(
                    float(p["aging"].get("0_30", 0.0)) + float(p["aging"].get("31_60", 0.0)) + float(p["aging"].get("61_90", 0.0)) + float(p["aging"].get("90_plus", 0.0))
                ),
                "is_override": bool(p.get("is_override", False)),
                "user_added": bool(p.get("user_added", False)),
                "auto_generated": bool(p.get("auto_generated", False)),
                "is_new": bool(is_new_this_month),
            })
        
        # FINAL ROUNDING AND ±1 RECONCILIATION AFTER SCALER AND ZEROING
        # Normalize integers and row totals
        for rp in combined_predictions:
            rp["0_30"] = float(int(round(rp.get("0_30", 0.0))))
            rp["31_60"] = float(int(round(rp.get("31_60", 0.0))))
            rp["61_90"] = float(int(round(rp.get("61_90", 0.0))))
            rp["90_plus"] = float(int(round(rp.get("90_plus", 0.0))))
            row_total = int(rp["0_30"] + rp["31_60"] + rp["61_90"] + rp["90_plus"]) 
            rp["total"] = float(row_total)
            rp["current"] = float(row_total)

        def _sum_col(rows, key):
            return int(sum(int(round(x.get(key, 0.0) or 0.0)) for x in rows))

        provided_targets = None
        if column_targets_payload:
            provided_targets = {
                "0_30": int(round(float(column_targets_payload.get("b0_30", 0)))),
                "31_60": int(round(float(column_targets_payload.get("b31_60", 0)))),
                "61_90": int(round(float(column_targets_payload.get("b61_90", 0)))),
                "90_plus": int(round(float(column_targets_payload.get("b90_plus", 0)))),
            }

        # First reconcile any single-bucket ±1 differences to match provided targets
        if provided_targets is not None:
            col_sums = {
                "0_30": _sum_col(combined_predictions, "0_30"),
                "31_60": _sum_col(combined_predictions, "31_60"),
                "61_90": _sum_col(combined_predictions, "61_90"),
                "90_plus": _sum_col(combined_predictions, "90_plus"),
            }
            for bucket in ["31_60", "61_90", "90_plus", "0_30"]:
                diff = provided_targets[bucket] - col_sums[bucket]
                if abs(diff) == 1:
                    sign = 1 if diff > 0 else -1
                    for rp in combined_predictions:
                        if bucket in ["31_60", "61_90", "90_plus"] and (rp.get("user_added") or rp.get("auto_generated") or rp.get("is_new")):
                            continue
                        rp[bucket] = float(int(round(rp[bucket])) + sign)
                        new_total = int(rp["0_30"] + rp["31_60"] + rp["61_90"] + rp["90_plus"]) 
                        rp["total"] = float(new_total)
                        rp["current"] = float(new_total)
                        break
                    col_sums[bucket] += sign

        # Then reconcile grand total if off by ±1
        total_target = sum(provided_targets.values()) if provided_targets is not None else int(round(effective_target_total))
        final_total = float(_sum_col(combined_predictions, "total"))
        grand_diff = total_target - int(final_total)
        if abs(grand_diff) == 1:
            sign = 1 if grand_diff > 0 else -1
            for candidate_bucket in ["31_60", "61_90", "90_plus", "0_30"]:
                for rp in combined_predictions:
                    if candidate_bucket in ["31_60", "61_90", "90_plus"] and (rp.get("user_added") or rp.get("auto_generated") or rp.get("is_new")):
                        continue
                    rp[candidate_bucket] = float(int(round(rp[candidate_bucket])) + sign)
                    new_total = int(rp["0_30"] + rp["31_60"] + rp["61_90"] + rp["90_plus"]) 
                    rp["total"] = float(new_total)
                    rp["current"] = float(new_total)
                    break
                else:
                    continue
                break

        # Final recompute of grand total
        final_total = float(_sum_col(combined_predictions, "total"))
        
        audit_snapshot_id = str(uuid4())
        
        return PredictionResponse(
            predictions=combined_predictions,
            model_meta={
                "model_version": model_doc.get("version", "unknown"),
                "metrics": model_doc.get("metrics", {}),
            },
            grand_total=final_total,
            warnings=warnings,
            audit_snapshot_id=audit_snapshot_id,
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/approve", response_model=ApproveResponse, dependencies=[Depends(verify_api_key)])
async def approve_predictions(request: ApproveRequest):
    """Approve and save predictions to database"""
    try:
        ar_data_collection = mongo_db["ar_data"]
        ml_jobs_collection = mongo_db["ml_jobs"]
        
        # Prepare documents for insertion
        documents = []
        for pred in request.predictions:
            doc = {
                "client_id": ObjectId(request.client_id),
                "description": pred["description"],
                "month": request.target_month,
                "aging": {
                    "current": pred.get("current", pred["0_30"]),  # Use current column if available, fallback to 0_30
                    "0_30": pred["0_30"],
                    "31_60": pred["31_60"],
                    "61_90": pred["61_90"],
                    "90_plus": pred["90_plus"]
                },
                "total": pred["total"],
                "predicted": True,
                "model_version": request.model_version,
                "approved_at": datetime.now(),
                "approved_by": "web_user",
                "approval_comment": request.comment or "",
                "origin": "web"
            }
            documents.append(doc)
        
        # Upsert documents
        for doc in documents:
            ar_data_collection.update_one(
                {
                    "client_id": doc["client_id"],
                    "description": doc["description"],
                    "month": doc["month"]
                },
                {"$set": doc},
                upsert=True
            )
        
        # Always retrain after approval to include the new month's data
        job_id = str(uuid4())
        job_doc = {
            "_id": job_id,
            "client_id": request.client_id,
            "status": "queued",
            "created_at": datetime.now(),
            "model_version": request.model_version,
            "triggered_by": "auto_retrain_after_approval",
            "logs": []
        }
        ml_jobs_collection.insert_one(job_doc)
        
        # Start retrain job in background to include newly approved data
        asyncio.create_task(run_retrain_job(job_id, request.client_id))
        
        logger.info(f"Approved predictions for client {request.client_id}, month {request.target_month}. Auto-retraining model with new data.")
        
        return ApproveResponse(
            success=True,
            job_id=job_id,
            message="Predictions approved and saved successfully. Model is being retrained with new data."
        )
        
    except Exception as e:
        logger.error(f"Approval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Approval failed: {str(e)}")

@app.post("/api/train", response_model=TrainJobResponse, dependencies=[Depends(verify_api_key)])
async def trigger_training(client_id: str = Form(...)):
    """Trigger model retraining for a specific client"""
    try:
        job_id = str(uuid4())
        ml_jobs_collection = mongo_db["ml_jobs"]
        
        # Create job record
        job_doc = {
            "_id": job_id,
            "client_id": client_id,
            "status": "queued",
            "created_at": datetime.now(),
            "triggered_by": "manual",
            "logs": []
        }
        ml_jobs_collection.insert_one(job_doc)
        
        # Start training job in background
        asyncio.create_task(run_retrain_job(job_id, client_id))
        
        return TrainJobResponse(
            job_id=job_id,
            status="queued",
            message="Training job queued successfully"
        )
        
    except Exception as e:
        logger.error(f"Training trigger failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training trigger failed: {str(e)}")

@app.get("/api/train_status/{job_id}", response_model=TrainJobResponse)
async def get_training_status(job_id: str):
    """Get status of a training job"""
    try:
        ml_jobs_collection = mongo_db["ml_jobs"]
        job = ml_jobs_collection.find_one({"_id": job_id})
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return TrainJobResponse(
            job_id=job_id,
            status=job["status"],
            message=job.get("message", ""),
            metrics=job.get("metrics")
        )
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@app.get("/api/client/{client_id}/last_month")
async def get_client_last_month(client_id: str):
    """Get the last available month for a specific client"""
    try:
        ar_data_collection = mongo_db["ar_data"]
        
        # Find the last month with any data (historical or predictions) for this client
        last_any = ar_data_collection.find_one(
            {"client_id": ObjectId(client_id)},
            sort=[("month", -1)]
        )

        # Find the last month with historical data only (predicted == False)
        last_hist = ar_data_collection.find_one(
            {"client_id": ObjectId(client_id), "predicted": False},
            sort=[("month", -1)]
        )
        
        return {
            "last_month": (last_any["month"] if last_any else None),
            "last_historical_month": (last_hist["month"] if last_hist else None)
        }
        
    except Exception as e:
        logger.error(f"Error fetching last month for client {client_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch last month")

@app.get("/api/history/{client_id}/{description}")
async def get_description_history(client_id: str, description: str):
    """Get historical data for a specific client/description for sparkline display"""
    try:
        ar_data_collection = mongo_db["ar_data"]
        
        # Get historical data for this client/description
        history = list(ar_data_collection.find(
            {
                "client_id": ObjectId(client_id),
                "description": description,
                "predicted": False  # Only historical data, not predictions
            },
            {"month": 1, "total": 1, "aging.0_30": 1, "aging.31_60": 1, "aging.61_90": 1, "aging.90_plus": 1},
            sort=[("month", 1)]
        ))
        
        # Format for sparkline
        sparkline_data = []
        for record in history:
            sparkline_data.append({
                "month": record["month"],
                "total": record["total"],
                "0_30": record["aging"]["0_30"],
                "31_60": record["aging"]["31_60"],
                "61_90": record["aging"]["61_90"],
                "90_plus": record["aging"]["90_plus"]
            })
        
        return {"history": sparkline_data}
        
    except Exception as e:
        logger.error(f"History fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"History fetch failed: {str(e)}")

@app.get("/api/client/{client_id}/history")
async def get_client_history(client_id: str, month: Optional[str] = None):
    """Get complete history for a client including both historical data and approved predictions"""
    try:
        ar_data_collection = mongo_db["ar_data"]
        clients_collection = mongo_db["clients"]
        
        # Get client info
        client = clients_collection.find_one({"_id": ObjectId(client_id)})
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Build query filter
        query_filter = {"client_id": ObjectId(client_id)}
        if month:
            query_filter["month"] = month
        
        # Get all data for this client (both historical and approved predictions)
        all_data = list(ar_data_collection.find(
            query_filter,
            {
                "description": 1, 
                "month": 1, 
                "aging": 1, 
                "total": 1, 
                "predicted": 1,
                "model_version": 1,
                "approved_at": 1,
                "approved_by": 1,
                "approval_comment": 1,
                "created_at": 1
            },
            sort=[("month", 1), ("description", 1)]
        ))
        
        # If requesting a specific month, return raw data for comparison
        if month:
            if not all_data:
                return []
            
            # Convert to the format expected by the frontend
            formatted_data = []
            for record in all_data:
                aging = record.get("aging", {})
                formatted_record = {
                    "description": record["description"],
                    "current": aging.get("current", 0),
                    "0_30": aging.get("0_30", 0),
                    "31_60": aging.get("31_60", 0),
                    "61_90": aging.get("61_90", 0),
                    "90_plus": aging.get("90_plus", 0),
                    "total": record["total"]
                }
                formatted_data.append(formatted_record)
            
            return formatted_data
        
        if not all_data:
            return {
                "client": {
                    "id": str(client["_id"]),
                    "name": client["name"],
                    "created_at": client.get("created_at")
                },
                "history": [],
                "summary": {
                    "total_months": 0,
                    "total_records": 0,
                    "historical_records": 0,
                    "predicted_records": 0,
                    "first_month": None,
                    "last_month": None
                }
            }
        
        # Group data by month for better organization
        monthly_data = {}
        for record in all_data:
            month = record["month"]
            if month not in monthly_data:
                monthly_data[month] = {
                    "month": month,
                    "records": [],
                    "total_amount": 0,
                    "has_predictions": False,
                    "historical_count": 0,
                    "predicted_count": 0
                }
            
            record_data = {
                "description": record["description"],
                "aging": record["aging"],
                "total": record["total"],
                "predicted": record.get("predicted", False),
                "model_version": record.get("model_version"),
                "approved_at": record.get("approved_at"),
                "approved_by": record.get("approved_by"),
                "approval_comment": record.get("approval_comment"),
                "created_at": record.get("created_at")
            }
            
            monthly_data[month]["records"].append(record_data)
            monthly_data[month]["total_amount"] += record["total"]
            
            if record.get("predicted", False):
                monthly_data[month]["has_predictions"] = True
                monthly_data[month]["predicted_count"] += 1
            else:
                monthly_data[month]["historical_count"] += 1
        
        # Convert to list and sort by month
        history = list(monthly_data.values())
        history.sort(key=lambda x: x["month"])
        
        # Calculate summary statistics
        total_records = len(all_data)
        historical_records = sum(1 for r in all_data if not r.get("predicted", False))
        predicted_records = sum(1 for r in all_data if r.get("predicted", False))
        months = sorted(monthly_data.keys())
        
        summary = {
            "total_months": len(months),
            "total_records": total_records,
            "historical_records": historical_records,
            "predicted_records": predicted_records,
            "first_month": months[0] if months else None,
            "last_month": months[-1] if months else None
        }
        
        return {
            "client": {
                "id": str(client["_id"]),
                "name": client["name"],
                "created_at": client.get("created_at")
            },
            "history": history,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Client history fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Client history fetch failed: {str(e)}")

@app.get("/api/clients/summary")
async def get_clients_summary():
    """Get summary information for all clients"""
    try:
        clients_collection = mongo_db["clients"]
        ar_data_collection = mongo_db["ar_data"]
        models_collection = mongo_db["models"]
        
        # Get all clients
        clients = list(clients_collection.find({}, {"_id": 1, "name": 1, "created_at": 1}))
        
        client_summaries = []
        for client in clients:
            client_id = client["_id"]
            
            # Get data counts
            total_records = ar_data_collection.count_documents({"client_id": client_id})
            historical_records = ar_data_collection.count_documents({"client_id": client_id, "predicted": False})
            predicted_records = ar_data_collection.count_documents({"client_id": client_id, "predicted": True})
            
            # Get month range
            first_record = ar_data_collection.find_one(
                {"client_id": client_id},
                sort=[("month", 1)]
            )
            last_record = ar_data_collection.find_one(
                {"client_id": client_id},
                sort=[("month", -1)]
            )
            
            # Get model info
            model_doc = models_collection.find_one({"client_id": client_id}, sort=[("created_at", -1)])
            
            # Get last prediction info
            last_prediction = ar_data_collection.find_one(
                {"client_id": client_id, "predicted": True},
                sort=[("month", -1)]
            )
            
            client_summaries.append({
                "id": str(client_id),
                "name": client["name"],
                "created_at": client.get("created_at"),
                "total_records": total_records,
                "historical_records": historical_records,
                "predicted_records": predicted_records,
                "first_month": first_record["month"] if first_record else None,
                "last_month": last_record["month"] if last_record else None,
                "has_model": model_doc is not None,
                "model_version": model_doc.get("version") if model_doc else None,
                "last_prediction": last_prediction["month"] if last_prediction else None,
                "last_prediction_date": last_prediction.get("approved_at") if last_prediction else None
            })
        
        return {
            "clients": client_summaries,
            "total_clients": len(client_summaries),
            "total_records": sum(c["total_records"] for c in client_summaries),
            "total_historical": sum(c["historical_records"] for c in client_summaries),
            "total_predictions": sum(c["predicted_records"] for c in client_summaries)
        }
        
    except Exception as e:
        logger.error(f"Clients summary fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Clients summary fetch failed: {str(e)}")

# Background task for retraining
async def run_retrain_job(job_id: str, client_id: str):
    """Run retrain job in background"""
    try:
        ml_jobs_collection = mongo_db["ml_jobs"]
        ar_data_collection = mongo_db["ar_data"]
        models_collection = mongo_db["models"]
        
        # Update job status to running
        ml_jobs_collection.update_one(
            {"_id": job_id},
            {"$set": {"status": "running", "started_at": datetime.now()}}
        )
        
        # Get ALL client data (historical + approved predictions for training)
        all_data = list(ar_data_collection.find(
            {"client_id": ObjectId(client_id)},  # Include both historical and approved predictions
            {"description": 1, "month": 1, "aging": 1, "total": 1}
        ))
        
        if not all_data:
            raise ValueError("No data found for retraining")
        
        # Convert to DataFrame in the format expected by new_train_model
        df = pd.DataFrame([
            {
                'month': r['month'],
                'description': r['description'],
                'current': float(r['aging'].get('current', 0.0)),
                '0_30': float(r['aging'].get('0_30', 0.0)),
                '31_60': float(r['aging'].get('31_60', 0.0)),
                '61_90': float(r['aging'].get('61_90', 0.0)),
                '90_plus': float(r['aging'].get('90_plus', 0.0)),
                'total': float(r.get('total', 0.0)),
            } for r in all_data
        ])

        # Train new model with new pipeline (including newly approved data)
        logger.info(f"Retraining model for client {client_id} with {len(df)} data points across {df['month'].nunique()} months")
        artifacts, metrics = new_train_model(df, client_name="SBS", models_dir=os.path.join(os.path.dirname(__file__), "models"))
        
        # Use GridFS for large model files instead of embedding in document
        from gridfs import GridFSBucket
        bucket = GridFSBucket(mongo_db)
        
        # Delete any existing model files for this client
        existing_files = list(bucket.find({"metadata.client_id": client_id}))
        for file_doc in existing_files:
            bucket.delete(file_doc._id)
            logger.info(f"Deleted existing GridFS model file {file_doc._id} for client {client_id}")
        
        # Store classifier in GridFS
        with open(artifacts.classifier_path, "rb") as f:
            clf_file_id = bucket.upload_from_stream(
                f"classifier_{client_id}_{artifacts.version}",
                f,
                metadata={"client_id": client_id, "model_type": "classifier", "version": artifacts.version}
            )
        
        # Store regressor in GridFS  
        with open(artifacts.regressor_path, "rb") as f:
            regr_file_id = bucket.upload_from_stream(
                f"regressor_{client_id}_{artifacts.version}",
                f,
                metadata={"client_id": client_id, "model_type": "regressor", "version": artifacts.version}
            )
        
        # Store model metadata (without binary data)
        model_doc = {
            "client_id": ObjectId(client_id),
            "version": artifacts.version,
            "metrics": metrics,
            "created_at": datetime.now(),
            "status": "trained",
            "gridfs_files": {
                "carry_classifier": clf_file_id,
                "bucket_regressor": regr_file_id,
            },
        }
        models_collection.insert_one(model_doc)
        logger.info(f"Stored model metadata and GridFS files for client {client_id}, version {artifacts.version}")
        
        # Clear model cache to force loading of new model
        cache_key = client_id
        if cache_key in client_models:
            del client_models[cache_key]
            logger.info(f"Cleared model cache for client {client_id}")
        
        # Remove files
        try:
            for p in [artifacts.classifier_path, artifacts.regressor_path, artifacts.registry_path]:
                if p and os.path.exists(p):
                    os.remove(p)
        except Exception:
            pass
        
        # Update job status to completed
        ml_jobs_collection.update_one(
            {"_id": job_id},
            {"$set": {
                "status": "completed",
                "completed_at": datetime.now(),
                "model_version": artifacts.version,
                "message": "Training completed successfully",
            }}
        )
        
        logger.info(f"Retrain job {job_id} completed successfully for client {client_id}")
        
    except Exception as e:
        # Update job status to failed
        ml_jobs_collection.update_one(
            {"_id": job_id},
            {"$set": {
                "status": "failed",
                "completed_at": datetime.now(),
                "message": f"Training failed: {str(e)}"
            }}
        )
        
        logger.error(f"Retrain job {job_id} failed with exception: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, reload=DEBUG)
