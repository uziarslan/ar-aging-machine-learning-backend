# AR Aging ML Backend

Integrated FastAPI backend with embedded machine learning pipeline for AR aging predictions.

## Features

- **Multi-client support** - Separate ML models per client
- **CSV upload** - Upload client data and auto-train models
- **Real-time predictions** - Generate AR aging predictions using ML
- **Auto-retrain** - Automatically retrain models when predictions are approved
- **Database-only workflow** - All operations work with MongoDB
- **RESTful API** - Clean API endpoints for frontend integration

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file:

```bash
cp env.example .env
```

Edit `.env`:
```
MONGO_URI=mongodb://localhost:27017/ar_aging
ADMIN_API_KEY=your-secure-api-key-here
```

### 3. Start the Server

```bash
MONGO_URI=mongodb://localhost:27017/ar_aging ADMIN_API_KEY=demo-api-key-123 uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## API Endpoints

### Core Endpoints

- `GET /api/health` - Health check
- `GET /api/clients` - List all clients with model status
- `POST /api/upload` - Upload CSV data and train models (requires API key)
- `POST /api/predict` - Generate AR aging predictions
- `POST /api/approve` - Approve predictions and optionally retrain (requires API key)
- `POST /api/train` - Manually trigger model retraining (requires API key)
- `GET /api/train_status/{job_id}` - Check training job status
- `GET /api/history/{client_id}/{description}` - Get historical data for sparklines

### Authentication

Protected endpoints (`/api/upload`, `/api/approve`, `/api/train`) require an API key in the header:
```
X-API-Key: your-api-key
```

## CSV Upload Format

When uploading client data, the CSV must have these columns:

```csv
month,description,b0_30,b31_60,b61_90,b90p,total
2021-09,Company A,1000,500,200,100,1800
2021-10,Company A,1200,600,250,150,2200
```

- `month`: Format YYYY-MM (e.g., 2021-09)
- `description`: Company/client name
- `b0_30`, `b31_60`, `b61_90`, `b90p`: Aging bucket amounts
- `total`: Total amount for the month

## Machine Learning Pipeline

The backend includes an integrated ML pipeline with:

1. **Feature Engineering**
   - Lag features (1, 2, 3 months)
   - Rolling statistics (mean, std)
   - Seasonal features (month sin/cos)
   - Trend analysis
   - Months since last non-zero

2. **Models**
   - **Carry-forward Classifier**: Predicts if a description will continue to next month
   - **Bucket Regressors**: Predicts amounts for each aging bucket

3. **Constrained Allocation**
   - Ensures predicted totals match target exactly
   - Proportional scaling when needed

## Database Collections

- `clients` - Client information and metadata
- `ar_data` - Historical AR data and predictions
- `models` - Trained model metadata and versions
- `ml_jobs` - Background training job status

## Development

### Running Tests

```bash
python test_integration.py
```

### Adding New Features

The backend is structured with:
- `app.py` - Main FastAPI application
- ML classes are embedded in the same file for simplicity
- All database operations use PyMongo
- Background tasks use asyncio

## Production Deployment

For production deployment:

1. Set secure environment variables
2. Use a production ASGI server (e.g., Gunicorn with Uvicorn workers)
3. Configure proper logging
4. Set up monitoring and alerting
5. Use a production MongoDB instance

```bash
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```