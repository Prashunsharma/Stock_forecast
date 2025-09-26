from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from prophet import Prophet
import datetime
import logging

app = FastAPI(title="Stock Forecast API")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allow Flutter frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In prod, restrict this to your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class PredictRequest(BaseModel):
    symbol: str = Field(..., example="INFY.NS")
    start_date: Optional[str] = Field(None, example="2024-01-01")
    end_date: Optional[str] = Field(None, example="2024-12-01")
    time_period_type: Optional[str] = Field(None, example="Months")  # Days/Weeks/Months/Years
    time_period_value: Optional[int] = Field(None, example=6)
    predict_days: int = Field(..., example=30)

class OHLCPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[int] = None

class ForecastPoint(BaseModel):
    date: str
    predicted: float
    lower: float
    upper: float

class PredictResponse(BaseModel):
    symbol: str
    historical: List[OHLCPoint]
    forecast: List[ForecastPoint]

# ---------- Utils ----------
def download_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download stock data with better error handling"""
    try:
        logger.info(f"Downloading data for {symbol} from {start} to {end}")
        
        # Add a small buffer to end date to ensure we get recent data
        end_dt = datetime.datetime.strptime(end, "%Y-%m-%d")
        end_dt += datetime.timedelta(days=1)  # Add one day buffer
        end_buffered = end_dt.strftime("%Y-%m-%d")
        
        df = yf.download(tickers=symbol, start=start, end=end_buffered, progress=False)
        
        if df.empty:
            logger.error(f"No data returned for {symbol}")
            return pd.DataFrame()
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
            
        logger.info(f"Downloaded {len(df)} rows of data")
        return df
        
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        raise HTTPException(500, f"Error downloading data: {str(e)}")

# ---------- Routes ----------
@app.get("/")
def root():
    return {"message": "✅ Stock Forecast API is running!", "timestamp": datetime.datetime.now().isoformat()}

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        symbol = req.symbol.strip().upper()
        logger.info(f"Processing prediction request for {symbol}")

        # Resolve dates
        if req.start_date and req.end_date:
            start, end = req.start_date, req.end_date
            # Validate date format and order
            try:
                start_dt = datetime.datetime.strptime(start, "%Y-%m-%d")
                end_dt = datetime.datetime.strptime(end, "%Y-%m-%d")
                if start_dt >= end_dt:
                    raise HTTPException(400, "start_date must be before end_date")
            except ValueError:
                raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD")
                
        elif req.time_period_type and req.time_period_value:
            days = req.time_period_value
            t = req.time_period_type.lower()
            
            if t == "years":
                days *= 365
            elif t == "months":
                days *= 30
            elif t == "weeks":
                days *= 7
            elif t == "days":
                days *= 1
            else:
                raise HTTPException(status_code=400, detail="Invalid time_period_type. Use: days, weeks, months, or years")

            # Fix: Calculate start date by going back from today
            end_dt = datetime.date.today()
            start_dt = end_dt - datetime.timedelta(days=days)
            start = start_dt.isoformat()
            end = end_dt.isoformat()
            
            logger.info(f"Calculated date range: {start} to {end}")

        else:
            # Default: last 6 months of data
            end_dt = datetime.date.today()
            start_dt = end_dt - datetime.timedelta(days=180)  # 6 months
            start = start_dt.isoformat()
            end = end_dt.isoformat()
            logger.info(f"Using default date range: {start} to {end}")

        # Download historical data
        raw = download_data(symbol, start, end)
        if raw is None or raw.empty:
            raise HTTPException(404, f"No data found for {symbol}. Please check if the symbol is correct and try a different date range.")

        # Check if we have sufficient data
        if len(raw) < 10:  # Need minimum data for meaningful prediction
            raise HTTPException(400, f"Insufficient data for {symbol}. Found only {len(raw)} data points. Need at least 10.")

        # Historical OHLC
        raw_reset = raw.reset_index()
        hist = []
        for _, r in raw_reset.iterrows():
            try:
                hist.append({
                    "date": pd.to_datetime(r["Date"]).strftime("%Y-%m-%d"),
                    "open": float(r["Open"]) if not pd.isna(r["Open"]) else 0.0,
                    "high": float(r["High"]) if not pd.isna(r["High"]) else 0.0,
                    "low": float(r["Low"]) if not pd.isna(r["Low"]) else 0.0,
                    "close": float(r["Close"]) if not pd.isna(r["Close"]) else 0.0,
                    "volume": int(r["Volume"]) if not pd.isna(r["Volume"]) else 0
                })
            except Exception as e:
                logger.warning(f"Skipping row due to error: {e}")
                continue

        if len(hist) == 0:
            raise HTTPException(400, "No valid historical data points found")

        # Prophet model
        df_prop = raw_reset[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"}).dropna()
        
        if len(df_prop) < 3:
            raise HTTPException(400, f"Not enough valid data for forecasting. Found {len(df_prop)} points, need at least 3.")

        logger.info(f"Training Prophet model with {len(df_prop)} data points")
        
        # Configure Prophet with better parameters for stock data
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='multiplicative',  # Better for stock prices
            changepoint_prior_scale=0.05  # More conservative changepoints
        )
        
        model.fit(df_prop)

        # Generate forecast
        future = model.make_future_dataframe(periods=req.predict_days)
        forecast = model.predict(future)

        # Extract only future predictions
        last_hist_date = pd.to_datetime(df_prop["ds"].max())
        fc_rows = forecast[forecast["ds"] > last_hist_date].head(req.predict_days)

        forecast_list = []
        for _, r in fc_rows.iterrows():
            forecast_list.append({
                "date": pd.to_datetime(r["ds"]).strftime("%Y-%m-%d"),
                "predicted": max(0, float(r["yhat"])),  # Ensure non-negative prices
                "lower": max(0, float(r["yhat_lower"])),
                "upper": max(0, float(r["yhat_upper"])),
            })

        logger.info(f"Generated {len(forecast_list)} forecast points")

        return PredictResponse(symbol=symbol, historical=hist, forecast=forecast_list)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.get("/symbols")
def get_popular_symbols():
    """Return some popular Indian stock symbols for testing"""
    return {
        "indian_stocks": [
            "TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", 
            "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "HINDUNILVR.NS"
        ],
        "us_stocks": [
            "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
