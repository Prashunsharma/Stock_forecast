from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from prophet import Prophet
import datetime
import logging
from io import StringIO
import sys

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
    start_date: Optional[str] = Field(None, example="2023-01-01")
    end_date: Optional[str] = Field(None, example="2024-09-25")
    time_period_type: Optional[str] = Field(None, example="Months")  # Days/Weeks/Months/Years
    time_period_value: Optional[int] = Field(None, example=12)
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
    debug_info: Optional[dict] = None

class DebugResponse(BaseModel):
    symbol: str
    raw_data_available: bool
    data_points: int
    date_range: dict
    error: Optional[str] = None

# ---------- Utils ----------
def test_symbol_availability(symbol: str) -> dict:
    """Test if a symbol is available and return debug info"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Try to get basic info
        try:
            info = ticker.info
            logger.info(f"Ticker info available for {symbol}")
        except Exception as e:
            logger.warning(f"No ticker info for {symbol}: {e}")
            info = {}
        
        # Try recent data (last 5 days)
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=5)
        
        try:
            recent_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            logger.info(f"Recent data shape for {symbol}: {recent_data.shape}")
        except Exception as e:
            logger.error(f"Failed to get recent data for {symbol}: {e}")
            recent_data = pd.DataFrame()
        
        # Try longer period (1 year)
        start_1y = end_date - datetime.timedelta(days=365)
        try:
            yearly_data = yf.download(symbol, start=start_1y, end=end_date, progress=False)
            logger.info(f"Yearly data shape for {symbol}: {yearly_data.shape}")
        except Exception as e:
            logger.error(f"Failed to get yearly data for {symbol}: {e}")
            yearly_data = pd.DataFrame()
        
        return {
            "symbol": symbol,
            "info_available": bool(info),
            "recent_data_points": len(recent_data),
            "yearly_data_points": len(yearly_data),
            "recent_date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "yearly_date_range": {
                "start": start_1y.isoformat(),
                "end": end_date.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Symbol test failed for {symbol}: {e}")
        return {
            "symbol": symbol,
            "error": str(e),
            "info_available": False,
            "recent_data_points": 0,
            "yearly_data_points": 0
        }

def download_data_robust(symbol: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """Download stock data with multiple retry strategies and debug info"""
    debug_info = {
        "symbol": symbol,
        "requested_range": {"start": start, "end": end},
        "attempts": [],
        "final_data_points": 0
    }
    
    # Strategy 1: Direct download with requested dates
    try:
        logger.info(f"Attempt 1: Direct download for {symbol} from {start} to {end}")
        df = yf.download(tickers=symbol, start=start, end=end, progress=False)
        
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            debug_info["attempts"].append({"method": "direct", "success": True, "data_points": len(df)})
            debug_info["final_data_points"] = len(df)
            logger.info(f"Direct download successful: {len(df)} points")
            return df, debug_info
        else:
            debug_info["attempts"].append({"method": "direct", "success": False, "error": "Empty dataframe"})
            
    except Exception as e:
        logger.warning(f"Direct download failed: {e}")
        debug_info["attempts"].append({"method": "direct", "success": False, "error": str(e)})
    
    # Strategy 2: Use period parameter instead of dates
    try:
        logger.info(f"Attempt 2: Using period parameter for {symbol}")
        df = yf.download(tickers=symbol, period="1y", progress=False)
        
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            debug_info["attempts"].append({"method": "period_1y", "success": True, "data_points": len(df)})
            debug_info["final_data_points"] = len(df)
            logger.info(f"Period download successful: {len(df)} points")
            return df, debug_info
        else:
            debug_info["attempts"].append({"method": "period_1y", "success": False, "error": "Empty dataframe"})
            
    except Exception as e:
        logger.warning(f"Period download failed: {e}")
        debug_info["attempts"].append({"method": "period_1y", "success": False, "error": str(e)})
    
    # Strategy 3: Try with Ticker object
    try:
        logger.info(f"Attempt 3: Using Ticker object for {symbol}")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end)
        
        if not df.empty:
            # Rename columns to match expected format
            if 'Close' not in df.columns and 'close' in df.columns:
                df = df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
            debug_info["attempts"].append({"method": "ticker_history", "success": True, "data_points": len(df)})
            debug_info["final_data_points"] = len(df)
            logger.info(f"Ticker history successful: {len(df)} points")
            return df, debug_info
        else:
            debug_info["attempts"].append({"method": "ticker_history", "success": False, "error": "Empty dataframe"})
            
    except Exception as e:
        logger.warning(f"Ticker history failed: {e}")
        debug_info["attempts"].append({"method": "ticker_history", "success": False, "error": str(e)})
    
    # Strategy 4: Try shorter period if long period failed
    try:
        logger.info(f"Attempt 4: Using shorter period (6m) for {symbol}")
        df = yf.download(tickers=symbol, period="6mo", progress=False)
        
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            debug_info["attempts"].append({"method": "period_6mo", "success": True, "data_points": len(df)})
            debug_info["final_data_points"] = len(df)
            logger.info(f"6-month period download successful: {len(df)} points")
            return df, debug_info
        else:
            debug_info["attempts"].append({"method": "period_6mo", "success": False, "error": "Empty dataframe"})
            
    except Exception as e:
        logger.warning(f"6-month period download failed: {e}")
        debug_info["attempts"].append({"method": "period_6mo", "success": False, "error": str(e)})
    
    # All strategies failed
    debug_info["final_error"] = "All download strategies failed"
    return pd.DataFrame(), debug_info

# ---------- Routes ----------
@app.get("/")
def root():
    return {
        "message": "✅ Stock Forecast API is running!", 
        "timestamp": datetime.datetime.now().isoformat(),
        "yfinance_version": yf.__version__ if hasattr(yf, '__version__') else "unknown"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

@app.get("/debug/{symbol}", response_model=DebugResponse)
def debug_symbol(symbol: str):
    """Debug endpoint to test symbol availability"""
    symbol = symbol.strip().upper()
    debug_info = test_symbol_availability(symbol)
    
    return DebugResponse(
        symbol=symbol,
        raw_data_available=debug_info.get("yearly_data_points", 0) > 0,
        data_points=debug_info.get("yearly_data_points", 0),
        date_range=debug_info.get("yearly_date_range", {}),
        error=debug_info.get("error")
    )

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
                if end_dt > datetime.datetime.now():
                    # Adjust end date to today if it's in the future
                    end = datetime.date.today().isoformat()
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

            # Calculate date range - go back from today
            end_dt = datetime.date.today()
            start_dt = end_dt - datetime.timedelta(days=days)
            start = start_dt.isoformat()
            end = end_dt.isoformat()
            
            logger.info(f"Calculated date range: {start} to {end} ({days} days)")

        else:
            # Default: last 1 year of data
            end_dt = datetime.date.today()
            start_dt = end_dt - datetime.timedelta(days=365)
            start = start_dt.isoformat()
            end = end_dt.isoformat()
            logger.info(f"Using default date range: {start} to {end}")

        # Download historical data with robust method
        raw, debug_info = download_data_robust(symbol, start, end)
        
        if raw is None or raw.empty:
            # Create detailed error message
            error_details = {
                "symbol": symbol,
                "requested_range": {"start": start, "end": end},
                "debug_info": debug_info,
                "suggestions": [
                    "Try a different symbol format (e.g., TCS.NS, TCS.BO for Indian stocks)",
                    "Check if the symbol exists on Yahoo Finance website",
                    "Try a different date range",
                    f"Use the debug endpoint: GET /debug/{symbol}"
                ]
            }
            raise HTTPException(
                404, 
                f"No data found for {symbol}. Please check the debug info and suggestions.",
                # details=error_details  # FastAPI will include this in response
            )

        # Check if we have sufficient data
        if len(raw) < 10:
            raise HTTPException(400, f"Insufficient data for {symbol}. Found only {len(raw)} data points. Need at least 10.")

        # Historical OHLC
        raw_reset = raw.reset_index()
        hist = []
        
        # Handle different column formats
        date_col = 'Date' if 'Date' in raw_reset.columns else raw_reset.index.name or 'Date'
        if date_col not in raw_reset.columns:
            raw_reset = raw_reset.reset_index()
            date_col = raw_reset.columns[0]
        
        for _, r in raw_reset.iterrows():
            try:
                hist.append({
                    "date": pd.to_datetime(r[date_col]).strftime("%Y-%m-%d"),
                    "open": float(r["Open"]) if not pd.isna(r["Open"]) else 0.0,
                    "high": float(r["High"]) if not pd.isna(r["High"]) else 0.0,
                    "low": float(r["Low"]) if not pd.isna(r["Low"]) else 0.0,
                    "close": float(r["Close"]) if not pd.isna(r["Close"]) else 0.0,
                    "volume": int(r["Volume"]) if "Volume" in r and not pd.isna(r["Volume"]) else 0
                })
            except Exception as e:
                logger.warning(f"Skipping row due to error: {e}")
                continue

        if len(hist) == 0:
            raise HTTPException(400, "No valid historical data points found after processing")

        # Prophet model
        df_prop = raw_reset[[date_col, "Close"]].rename(columns={date_col: "ds", "Close": "y"}).dropna()
        
        if len(df_prop) < 10:
            raise HTTPException(400, f"Not enough valid data for forecasting. Found {len(df_prop)} points, need at least 10.")

        logger.info(f"Training Prophet model with {len(df_prop)} data points")
        
        # Configure Prophet with better parameters for stock data
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            interval_width=0.8  # 80% confidence interval
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
                "predicted": max(0, float(r["yhat"])),
                "lower": max(0, float(r["yhat_lower"])),
                "upper": max(0, float(r["yhat_upper"])),
            })

        logger.info(f"Generated {len(forecast_list)} forecast points")

        return PredictResponse(
            symbol=symbol, 
            historical=hist, 
            forecast=forecast_list,
            debug_info=debug_info
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.get("/symbols")
def get_popular_symbols():
    """Return some popular stock symbols for testing"""
    return {
        "indian_stocks_nse": [
            "TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", 
            "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "HINDUNILVR.NS"
        ],
        "indian_stocks_bse": [
            "TCS.BO", "INFY.BO", "RELIANCE.BO", "HDFCBANK.BO"
        ],
        "us_stocks": [
            "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA"
        ],
        "note": "Use .NS for NSE (National Stock Exchange) or .BO for BSE (Bombay Stock Exchange) for Indian stocks"
    }

@app.get("/test-download/{symbol}")
def test_download(symbol: str):
    """Quick test endpoint to check if data download works"""
    try:
        symbol = symbol.strip().upper()
        raw, debug_info = download_data_robust(symbol, "2023-01-01", "2024-09-25")
        
        return {
            "symbol": symbol,
            "success": not raw.empty,
            "data_points": len(raw),
            "columns": list(raw.columns) if not raw.empty else [],
            "date_range": {
                "start": str(raw.index.min()) if not raw.empty else None,
                "end": str(raw.index.max()) if not raw.empty else None
            },
            "debug_info": debug_info
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
