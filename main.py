from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from prophet import Prophet
import datetime
import logging
import requests
import time
import random

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

# Custom session with proper headers to avoid Yahoo Finance blocking
def create_session():
    session = requests.Session()
    # Updated headers to bypass Yahoo Finance blocking
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    })
    return session

# ---------- Models ----------
class PredictRequest(BaseModel):
    symbol: str = Field(..., example="INFY.NS")
    start_date: Optional[str] = Field(None, example="2023-01-01")
    end_date: Optional[str] = Field(None, example="2024-09-25")
    time_period_type: Optional[str] = Field(None, example="Months")
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
def download_data_with_fallbacks(symbol: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """Download stock data with multiple fallback strategies"""
    debug_info = {
        "symbol": symbol,
        "requested_range": {"start": start, "end": end},
        "attempts": [],
        "final_data_points": 0
    }
    
    # Strategy 1: yfinance with custom session and headers
    try:
        logger.info(f"Attempt 1: yfinance with custom session for {symbol}")
        
        # Create a custom session
        session = create_session()
        
        # Add random delay to avoid rate limiting
        time.sleep(random.uniform(0.1, 0.5))
        
        # Try with custom session
        df = yf.download(
            tickers=symbol, 
            start=start, 
            end=end, 
            progress=False,
            session=session,
            timeout=30
        )
        
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            debug_info["attempts"].append({"method": "custom_session", "success": True, "data_points": len(df)})
            debug_info["final_data_points"] = len(df)
            logger.info(f"Custom session download successful: {len(df)} points")
            return df, debug_info
        else:
            debug_info["attempts"].append({"method": "custom_session", "success": False, "error": "Empty dataframe"})
            
    except Exception as e:
        logger.warning(f"Custom session download failed: {e}")
        debug_info["attempts"].append({"method": "custom_session", "success": False, "error": str(e)})
    
    # Strategy 2: Direct yfinance with period
    try:
        logger.info(f"Attempt 2: yfinance with period for {symbol}")
        time.sleep(random.uniform(0.1, 0.5))
        
        df = yf.download(tickers=symbol, period="1y", progress=False, timeout=30)
        
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
    
    # Strategy 3: Ticker object with custom session
    try:
        logger.info(f"Attempt 3: Ticker object with custom session for {symbol}")
        time.sleep(random.uniform(0.1, 0.5))
        
        session = create_session()
        ticker = yf.Ticker(symbol, session=session)
        df = ticker.history(start=start, end=end, timeout=30)
        
        if not df.empty:
            debug_info["attempts"].append({"method": "ticker_custom_session", "success": True, "data_points": len(df)})
            debug_info["final_data_points"] = len(df)
            logger.info(f"Ticker custom session successful: {len(df)} points")
            return df, debug_info
        else:
            debug_info["attempts"].append({"method": "ticker_custom_session", "success": False, "error": "Empty dataframe"})
            
    except Exception as e:
        logger.warning(f"Ticker custom session failed: {e}")
        debug_info["attempts"].append({"method": "ticker_custom_session", "success": False, "error": str(e)})
    
    # Strategy 4: Try with different periods using Ticker
    periods = ["1y", "6mo", "3mo", "1mo"]
    for period in periods:
        try:
            logger.info(f"Attempt 4.{periods.index(period)+1}: Ticker with {period} period for {symbol}")
            time.sleep(random.uniform(0.1, 0.5))
            
            session = create_session()
            ticker = yf.Ticker(symbol, session=session)
            df = ticker.history(period=period, timeout=30)
            
            if not df.empty:
                debug_info["attempts"].append({"method": f"ticker_{period}", "success": True, "data_points": len(df)})
                debug_info["final_data_points"] = len(df)
                logger.info(f"Ticker {period} successful: {len(df)} points")
                return df, debug_info
            else:
                debug_info["attempts"].append({"method": f"ticker_{period}", "success": False, "error": "Empty dataframe"})
                
        except Exception as e:
            logger.warning(f"Ticker {period} failed: {e}")
            debug_info["attempts"].append({"method": f"ticker_{period}", "success": False, "error": str(e)})
    
    # Strategy 5: Try alternative data source (fallback to US symbol if Indian)
    if symbol.endswith('.NS') or symbol.endswith('.BO'):
        alt_symbol = symbol.replace('.NS', '').replace('.BO', '')
        try:
            logger.info(f"Attempt 5: Trying alternative symbol {alt_symbol} for {symbol}")
            time.sleep(random.uniform(0.1, 0.5))
            
            session = create_session()
            df = yf.download(tickers=alt_symbol, period="1y", progress=False, session=session, timeout=30)
            
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(0)
                debug_info["attempts"].append({"method": "alternative_symbol", "success": True, "data_points": len(df), "alt_symbol": alt_symbol})
                debug_info["final_data_points"] = len(df)
                logger.info(f"Alternative symbol {alt_symbol} successful: {len(df)} points")
                return df, debug_info
            else:
                debug_info["attempts"].append({"method": "alternative_symbol", "success": False, "error": "Empty dataframe", "alt_symbol": alt_symbol})
                
        except Exception as e:
            logger.warning(f"Alternative symbol {alt_symbol} failed: {e}")
            debug_info["attempts"].append({"method": "alternative_symbol", "success": False, "error": str(e), "alt_symbol": alt_symbol})
    
    # Strategy 6: Manual Yahoo Finance URL (last resort)
    try:
        logger.info(f"Attempt 6: Manual Yahoo Finance API for {symbol}")
        time.sleep(random.uniform(0.1, 0.5))
        
        # Convert dates to timestamps
        start_ts = int(datetime.datetime.strptime(start, "%Y-%m-%d").timestamp())
        end_ts = int(datetime.datetime.strptime(end, "%Y-%m-%d").timestamp())
        
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={start_ts}&period2={end_ts}&interval=1d&events=history"
        
        session = create_session()
        response = session.get(url, timeout=30)
        
        if response.status_code == 200 and response.text and not response.text.startswith('<!DOCTYPE html>'):
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            
            if not df.empty:
                debug_info["attempts"].append({"method": "manual_yahoo_api", "success": True, "data_points": len(df)})
                debug_info["final_data_points"] = len(df)
                logger.info(f"Manual Yahoo API successful: {len(df)} points")
                return df, debug_info
            else:
                debug_info["attempts"].append({"method": "manual_yahoo_api", "success": False, "error": "Empty dataframe from API"})
        else:
            debug_info["attempts"].append({"method": "manual_yahoo_api", "success": False, "error": f"HTTP {response.status_code}"})
            
    except Exception as e:
        logger.warning(f"Manual Yahoo API failed: {e}")
        debug_info["attempts"].append({"method": "manual_yahoo_api", "success": False, "error": str(e)})
    
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
    
    try:
        # Quick test with the new download function
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=30)  # Last 30 days
        
        df, debug_info = download_data_with_fallbacks(symbol, start_date.isoformat(), end_date.isoformat())
        
        return DebugResponse(
            symbol=symbol,
            raw_data_available=len(df) > 0,
            data_points=len(df),
            date_range={
                "start": str(df.index.min()) if not df.empty else None,
                "end": str(df.index.max()) if not df.empty else None
            },
            error=debug_info.get("final_error") if df.empty else None
        )
    except Exception as e:
        return DebugResponse(
            symbol=symbol,
            raw_data_available=False,
            data_points=0,
            date_range={},
            error=str(e)
        )

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        symbol = req.symbol.strip().upper()
        logger.info(f"Processing prediction request for {symbol}")

        # Resolve dates
        if req.start_date and req.end_date:
            start, end = req.start_date, req.end_date
            try:
                start_dt = datetime.datetime.strptime(start, "%Y-%m-%d")
                end_dt = datetime.datetime.strptime(end, "%Y-%m-%d")
                if start_dt >= end_dt:
                    raise HTTPException(400, "start_date must be before end_date")
                if end_dt > datetime.datetime.now():
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
                raise HTTPException(400, "Invalid time_period_type. Use: days, weeks, months, or years")

            end_dt = datetime.date.today()
            start_dt = end_dt - datetime.timedelta(days=days)
            start = start_dt.isoformat()
            end = end_dt.isoformat()
            
            logger.info(f"Calculated date range: {start} to {end} ({days} days)")

        else:
            end_dt = datetime.date.today()
            start_dt = end_dt - datetime.timedelta(days=365)
            start = start_dt.isoformat()
            end = end_dt.isoformat()
            logger.info(f"Using default date range: {start} to {end}")

        # Download historical data with all fallback strategies
        raw, debug_info = download_data_with_fallbacks(symbol, start, end)
        
        if raw is None or raw.empty:
            raise HTTPException(
                404, 
                f"Unable to fetch data for {symbol}. All download methods failed. "
                f"Please check if the symbol is correct or try a different symbol. "
                f"Debug info: {debug_info.get('final_error', 'Unknown error')}"
            )

        if len(raw) < 10:
            raise HTTPException(400, f"Insufficient data for {symbol}. Found only {len(raw)} data points. Need at least 10.")

        # Process historical OHLC data
        raw_reset = raw.reset_index()
        hist = []
        
        # Handle different column formats
        date_col = 'Date' if 'Date' in raw_reset.columns else raw_reset.columns[0]
        
        for _, r in raw_reset.iterrows():
            try:
                hist.append({
                    "date": pd.to_datetime(r[date_col]).strftime("%Y-%m-%d"),
                    "open": float(r["Open"]) if "Open" in r and not pd.isna(r["Open"]) else 0.0,
                    "high": float(r["High"]) if "High" in r and not pd.isna(r["High"]) else 0.0,
                    "low": float(r["Low"]) if "Low" in r and not pd.isna(r["Low"]) else 0.0,
                    "close": float(r["Close"]) if "Close" in r and not pd.isna(r["Close"]) else 0.0,
                    "volume": int(r["Volume"]) if "Volume" in r and not pd.isna(r["Volume"]) else 0
                })
            except Exception as e:
                logger.warning(f"Skipping row due to error: {e}")
                continue

        if len(hist) == 0:
            raise HTTPException(400, "No valid historical data points found after processing")

        # Prepare data for Prophet
        df_prop = raw_reset[[date_col, "Close"]].rename(columns={date_col: "ds", "Close": "y"}).dropna()
        
        if len(df_prop) < 10:
            raise HTTPException(400, f"Not enough valid data for forecasting. Found {len(df_prop)} points, need at least 10.")

        logger.info(f"Training Prophet model with {len(df_prop)} data points")
        
        # Configure Prophet
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            interval_width=0.8
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
        "note": "Try US stocks first to test if the API works, then try Indian stocks"
    }

@app.get("/test-download/{symbol}")
def test_download(symbol: str):
    """Quick test endpoint to check if data download works"""
    try:
        symbol = symbol.strip().upper()
        raw, debug_info = download_data_with_fallbacks(symbol, "2023-01-01", "2024-09-25")
        
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
