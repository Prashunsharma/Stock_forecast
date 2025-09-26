from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from prophet import Prophet
import datetime

app = FastAPI(title="Stock Forecast API")

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
    end_date: Optional[str] = Field(None, example="2025-01-01")
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
    df = yf.download(tickers=symbol, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    return df

# ---------- Routes ----------
@app.get("/")
def root():
    return {"message": "✅ Stock Forecast API is running on Render!"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    symbol = req.symbol.strip().upper()

    # Resolve dates
    if req.start_date and req.end_date:
        start, end = req.start_date, req.end_date
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
            raise HTTPException(status_code=400, detail="Invalid time_period_type")

        end_dt = datetime.date.today() + datetime.timedelta(days=1)  # tomorrow
        start_dt = end_dt - datetime.timedelta(days=days)
        start = start_dt.isoformat()
        end = end_dt.isoformat()


    else:
        raise HTTPException(400, "Provide either (start_date,end_date) or (time_period_type,value)")

    # Download historical data
    raw = download_data(symbol, start, end)
    if raw is None or raw.empty:
        raise HTTPException(404, f"No data for {symbol} between {start} and {end}")

    # Historical OHLC
    raw_reset = raw.reset_index()
    hist = []
    for _, r in raw_reset.iterrows():
        hist.append({
            "date": pd.to_datetime(r["Date"]).strftime("%Y-%m-%d"),
            "open": float(r["Open"]),
            "high": float(r["High"]),
            "low": float(r["Low"]),
            "close": float(r["Close"]),
            "volume": int(r["Volume"]) if not pd.isna(r["Volume"]) else 0
        })

    # Prophet model
    df_prop = raw_reset[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"}).dropna()
    if len(df_prop) < 3:
        raise HTTPException(400, "Not enough data for forecasting")

    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df_prop)

    future = model.make_future_dataframe(periods=req.predict_days)
    forecast = model.predict(future)

    last_hist_date = pd.to_datetime(df_prop["ds"].max())
    fc_rows = forecast[forecast["ds"] > last_hist_date].head(req.predict_days)

    forecast_list = []
    for _, r in fc_rows.iterrows():
        forecast_list.append({
            "date": pd.to_datetime(r["ds"]).strftime("%Y-%m-%d"),
            "predicted": float(r["yhat"]),
            "lower": float(r["yhat_lower"]),
            "upper": float(r["yhat_upper"]),
        })

    return PredictResponse(symbol=symbol, historical=hist, forecast=forecast_list)



