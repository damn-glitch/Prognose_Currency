# Alternative version using multiple data sources
import streamlit as st
from datetime import date, datetime, timedelta
import yfinance
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go
import pandas as pd
import requests
import time

START = "2009-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("CryptoProphet Infinitum Intelligence")

stocks = ("ETH-USD", "BTC-USD", "GOOG", "AAPL", "MSFT", "GME")
selected_stock = st.selectbox("Select Stock", stocks)

n_years = st.slider("Years of Prediction", 1, 100)
period = n_years * 365

@st.cache_data
def load_data_yfinance(stock):
    """Try to load data using yfinance"""
    try:
        st.info(f"Trying yfinance for {stock}...")
        data = yfinance.download(stock, start=START, end=TODAY, progress=False)
        if not data.empty:
            data.reset_index(inplace=True)
            return data, "yfinance"
    except Exception as e:
        st.warning(f"yfinance failed: {str(e)}")
    return pd.DataFrame(), None

@st.cache_data  
def create_sample_data(stock):
    """Create sample data if real data unavailable"""
    st.warning("Creating sample data for demonstration...")
    
    # Create sample data with realistic patterns
    dates = pd.date_range(start="2023-01-01", end=TODAY, freq='D')
    
    # Base price depending on stock
    if "ETH" in stock:
        base_price = 2000
    elif "BTC" in stock:
        base_price = 40000
    elif stock == "AAPL":
        base_price = 150
    else:
        base_price = 100
    
    # Generate realistic price movements
    import numpy as np
    np.random.seed(42)  # For reproducible results
    
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
    prices = [base_price]
    
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    return data, "sample"

# Try to load data
data, source = load_data_yfinance(selected_stock)

if data.empty:
    st.error("Unable to download real data. Using sample data for demonstration.")
    data, source = create_sample_data(selected_stock)

# Display data source info
if source == "sample":
    st.warning("⚠️ Using sample data - not real market data!")
else:
    st.success(f"✅ Using real data from {source}")

st.write(f"Data shape: {data.shape}")
st.write(f"Date range: {data['Date'].min()} to {data['Date'].max()}")

# Continue with the rest of your app...
default_rows = min(50, len(data))
stock_view = st.number_input("Enter the number of rows to view", 
                           min_value=1, 
                           max_value=len(data), 
                           value=default_rows, 
                           step=1)

st.subheader(f"Stock data of last {stock_view} rows")
st.write(data.tail(stock_view))

# Create a plot for historical stock data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="stock_open"))
fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="stock_close"))

fig.update_layout(
    title_text="Time Series data",
    xaxis_rangeslider_visible=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig)

# Prepare data for forecasting
df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

if len(df_train) < 2:
    st.error("Not enough data points for forecasting.")
    st.stop()

# Prophet forecasting
