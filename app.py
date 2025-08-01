# Import necessary libraries
import streamlit as st
from datetime import date
import yfinance
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go
import pandas as pd
import base64

# Set start date and today's date
START = "2009-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Set Streamlit app title
st.title("CryptoProphet Infinitum Intelligence")

# Define a list of stocks for user selection
stocks = ("ETH-USD", "BTC-USD", "GOOG", "AAPL", "MSFT", "GME")
selected_stock = st.selectbox("Select Stock", stocks)

# Slider for selecting the number of years for prediction
n_years = st.slider("Years of Prediction", 1, 100)
period = n_years * 365

# Function to load historical stock data using Yahoo Finance
@st.cache_data
def load_data(stock):
    try:
        st.info(f"Downloading data for {stock} from {START} to {TODAY}...")
        
        # Try downloading with different parameters
        data = yfinance.download(stock, start=START, end=TODAY, progress=False)
        
        if data.empty:
            st.warning(f"No data returned for {stock}. Trying shorter date range...")
            # Try with a shorter date range (last 2 years)
            short_start = "2022-01-01"
            data = yfinance.download(stock, start=short_start, end=TODAY, progress=False)
        
        if not data.empty:
            data.reset_index(inplace=True)
            st.success(f"Successfully downloaded {len(data)} rows of data for {stock}")
        else:
            st.error(f"Failed to download data for {stock}")
            
        return data
        
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return pd.DataFrame()

# Load historical stock data for the selected stock
data = load_data(selected_stock)

# Debug information
st.subheader("Debug Information")
st.write(f"Selected stock: {selected_stock}")
st.write(f"Data shape: {data.shape if not data.empty else 'No data'}")
if not data.empty:
    st.write(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    st.write(f"Columns: {list(data.columns)}")

# Check if data is available
if data.empty:
    st.error("No data available for the selected stock. This could be due to:")
    st.write("1. Network connectivity issues")
    st.write("2. Yahoo Finance API temporarily unavailable") 
    st.write("3. Invalid stock symbol")
    st.write("4. Data not available for selected date range")
    
    # Try manual test
    st.subheader("Manual Test")
    if st.button("Test yfinance connection"):
        test_stock = "AAPL"  # Known working stock
        test_data = yfinance.download(test_stock, start="2023-01-01", end=TODAY, progress=False)
        if not test_data.empty:
            st.success(f"yfinance is working - downloaded {len(test_data)} rows for {test_stock}")
        else:
            st.error("yfinance appears to be having issues")
    
    st.stop()

# User input for the number of rows to view in the stock data
# Fix: Set default value to minimum of 50 and actual data length
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

# Update layout for dark background
fig.update_layout(
    title_text="Time Series data",
    xaxis_rangeslider_visible=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig)

# Prepare data for forecasting using Prophet
df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Check if we have enough data for forecasting
if len(df_train) < 2:
    st.error("Not enough data points for forecasting. Need at least 2 data points.")
    st.stop()

# Initialize and fit the Prophet model
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# User input for the number of rows to view in the forecasted data
# Fix: Set default value to minimum of 50 and actual forecast length
default_forecast_rows = min(50, len(forecast))
rows_to_view = st.number_input("Enter the number of rows to view for forecast", 
                              min_value=1, 
                              max_value=len(forecast), 
                              value=default_forecast_rows, 
                              step=1)

# View specified number of rows of forecasted data as DataFrame
st.subheader(f"Last {rows_to_view} Rows of Forecasted Data")
st.dataframe(forecast.tail(rows_to_view))

# Plot the forecasted data
st.subheader("Forecast data")
fig2 = plot_plotly(m, forecast)

# Update layout for dark background
fig2.update_layout(
    title_text="Forecast Time Series data",
    xaxis_rangeslider_visible=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig2)
