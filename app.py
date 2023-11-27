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
st.title("ISTE 470 Project Group 7")

# Define a list of stocks for user selection
stocks = ("ETH-USD", "BTC-USD", "GOOG", "AAPL", "MSFT", "GME")
selected_stock = st.selectbox("Select Stock", stocks)

# Slider for selecting the number of years for prediction
n_years = st.slider("Years of Prediction", 1, 100)
period = n_years * 365

# Function to load historical stock data using Yahoo Finance
@st.cache_data
def load_data(stock):
    data = yfinance.download(stock, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load historical stock data for the selected stock
data = load_data(selected_stock)

# User input for the number of rows to view in the stock data
stock_view = st.number_input("Enter the number of rows to view", min_value=1, max_value=len(data), value=50, step=1)
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

# Initialize and fit the Prophet model
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# User input for the number of rows to view in the forecasted data
rows_to_view = st.number_input("Enter the number of rows to view", min_value=1, max_value=len(forecast), value=50, step=1)

# View specified number of rows of forecasted data as DataFrame
st.subheader(f"Last {rows_to_view} Rows of Forecasted Data")
st.dataframe(forecast.tail(rows_to_view))

# Plot the forecasted data
st.subheader("Forecast data")
fig2 = plot_plotly(m, forecast)
# Update layout for dark background
fig2.update_layout(
    title_text="Time Series data",
    xaxis_rangeslider_visible=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig2)
