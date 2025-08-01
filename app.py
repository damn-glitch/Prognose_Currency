# Import necessary libraries
import streamlit as st
from datetime import date, datetime, timedelta
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="CryptoProphet Ultimate",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .prediction-highlight {
        background: linear-gradient(90deg, #ffeaa7, #fdcb6e);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #e17055;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Main title with custom styling
st.markdown('<h1 class="main-header">🚀 CryptoProphet Ultimate Intelligence 🚀</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("⚙️ Configuration Panel")

# Enhanced stock selection with categories
crypto_stocks = ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "SOL-USD", "MATIC-USD", "AVAX-USD", "LINK-USD"]
tech_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
traditional_stocks = ["SPY", "QQQ", "VOO", "JPM", "JNJ", "PG", "KO", "WMT"]

category = st.sidebar.radio("📊 Select Category", ["Cryptocurrencies", "Tech Stocks", "Traditional Stocks"])

if category == "Cryptocurrencies":
    stocks = crypto_stocks
elif category == "Tech Stocks":
    stocks = tech_stocks
else:
    stocks = traditional_stocks

selected_stock = st.sidebar.selectbox("🎯 Select Asset", stocks)

# Enhanced prediction options
st.sidebar.subheader("🔮 Prediction Settings")
n_years = st.sidebar.slider("Years of Prediction", 1, 10, 2)
period = n_years * 365

# Advanced options
show_components = st.sidebar.checkbox("📈 Show Trend Components", True)
show_confidence = st.sidebar.checkbox("📊 Show Confidence Intervals", True)
include_volume = st.sidebar.checkbox("📊 Include Volume Analysis", True)

# Data loading with multiple fallback strategies
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data_advanced(stock):
    """Advanced data loading with multiple strategies and error handling"""
    
    strategies = [
        ("Full Range", START, TODAY),
        ("5 Years", "2019-01-01", TODAY),
        ("3 Years", "2021-01-01", TODAY),
        ("2 Years", "2022-01-01", TODAY),
        ("1 Year", "2023-01-01", TODAY)
    ]
    
    for strategy_name, start_date, end_date in strategies:
        try:
            with st.spinner(f"📥 Trying {strategy_name} data for {stock}..."):
                data = yf.download(stock, start=start_date, end=end_date, progress=False)
                
                if not data.empty and len(data) > 30:  # Need at least 30 days
                    data.reset_index(inplace=True)
                    st.success(f"✅ Successfully loaded {len(data)} days of data using {strategy_name} strategy")
                    return data, strategy_name
                    
        except Exception as e:
            continue
    
    # If all else fails, create realistic sample data
    st.warning("⚠️ Creating sample data for demonstration")
    return create_realistic_sample_data(stock), "Sample Data"

def create_realistic_sample_data(stock):
    """Create realistic sample data based on asset type"""
    
    # Determine asset characteristics
    if "BTC" in stock:
        base_price, volatility = 45000, 0.04
    elif "ETH" in stock:
        base_price, volatility = 2500, 0.05
    elif stock in ["AAPL", "MSFT", "GOOGL"]:
        base_price, volatility = 150, 0.02
    else:
        base_price, volatility = 100, 0.025
    
    # Generate realistic price data
    dates = pd.date_range(start="2022-01-01", end=TODAY, freq='D')
    np.random.seed(42)
    
    returns = np.random.normal(0.0008, volatility, len(dates))
    prices = [base_price]
    
    for return_rate in returns[1:]:
        new_price = prices[-1] * (1 + return_rate)
        prices.append(max(new_price, base_price * 0.1))  # Prevent unrealistic crashes
    
    # Add some trend and seasonality
    trend = np.linspace(0, 0.3, len(dates))
    seasonal = 0.1 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    prices = np.array(prices) * (1 + trend + seasonal)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices * np.random.uniform(0.99, 1.01, len(prices)),
        'High': prices * np.random.uniform(1.00, 1.05, len(prices)),
        'Low': prices * np.random.uniform(0.95, 1.00, len(prices)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 50000000, len(prices))
    })
    
    return data

# Load data
data, data_source = load_data_advanced(selected_stock)

# Display data source information
if data_source == "Sample Data":
    st.error("🚨 Using sample data - not real market data!")
else:
    st.info(f"📊 Data source: {data_source} | Last updated: {data['Date'].max().strftime('%Y-%m-%d')}")

# Main dashboard layout
col1, col2, col3, col4 = st.columns(4)

current_price = data['Close'].iloc[-1]
prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
price_change = current_price - prev_price
price_change_pct = (price_change / prev_price) * 100

with col1:
    st.metric("💰 Current Price", f"${current_price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)")

with col2:
    st.metric("📊 Volume", f"{data['Volume'].iloc[-1]:,.0f}")

with col3:
    high_52w = data['High'].tail(252).max() if len(data) >= 252 else data['High'].max()
    st.metric("📈 52W High", f"${high_52w:.2f}")

with col4:
    low_52w = data['Low'].tail(252).min() if len(data) >= 252 else data['Low'].min()
    st.metric("📉 52W Low", f"${low_52w:.2f}")

# Data overview section
st.subheader("📊 Data Overview")

tab1, tab2, tab3 = st.tabs(["📈 Price Chart", "📋 Data Table", "📊 Statistics"])

with tab1:
    # Enhanced price chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price Movement', 'Volume'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True
    )
    
    # Price traces
    fig.add_trace(go.Scatter(
        x=data["Date"], y=data["Close"], 
        name="Close Price", line=dict(color="#00cc96", width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=data["Date"], y=data["Open"], 
        name="Open Price", line=dict(color="#ff6692", width=1),
        opacity=0.7
    ), row=1, col=1)
    
    # Volume
    if include_volume:
        fig.add_trace(go.Bar(
            x=data["Date"], y=data["Volume"],
            name="Volume", marker_color="#ab63fa"
        ), row=2, col=1)
    
    fig.update_layout(
        title=f"📈 {selected_stock} - Historical Price & Volume Analysis",
        xaxis_rangeslider_visible=True,
        template="plotly_dark",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Interactive data table
    display_rows = st.slider("Rows to display", 10, min(100, len(data)), 20)
    
    # Calculate additional metrics
    data_display = data.tail(display_rows).copy()
    data_display['Daily Change %'] = ((data_display['Close'] - data_display['Open']) / data_display['Open'] * 100).round(2)
    data_display['High-Low %'] = ((data_display['High'] - data_display['Low']) / data_display['Low'] * 100).round(2)
    
    st.dataframe(
        data_display[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Daily Change %', 'High-Low %']],
        use_container_width=True
    )

with tab3:
    # Statistical analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Price Statistics")
        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
            'Value': [
                f"${data['Close'].mean():.2f}",
                f"${data['Close'].median():.2f}",
                f"${data['Close'].std():.2f}",
                f"${data['Close'].min():.2f}",
                f"${data['Close'].max():.2f}",
                f"{data['Close'].skew():.3f}",
                f"{data['Close'].kurtosis():.3f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("📈 Returns Analysis")
        returns = data['Close'].pct_change().dropna()
        returns_stats = pd.DataFrame({
            'Metric': ['Daily Return Mean', 'Daily Return Std', 'Sharpe Ratio (approx)', 'Max Daily Gain', 'Max Daily Loss'],
            'Value': [
                f"{returns.mean()*100:.3f}%",
                f"{returns.std()*100:.3f}%",
                f"{(returns.mean()/returns.std())*np.sqrt(252):.2f}",
                f"{returns.max()*100:.2f}%",
                f"{returns.min()*100:.2f}%"
            ]
        })
        st.dataframe(returns_stats, use_container_width=True, hide_index=True)

# Prophet Forecasting Section
st.subheader("🔮 AI-Powered Forecasting with Prophet")

# Prepare data for Prophet
df_train = data[["Date", "Close"]].copy()
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Enhanced Prophet model with custom parameters
with st.spinner("🤖 Training AI model..."):
    m = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        holidays_prior_scale=10,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.8 if show_confidence else 0.95
    )
    m.fit(df_train)

# Make future predictions
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Forecast visualization
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Forecast Visualization")
    
    # Create enhanced forecast plot
    fig_forecast = go.Figure()
    
    # Historical data
    fig_forecast.add_trace(go.Scatter(
        x=df_train['ds'], y=df_train['y'],
        mode='lines', name='Historical Data',
        line=dict(color='#00cc96', width=2)
    ))
    
    # Forecast line
    future_data = forecast[forecast['ds'] > df_train['ds'].max()]
    fig_forecast.add_trace(go.Scatter(
        x=future_data['ds'], y=future_data['yhat'],
        mode='lines', name='Forecast',
        line=dict(color='#ff6692', width=3, dash='dash')
    ))
    
    # Confidence intervals
    if show_confidence:
        fig_forecast.add_trace(go.Scatter(
            x=future_data['ds'], y=future_data['yhat_upper'],
            fill=None, mode='lines',
            line=dict(color='rgba(255, 102, 146, 0)'),
            showlegend=False
        ))
        fig_forecast.add_trace(go.Scatter(
            x=future_data['ds'], y=future_data['yhat_lower'],
            fill='tonexty', mode='lines',
            fillcolor='rgba(255, 102, 146, 0.2)',
            line=dict(color='rgba(255, 102, 146, 0)'),
            name='Confidence Interval'
        ))
    
    fig_forecast.update_layout(
        title=f"🔮 {selected_stock} Price Forecast - Next {n_years} Year(s)",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_dark",
        height=500
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)

with col2:
    st.subheader("🎯 Key Predictions")
    
    # Calculate key prediction metrics
    current_price = df_train['y'].iloc[-1]
    
    # Future predictions at specific intervals
    future_30d = forecast[forecast['ds'] == (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')]
    future_90d = forecast[forecast['ds'] == (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')]
    future_1y = forecast[forecast['ds'] == (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')]
    
    def get_prediction_value(pred_df):
        return pred_df['yhat'].iloc[0] if not pred_df.empty else None
    
    pred_30d = get_prediction_value(future_30d)
    pred_90d = get_prediction_value(future_90d)
    pred_1y = get_prediction_value(future_1y)
    
    if pred_30d:
        change_30d = ((pred_30d - current_price) / current_price) * 100
        st.markdown(f"""
        <div class="prediction-highlight">
        <h4>📅 30 Days</h4>
        <p><strong>${pred_30d:.2f}</strong> ({change_30d:+.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    if pred_90d:
        change_90d = ((pred_90d - current_price) / current_price) * 100
        st.markdown(f"""
        <div class="prediction-highlight">
        <h4>📅 90 Days</h4>
        <p><strong>${pred_90d:.2f}</strong> ({change_90d:+.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    if pred_1y:
        change_1y = ((pred_1y - current_price) / current_price) * 100
        st.markdown(f"""
        <div class="prediction-highlight">
        <h4>📅 1 Year</h4>
        <p><strong>${pred_1y:.2f}</strong> ({change_1y:+.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)

# Show trend components
if show_components:
    st.subheader("📊 Trend Component Analysis")
    
    fig_components = m.plot_components(forecast)
    st.pyplot(fig_components)

# Detailed forecast table
st.subheader("📋 Detailed Forecast Table")

# Future predictions table
future_predictions = forecast[forecast['ds'] > df_train['ds'].max()].copy()
future_predictions['Price Change'] = future_predictions['yhat'] - current_price
future_predictions['Price Change %'] = (future_predictions['Price Change'] / current_price) * 100

# Display options for the table
table_cols = st.columns(3)
with table_cols[0]:
    table_rows = st.selectbox("Rows to show", [10, 25, 50, 100], index=1)
with table_cols[1]:
    show_confidence_table = st.checkbox("Show confidence bounds in table", True)
with table_cols[2]:
    date_filter = st.selectbox("Filter by", ["All", "Next 30 days", "Next 90 days", "Next 1 year"])

# Apply filters
if date_filter == "Next 30 days":
    cutoff_date = datetime.now() + timedelta(days=30)
    future_predictions = future_predictions[future_predictions['ds'] <= cutoff_date.strftime('%Y-%m-%d')]
elif date_filter == "Next 90 days":
    cutoff_date = datetime.now() + timedelta(days=90)
    future_predictions = future_predictions[future_predictions['ds'] <= cutoff_date.strftime('%Y-%m-%d')]
elif date_filter == "Next 1 year":
    cutoff_date = datetime.now() + timedelta(days=365)
    future_predictions = future_predictions[future_predictions['ds'] <= cutoff_date.strftime('%Y-%m-%d')]

# Select columns for display
display_columns = ['ds', 'yhat', 'Price Change', 'Price Change %']
if show_confidence_table:
    display_columns.extend(['yhat_lower', 'yhat_upper'])

# Format the table
table_data = future_predictions[display_columns].head(table_rows).copy()
table_data.columns = ['Date', 'Predicted Price', 'Price Change ($)', 'Price Change (%)', 'Lower Bound', 'Upper Bound'][:len(display_columns)]

# Round numerical columns
for col in table_data.columns:
    if col != 'Date':
        table_data[col] = table_data[col].round(2)

st.dataframe(table_data, use_container_width=True, hide_index=True)

# Export functionality
st.subheader("💾 Export Data")

col1, col2, col3 = st.columns(3)

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

with col1:
    csv_historical = convert_df_to_csv(data)
    st.download_button(
        label="📥 Download Historical Data",
        data=csv_historical,
        file_name=f"{selected_stock}_historical_data.csv",
        mime="text/csv"
    )

with col2:
    csv_forecast = convert_df_to_csv(future_predictions)
    st.download_button(
        label="📥 Download Forecast Data",
        data=csv_forecast,
        file_name=f"{selected_stock}_forecast_data.csv",
        mime="text/csv"
    )

with col3:
    # Create a summary report
    summary_data = {
        'Asset': [selected_stock],
        'Current Price': [f"${current_price:.2f}"],
        'Data Source': [data_source],
        'Forecast Period': [f"{n_years} year(s)"],
        'Generated On': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    }
    summary_df = pd.DataFrame(summary_data)
    csv_summary = convert_df_to_csv(summary_df)
    st.download_button(
        label="📥 Download Summary",
        data=csv_summary,
        file_name=f"{selected_stock}_summary.csv",
        mime="text/csv"
    )

