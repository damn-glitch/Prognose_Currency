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

# Safe price calculations with error handling
def safe_format_price(value, default="N/A"):
    """Safely format price values, handling NaN and None"""
    try:
        if pd.isna(value) or value is None:
            return default
        return f"${float(value):.2f}"
    except:
        return default

def safe_format_number(value, default="N/A"):
    """Safely format numbers, handling NaN and None"""
    try:
        if pd.isna(value) or value is None:
            return default
        return f"{float(value):,.0f}"
    except:
        return default

def safe_format_change(current, previous):
    """Safely calculate and format price changes"""
    try:
        if pd.isna(current) or pd.isna(previous) or current is None or previous is None or previous == 0:
            return "N/A", "N/A"
        
        change = float(current) - float(previous)
        change_pct = (change / float(previous)) * 100
        return f"{change:+.2f}", f"{change_pct:+.2f}%"
    except:
        return "N/A", "N/A"

# Calculate metrics safely
current_price = data['Close'].iloc[-1] if not data.empty else None
prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
current_volume = data['Volume'].iloc[-1] if not data.empty else None

# Calculate price changes
price_change_str, price_change_pct_str = safe_format_change(current_price, prev_price)

with col1:
    price_display = safe_format_price(current_price)
    if price_change_str != "N/A" and price_change_pct_str != "N/A":
        st.metric("💰 Current Price", price_display, f"{price_change_str} ({price_change_pct_str})")
    else:
        st.metric("💰 Current Price", price_display)

with col2:
    volume_display = safe_format_number(current_volume)
    st.metric("📊 Volume", volume_display)

with col3:
    try:
        high_52w = data['High'].tail(252).max() if len(data) >= 252 else data['High'].max()
        high_52w_display = safe_format_price(high_52w)
    except:
        high_52w_display = "N/A"
    st.metric("📈 52W High", high_52w_display)

with col4:
    try:
        low_52w = data['Low'].tail(252).min() if len(data) >= 252 else data['Low'].min()
        low_52w_display = safe_format_price(low_52w)
    except:
        low_52w_display = "N/A"
    st.metric("📉 52W Low", low_52w_display)

# Data overview section
st.subheader("📊 Data Overview")

tab1, tab2, tab3 = st.tabs(["📈 Price Chart", "📋 Data Table", "📊 Statistics"])

with tab1:
    # Enhanced price chart
    try:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price Movement', 'Volume'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True
        )
        
        # Clean data for plotting
        plot_data = data.dropna(subset=['Date', 'Close', 'Open'])
        
        if not plot_data.empty:
            # Price traces
            fig.add_trace(go.Scatter(
                x=plot_data["Date"], y=plot_data["Close"], 
                name="Close Price", line=dict(color="#00cc96", width=2)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=plot_data["Date"], y=plot_data["Open"], 
                name="Open Price", line=dict(color="#ff6692", width=1),
                opacity=0.7
            ), row=1, col=1)
            
            # Volume
            if include_volume and 'Volume' in plot_data.columns:
                volume_data = plot_data.dropna(subset=['Volume'])
                if not volume_data.empty:
                    fig.add_trace(go.Bar(
                        x=volume_data["Date"], y=volume_data["Volume"],
                        name="Volume", marker_color="#ab63fa"
                    ), row=2, col=1)
            
            fig.update_layout(
                title=f"📈 {selected_stock} - Historical Price & Volume Analysis",
                xaxis_rangeslider_visible=True,
                template="plotly_dark",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Insufficient data for price chart")
            
    except Exception as e:
        st.error("Unable to create price chart. Please try a different asset or time period.")

with tab2:
    # Interactive data table
    display_rows = st.slider("Rows to display", 10, min(100, len(data)), 20)
    
    # Calculate additional metrics safely
    try:
        data_display = data.tail(display_rows).copy()
        
        # Calculate daily changes with error handling
        data_display['Daily Change %'] = ((data_display['Close'] - data_display['Open']) / data_display['Open'] * 100)
        data_display['High-Low %'] = ((data_display['High'] - data_display['Low']) / data_display['Low'] * 100)
        
        # Handle NaN values
        data_display['Daily Change %'] = data_display['Daily Change %'].fillna(0).round(2)
        data_display['High-Low %'] = data_display['High-Low %'].fillna(0).round(2)
        
        # Select and display columns
        display_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Daily Change %', 'High-Low %']
        available_columns = [col for col in display_columns if col in data_display.columns]
        
        st.dataframe(
            data_display[available_columns],
            use_container_width=True
        )
    except Exception as e:
        st.error("Unable to display data table. Showing raw data instead.")
        st.dataframe(data.tail(display_rows), use_container_width=True)

with tab3:
    # Statistical analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Price Statistics")
        try:
            close_prices = data['Close'].dropna()
            if len(close_prices) > 0:
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
                    'Value': [
                        f"${close_prices.mean():.2f}",
                        f"${close_prices.median():.2f}",
                        f"${close_prices.std():.2f}",
                        f"${close_prices.min():.2f}",
                        f"${close_prices.max():.2f}",
                        f"{close_prices.skew():.3f}",
                        f"{close_prices.kurtosis():.3f}"
                    ]
                })
            else:
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
                    'Value': ['N/A'] * 7
                })
        except Exception as e:
            stats_df = pd.DataFrame({
                'Metric': ['Error'],
                'Value': ['Unable to calculate statistics']
            })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("📈 Returns Analysis")
        try:
            returns = data['Close'].pct_change().dropna()
            if len(returns) > 0 and not returns.empty:
                sharpe_ratio = (returns.mean()/returns.std())*np.sqrt(252) if returns.std() != 0 else 0
                returns_stats = pd.DataFrame({
                    'Metric': ['Daily Return Mean', 'Daily Return Std', 'Sharpe Ratio (approx)', 'Max Daily Gain', 'Max Daily Loss'],
                    'Value': [
                        f"{returns.mean()*100:.3f}%",
                        f"{returns.std()*100:.3f}%",
                        f"{sharpe_ratio:.2f}",
                        f"{returns.max()*100:.2f}%",
                        f"{returns.min()*100:.2f}%"
                    ]
                })
            else:
                returns_stats = pd.DataFrame({
                    'Metric': ['Daily Return Mean', 'Daily Return Std', 'Sharpe Ratio (approx)', 'Max Daily Gain', 'Max Daily Loss'],
                    'Value': ['N/A'] * 5
                })
        except Exception as e:
            returns_stats = pd.DataFrame({
                'Metric': ['Error'],
                'Value': ['Unable to calculate returns']
            })
        st.dataframe(returns_stats, use_container_width=True, hide_index=True)

# Prophet Forecasting Section
st.subheader("🔮 AI-Powered Forecasting with Prophet")

# Check if we have the required data
if data.empty:
    st.error("❌ No data available for forecasting.")
    st.stop()

# Verify required columns exist
required_columns = ['Date', 'Close']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    st.error(f"❌ Missing required columns: {missing_columns}")
    st.write(f"Available columns: {list(data.columns)}")
    st.stop()

# Prepare data for Prophet with comprehensive cleaning
try:
    # Extract only the columns we need
    df_train = data[["Date", "Close"]].copy()
    
    # Check if we got data
    if df_train.empty:
        st.error("❌ No data available after extracting Date and Close columns.")
        st.stop()
    
    # Remove any NaN values first
    df_train = df_train.dropna()
    
    # Rename columns for Prophet
    df_train.columns = ['ds', 'y']
    
    # Verify the rename worked by checking columns again
    if 'ds' not in df_train.columns or 'y' not in df_train.columns:
        st.error("❌ Column renaming failed.")
        st.write(f"Columns after rename: {list(df_train.columns)}")
        st.stop()
        
except Exception as e:
    st.error(f"❌ Error preparing data for Prophet: {str(e)}")
    st.write(f"Data info: Shape={data.shape if not data.empty else 'No data'}, Columns={list(data.columns) if not data.empty else 'No data'}")
    st.stop()

# Comprehensive data cleaning for Prophet
st.info("🧹 Preprocessing data for AI model...")

try:
    # Store initial row count
    initial_rows = len(df_train)
    
    # Remove any infinite values
    df_train = df_train[np.isfinite(df_train['y'])]
    
    # Ensure we have valid dates
    df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce')
    df_train = df_train.dropna(subset=['ds'])
    
    # Remove any zero or negative prices (invalid for financial data)
    df_train = df_train[df_train['y'] > 0]
    
    # Sort by date to ensure chronological order
    df_train = df_train.sort_values('ds').reset_index(drop=True)
    
    # Remove duplicate dates (keep the last value for each date)
    df_train = df_train.drop_duplicates(subset=['ds'], keep='last')
    
    cleaned_rows = len(df_train)
    
    # Debug information
    st.write(f"📊 **Data Summary for AI Training:**")
    st.write(f"- Initial data points: {initial_rows}")
    st.write(f"- After cleaning: {cleaned_rows}")
    st.write(f"- Data removed: {initial_rows - cleaned_rows} points")
    
    if not df_train.empty:
        try:
            st.write(f"- Date range: {df_train['ds'].min().strftime('%Y-%m-%d')} to {df_train['ds'].max().strftime('%Y-%m-%d')}")
            st.write(f"- Price range: ${df_train['y'].min():.2f} to ${df_train['y'].max():.2f}")
            st.write(f"- Data quality: {'✅ Good' if len(df_train) > 30 else '⚠️ Limited'}")
        except Exception as e:
            st.write(f"- Error displaying data summary: {str(e)}")
            
except Exception as e:
    st.error(f"❌ Error during data cleaning: {str(e)}")
    st.stop()

# Check if we have sufficient data
if df_train.empty:
    st.error("❌ No valid data available for forecasting after cleaning.")
    st.info("💡 **Possible solutions:**")
    st.write("1. Try selecting a different asset")
    st.write("2. Check your internet connection")
    st.write("3. The selected asset might not have sufficient historical data")
    st.stop()

elif len(df_train) < 10:
    st.error(f"❌ Insufficient data for reliable forecasting. Found only {len(df_train)} valid data points.")
    st.info("💡 **Minimum requirement:** At least 10 data points needed for forecasting.")
    st.write("**Available data:**")
    st.dataframe(df_train.head(10))
    st.stop()

elif len(df_train) < 30:
    st.warning(f"⚠️ Limited data available ({len(df_train)} points). Predictions may be less accurate.")

# Check date consistency
try:
    date_diff = df_train['ds'].diff().dt.days.median()
    if pd.isna(date_diff) or date_diff > 7:
        st.warning("⚠️ Irregular data frequency detected. This may affect prediction accuracy.")
except:
    st.warning("⚠️ Unable to analyze date consistency.")

# Enhanced Prophet model with custom parameters and error handling
with st.spinner("🤖 Training AI model..."):
    try:
        # Start with basic Prophet model
        m = Prophet(
            changepoint_prior_scale=0.01,  # More conservative
            seasonality_prior_scale=1,     # Less aggressive seasonality
            daily_seasonality=False,
            weekly_seasonality=len(df_train) > 14,  # Only if we have enough data
            yearly_seasonality=len(df_train) > 365,  # Only if we have enough data
            interval_width=0.95
        )
        
        # Add monthly seasonality if we have enough data
        if len(df_train) > 60:
            m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        # Fit the model
        m.fit(df_train)
        
        # Make future predictions
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)
        
        st.success("✅ AI model trained successfully!")
        
        # Validate forecast results
        if forecast.empty or forecast['yhat'].isna().all():
            raise Exception("Forecast generated no valid predictions")
        
        # Check for reasonable predictions (not too extreme)
        current_price = df_train['y'].iloc[-1]
        max_prediction = forecast['yhat'].max()
        min_prediction = forecast['yhat'].min()
        
        if max_prediction > current_price * 100 or min_prediction < current_price * 0.01:
            st.warning("⚠️ Model generated extreme predictions. Results may be unreliable.")
        
    except Exception as e:
        st.error("❌ Unable to train forecasting model.")
        
        # Provide detailed error information and solutions
        st.info("🔧 **Troubleshooting Information:**")
        error_msg = str(e).lower()
        
        if "insufficient" in error_msg or "empty" in error_msg:
            st.write("**Issue:** Not enough valid data points")
            st.write("**Solutions:**")
            st.write("- Try a different asset with more trading history")
            st.write("- Reduce the prediction period")
        elif "date" in error_msg or "time" in error_msg:
            st.write("**Issue:** Date formatting or consistency problems")
            st.write("**Solutions:**")
            st.write("- The data source might have irregular date formats")
            st.write("- Try refreshing the app or selecting a different asset")
        else:
            st.write("**Issue:** Model configuration or data quality problem")
            st.write("**Solutions:**")
            st.write("- Try reducing the prediction period to 1-2 years")
            st.write("- Select a more established asset (like BTC-USD or AAPL)")
            st.write("- Check if the asset is actively traded")
        
        # Show available data for debugging
        st.write("**Available data sample:**")
        try:
            st.dataframe(df_train.head(10))
        except:
            st.write("Unable to display data sample")
        
        # Try simple prediction as fallback
        st.subheader("📈 Simple Trend Analysis (Fallback)")
        if len(df_train) >= 2:
            try:
                # Calculate simple moving average trend
                recent_data = df_train.tail(min(30, len(df_train)))
                trend = recent_data['y'].diff().mean()
                current = df_train['y'].iloc[-1]
                
                st.write(f"**Current Price:** ${current:.2f}")
                st.write(f"**Recent Trend:** {'+' if trend > 0 else ''}{trend:.2f} per day")
                
                # Simple projections
                days_30 = max(0, current + (trend * 30))
                days_90 = max(0, current + (trend * 90))
                days_365 = max(0, current + (trend * 365))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("30 Days (Simple)", f"${days_30:.2f}", f"{((days_30-current)/current*100):+.1f}%")
                with col2:
                    st.metric("90 Days (Simple)", f"${days_90:.2f}", f"{((days_90-current)/current*100):+.1f}%")
                with col3:
                    st.metric("1 Year (Simple)", f"${days_365:.2f}", f"{((days_365-current)/current*100):+.1f}%")
                
                st.info("📝 **Note:** These are simple linear projections, not AI predictions. For better accuracy, try using assets with more historical data.")
                
                # Create simple trend chart
                fig_simple = go.Figure()
                
                # Historical data
                fig_simple.add_trace(go.Scatter(
                    x=df_train['ds'], y=df_train['y'],
                    mode='lines', name='Historical Price',
                    line=dict(color='#00cc96', width=2)
                ))
                
                # Simple trend line
                future_dates = pd.date_range(
                    start=df_train['ds'].max() + timedelta(days=1), 
                    periods=min(365, period), 
                    freq='D'
                )
                future_prices = [current + (trend * i) for i in range(1, len(future_dates)+1)]
                future_prices = [max(0, price) for price in future_prices]  # Ensure non-negative
                
                fig_simple.add_trace(go.Scatter(
                    x=future_dates, y=future_prices,
                    mode='lines', name='Simple Trend',
                    line=dict(color='#ff6692', width=2, dash='dash')
                ))
                
                fig_simple.update_layout(
                    title=f"📈 {selected_stock} - Simple Trend Projection",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig_simple, use_container_width=True)
                
            except Exception as trend_error:
                st.write(f"Unable to create simple trend analysis: {str(trend_error)}")
        
        # Don't stop here - let the user continue with the app
        st.warning("⚠️ Continuing with limited functionality. Some features may not be available.")
        
        # Create minimal forecast data for the rest of the app to work
        forecast = pd.DataFrame({
            'ds': df_train['ds'],
            'yhat': df_train['y'],
            'yhat_lower': df_train['y'] * 0.9,
            'yhat_upper': df_train['y'] * 1.1
        })

# Forecast visualization (only if forecast exists)
if 'forecast' in locals():
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Forecast Visualization")
        
        try:
            # Create enhanced forecast plot
            fig_forecast = go.Figure()
            
            # Clean training data
            df_train_clean = df_train.dropna()
            forecast_clean = forecast.dropna(subset=['ds', 'yhat'])
            
            if not df_train_clean.empty and not forecast_clean.empty:
                # Historical data
                fig_forecast.add_trace(go.Scatter(
                    x=df_train_clean['ds'], y=df_train_clean['y'],
                    mode='lines', name='Historical Data',
                    line=dict(color='#00cc96', width=2)
                ))
                
                # Forecast line
                future_data = forecast_clean[forecast_clean['ds'] > df_train_clean['ds'].max()]
                
                if not future_data.empty:
                    fig_forecast.add_trace(go.Scatter(
                        x=future_data['ds'], y=future_data['yhat'],
                        mode='lines', name='Forecast',
                        line=dict(color='#ff6692', width=3, dash='dash')
                    ))
                    
                    # Confidence intervals
                    if show_confidence and 'yhat_upper' in future_data.columns and 'yhat_lower' in future_data.columns:
                        # Clean confidence interval data
                        conf_data = future_data.dropna(subset=['yhat_upper', 'yhat_lower'])
                        
                        if not conf_data.empty:
                            fig_forecast.add_trace(go.Scatter(
                                x=conf_data['ds'], y=conf_data['yhat_upper'],
                                fill=None, mode='lines',
                                line=dict(color='rgba(255, 102, 146, 0)'),
                                showlegend=False
                            ))
                            fig_forecast.add_trace(go.Scatter(
                                x=conf_data['ds'], y=conf_data['yhat_lower'],
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
            else:
                st.warning("⚠️ Unable to create forecast chart - insufficient data")
                
        except Exception as e:
            st.error("Unable to generate forecast visualization. Please try a different asset or time period.")

    with col2:
        st.subheader("🎯 Key Predictions")
        
        # Calculate key prediction metrics safely
        try:
            current_price_pred = float(df_train['y'].iloc[-1]) if not df_train.empty else None
            
            if current_price_pred is None or pd.isna(current_price_pred):
                st.warning("⚠️ Unable to calculate predictions - insufficient price data")
            else:
                # Future predictions at specific intervals
                future_30d = forecast[forecast['ds'] == (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')]
                future_90d = forecast[forecast['ds'] == (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')]
                future_1y = forecast[forecast['ds'] == (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')]
                
                def get_prediction_value(pred_df):
                    try:
                        if not pred_df.empty and not pd.isna(pred_df['yhat'].iloc[0]):
                            return float(pred_df['yhat'].iloc[0])
                        return None
                    except:
                        return None
                
                pred_30d = get_prediction_value(future_30d)
                pred_90d = get_prediction_value(future_90d)
                pred_1y = get_prediction_value(future_1y)
                
                if pred_30d and not pd.isna(pred_30d):
                    change_30d = ((pred_30d - current_price_pred) / current_price_pred) * 100
                    st.markdown(f"""
                    <div class="prediction-highlight">
                    <h4>📅 30 Days</h4>
                    <p><strong>${pred_30d:.2f}</strong> ({change_30d:+.1f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("30-day prediction not available")
                
                if pred_90d and not pd.isna(pred_90d):
                    change_90d = ((pred_90d - current_price_pred) / current_price_pred) * 100
                    st.markdown(f"""
                    <div class="prediction-highlight">
                    <h4>📅 90 Days</h4>
                    <p><strong>${pred_90d:.2f}</strong> ({change_90d:+.1f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("90-day prediction not available")
                
                if pred_1y and not pd.isna(pred_1y):
                    change_1y = ((pred_1y - current_price_pred) / current_price_pred) * 100
                    st.markdown(f"""
                    <div class="prediction-highlight">
                    <h4>📅 1 Year</h4>
                    <p><strong>${pred_1y:.2f}</strong> ({change_1y:+.1f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("1-year prediction not available")
                    
        except Exception as e:
            st.error("Unable to generate predictions. Please try a different asset or time period.")

    # Show trend components (only if model 'm' exists)
    if show_components and 'm' in locals():
        st.subheader("📊 Trend Component Analysis")
        
        try:
            fig_components = m.plot_components(forecast)
            st.pyplot(fig_components)
        except Exception as e:
            st.warning("⚠️ Unable to display trend components. This may be due to insufficient data or model complexity.")

# Detailed forecast table
st.subheader("📋 Detailed Forecast Table")

# Future predictions table
try:
    if 'forecast' in locals() and not forecast.empty and not df_train.empty:
        future_predictions = forecast[forecast['ds'] > df_train['ds'].max()].copy()
        current_price_table = float(df_train['y'].iloc[-1]) if not df_train.empty else 0
        
        if not future_predictions.empty and current_price_table > 0:
            future_predictions['Price Change'] = future_predictions['yhat'] - current_price_table
            future_predictions['Price Change %'] = (future_predictions['Price Change'] / current_price_table) * 100
            
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
            if show_confidence_table and 'yhat_lower' in future_predictions.columns and 'yhat_upper' in future_predictions.columns:
                display_columns.extend(['yhat_lower', 'yhat_upper'])

            # Format the table
            if not future_predictions.empty:
                table_data = future_predictions[display_columns].head(table_rows).copy()
                column_names = ['Date', 'Predicted Price', 'Price Change ($)', 'Price Change (%)']
                if show_confidence_table and 'yhat_lower' in display_columns and 'yhat_upper' in display_columns:
                    column_names.extend(['Lower Bound', 'Upper Bound'])
                table_data.columns = column_names

                # Round numerical columns and handle NaN values
                for col in table_data.columns:
                    if col != 'Date':
                        table_data[col] = table_data[col].fillna(0).round(2)

                st.dataframe(table_data, use_container_width=True, hide_index=True)
            else:
                st.info("No future predictions to display.")
        else:
            st.warning("⚠️ Unable to generate forecast table - insufficient data")
    else:
        st.info("No forecast data available yet.")
        
except Exception as e:
    st.error("Unable to generate forecast table. Please try refreshing or selecting a different asset.")

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
    if 'future_predictions' in locals() and not future_predictions.empty:
        csv_forecast = convert_df_to_csv(future_predictions)
        st.download_button(
            label="📥 Download Forecast Data",
            data=csv_forecast,
            file_name=f"{selected_stock}_forecast_data.csv",
            mime="text/csv"
        )
    else:
        st.info("Forecast data not available for download")

with col3:
    # Create a summary report
    summary_data = {
        'Asset': [selected_stock],
        'Current Price': [safe_format_price(current_price)],
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
