import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import ta
from datetime import datetime, timedelta
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Set page config
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# Define company names
COMPANY_NAMES = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'TSLA': 'Tesla Inc.',
    'META': 'Meta Platforms Inc.',
    'NVDA': 'NVIDIA Corporation',
    'NFLX': 'Netflix Inc.',
    'DIS': 'Walt Disney Company',
    'PYPL': 'PayPal Holdings Inc.'
}

@st.cache_data
def load_data(ticker, start_date, end_date):
    """Load and preprocess stock data"""
    try:
        # Download data
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            st.error(f"No data found for {ticker}")
            return None
        
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Missing required columns for {ticker}")
            return None
        
        # Add technical indicators
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        
        # Price change and volatility
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Drop NaN values
        df = df.dropna()
        
        if len(df) < 100:
            st.error(f"Not enough data for {ticker}. Need at least 100 days.")
            return None
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        return None

def create_sequences(data, window_size):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

@st.cache_data
def build_and_train_model(df, window_size, epochs):
    """Build and train LSTM model"""
    try:
        # Use only Close price for prediction
        data = df[['Close']].values
        
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Create sequences
        X, y = create_sequences(scaled_data, window_size)
        
        if len(X) == 0:
            raise ValueError("Not enough data to create sequences")
        
        # Reshape for LSTM
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split data (80% train, 20% test)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Build model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=0)
        
        # Make predictions
        predictions = model.predict(X_test, verbose=0)
        
        # Inverse transform predictions
        predictions = scaler.inverse_transform(predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
        mae = mean_absolute_error(y_test_actual, predictions)
        mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100
        
        # Predict next price
        last_sequence = scaled_data[-window_size:]
        last_sequence = np.reshape(last_sequence, (1, window_size, 1))
        next_price_scaled = model.predict(last_sequence, verbose=0)
        next_price = scaler.inverse_transform(next_price_scaled)[0, 0]
        
        return y_test_actual.flatten(), predictions.flatten(), next_price, rmse, mae, mape
        
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None, None, None, None, None, None

def calculate_stock_score(df, next_price, current_price, mape):
    """Calculate stock score based on various factors"""
    try:
        expected_return = ((next_price - current_price) / current_price) * 100
        
        # Technical indicators
        rsi = df['RSI'].iloc[-1]
        
        # Score components
        prediction_score = min(max(expected_return * 10, -50), 50)  # Cap at ±50
        rsi_score = 100 - abs(rsi - 50) * 2  # Closer to 50 is better
        accuracy_score = max(100 - mape, 0)
        
        total_score = (prediction_score * 0.5 + rsi_score * 0.3 + accuracy_score * 0.2)
        
        return {
            'total_score': total_score,
            'expected_return': expected_return,
            'current_price': current_price,
            'predicted_price': next_price,
            'rsi': rsi,
            'accuracy': accuracy_score
        }
    except Exception as e:
        st.error(f"Error calculating stock score: {str(e)}")
        return None

def add_watermark(fig):
    """Add watermark to plot"""
    fig.add_annotation(
        text="Priyanshu Joarder",
        xref="paper", yref="paper",
        x=0.98, y=0.02,
        xanchor='right', yanchor='bottom',
        showarrow=False,
        font=dict(size=10, color="rgba(128,128,128,0.5)"),
        bgcolor="rgba(255,255,255,0.3)",
        bordercolor="rgba(128,128,128,0.3)",
        borderwidth=1
    )
    return fig

def create_chart(df, ticker):
    """Create price chart with moving averages"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['Close'], 
        name='Close Price',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['MA20'], 
        name='MA20',
        line=dict(color='orange', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['MA50'], 
        name='MA50',
        line=dict(color='green', width=1)
    ))
    
    company_name = COMPANY_NAMES.get(ticker, ticker)
    fig.update_layout(
        title=f'{company_name} Stock Price',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=400,
        template='plotly_white'
    )
    
    return add_watermark(fig)

def create_volume_chart(df, ticker):
    """Create volume analysis chart"""
    fig = go.Figure()
    
    # Color bars based on price change
    colors = ['green' if val > 0 else 'red' for val in df['Returns']]
    
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color=colors,
        opacity=0.6
    ))
    
    # Add volume moving average
    volume_ma = df['Volume'].rolling(window=20).mean()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=volume_ma,
        name='Volume MA20',
        line=dict(color='blue', width=2)
    ))
    
    company_name = COMPANY_NAMES.get(ticker, ticker)
    fig.update_layout(
        title=f'{company_name} Volume Analysis',
        xaxis_title='Date',
        yaxis_title='Volume',
        height=400,
        template='plotly_white'
    )
    
    return add_watermark(fig)

def create_rsi_macd_chart(df, ticker):
    """Create combined RSI and MACD chart"""
    company_name = COMPANY_NAMES.get(ticker, ticker)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['RSI', 'MACD'],
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )
    
    # RSI plot
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        name='RSI',
        line=dict(color='purple', width=2)
    ), row=1, col=1)
    
    # RSI overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=1, col=1)
    
    # MACD plot
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        name='MACD Line',
        line=dict(color='blue', width=2)
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD_signal'],
        name='Signal Line',
        line=dict(color='red', width=1)
    ), row=2, col=1)
    
    # MACD histogram
    macd_histogram = df['MACD'] - df['MACD_signal']
    colors = ['green' if val > 0 else 'red' for val in macd_histogram]
    fig.add_trace(go.Bar(
        x=df.index,
        y=macd_histogram,
        name='MACD Histogram',
        marker_color=colors,
        opacity=0.5
    ), row=2, col=1)
    
    fig.update_layout(
        title=f'{company_name} Technical Indicators (RSI & MACD)',
        height=400,
        showlegend=True,
        template='plotly_white'
    )
    
    fig.update_yaxes(title_text="RSI", row=1, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    return add_watermark(fig)

def create_prediction_chart(actual, predicted, ticker):
    """Create prediction comparison chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=actual, 
        name='Actual',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        y=predicted, 
        name='Predicted',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    company_name = COMPANY_NAMES.get(ticker, ticker)
    fig.update_layout(
        title=f'{company_name} - Actual vs Predicted',
        xaxis_title='Time',
        yaxis_title='Price ($)',
        height=400,
        template='plotly_white'
    )
    
    return add_watermark(fig)

def main():
    """Main application function"""
    st.title("📈 Stock Prediction Dashboard")
    st.markdown("""
    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;'>
        <strong style='font-size: 18px;'>LSTM-based Stock Prediction App</strong>
        <strong style='font-size: 18px; color: #666;'>by Priyanshu Joarder</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("📊 Configuration")
        
        selected_stocks = st.multiselect(
            "Select Stocks",
            options=list(COMPANY_NAMES.keys()),
            default=['AAPL', 'MSFT'],
            max_selections=3,
            help="Select up to 3 stocks for analysis"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date", 
                value=datetime(2022, 1, 1),
                max_value=datetime.now() - timedelta(days=365)
            )
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=datetime.now(),
                min_value=start_date + timedelta(days=365)
            )
        
        st.divider()
        
        window_size = st.slider(
            "Window Size (days)", 
            min_value=30, 
            max_value=120, 
            value=60,
            help="Number of days to look back for prediction"
        )
        
        epochs = st.slider(
            "Training Epochs", 
            min_value=10, 
            max_value=100, 
            value=30,
            help="Number of training iterations"
        )
        
        st.divider()
        
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        if run_analysis and not selected_stocks:
            st.error("Please select at least one stock!")
    
    # Main content
    if not selected_stocks:
        st.info("👆 Please select stocks from the sidebar to begin analysis")
        
        # Show sample information
        st.subheader("Available Stocks")
        sample_df = pd.DataFrame([
            {"Symbol": k, "Company": v} 
            for k, v in COMPANY_NAMES.items()
        ])
        st.dataframe(sample_df, use_container_width=True)
        
        return
    
    if run_analysis:
        results = {}
        stock_scores = {}
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(selected_stocks):
            company_name = COMPANY_NAMES.get(ticker, ticker)
            status_text.text(f"Processing {company_name}... ({i+1}/{len(selected_stocks)})")
            
            # Load data
            df = load_data(ticker, start_date, end_date)
            if df is None:
                continue
            
            # Train model and get predictions
            with st.spinner(f'Training model for {company_name}...'):
                model_results = build_and_train_model(df, window_size, epochs)
                
            if model_results[0] is None:
                st.error(f"Failed to train model for {company_name}")
                continue
                
            actual, predicted, next_price, rmse, mae, mape = model_results
            current_price = df['Close'].iloc[-1]
            
            results[ticker] = {
                'actual': actual,
                'predicted': predicted,
                'next_price': next_price,
                'current_price': current_price,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'df': df
            }
            
            # Calculate stock score
            score = calculate_stock_score(df, next_price, current_price, mape)
            if score:
                stock_scores[ticker] = score
            
            progress_bar.progress((i + 1) / len(selected_stocks))
        
        status_text.empty()
        progress_bar.empty()
        
        if not results:
            st.error("No stocks were analyzed successfully. Please try again.")
            return
        
        # Display results
        st.header("📊 Analysis Results")
        
        # Best stock recommendation
        if stock_scores:
            sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
            best_stock = sorted_stocks[0]
            best_ticker = best_stock[0]
            best_company = COMPANY_NAMES.get(best_ticker, best_ticker)
            
            st.markdown("## 🏆 TOP RECOMMENDATION")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #4CAF50, #45a049); 
                           padding: 20px; border-radius: 10px; color: white; margin-bottom: 10px;">
                    <h2 style="margin: 0; color: white;">{best_company}</h2>
                    <h3 style="margin: 5px 0; color: white;">({best_ticker})</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Overall Score", f"{best_stock[1]['total_score']:.1f}/100")
                st.metric("Expected Return", f"{best_stock[1]['expected_return']:.2f}%")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${best_stock[1]['current_price']:.2f}")
            with col2:
                st.metric("Target Price", f"${best_stock[1]['predicted_price']:.2f}")
            with col3:
                st.metric("RSI", f"{best_stock[1]['rsi']:.1f}")
            with col4:
                st.metric("Model Accuracy", f"{100 - results[best_ticker]['mape']:.1f}%")
            
            st.warning("⚠️ **Disclaimer**: This is an AI-generated recommendation for educational purposes only. Always conduct your own research and consult with financial advisors before making investment decisions.")
        
        # Comparison table
        if len(results) > 1:
            st.subheader("📈 Stock Comparison")
            comparison_data = []
            for ticker, score_data in sorted(stock_scores.items(), key=lambda x: x[1]['total_score'], reverse=True):
                company_name = COMPANY_NAMES.get(ticker, ticker)
                comparison_data.append({
                    'Rank': len(comparison_data) + 1,
                    'Company': company_name,
                    'Symbol': ticker,
                    'Score': f"{score_data['total_score']:.1f}",
                    'Expected Return (%)': f"{score_data['expected_return']:.2f}",
                    'Current Price ($)': f"{score_data['current_price']:.2f}",
                    'Target Price ($)': f"{score_data['predicted_price']:.2f}",
                    'Model Accuracy (%)': f"{100 - results[ticker]['mape']:.1f}"
                })
            
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        # Detailed individual analysis
        st.header("🔍 Detailed Analysis")
        
        for ticker in results:
            company_name = COMPANY_NAMES.get(ticker, ticker)
            
            with st.expander(f"📊 {company_name} ({ticker}) Analysis", expanded=len(results) == 1):
                # Price and prediction charts
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = create_chart(results[ticker]['df'], ticker)
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    fig2 = create_prediction_chart(
                        results[ticker]['actual'], 
                        results[ticker]['predicted'], 
                        ticker
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Volume and technical indicators
                col3, col4 = st.columns(2)
                
                with col3:
                    fig3 = create_volume_chart(results[ticker]['df'], ticker)
                    st.plotly_chart(fig3, use_container_width=True)
                
                with col4:
                    fig4 = create_rsi_macd_chart(results[ticker]['df'], ticker)
                    st.plotly_chart(fig4, use_container_width=True)
                
                # Performance metrics
                st.subheader("📈 Model Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("RMSE", f"{results[ticker]['rmse']:.4f}")
                with col2:
                    st.metric("MAE", f"{results[ticker]['mae']:.4f}")
                with col3:
                    st.metric("MAPE", f"{results[ticker]['mape']:.2f}%")
                with col4:
                    accuracy = 100 - results[ticker]['mape']
                    st.metric("Accuracy", f"{accuracy:.1f}%")

if __name__ == "__main__":
    main()
