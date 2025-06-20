import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

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
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            st.error(f"No data found for {ticker}")
            return None
        
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Ensure required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Missing required columns for {ticker}")
            return None
        
        # Add technical indicators
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['Close'], 14)
        df['MACD'], df['MACD_signal'] = calculate_macd(df['Close'])
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

def calculate_rsi(prices, window=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    return macd, macd_signal

def create_features(df, window_size):
    """Create features for machine learning model"""
    features = []
    targets = []
    
    # Feature columns
    feature_cols = ['Close', 'Volume', 'MA20', 'MA50', 'RSI', 'MACD', 'Volatility']
    
    for i in range(window_size, len(df)):
        # Create window of features
        window_features = []
        for col in feature_cols:
            window_data = df[col].iloc[i-window_size:i].values
            window_features.extend([
                np.mean(window_data),
                np.std(window_data),
                window_data[-1],  # Latest value
                (window_data[-1] - window_data[0]) / window_data[0] if window_data[0] != 0 else 0  # Change ratio
            ])
        
        features.append(window_features)
        targets.append(df['Close'].iloc[i])
    
    return np.array(features), np.array(targets)

@st.cache_data
def build_and_train_model(df, window_size, model_type='neural_network'):
    """Build and train ML model"""
    try:
        # Create features
        X, y = create_features(df, window_size)
        
        if len(X) == 0:
            raise ValueError("Not enough data to create features")
        
        # Scale features
        scaler_X = StandardScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Split data (80% train, 20% test)
        split = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y_scaled[:split], y_scaled[split:]
        
        # Build and train model
        if model_type == 'neural_network':
            model = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        else:  # random_forest
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_scaled = model.predict(X_test)
        
        # Inverse transform predictions
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
        mae = mean_absolute_error(y_test_actual, y_pred)
        mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100
        r2 = r2_score(y_test_actual, y_pred)
        
        # Predict next price
        last_features = X_scaled[-1:]
        next_price_scaled = model.predict(last_features)
        next_price = scaler_y.inverse_transform(next_price_scaled.reshape(-1, 1))[0, 0]
        
        return y_test_actual, y_pred, next_price, rmse, mae, mape, r2
        
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None, None, None, None, None, None, None

def calculate_stock_score(df, next_price, current_price, mape, r2):
    """Calculate stock score based on various factors"""
    try:
        expected_return = ((next_price - current_price) / current_price) * 100
        
        # Technical indicators
        rsi = df['RSI'].iloc[-1]
        
        # Score components
        prediction_score = min(max(expected_return * 10, -50), 50)
        rsi_score = 100 - abs(rsi - 50) * 2
        accuracy_score = max(100 - mape, 0)
        r2_score_normalized = max(r2 * 100, 0)
        
        total_score = (prediction_score * 0.4 + rsi_score * 0.2 + 
                      accuracy_score * 0.2 + r2_score_normalized * 0.2)
        
        return {
            'total_score': total_score,
            'expected_return': expected_return,
            'current_price': current_price,
            'predicted_price': next_price,
            'rsi': rsi,
            'accuracy': accuracy_score,
            'r2': r2
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
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='orange', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='green', width=1)))
    
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
    
    colors = ['green' if val > 0 else 'red' for val in df['Returns']]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors, opacity=0.6))
    
    volume_ma = df['Volume'].rolling(window=20).mean()
    fig.add_trace(go.Scatter(x=df.index, y=volume_ma, name='Volume MA20', line=dict(color='blue', width=2)))
    
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
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['RSI', 'MACD'],
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )
    
    # RSI plot
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=2)), row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=1, col=1)
    
    # MACD plot
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD Line', line=dict(color='blue', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal Line', line=dict(color='red', width=1)), row=2, col=1)
    
    macd_histogram = df['MACD'] - df['MACD_signal']
    colors = ['green' if val > 0 else 'red' for val in macd_histogram]
    fig.add_trace(go.Bar(x=df.index, y=macd_histogram, name='MACD Histogram', marker_color=colors, opacity=0.5), row=2, col=1)
    
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
    
    fig.add_trace(go.Scatter(y=actual, name='Actual', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(y=predicted, name='Predicted', line=dict(color='red', width=2, dash='dash')))
    
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
        <strong style='font-size: 18px;'>ML-based Stock Prediction App</strong>
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
            start_date = st.date_input("Start Date", value=datetime(2022, 1, 1), max_value=datetime.now() - timedelta(days=365))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now(), min_value=start_date + timedelta(days=365))
        
        st.divider()
        
        window_size = st.slider("Window Size (days)", min_value=30, max_value=120, value=60, help="Number of days to look back for prediction")
        model_type = st.selectbox("Model Type", options=['neural_network', 'random_forest'], index=0, help="Choose ML algorithm")
        
        st.divider()
        
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        if run_analysis and not selected_stocks:
            st.error("Please select at least one stock!")
    
    # Main content
    if not selected_stocks:
        st.info("👆 Please select stocks from the sidebar to begin analysis")
        st.subheader("Available Stocks")
        sample_df = pd.DataFrame([{"Symbol": k, "Company": v} for k, v in COMPANY_NAMES.items()])
        st.dataframe(sample_df, use_container_width=True)
        return
    
    if run_analysis:
        results = {}
        stock_scores = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(selected_stocks):
            company_name = COMPANY_NAMES.get(ticker, ticker)
            status_text.text(f"Processing {company_name}... ({i+1}/{len(selected_stocks)})")
            
            df = load_data(ticker, start_date, end_date)
            if df is None:
                continue
            
            with st.spinner(f'Training model for {company_name}...'):
                model_results = build_and_train_model(df, window_size, model_type)
                
            if model_results[0] is None:
                st.error(f"Failed to train model for {company_name}")
                continue
                
            actual, predicted, next_price, rmse, mae, mape, r2 = model_results
            current_price = df['Close'].iloc[-1]
            
            results[ticker] = {
                'actual': actual, 'predicted': predicted, 'next_price': next_price,
                'current_price': current_price, 'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2, 'df': df
            }
            
            score = calculate_stock_score(df, next_price, current_price, mape, r2)
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
                st.metric("R² Score", f"{best_stock[1]['r2']:.3f}")
            
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
                    'R² Score': f"{score_data['r2']:.3f}"
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
                    fig2 = create_prediction_chart(results[ticker]['actual'], results[ticker]['predicted'], ticker)
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
                    st.metric("R² Score", f"{results[ticker]['r2']:.3f}")

if __name__ == "__main__":
    main()