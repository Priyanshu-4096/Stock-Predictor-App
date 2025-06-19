import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Try importing TensorFlow and ta, with fallbacks
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.error("TensorFlow not available. Please check your requirements.txt")

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    st.warning("Technical Analysis library not available. Some features may be limited.")

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

COMPANY_NAMES = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'TSLA': 'Tesla Inc.',
    'META': 'Meta Platforms Inc.',
    'NVDA': 'NVIDIA Corporation'
}

@st.cache_data
def load_data(ticker, start_date, end_date):
    """Load and preprocess stock data"""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Add basic technical indicators
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # Add technical indicators if ta is available
        if TA_AVAILABLE:
            df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
            
            # MACD
            macd = ta.trend.MACD(close=df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
        else:
            # Simple RSI calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Simple MACD calculation
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        
        # Price change
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        return df.dropna()
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        return None

@st.cache_data
def build_and_train_model(df, window_size, epochs):
    """Build and train LSTM model"""
    if not TENSORFLOW_AVAILABLE:
        st.error("TensorFlow is required for model training")
        return None, None, None, None, None, None
    
    try:
        # Use only Close price for prediction
        data = df[['Close']].values
        
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        def create_sequences(data, window):
            X, y = [], []
            for i in range(window, len(data)):
                X.append(data[i-window:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)
        
        X, y = create_sequences(scaled_data, window_size)
        
        if len(X) == 0:
            st.error("Insufficient data for model training")
            return None, None, None, None, None, None
        
        # Reshape for LSTM
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split data
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
        with st.spinner('Training model...'):
            model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=0)
        
        # Make predictions
        predictions = model.predict(X_test)
        
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
        next_price_scaled = model.predict(last_sequence)
        next_price = scaler.inverse_transform(next_price_scaled)[0, 0]
        
        return y_test_actual.flatten(), predictions.flatten(), next_price, rmse, mae, mape
    
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None, None, None, None, None, None

def calculate_stock_score(df, next_price, current_price, mape):
    """Calculate stock score"""
    try:
        expected_return = ((next_price - current_price) / current_price) * 100
        
        # Technical indicators
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        
        # Score components
        prediction_score = min(max(expected_return * 10, -50), 50)
        rsi_score = 100 - abs(rsi - 50) * 2
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
    """Create price chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['Close'], 
        name='Close Price',
        line=dict(color='blue', width=2)
    ))
    
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['MA20'], 
            name='MA20',
            line=dict(color='orange', width=1)
        ))
    
    if 'MA50' in df.columns:
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
        height=400
    )
    
    fig = add_watermark(fig)
    return fig

def create_volume_chart(df, ticker):
    """Create volume analysis chart"""
    fig = go.Figure()
    
    # Color bars based on price change
    colors = ['green' if df['Returns'].iloc[i] > 0 else 'red' 
              for i in range(len(df))]
    
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
        height=400
    )
    
    fig = add_watermark(fig)
    return fig

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
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI'],
            name='RSI',
            line=dict(color='purple', width=2)
        ), row=1, col=1)
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=1, col=1)
    
    # MACD plot
    if 'MACD' in df.columns and 'MACD_signal' in df.columns:
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
        title=f'{company_name} Technical Indicators',
        height=400,
        showlegend=True
    )
    
    fig.update_yaxes(title_text="RSI", row=1, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    fig = add_watermark(fig)
    return fig

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
        height=400
    )
    
    fig = add_watermark(fig)
    return fig

def main():
    st.title("📈 Stock Prediction Dashboard")
    st.markdown("""
    <div style='display: flex; justify-content: space-between;'>
        <strong>LSTM-based Stock Prediction App</strong>
        <strong>by Priyanshu Joarder</strong>
    </div>
    """, unsafe_allow_html=True)
    
    if not TENSORFLOW_AVAILABLE:
        st.error("⚠️ TensorFlow is not available. Please check your deployment configuration.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        selected_stocks = st.multiselect(
            "Select Stocks",
            options=list(COMPANY_NAMES.keys()),
            default=['AAPL', 'MSFT'],
            max_selections=3
        )
        
        start_date = st.date_input("Start Date", value=datetime(2022, 1, 1))
        end_date = st.date_input("End Date", value=datetime.now())
        
        window_size = st.slider("Window Size", 30, 90, 60)
        epochs = st.slider("Training Epochs", 10, 50, 20)
        
        run_analysis = st.button("🚀 Run Analysis", type="primary")
    
    if not selected_stocks:
        st.warning("Please select at least one stock")
        return
    
    if run_analysis:
        results = {}
        stock_scores = {}
        
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(selected_stocks):
            company_name = COMPANY_NAMES.get(ticker, ticker)
            st.text(f"Processing {company_name}...")
            
            # Load data
            df = load_data(ticker, start_date, end_date)
            if df is None:
                continue
            
            try:
                # Train model
                model_result = build_and_train_model(df, window_size, epochs)
                if model_result[0] is None:
                    continue
                
                actual, predicted, next_price, rmse, mae, mape = model_result
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
                
                # Calculate score
                score_result = calculate_stock_score(df, next_price, current_price, mape)
                if score_result:
                    stock_scores[ticker] = score_result
                
            except Exception as e:
                st.error(f"Error analyzing {company_name}: {str(e)}")
                continue
            
            progress_bar.progress((i + 1) / len(selected_stocks))
        
        if not results:
            st.error("No stocks analyzed successfully")
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
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #4CAF50, #45a049); 
                           padding: 20px; border-radius: 10px; color: white;">
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
                st.metric("Model Accuracy", f"{results[best_ticker]['mape']:.1f}% MAPE")
            
            st.warning("⚠️ **Disclaimer**: AI-generated recommendation. Always do your own research!")
            
            # Comparison table
            st.subheader("Stock Comparison")
            comparison_data = []
            for ticker, score_data in sorted_stocks:
                company_name = COMPANY_NAMES.get(ticker, ticker)
                comparison_data.append({
                    'Company': company_name,
                    'Score': f"{score_data['total_score']:.1f}",
                    'Expected Return (%)': f"{score_data['expected_return']:.2f}",
                    'Current Price ($)': f"{score_data['current_price']:.2f}",
                    'Target Price ($)': f"{score_data['predicted_price']:.2f}",
                    'MAPE (%)': f"{results[ticker]['mape']:.2f}"
                })
            
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        # Individual analysis
        st.header("Detailed Analysis")
        
        for ticker in results:
            company_name = COMPANY_NAMES.get(ticker, ticker)
            
            with st.expander(f"{company_name} Analysis", expanded=False):
                # Charts
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
                
                col3, col4 = st.columns(2)
                
                with col3:
                    fig3 = create_volume_chart(results[ticker]['df'], ticker)
                    st.plotly_chart(fig3, use_container_width=True)
                
                with col4:
                    fig4 = create_rsi_macd_chart(results[ticker]['df'], ticker)
                    st.plotly_chart(fig4, use_container_width=True)
                
                # Metrics
                st.subheader("Model Performance Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"{results[ticker]['rmse']:.4f}")
                with col2:
                    st.metric("MAE", f"{results[ticker]['mae']:.4f}")
                with col3:
                    st.metric("MAPE", f"{results[ticker]['mape']:.2f}%")

if __name__ == "__main__":
    main()