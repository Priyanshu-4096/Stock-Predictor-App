import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
from datetime import datetime, timedelta
import ta

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def get_stock_data(symbol, period="2y"):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    try:
        # Moving Averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_lower'] = bollinger.bollinger_lband()
        df['BB_middle'] = bollinger.bollinger_mavg()
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # Price features
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        return df
    except Exception as e:
        st.error(f"Error adding technical indicators: {str(e)}")
        return df

def prepare_features(df, lookback_days=30):
    """Prepare features for machine learning"""
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Create lagged features
    feature_columns = ['Close', 'Volume', 'MA_5', 'MA_10', 'MA_20', 'RSI', 'MACD', 'High_Low_Pct']
    
    for col in feature_columns:
        if col in df.columns:
            for i in range(1, lookback_days + 1):
                df[f'{col}_lag_{i}'] = df[col].shift(i)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def create_ml_features(df):
    """Create features for machine learning models"""
    feature_cols = [col for col in df.columns if 
                   col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
                   and not col.startswith('BB_')]
    
    # If no lag features, create some basic ones
    if not any('lag' in col for col in feature_cols):
        df['Close_lag_1'] = df['Close'].shift(1)
        df['Close_lag_2'] = df['Close'].shift(2)
        df['Close_lag_3'] = df['Close'].shift(3)
        df['Volume_lag_1'] = df['Volume'].shift(1)
        feature_cols.extend(['Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Volume_lag_1'])
    
    return df[feature_cols].dropna(), df['Close'][df[feature_cols].dropna().index]

def train_models(X, y):
    """Train multiple ML models"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'scaler': scaler,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred,
            'actual': y_test
        }
    
    return results

def predict_future_prices(model, scaler, last_features, days=30):
    """Predict future stock prices"""
    predictions = []
    current_features = last_features.copy()
    
    for _ in range(days):
        # Scale current features
        current_features_scaled = scaler.transform([current_features])
        
        # Make prediction
        pred = model.predict(current_features_scaled)[0]
        predictions.append(pred)
        
        # Update features for next prediction (simplified approach)
        # In practice, you'd want a more sophisticated way to update features
        current_features = np.roll(current_features, 1)
        current_features[0] = pred
    
    return predictions

def main():
    st.markdown("<h1 class='main-header'>📈 Stock Price Predictor</h1>", unsafe_allow_html=True)
    st.markdown("### Predict stock prices using Machine Learning (Python 3.13.5 Compatible)")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Stock symbol input
    symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL", help="e.g., AAPL, GOOGL, MSFT").upper()
    
    # Time period
    period_options = {"1 Year": "1y", "2 Years": "2y", "5 Years": "5y", "10 Years": "10y"}
    period = st.sidebar.selectbox("Select Time Period", list(period_options.keys()))
    
    # Prediction days
    pred_days = st.sidebar.slider("Days to Predict", 1, 60, 30)
    
    if st.sidebar.button("Analyze Stock", type="primary"):
        with st.spinner(f"Fetching data for {symbol}..."):
            # Get stock data
            data = get_stock_data(symbol, period_options[period])
            
            if data is not None and not data.empty:
                st.success(f"Successfully loaded {len(data)} days of data for {symbol}")
                
                # Display basic info
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
                
                with col2:
                    price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                    st.metric("Daily Change", f"${price_change:.2f}")
                
                with col3:
                    st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
                
                with col4:
                    volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                    st.metric("Volatility", f"{volatility:.1f}%")
                
                # Plot stock price
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig.update_layout(
                    title=f"{symbol} Stock Price History",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Prepare data for ML
                with st.spinner("Preparing data and training models..."):
                    try:
                        # Add technical indicators
                        data_with_indicators = add_technical_indicators(data.copy())
                        
                        # Create features
                        X, y = create_ml_features(data_with_indicators)
                        
                        if len(X) > 50:  # Ensure we have enough data
                            # Train models
                            results = train_models(X, y)
                            
                            # Display model performance
                            st.header("Model Performance")
                            
                            perf_col1, perf_col2, perf_col3 = st.columns(3)
                            
                            for i, (name, result) in enumerate(results.items()):
                                col = [perf_col1, perf_col2, perf_col3][i]
                                with col:
                                    st.markdown(f"**{name}**")
                                    st.write(f"R² Score: {result['r2']:.3f}")
                                    st.write(f"RMSE: ${result['rmse']:.2f}")
                                    st.write(f"MAE: ${result['mae']:.2f}")
                            
                            # Select best model
                            best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
                            best_model = results[best_model_name]
                            
                            st.success(f"Best performing model: {best_model_name} (R² = {best_model['r2']:.3f})")
                            
                            # Make future predictions
                            st.header("Future Price Predictions")
                            
                            # Get last features for prediction
                            last_features = X.iloc[-1].values
                            
                            # Predict future prices
                            future_predictions = predict_future_prices(
                                best_model['model'], 
                                best_model['scaler'], 
                                last_features, 
                                pred_days
                            )
                            
                            # Create future dates
                            last_date = data.index[-1]
                            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=pred_days)
                            
                            # Plot predictions
                            fig_pred = go.Figure()
                            
                            # Historical data
                            fig_pred.add_trace(go.Scatter(
                                x=data.index[-60:],  # Last 60 days
                                y=data['Close'].iloc[-60:],
                                mode='lines',
                                name='Historical',
                                line=dict(color='#1f77b4', width=2)
                            ))
                            
                            # Predictions
                            fig_pred.add_trace(go.Scatter(
                                x=future_dates,
                                y=future_predictions,
                                mode='lines+markers',
                                name='Predictions',
                                line=dict(color='#ff7f0e', width=2, dash='dash'),
                                marker=dict(size=6)
                            ))
                            
                            fig_pred.update_layout(
                                title=f"{symbol} Price Prediction ({pred_days} days)",
                                xaxis_title="Date",
                                yaxis_title="Price ($)",
                                template="plotly_white",
                                height=500
                            )
                            
                            st.plotly_chart(fig_pred, use_container_width=True)
                            
                            # Display prediction summary
                            current_price = data['Close'].iloc[-1]
                            predicted_price = future_predictions[-1]
                            price_change_pred = predicted_price - current_price
                            price_change_pct = (price_change_pred / current_price) * 100
                            
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h3>Prediction Summary</h3>
                                <p><strong>Current Price:</strong> ${current_price:.2f}</p>
                                <p><strong>Predicted Price ({pred_days} days):</strong> ${predicted_price:.2f}</p>
                                <p><strong>Expected Change:</strong> ${price_change_pred:.2f} ({price_change_pct:+.1f}%)</p>
                                <p><strong>Model Used:</strong> {best_model_name}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Technical indicators chart
                            st.header("Technical Analysis")
                            
                            fig_tech = make_subplots(
                                rows=2, cols=2,
                                subplot_titles=('Moving Averages', 'RSI', 'MACD', 'Volume'),
                                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                       [{"secondary_y": False}, {"secondary_y": False}]]
                            )
                            
                            # Moving averages
                            recent_data = data_with_indicators.tail(100)
                            fig_tech.add_trace(go.Scatter(x=recent_data.index, y=recent_data['Close'], name='Close', line=dict(color='blue')), row=1, col=1)
                            fig_tech.add_trace(go.Scatter(x=recent_data.index, y=recent_data['MA_20'], name='MA 20', line=dict(color='red')), row=1, col=1)
                            fig_tech.add_trace(go.Scatter(x=recent_data.index, y=recent_data['MA_50'], name='MA 50', line=dict(color='green')), row=1, col=1)
                            
                            # RSI
                            fig_tech.add_trace(go.Scatter(x=recent_data.index, y=recent_data['RSI'], name='RSI', line=dict(color='purple')), row=1, col=2)
                            fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=2)
                            fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=2)
                            
                            # MACD
                            fig_tech.add_trace(go.Scatter(x=recent_data.index, y=recent_data['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
                            fig_tech.add_trace(go.Scatter(x=recent_data.index, y=recent_data['MACD_signal'], name='Signal', line=dict(color='red')), row=2, col=1)
                            
                            # Volume
                            fig_tech.add_trace(go.Bar(x=recent_data.index, y=recent_data['Volume'], name='Volume', marker_color='lightblue'), row=2, col=2)
                            
                            fig_tech.update_layout(height=600, showlegend=False, template="plotly_white")
                            st.plotly_chart(fig_tech, use_container_width=True)
                            
                        else:
                            st.error("Not enough data to train the model. Please try a longer time period.")
                            
                    except Exception as e:
                        st.error(f"Error in model training: {str(e)}")
                        st.info("This might be due to insufficient data or technical indicator calculation issues.")
            
            else:
                st.error(f"Could not fetch data for {symbol}. Please check the symbol and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>⚠️ <strong>Disclaimer:</strong> This is for educational purposes only. Stock predictions are not guaranteed and should not be used as the sole basis for investment decisions.</p>
        <p>Built with ❤️ using Streamlit • Compatible with Python 3.13.5</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

