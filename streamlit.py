import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
import warnings
import io

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Enhanced Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
COMPANY_NAMES = {
    'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corporation', 'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.', 'TSLA': 'Tesla Inc.', 'META': 'Meta Platforms Inc.',
    'NVDA': 'NVIDIA Corporation', 'JPM': 'JPMorgan Chase & Co.', 
    'NFLX': 'Netflix Inc.', 'ADBE': 'Adobe Inc.'
}

def add_watermark(fig):
    """Add watermark to plot"""
    fig.add_annotation(
        text="Priyanshu Joarder", xref="paper", yref="paper",
        x=0.98, y=0.02, xanchor='right', yanchor='bottom',
        showarrow=False, font=dict(size=10, color="rgba(128,128,128,0.5)"),
        bgcolor="rgba(255,255,255,0.3)", bordercolor="rgba(128,128,128,0.3)", borderwidth=1
    )
    return fig

@st.cache_data
def load_data(ticker, start_date, end_date):
    """Load and preprocess stock data with enhanced features"""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty or isinstance(df.columns, pd.MultiIndex):
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
        
        # Enhanced technical indicators
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # Price momentum indicators
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        df['RSI_fast'] = ta.momentum.RSIIndicator(close=df['Close'], window=7).rsi()
        df['RSI_slow'] = ta.momentum.RSIIndicator(close=df['Close'], window=21).rsi()
        
        # MACD variations
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_hist'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=df['Close'], window=20)
        df['BB_high'] = bb.bollinger_hband()
        df['BB_low'] = bb.bollinger_lband()
        df['BB_mid'] = bb.bollinger_mavg()
        df['BB_width'] = (df['BB_high'] - df['BB_low']) / df['BB_mid']
        df['BB_position'] = (df['Close'] - df['BB_low']) / (df['BB_high'] - df['BB_low'])
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Returns_5d'] = df['Close'].pct_change(5)
        df['Returns_10d'] = df['Close'].pct_change(10)
        
        # Volatility measures
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['Volatility_short'] = df['Returns'].rolling(window=5).std()
        df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
        
        # Volume indicators
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_MA20']
        df['Volume_price_trend'] = ta.volume.VolumePriceTrendIndicator(close=df['Close'], volume=df['Volume']).volume_price_trend()
        
        # Trend strength
        df['ADX'] = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close']).adx()
        
        # Price position relative to ranges
        df['High_Low_Pct'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['Close_Open_Pct'] = (df['Close'] - df['Open']) / df['Open']
        
        # Support/Resistance levels
        df['Support_20'] = df['Low'].rolling(window=20).min()
        df['Resistance_20'] = df['High'].rolling(window=20).max()
        df['Support_distance'] = (df['Close'] - df['Support_20']) / df['Close']
        df['Resistance_distance'] = (df['Resistance_20'] - df['Close']) / df['Close']
        
        return df.dropna()
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        return None

def create_advanced_features(df, window_size):
    """Create advanced features for improved neural network"""
    features = []
    targets = []
    
    # Enhanced feature set
    feature_columns = [
        'Close', 'Volume', 'MA5', 'MA10', 'MA20', 'MA50', 'MA200',
        'RSI', 'RSI_fast', 'RSI_slow', 'MACD', 'MACD_signal', 'MACD_hist',
        'BB_high', 'BB_low', 'BB_width', 'BB_position',
        'Returns', 'Returns_5d', 'Returns_10d', 'Volatility', 'Volatility_short', 'ATR',
        'Volume_ratio', 'Volume_price_trend', 'ADX',
        'High_Low_Pct', 'Close_Open_Pct', 'Support_distance', 'Resistance_distance'
    ]
    
    # Fill missing values with forward fill then backward fill
    for col in feature_columns:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        else:
            df[col] = df['Close'].fillna(method='ffill').fillna(method='bfill')
    
    # Normalize features relative to current price for better scaling
    price_cols = ['Close', 'MA5', 'MA10', 'MA20', 'MA50', 'MA200', 'BB_high', 'BB_low', 'Support_20', 'Resistance_20']
    for col in price_cols:
        if col in df.columns:
            df[f'{col}_norm'] = df[col] / df['Close']
            if f'{col}_norm' not in feature_columns:
                feature_columns.append(f'{col}_norm')
    
    data = df[feature_columns].values
    target_data = df['Close'].values
    
    # Create sequences with overlap for better training
    for i in range(window_size, len(data)):
        feature_window = data[i-window_size:i].flatten()
        features.append(feature_window)
        targets.append(target_data[i])
    
    return np.array(features), np.array(targets)

def build_enhanced_model(df, window_size, max_iterations):
    """Build neural network with bias correction"""
    np.random.seed(42)
    
    X, y = create_advanced_features(df, window_size)
    
    if len(X) == 0:
        raise ValueError("Not enough data to create features")
    
    # Use RobustScaler for better handling of outliers
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # Enhanced train/validation/test split
    train_size = int(len(X_scaled) * 0.7)
    val_size = int(len(X_scaled) * 0.15)
    
    X_train = X_scaled[:train_size]
    X_val = X_scaled[train_size:train_size+val_size]
    X_test = X_scaled[train_size+val_size:]
    
    y_train = y_scaled[:train_size]
    y_val = y_scaled[train_size:train_size+val_size]
    y_test = y_scaled[train_size+val_size:]
    
    # Enhanced Neural Network with dropout-like regularization
    model = MLPRegressor(
        hidden_layer_sizes=(200, 100, 50, 25),  # Deeper network
        activation='relu',  # ReLU for better gradient flow
        solver='adam',
        learning_rate='adaptive',  # Adaptive learning rate
        learning_rate_init=0.001,
        max_iter=max_iterations,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,  # More patience
        alpha=0.001,  # Stronger regularization
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    
    # Ensemble approach: Train multiple models
    models = []
    predictions_ensemble = []
    
    for seed in [42, 123, 456]:  # Multiple random seeds
        model_temp = MLPRegressor(
            hidden_layer_sizes=(150, 75, 40),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=max_iterations,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            alpha=0.001
        )
        
        model_temp.fit(X_train, y_train)
        pred_temp = model_temp.predict(X_test)
        predictions_ensemble.append(pred_temp)
        models.append(model_temp)
    
    # Average ensemble predictions
    y_pred_scaled = np.mean(predictions_ensemble, axis=0)
    
    # Transform back to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    mae = mean_absolute_error(y_test_actual, y_pred)
    mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100
    
    # Future prediction with trend consideration
    recent_data = df.tail(window_size * 2)  # Use more recent data
    recent_trend = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[-window_size]) / recent_data['Close'].iloc[-window_size]
    
    # Get last features for prediction
    last_features, _ = create_advanced_features(df.tail(window_size + 1), window_size)
    if len(last_features) > 0:
        last_features_scaled = scaler_X.transform(last_features[-1:])
        
        # Ensemble prediction
        next_predictions = []
        for model_temp in models:
            pred_temp = model_temp.predict(last_features_scaled)
            next_predictions.append(pred_temp[0])
        
        next_price_scaled = np.mean(next_predictions)
        next_price_raw = scaler_y.inverse_transform([[next_price_scaled]])[0, 0]
        
        # Apply trend correction and bias adjustment
        current_price = df['Close'].iloc[-1]
        
        # Bias correction: Adjust for historical bias
        prediction_bias = np.mean(y_pred - y_test_actual) / np.mean(y_test_actual)
        
        # Trend-aware prediction
        trend_factor = 1 + (recent_trend * 0.3)  # Moderate trend influence
        next_price = next_price_raw * trend_factor
        
        # Bias correction
        next_price = next_price * (1 - prediction_bias)
        
        # Ensure reasonable bounds (within ±20% of current price for single-day prediction)
        price_change_limit = 0.20
        max_price = current_price * (1 + price_change_limit)
        min_price = current_price * (1 - price_change_limit)
        next_price = np.clip(next_price, min_price, max_price)
        
        # Add small positive bias for growth expectation (market tends to grow over time)
        growth_bias = 0.0002  # 0.02% daily growth expectation
        next_price = next_price * (1 + growth_bias)
        
    else:
        # Fallback prediction
        current_price = df['Close'].iloc[-1]
        ma_short = df['MA5'].iloc[-1]
        ma_long = df['MA20'].iloc[-1]
        
        if ma_short > ma_long:  # Uptrend
            next_price = current_price * 1.005  # 0.5% increase
        else:  # Downtrend or sideways
            next_price = current_price * 1.001  # 0.1% increase
    
    return y_test_actual, y_pred, next_price, rmse, mae, mape

def calculate_enhanced_score(df, next_price, current_price, mape):
    """scoring system"""
    expected_return = ((next_price - current_price) / current_price) * 100
    
    # Technical indicators
    rsi = df['RSI'].iloc[-1]
    macd_signal = 1 if df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1] else -1
    bb_position = df['BB_position'].iloc[-1] if 'BB_position' in df.columns else 0.5
    volume_trend = df['Volume_ratio'].iloc[-1] if 'Volume_ratio' in df.columns else 1
    
    # Scoring components
    return_score = min(max(expected_return * 8, -40), 40)  # Cap extreme values
    rsi_score = 100 - abs(rsi - 50) * 1.5  # Favor moderate RSI
    macd_score = macd_signal * 20  # MACD signal strength
    bb_score = (1 - abs(bb_position - 0.5)) * 40  # Favor middle of BB
    volume_score = min(volume_trend * 10, 20)  # Volume confirmation
    accuracy_score = max(100 - mape, 0)
    
    # Weighted total score
    total_score = (
        return_score * 0.35 +
        rsi_score * 0.15 +
        macd_score * 0.15 +
        bb_score * 0.10 +
        volume_score * 0.10 +
        accuracy_score * 0.15
    )
    
    return {
        'total_score': max(total_score, 0), 'expected_return': expected_return,
        'current_price': current_price, 'predicted_price': next_price,
        'rsi': rsi, 'accuracy': accuracy_score
    }

def create_enhanced_charts(df, actual, predicted, ticker):
    """Create visualization charts"""
    company_name = COMPANY_NAMES.get(ticker, ticker)
    
    # Price chart with predictions
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[f'{company_name} - Price & Moving Averages', 'Volume Analysis', 'Technical Indicators'],
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price data
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='green', width=1)), row=1, col=1)
    
    if 'BB_high' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_high'], name='BB High', line=dict(color='red', dash='dash', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_low'], name='BB Low', line=dict(color='red', dash='dash', width=1)), row=1, col=1)
    
    # Volume
    colors = ['green' if df['Returns'].iloc[i] > 0 else 'red' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors, opacity=0.6), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Volume_MA20'], name='Volume MA20', line=dict(color='blue', width=2)), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1,  annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, annotation_text="Oversold")
    
    fig.update_layout(title=f'{company_name} - Analysis', height=800, template='plotly_white')
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    
    return add_watermark(fig)

def create_csv_download(summary_df, selected_stocks, start_date, end_date):
    """Create CSV content for download"""
    # Add metadata
    metadata_df = pd.DataFrame({
        'Analysis Details': [
            f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            f'Analysis Period: {start_date} to {end_date}',
            f'Stocks Analyzed: {", ".join(selected_stocks)}',
            f'Total Stocks: {len(selected_stocks)}',
            '',
            'Stock Analysis Results:'
        ]
    })
    
    # Create buffer
    output = io.StringIO()
    
    # Write metadata
    metadata_df.to_csv(output, index=False, header=False)
    output.write('\n')
    
    # Write summary data
    summary_df.to_csv(output, index=False)
    
    # Additional analysis notes
    output.write('\n\nNotes:\n')
    output.write('- Score: Overall prediction confidence (0-100)\n')
    output.write('- Expected Return: Predicted percentage change\n')
    output.write('- MAPE: Mean Absolute Percentage Error (lower is better)\n')
    output.write('- This analysis is for educational purposes only\n')
    output.write('- Not financial advice - please do your own research\n')
    
    return output.getvalue()

def create_stock_comparison_charts(results_dict, stock_scores_dict, selected_tickers):
    """
    Create comprehensive stock comparison charts
   
    """
    
    # Color palette for different stocks
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 1. Price Comparison Chart (Normalized)
    def create_price_comparison():
        fig = go.Figure()
        
        for i, ticker in enumerate(selected_tickers):
            if ticker in results_dict:
                df = results_dict[ticker]['df']
                # Normalize prices to start from 100 for comparison
                normalized_prices = (df['Close'] / df['Close'].iloc[0]) * 100
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=normalized_prices,
                    mode='lines',
                    name=f'{COMPANY_NAMES.get(ticker, ticker)} ({ticker})',
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f'<b>{ticker}</b><br>' +
                                'Date: %{x}<br>' +
                                'Normalized Price: %{y:.2f}<br>' +
                                '<extra></extra>'
                ))
        
        fig.update_layout(
            title='📈 Stock Price Comparison (Normalized to 100)',
            xaxis_title='Date',
            yaxis_title='Normalized Price (Base = 100)',
            template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return add_watermark(fig)
    
    # 2. Volume Comparison Chart
    def create_volume_comparison():
        fig = go.Figure()
        
        for i, ticker in enumerate(selected_tickers):
            if ticker in results_dict:
                df = results_dict[ticker]['df']
                # Use volume ratio for better comparison
                volume_ratio = df['Volume_ratio'] if 'Volume_ratio' in df.columns else df['Volume'] / df['Volume'].rolling(20).mean()
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=volume_ratio,
                    mode='lines',
                    name=f'{ticker}',
                    line=dict(color=colors[i % len(colors)], width=1.5),
                    opacity=0.8
                ))
        
        fig.add_hline(y=1, line_dash="dash", line_color="gray", 
                     annotation_text="Average Volume")
        
        fig.update_layout(
            title='📊 Volume Ratio Comparison',
            xaxis_title='Date',
            yaxis_title='Volume Ratio (vs 20-day MA)',
            template='plotly_white',
            hovermode='x unified'
        )
        
        return add_watermark(fig)
    
    # 3. Technical Indicators Comparison
    def create_technical_comparison():
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['RSI Comparison', 'MACD Comparison', 'Bollinger Band Position', 'Volatility Comparison'],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        for i, ticker in enumerate(selected_tickers):
            if ticker in results_dict:
                df = results_dict[ticker]['df']
                color = colors[i % len(colors)]
                
                # RSI
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['RSI'], name=f'{ticker} RSI',
                    line=dict(color=color, width=1.5), legendgroup=ticker
                ), row=1, col=1)
                
                # MACD
                if 'MACD' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['MACD'], name=f'{ticker} MACD',
                        line=dict(color=color, width=1.5), legendgroup=ticker,
                        showlegend=False
                    ), row=1, col=2)
                
                # Bollinger Band Position
                if 'BB_position' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['BB_position'], name=f'{ticker} BB Pos',
                        line=dict(color=color, width=1.5), legendgroup=ticker,
                        showlegend=False
                    ), row=2, col=1)
                
                # Volatility
                if 'Volatility' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['Volatility'] * 100, name=f'{ticker} Vol',
                        line=dict(color=color, width=1.5), legendgroup=ticker,
                        showlegend=False
                    ), row=2, col=2)
        
        # Add reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=2, col=1)
        
        fig.update_layout(
            title='📊 Technical Indicators Comparison',
            height=600,
            template='plotly_white'
        )
        
        return add_watermark(fig)
    
    # 4. Performance Metrics Comparison
    def create_performance_comparison():
        # Prepare data for comparison
        comparison_data = []
        for ticker in selected_tickers:
            if ticker in results_dict and ticker in stock_scores_dict:
                result = results_dict[ticker]
                score = stock_scores_dict[ticker]
                
                comparison_data.append({
                    'Ticker': ticker,
                    'Company': COMPANY_NAMES.get(ticker, ticker),
                    'Score': score['total_score'],
                    'Expected_Return': score['expected_return'],
                    'MAPE': result['mape'],
                    'RMSE': result['rmse'],
                    'Current_Price': score['current_price'],
                    'Predicted_Price': score['predicted_price'],
                    'RSI': score['rsi']
                })
        
        df_comp = pd.DataFrame(comparison_data)
        
        # Create radar chart for performance comparison
        fig = go.Figure()
        
        for i, row in df_comp.iterrows():
            # Normalize metrics for radar chart (0-100 scale)
            score_norm = row['Score']
            return_norm = min(max((row['Expected_Return'] + 5) * 10, 0), 100)  # Scale return
            accuracy_norm = max(100 - row['MAPE'], 0)  # Invert MAPE
            rsi_norm = 100 - abs(row['RSI'] - 50) * 2  # Favor RSI around 50
            
            fig.add_trace(go.Scatterpolar(
                r=[score_norm, return_norm, accuracy_norm, rsi_norm, score_norm],
                theta=['Overall Score', 'Expected Return', 'Accuracy', 'RSI Health', 'Overall Score'],
                fill='toself',
                name=f"{row['Company']} ({row['Ticker']})",
                line=dict(color=colors[i % len(colors)]),
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="📊 Multi-Metric Performance Comparison"
        )
        
        return add_watermark(fig)
    
    # 5. Prediction Accuracy Comparison
    def create_accuracy_comparison():
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['RMSE Comparison', 'MAPE Comparison'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        tickers = []
        rmse_values = []
        mape_values = []
        
        for ticker in selected_tickers:
            if ticker in results_dict:
                tickers.append(ticker)
                rmse_values.append(results_dict[ticker]['rmse'])
                mape_values.append(results_dict[ticker]['mape'])
        
        # RMSE Bar Chart
        fig.add_trace(go.Bar(
            x=tickers,
            y=rmse_values,
            name='RMSE',
            marker=dict(color=colors[:len(tickers)]),
            text=[f'{val:.2f}' for val in rmse_values],
            textposition='auto'
        ), row=1, col=1)
        
        # MAPE Bar Chart
        fig.add_trace(go.Bar(
            x=tickers,
            y=mape_values,
            name='MAPE (%)',
            marker=dict(color=colors[:len(tickers)]),
            text=[f'{val:.1f}%' for val in mape_values],
            textposition='auto',
            showlegend=False
        ), row=1, col=2)
        
        fig.update_layout(
            title='📊 Model Accuracy Comparison',
            height=400,
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="RMSE", row=1, col=1)
        fig.update_yaxes(title_text="MAPE (%)", row=1, col=2)
        
        return add_watermark(fig)
    
    # 6. Price Target vs Current Price Comparison
    def create_price_target_comparison():
        fig = go.Figure()
        
        tickers = []
        current_prices = []
        target_prices = []
        expected_returns = []
        
        for ticker in selected_tickers:
            if ticker in stock_scores_dict:
                score = stock_scores_dict[ticker]
                tickers.append(ticker)
                current_prices.append(score['current_price'])
                target_prices.append(score['predicted_price'])
                expected_returns.append(score['expected_return'])
        
        # Current Price bars
        fig.add_trace(go.Bar(
            x=tickers,
            y=current_prices,
            name='Current Price',
            marker=dict(color='lightblue'),
            text=[f'${val:.2f}' for val in current_prices],
            textposition='auto'
        ))
        
        # Target Price bars  
        fig.add_trace(go.Bar(
            x=tickers,
            y=target_prices,
            name='Target Price',
            marker=dict(color='orange'),
            text=[f'${val:.2f}' for val in target_prices],
            textposition='auto'
        ))
        
        # Add expected return as annotations
        for i, (ticker, return_val) in enumerate(zip(tickers, expected_returns)):
            fig.add_annotation(
                x=ticker,
                y=max(current_prices[i], target_prices[i]) * 1.1,
                text=f'{return_val:+.2f}%',
                showarrow=False,
                font=dict(size=12, color='red' if return_val < 0 else 'green')
            )
        
        fig.update_layout(
            title='💰 Current vs Target Price Comparison',
            xaxis_title='Stock Ticker',
            yaxis_title='Price ($)',
            template='plotly_white',
            barmode='group'
        )
        
        return add_watermark(fig)
    
    # Return all charts as a dictionary
    charts = {
        'price_comparison': create_price_comparison(),
        'volume_comparison': create_volume_comparison(),
        'technical_comparison': create_technical_comparison(),
        'performance_comparison': create_performance_comparison(),
        'accuracy_comparison': create_accuracy_comparison(),
        'price_target_comparison': create_price_target_comparison()
    }
    
    return charts


# Usage example function
def display_comparison_charts(results, stock_scores, selected_stocks):
    """
    Display all comparison charts in Streamlit
    
    """
    
    if len(selected_stocks) < 2:
        st.warning("Need at least 2 stocks for comparison charts.")
        return
    
    st.markdown("## 📊 Stock Comparison Analysis")
    
    # Create all comparison charts
    charts = create_stock_comparison_charts(results, stock_scores, selected_stocks)
    
    # Display charts in tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Price Trends", "📊 Volume", "🔧 Technical", 
        "🎯 Performance", "📏 Accuracy", "💰 Price Targets"
    ])
    
    with tab1:
        st.plotly_chart(charts['price_comparison'], use_container_width=True)
        st.info("💡 This chart shows normalized price movements. All stocks start at 100 for easy comparison of relative performance.")
    
    with tab2:
        st.plotly_chart(charts['volume_comparison'], use_container_width=True)
        st.info("💡 Volume ratio shows trading activity relative to the 20-day average. Values above 1 indicate higher than average volume.")
    
    with tab3:
        st.plotly_chart(charts['technical_comparison'], use_container_width=True)
        st.info("💡 Technical indicators help identify overbought/oversold conditions and trend strength across stocks.")
    
    with tab4:
        st.plotly_chart(charts['performance_comparison'], use_container_width=True)
        st.info("💡 Radar chart comparing multiple performance metrics. Larger areas indicate better overall performance.")
    
    with tab5:
        st.plotly_chart(charts['accuracy_comparison'], use_container_width=True)
        st.info("💡 Model accuracy comparison. Lower RMSE and MAPE values indicate more accurate predictions.")
    
    with tab6:
        st.plotly_chart(charts['price_target_comparison'], use_container_width=True)
        st.info("💡 Comparison of current prices vs predicted target prices. Percentages show expected returns.")

def main():
    st.markdown('<h1 class="main-header">📈 Stock Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: blue;">ML-based Stock Prediction App by Priyanshu Joarder</p>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown(
    "<h2 style=' color: #00CED1; font-size: 24px;'>Configuration</h2>",
    unsafe_allow_html=True
    )


    selected_stocks = st.sidebar.multiselect(
        "Select Stocks (Max 3)",
        list(COMPANY_NAMES.keys()),
        default=['AAPL', 'NVDA'],
        max_selections=3
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2021, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    window_size = st.sidebar.slider("Prediction Window (days)", 60, 200, 120)
    max_iterations = st.sidebar.slider("Max Training Iterations", 200, 1500, 800)
    
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        if not selected_stocks:
            st.warning("Please select at least one stock.")
            return
            
        with st.spinner("Running analysis..."):
            results = {}
            stock_scores = {}
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, ticker in enumerate(selected_stocks):
                company_name = COMPANY_NAMES.get(ticker, ticker)
                # Update progress
                progress = (i + 1) / len(selected_stocks)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing {company_name}... ({i+1}/{len(selected_stocks)})")
                
                df = load_data(ticker, start_date, end_date)
                if df is None or len(df) < window_size + 100:
                    st.warning(f"Insufficient data for {company_name}")
                    continue
                
                try:
                    actual, predicted, next_price, rmse, mae, mape = build_enhanced_model(df, window_size, max_iterations)
                    current_price = df['Close'].iloc[-1]
                    
                    results[ticker] = {
                        'actual': actual, 'predicted': predicted, 'next_price': next_price,
                        'current_price': current_price, 'rmse': rmse, 'mae': mae, 'mape': mape, 'df': df
                    }
                    
                    stock_scores[ticker] = calculate_enhanced_score(df, next_price, current_price, mape)
                    
                except Exception as e:
                    st.error(f"Error analyzing {company_name}: {str(e)}")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            if not results:
                st.error("No stocks analyzed successfully.")
                return
            
            # Results display
            st.success(f"✅ Analyzed {len(results)} stocks successfully!")
            
            # Top recommendation
            sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
            if sorted_stocks:
                best_stock = sorted_stocks[0]
                best_ticker = best_stock[0]
                best_company = COMPANY_NAMES.get(best_ticker, best_ticker)
                
                st.markdown("## 🏆 Top Recommendation")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f'<div class="metric-card success-metric"><h3>{best_company}</h3><p>({best_ticker})</p></div>', unsafe_allow_html=True)
                with col2:
                    st.metric("Score", f"{best_stock[1]['total_score']:.1f}/100")
                with col3:
                    st.metric("Expected Return", f"{best_stock[1]['expected_return']:.2f}%")
                with col4:
                    st.metric("Target Price", f"${best_stock[1]['predicted_price']:.2f}")
            
            # Summary table
            st.markdown("## 📊 Analysis Summary")
            summary_data = []
            for ticker, score_data in stock_scores.items():
                result_data = results[ticker]
                summary_data.append({
                    'Ticker': ticker,
                    'Company': COMPANY_NAMES.get(ticker, ticker),
                    'Score': f"{score_data['total_score']:.1f}",
                    'Expected Return (%)': f"{score_data['expected_return']:.2f}",
                    'Current Price ($)': f"{score_data['current_price']:.2f}",
                    'Target Price ($)': f"{score_data['predicted_price']:.2f}",
                    'MAPE (%)': f"{result_data['mape']:.1f}",
                    'RMSE': f"{result_data['rmse']:.2f}",
                    'MAE': f"{result_data['mae']:.2f}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('Score', key=lambda x: x.astype(float), ascending=False)
            st.dataframe(summary_df.reset_index(drop=True), use_container_width=True)
            
             # Stock Comparison Charts Section (ADD THIS)
            if len(selected_stocks) >= 2 and len(results) >= 2:
                st.markdown("---")  # Add separator line
                display_comparison_charts(results, stock_scores, selected_stocks)
            elif len(selected_stocks) >= 2:
                st.info("💡 Select and successfully analyze at least 2 stocks to see comparison charts.")

            # CSV Download Section
            st.markdown("## 📥 Download Results")
            
            # Create CSV content
            csv_content = create_csv_download(summary_df, selected_stocks, start_date, end_date)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stock_analysis_summary_{timestamp}.csv"
            
            # Download button
            st.download_button(
                label="📊 Download Summary as CSV",
                data=csv_content,
                file_name=filename,
                mime="text/csv",
                help="Download the complete analysis summary including metadata and results"
            )
            
            # Show file info
            #st.info(f"📄 File will be saved as: {filename}")
            
            # Individual analysis
            for ticker in results:
                company_name = COMPANY_NAMES.get(ticker, ticker)
                with st.expander(f"📊 {company_name} Detailed Analysis"):
                    result = results[ticker]
                    score = stock_scores[ticker]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Score", f"{score['total_score']:.1f}")
                    with col2:
                        st.metric("Expected Return", f"{score['expected_return']:.2f}%")
                    with col3:
                        st.metric("RMSE", f"{result['rmse']:.2f}")
                    with col4:
                        st.metric("MAPE", f"{result['mape']:.1f}%")
                    
                    fig = create_enhanced_charts(result['df'], result['actual'], result['predicted'], ticker)
                    st.plotly_chart(fig, use_container_width=True)
            
            st.warning("⚠️ **Disclaimer**: This is a Machine Learning generated prediction. Please consult and do you own research before investing.")
    
    # Information section
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        ### How It Works
        This advanced app uses an **Ensemble Multi-Layer Perceptron (MLP) Neural Network** with bias correction to predict stock prices based on:
    
        ### 📊 Market Data
        - **Historical Prices**: OHLC (Open, High, Low, Close) data
        - **Volume Analysis**: Trading volume, volume ratios, and volume-price trends
        - **Moving Averages**: MA5, MA10, MA20, MA50, MA200
        - **Momentum Oscillators**: RSI (3 timeframes), MACD with signal and histogram
        - **Volatility Measures**: Bollinger Bands, ATR, rolling volatility

    
        ### Scoring System
        The comprehensive score (0-100) combines multiple factors:
    
        - **Expected Return (35%)**: Predicted price change potential
        - **RSI Health (15%)**: Technical momentum indicator (favors moderate RSI)
        - **MACD Signal (15%)**: Trend direction confirmation
        - **Bollinger Position (10%)**: Price position within volatility bands
        - **Volume Confirmation (10%)**: Trading volume support
        - **Model Accuracy (15%)**: Prediction reliability (inverse of MAPE)
    
        ### Key Features
    
        **📈 Analysis Capabilities**
        - Real-time data from Yahoo Finance via yfinance
        - Advanced technical analysis with 30+ indicators
        - Multi-stock comparative analysis (up to 3 stocks)
        - Ensemble neural network predictions
        - Bias-corrected price targets
    
    
        ### Limitations & Considerations
    
        **⚠️ Technical Limitations**
        - Maximum 3 stocks per analysis session
        - Requires minimum 160+ days of historical data
        - Processing time increases with more stocks and longer training
        - Model performance varies with market conditions
    
    
        ---
        *Developed by Priyanshu Joarder | Enhanced ML Stock Prediction App*
        """)

if __name__ == "__main__":
    main()