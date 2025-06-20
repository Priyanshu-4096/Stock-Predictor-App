# 📈 Stock Prediction Dashboard

A sophisticated machine learning-based stock price prediction web application built with Streamlit. This dashboard provides AI-powered stock analysis, technical indicators, and price forecasting for major tech stocks using Neural Networks and Random Forest algorithms.

## 🚀 Features

- **Advanced ML Models**: Neural Network (MLP) and Random Forest algorithms for stock price prediction
- **Technical Analysis**: RSI, MACD, Moving Averages, and Volume analysis
- **Interactive Charts**: Beautiful, responsive charts powered by Plotly
- **Stock Comparison**: Compare up to 3 stocks side-by-side
- **Smart Recommendations**: AI-powered stock scoring and recommendations
- **Real-time Data**: Live stock data from Yahoo Finance
- **Performance Metrics**: RMSE, MAE, MAPE, and R² score analysis

## 📊 Supported Stocks

- Apple Inc. (AAPL)
- Microsoft Corporation (MSFT)
- Alphabet Inc. (GOOGL)
- Amazon.com Inc. (AMZN)
- Tesla Inc. (TSLA)
- Meta Platforms Inc. (META)
- NVIDIA Corporation (NVDA)
- Netflix Inc. (NFLX)
- Walt Disney Company (DIS)
- PayPal Holdings Inc. (PYPL)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/stock-prediction-dashboard.git
   cd stock-prediction-dashboard
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv stock_env
   source stock_env/bin/activate  # On Windows: stock_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## 🌐 Live Demo

Visit the live application: https://stock-predictor-app-pj.streamlit.app/

## 🎯 How to Use

1. **Select Stocks**: Choose up to 3 stocks from the sidebar
2. **Configure Parameters**: 
   - Set date range for historical data (minimum 1 year required)
   - Adjust window size for prediction (30-120 days)
   - Choose between Neural Network or Random Forest model
3. **Run Analysis**: Click "🚀 Run Analysis" to start prediction
4. **View Results**: 
   - See top stock recommendation with overall score
   - Compare all selected stocks in ranking table
   - Explore detailed technical analysis with interactive charts

## 📈 Model Architecture

The application offers two powerful machine learning models:

### Neural Network (MLP)
- **Architecture**: Multi-layer perceptron with 3 hidden layers (100, 50, 25 neurons)
- **Activation**: ReLU activation function
- **Optimizer**: Adam optimizer with adaptive learning rate
- **Regularization**: L2 regularization (alpha=0.001)
- **Best for**: Complex pattern recognition in stock data

### Random Forest
- **Estimators**: 100 decision trees
- **Max Depth**: 10 levels for optimal performance
- **Parallel Processing**: Multi-threaded execution
- **Best for**: Robust predictions with feature importance analysis

## 📊 Technical Indicators

- **Moving Averages**: 20-day and 50-day MA for trend analysis
- **RSI**: Relative Strength Index (14-day) for momentum analysis
- **MACD**: Moving Average Convergence Divergence with signal line
- **Volume Analysis**: Trading volume with 20-day moving average
- **Price Returns**: Daily percentage changes and volatility
- **Volatility**: 20-day rolling standard deviation

## 🔍 Feature Engineering

The model uses advanced feature engineering including:
- Price-based features (Open, High, Low, Close)
- Volume and volatility measures
- Technical indicators (RSI, MACD, Moving Averages)
- Statistical aggregations (mean, std, change ratios)
- Time-series window features for pattern recognition

## 📏 Performance Metrics

- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (average prediction error)
- **MAPE**: Mean Absolute Percentage Error (percentage-based accuracy)
- **R² Score**: Coefficient of determination (closer to 1 is better)

## ⚠️ Disclaimer

**This application is for educational and research purposes only. Stock market predictions are inherently uncertain and this tool should not be used as the sole basis for investment decisions. Always conduct your own research and consult with financial advisors before making investment choices.**

## 🛡️ Risk Warnings

- Past performance does not guarantee future results
- Stock prices can be highly volatile and unpredictable
- Machine learning models may not capture all market dynamics
- External factors (news, events, economic conditions) significantly impact stock prices
- Model predictions are based on historical patterns and technical indicators only

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Priyanshu Joarder**
- GitHub: https://github.com/Priyanshu-4096/
- LinkedIn: https://www.linkedin.com/in/priyanshu-joarder-308724144/

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web app framework
- [Yahoo Finance](https://finance.yahoo.com/) for providing free stock data via yfinance
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- [Plotly](https://plotly.com/) for interactive visualizations
- [Pandas](https://pandas.pydata.org/) for data manipulation and analysis

⭐ **If you found this project helpful, please give it a star!** ⭐
