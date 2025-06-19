# 📈 Stock Predictor App

A sophisticated LSTM-based stock price prediction web application built with Streamlit. This dashboard provides AI-powered stock analysis, technical indicators, and price forecasting for major tech stocks.

## 🚀 Features

- **LSTM Neural Network**: Advanced deep learning model for stock price prediction
- **Technical Analysis**: RSI, MACD, Moving Averages, and Volume analysis
- **Interactive Charts**: Beautiful, responsive charts powered by Plotly
- **Stock Comparison**: Compare multiple stocks side-by-side
- **Smart Recommendations**: AI-powered stock scoring and recommendations
- **Real-time Data**: Live stock data from Yahoo Finance

## 📊 Supported Stocks

- Apple Inc. (AAPL)
- Microsoft Corporation (MSFT)
- Alphabet Inc. (GOOGL)
- Amazon.com Inc. (AMZN)
- Tesla Inc. (TSLA)
- Meta Platforms Inc. (META)
- NVIDIA Corporation (NVDA)

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

Visit the live application: [Stock Prediction Dashboard](https://your-deployed-app-url.streamlit.app)

## 🎯 How to Use

1. **Select Stocks**: Choose up to 3 stocks from the sidebar
2. **Configure Parameters**: 
   - Set date range for historical data
   - Adjust LSTM window size (30-90 days)
   - Set training epochs (10-50)
3. **Run Analysis**: Click "🚀 Run Analysis" to start prediction
4. **View Results**: 
   - See top stock recommendation
   - Compare all selected stocks
   - Explore detailed technical analysis

## 📈 Model Architecture

The application uses an LSTM (Long Short-Term Memory) neural network with:
- **Input Layer**: Sequential stock price data
- **LSTM Layers**: 2 layers with 50 units each
- **Dropout Layers**: 20% dropout for regularization
- **Dense Layers**: Fully connected layers for final prediction
- **Output**: Next day price prediction

## 📊 Technical Indicators

- **Moving Averages**: 20-day and 50-day MA
- **RSI**: Relative Strength Index (14-day)
- **MACD**: Moving Average Convergence Divergence
- **Volume Analysis**: Trading volume with moving average
- **Price Returns**: Daily percentage changes
- **Volatility**: 20-day rolling standard deviation

## ⚠️ Disclaimer

**This application is for educational and research purposes only. Stock market predictions are inherently uncertain and this tool should not be used as the sole basis for investment decisions. Always conduct your own research and consult with financial advisors before making investment choices.**

## 🛡️ Risk Warnings

- Past performance does not guarantee future results
- Stock prices can be highly volatile and unpredictable
- Machine learning models may not capture all market dynamics
- External factors (news, events, etc.) significantly impact stock prices

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
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web app framework
- [Yahoo Finance](https://finance.yahoo.com/) for providing free stock data
- [TensorFlow](https://tensorflow.org/) for the deep learning capabilities
- [Plotly](https://plotly.com/) for interactive visualizations


---

⭐ **If you found this project helpful, please give it a star!** ⭐
