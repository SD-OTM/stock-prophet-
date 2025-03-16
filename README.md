# Stock Prophet

A sophisticated stock analysis and forecasting application designed to provide comprehensive financial insights using Python.

## Features

- **Stock Analysis**: Get detailed analysis of stocks using technical indicators (RSI, EMAs, Bollinger Bands, MACD, Stochastic, ADX)
- **Price Forecasting**: 5-hour price forecasts using ARIMA model
- **Trading Signals**: Buy/sell recommendations based on technical analysis
- **Watchlist Management**: Track your favorite stocks with automatic notifications
- **Portfolio Management**: Track your investments with profit/loss calculations
- **Telegram Bot Interface**: Easy-to-use commands for all features

## Telegram Bot Commands

- `/start` - Start the bot and get a welcome message
- `/help` - Show help message
- `/ticker SYMBOL` - Analyze a stock ticker (e.g., `/ticker AAPL`)

### Watchlist Commands
- `/add SYMBOL` - Add a stock to your watchlist
- `/remove SYMBOL` - Remove a stock from your watchlist
- `/watchlist` - View your current watchlist

### Portfolio Commands
- `/portfolio` - Show your current portfolio and performance
- `/buy TICKER PRICE QUANTITY` - Add a stock to your portfolio
- `/sell TICKER` - Remove a stock from your portfolio

## Technical Details

- **Data Source**: Yahoo Finance (yfinance)
- **Technical Analysis**: pandas-ta for indicators
- **Forecasting**: statsmodels ARIMA
- **Bot Framework**: python-telegram-bot
- **Storage**: JSON files for persistence

## Setup Instructions

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your Telegram bot token:
   - Create a bot with BotFather on Telegram
   - Set the token as an environment variable: `export TELEGRAM_BOT_TOKEN=your_token_here`
4. Run the bot: `python main.py`

## Testing

Run the built-in test suite:
- Test stock analysis: `python test_stock.py AAPL`
- Test portfolio management: `python test_stock.py portfolio`

## License

Private repository - All rights reserved.