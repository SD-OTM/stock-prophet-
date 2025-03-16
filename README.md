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

### Basic Commands
- `/start` - Start the bot and get a welcome message
- `/help` - Show help message with all available commands
- `/ticker SYMBOL` - Analyze a stock ticker with technical indicators and price forecast

### Examples:
```
/ticker TSLA
```
The above command will provide a comprehensive analysis of Tesla stock including:
- Current price and trend direction (bullish, bearish, or neutral)
- Technical indicators (RSI, EMA, Bollinger Bands, MACD, Stochastic)
- 5-hour price forecast with percentage change prediction
- Trading signals based on your selected strategy
- Recommended trading thresholds (take profit and stop loss percentages)

```
/ticker NVDA
```
This command will analyze NVIDIA stock with all technical indicators and provide buy/sell signals based on the combined strategy.

### Watchlist Commands
- `/add SYMBOL` - Add a stock to your watchlist (e.g., `/add MSFT`)
- `/remove SYMBOL` - Remove a stock from your watchlist (e.g., `/remove MSFT`)
- `/watchlist` - View your current watchlist with latest price updates

### Portfolio Commands
- `/portfolio` - Show your current portfolio and performance metrics
- `/buy TICKER PRICE QUANTITY` - Add a stock to your portfolio (e.g., `/buy AMZN 178.25 10`)
- `/sell TICKER` - Remove a stock from your portfolio (e.g., `/sell AMZN`)

### Strategy Commands
- `/strategies` - Show all available trading strategies
- `/strategy STRATEGY_NAME` - Set your preferred trading strategy (e.g., `/strategy combined`)

## Technical Details

- **Data Source**: Yahoo Finance (yfinance)
- **Technical Analysis**: pandas-ta for indicators
- **Forecasting**: statsmodels ARIMA
- **Bot Framework**: python-telegram-bot
- **Storage**: JSON files for persistence

## Setup Instructions

### Local Development

1. Clone this repository:
   ```bash
   git clone https://github.com/[your-username]/stock-prophet.git
   cd stock-prophet
   ```

2. Install dependencies:
   ```bash
   pip install -e .  # Install in development mode
   # OR
   pip install -r requirements-ci.txt
   ```

3. Set up your Telegram bot token:
   - Create a bot with BotFather on Telegram (@BotFather)
   - Get your token and set it as an environment variable:
   ```bash
   export TELEGRAM_BOT_TOKEN=your_token_here
   ```

4. Run the bot:
   ```bash
   python main.py
   ```

### GitHub Repository Setup

#### Option 1: Using the Setup Script (Recommended)

We've included a convenient setup script that initializes your GitHub repository and configures all necessary secrets:

```bash
# Make the script executable if it's not already
chmod +x github_setup.sh

# Run the setup script
./github_setup.sh
```

The script will:
- Create a new GitHub repository
- Set up GitHub secrets for your tokens
- Initialize git and push your code
- Configure GitHub Actions for CI/CD

#### Option 2: Manual Setup

1. Create a new GitHub repository
2. Push your local code to GitHub:
   ```bash
   git remote add origin https://github.com/[your-username]/stock-prophet.git
   git branch -M main
   git push -u origin main
   ```

3. GitHub Actions will automatically run tests on each push

4. For private repositories, set up your Telegram Bot Token as a repository secret:
   - Go to your repository on GitHub
   - Navigate to Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Add `TELEGRAM_BOT_TOKEN` as the name and your token as the value

5. (Optional) To enable SMS notifications, set up Twilio credentials as repository secrets:
   ```
   TWILIO_ACCOUNT_SID
   TWILIO_AUTH_TOKEN
   TWILIO_PHONE_NUMBER
   ```

## Testing

Run the built-in test suite:
- Test stock analysis: `python test_stock.py AAPL`
- Test portfolio management: `python test_stock.py portfolio`

## Continuous Integration

This project uses GitHub Actions for continuous integration. Every push to the main branch or pull request will automatically:

1. Set up a Python 3.11 environment
2. Install all dependencies
3. Run stock analysis tests
4. Run portfolio management tests

You can also manually trigger the tests by going to the "Actions" tab in GitHub and selecting "Run workflow" on the Stock Prophet Tests workflow.

[![Stock Prophet Tests](https://github.com/[your-username]/stock-prophet/actions/workflows/stock-prophet-tests.yml/badge.svg)](https://github.com/[your-username]/stock-prophet/actions/workflows/stock-prophet-tests.yml)

## License

Private repository - All rights reserved.