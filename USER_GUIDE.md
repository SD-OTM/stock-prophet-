# Stock Prophet: User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Telegram Bot Commands](#telegram-bot-commands)
4. [Analyzing Stocks](#analyzing-stocks)
5. [Managing Your Watchlist](#managing-your-watchlist)
6. [Managing Your Portfolio](#managing-your-portfolio)
7. [Trading Strategies](#trading-strategies)
8. [Understanding Analysis Results](#understanding-analysis-results)
9. [Receiving Notifications](#receiving-notifications)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Usage](#advanced-usage)

## Introduction

Stock Prophet is a powerful stock analysis and forecasting tool that provides technical analysis, price predictions, and trading signals through a convenient Telegram bot interface. It helps you make informed trading decisions by:

- Analyzing stocks using multiple technical indicators
- Forecasting future price movements
- Providing actionable buy/sell signals
- Tracking your watchlist and portfolio
- Sending automated notifications about significant price movements

## Getting Started

### Prerequisites

Before you begin, you'll need:

1. A Telegram account
2. The Telegram app installed on your phone or desktop

### Setting Up

1. **Find the Bot**: Search for the Stock Prophet bot in Telegram (your bot's username)
2. **Start the Conversation**: Click the "Start" button or send the `/start` command
3. **Get Help**: Send the `/help` command to see all available commands

## Telegram Bot Commands

Stock Prophet understands the following commands:

| Command | Description | Example |
|---------|-------------|---------|
| `/start` | Introduction to the bot | `/start` |
| `/help` | List all available commands | `/help` |
| `/ticker SYMBOL` | Analyze a stock | `/ticker AAPL` |
| `/add SYMBOL` | Add a stock to your watchlist | `/add AAPL` |
| `/remove SYMBOL` | Remove a stock from your watchlist | `/remove AAPL` |
| `/watchlist` | View your watchlist | `/watchlist` |
| `/buy SYMBOL PRICE QUANTITY` | Add a position to your portfolio | `/buy AAPL 150.50 10` |
| `/sell SYMBOL [QUANTITY]` | Remove a position from your portfolio | `/sell AAPL` or `/sell AAPL 5` |
| `/portfolio` | View your portfolio | `/portfolio` |
| `/strategies` | List all available trading strategies | `/strategies` |
| `/strategy STRATEGY_NAME` | Change your trading strategy | `/strategy combined` |

## Analyzing Stocks

### Basic Analysis

To analyze a stock, use the `/ticker` command followed by the stock symbol:

```
/ticker AAPL
```

The bot will respond with:

1. Current trend (bullish, bearish, or neutral)
2. Technical indicators (RSI, EMA, Bollinger Bands, MACD, Stochastic)
3. Price forecast for the next 5 hours
4. Trading signals based on your selected strategy

### Multiple Stock Analysis

You can analyze multiple stocks in succession:

```
/ticker AAPL
/ticker MSFT
/ticker TSLA
```

### Example Analysis

```
Mr. Otmane, here's your analysis for AAPL:
📊 Trend: Bullish
📉 Technical Indicators:
RSI: 65.42
EMA (9): 174.25
EMA (21): 172.80
Bollinger Bands (Upper/Middle/Lower): 176.32/174.25/172.18
MACD: 1.4522 / Signal: 0.9844
Stochastic %K/%D: 82.45/76.92
📈 Price Forecast (next 5 hours):
1h: $175.12
2h: $175.35
3h: $175.58
4h: $175.82
5h: $176.05
📈 Price is predicted to rise by 1.03% over 5 hours.
Standard trading thresholds applied (take profit: 3%, stop loss: 4%).
🚀 Trading Signals (using Combined Strategy):
🔔 BUY: Multiple indicators showing bullish momentum.
⚠️ Set stop loss at $167.28 (-4.0%)
🎯 Set take profit at $179.25 (+3.0%)
```

## Managing Your Watchlist

Your watchlist allows you to track stocks you're interested in.

### Adding Stocks

```
/add AAPL
```

Response: "Added AAPL to your watchlist."

### Removing Stocks

```
/remove AAPL
```

Response: "Removed AAPL from your watchlist."

### Viewing Your Watchlist

```
/watchlist
```

Response:
```
Your Watchlist:
1. AAPL: $174.50 (1.2% ↑)
2. MSFT: $328.75 (0.5% ↑)
3. TSLA: $242.10 (2.1% ↓)
```

## Managing Your Portfolio

The portfolio feature allows you to track your investments.

### Adding Positions

To add a stock position:

```
/buy AAPL 150.50 10
```

This adds 10 shares of AAPL purchased at $150.50 per share.

### Removing Positions

To sell all shares of a stock:

```
/sell AAPL
```

To sell a specific quantity:

```
/sell AAPL 5
```

This sells 5 shares of AAPL from your portfolio.

### Viewing Your Portfolio

```
/portfolio
```

Response:
```
Your Portfolio:
AAPL: 10 shares @ $150.50
Current Price: $174.50
Value: $1,745.00
P/L: +$240.00 (+16.08%)

MSFT: 5 shares @ $300.25
Current Price: $328.75
Value: $1,643.75
P/L: +$142.50 (+9.49%)

Portfolio Summary:
Total Cost: $2,502.50
Total Value: $3,388.75
Total P/L: +$886.25 (+35.41%)
```

## Trading Strategies

Stock Prophet offers multiple trading strategies to generate signals.

### Available Strategies

- **RSI Strategy**: Buy when RSI is oversold, sell when overbought
- **Bollinger Bands Strategy**: Buy at lower band, sell at upper band
- **MACD Strategy**: Buy on MACD crossover, sell on crossunder
- **Combined Strategy**: Requires confirmation from multiple indicators
- **Gold Strategy**: Specialized for gold commodities and ETFs with optimized parameters

### Specialized Gold Trading

Stock Prophet includes a dedicated strategy for gold and precious metals trading:

- **Gold Commodity Trading**: Optimized for gold commodities (XAU, GC=F, XAUUSD=X) with more aggressive parameters
- **Gold ETF Trading**: Configured for gold ETFs (GLD, IAU, SGOL, etc.) with more conservative parameters
- **Automatic Detection**: The system automatically identifies whether you're analyzing a gold commodity or ETF

When you analyze gold-related assets, the Gold Strategy is automatically applied with the appropriate parameter set:

```
/ticker GLD
/ticker GC=F
```

The system will:
1. Identify the asset type (commodity or ETF)
2. Apply the appropriate strategy parameters
3. Generate optimized trading signals

### Viewing Strategies

```
/strategies
```

Response:
```
Available Strategies:
1. rsi - Buy on RSI oversold (30), sell on overbought (70)
2. bollinger - Buy at lower band, sell at upper band
3. macd - Buy on MACD line crossing above signal, sell when below
4. combined - Multiple indicator confirmation (default)
5. gold - Specialized strategy for gold and precious metals trading

Current strategy: combined
Use /strategy NAME to change your strategy
```

### Changing Your Strategy

```
/strategy macd
```

Response: "Strategy changed to: MACD Strategy"

## Understanding Analysis Results

### Trend Analysis

The trend can be:
- **Bullish**: Prices are expected to rise
- **Bearish**: Prices are expected to fall
- **Neutral/Sideways**: No clear direction

### Technical Indicators

- **RSI (Relative Strength Index)**: Values above 70 are considered overbought, below 30 oversold
- **EMA (Exponential Moving Average)**: Shows price trend over time
- **Bollinger Bands**: Price volatility and potential reversal points
- **MACD (Moving Average Convergence Divergence)**: Trend direction and momentum
- **Stochastic Oscillator**: Momentum indicator comparing close price to range

### Price Forecast

The forecast shows predicted prices for the next 5 hours and the overall expected price direction.

### Trading Signals

Signals are generated based on your selected strategy:
- **BUY**: Suggests entering a long position
- **SELL**: Suggests exiting or shorting
- **HOLD/NEUTRAL**: No strong signal either way

Each signal includes recommended take profit and stop loss levels.

### Trading Thresholds

The system uses adaptive profit/loss thresholds:
- **Standard** (uptrends): 3% take profit, 4% stop loss
- **Conservative** (downtrends): 1% take profit, 5% stop loss
- **Gold Commodity**: 2.0% take profit, 2.5% stop loss (optimized for higher volatility)
- **Gold ETF**: 1.5% take profit, 2.0% stop loss (optimized for lower volatility)

## Receiving Notifications

Stock Prophet automatically monitors your watchlist and sends alerts for:

1. Significant price movements
2. New trading signals
3. Technical indicator triggers

Notifications are sent every 30 seconds if conditions are met.

## Troubleshooting

### Common Issues

**Problem**: Bot not responding
**Solution**: Try sending /start to restart the conversation

**Problem**: Stock symbol not recognized
**Solution**: Make sure you're using the correct ticker symbol (e.g., AAPL, not Apple)

**Problem**: No trading signals generated
**Solution**: Try a different strategy or check during market hours

## Advanced Usage

### Command Line Testing

For developers or advanced users, you can test the application via command line:

```bash
# Test stock analysis
python test_stock.py AAPL

# Test portfolio management
python test_stock.py portfolio

# Test backtesting functionality
python test_stock.py backtest AAPL combined 2024-01-01 2024-03-15 1d
```

### CI/CD Pipeline

The application includes an automated testing pipeline through GitHub Actions:

1. Each push to the main branch runs automated tests
2. Tests verify stock analysis, portfolio management, and backtesting
3. All trading strategies are validated
4. The CI environment is automatically detected and optimized

To view these tests, go to the "Actions" tab in the GitHub repository.

### Running in CI Environment

The application is designed to run efficiently in Continuous Integration environments:

1. Set the `CI=true` environment variable to enable CI mode
2. In CI mode, the application optimizes resource usage and skips non-essential operations
3. Testing parameters are automatically adjusted for faster execution

To simulate CI mode locally:

```bash
CI=true python test_stock.py
```

### Custom Parameters

Advanced users can modify strategy parameters by editing the strategies.py file:

```python
# Example: Make RSI Strategy more aggressive
params = {
    "oversold": 35,  # Default: 30
    "overbought": 65,  # Default: 70
    "take_profit": 4.0,  # Default: 3.0
    "stop_loss": 3.0,  # Default: 4.0
}
```

## Conclusion

Stock Prophet is a powerful tool for technical analysis and trading signal generation. By combining multiple technical indicators with price forecasting, it provides valuable insights for your trading decisions.

For technical details about how the application works, refer to DOCUMENTATION.md for a comprehensive explanation of each component.