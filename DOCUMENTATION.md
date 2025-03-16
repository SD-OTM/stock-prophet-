# Stock Prophet: Code Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Core Files](#core-files)
3. [Main Application (main.py)](#main-application-mainpy)
4. [Trading Strategies (strategies.py)](#trading-strategies-strategiespy)
5. [Visualization (visualization.py)](#visualization-visualizationpy)
6. [Testing (test_stock.py)](#testing-test_stockpy)
7. [CI/CD Pipeline (GitHub Actions)](#cicd-pipeline-github-actions)
8. [GitHub Repository Setup](#github-repository-setup)

## Project Overview

Stock Prophet is a comprehensive stock analysis and trading signal application that leverages technical indicators, machine learning forecasting, and a Telegram bot interface to provide users with actionable insights on stock trends and potential trading opportunities.

## Core Files

The application consists of the following core files:

- **main.py**: The primary application file containing stock analysis logic, Telegram bot implementation, and user data management
- **strategies.py**: Collection of trading strategies with configurable parameters
- **visualization.py**: Chart generation for technical indicators and price forecasts
- **test_stock.py**: Testing utilities for both stock analysis and portfolio management
- **user_portfolios.json**: Persistent storage for user portfolio data
- **github_setup.sh**: Helper script for GitHub repository initialization

## Main Application (main.py)

The main application file contains several key components:

### Data Management Functions

```python
def load_watchlists():
    """Load user watchlists from persistent storage"""
    
def save_watchlists():
    """Save user watchlists to persistent storage"""
    
def load_portfolios():
    """Load user portfolios from persistent storage"""
    
def save_portfolios():
    """Save user portfolios to persistent storage"""
```

These functions handle the persistence of user data between application restarts, ensuring that watchlists and portfolios are maintained.

### Stock Analysis Functions

```python
def get_stock_data(ticker, period="1d", interval="1h"):
    """Fetch stock data from Yahoo Finance"""
    
def calculate_indicators(data):
    """Calculate technical indicators for the provided stock data"""
    
def determine_trend(data):
    """Determine the trend direction (bullish, bearish, or neutral)"""
    
def forecast(data, steps=5):
    """Forecast future stock prices using ARIMA model"""
```

These functions form the core analysis engine, fetching stock data and applying technical analysis to it.

### Trading Strategy Management

```python
def load_user_strategies():
    """Load user strategy preferences"""
    
def save_user_strategies():
    """Save user strategy preferences"""
    
def generate_signals(data, user_id, ticker):
    """Generate trading signals based on selected strategy"""
    
def set_user_strategy(user_id, strategy_name):
    """Set a user's preferred trading strategy"""
    
def get_user_strategy(user_id):
    """Get a user's current trading strategy"""
```

These functions manage user preferences for trading strategies and generate signals based on the selected strategy.

### Telegram Bot Command Handlers

```python
def start(update: Update, context: CallbackContext):
    """Handle the /start command"""
    
def analyze_ticker(ticker, user_id="test_user"):
    """Analyze a stock ticker and generate a report"""
    
def handle_ticker(update: Update, context: CallbackContext):
    """Handle the /ticker command"""
    
def add_to_watchlist(update: Update, context: CallbackContext):
    """Handle the /add command"""
    
def remove_from_watchlist(update: Update, context: CallbackContext):
    """Handle the /remove command"""
    
def show_watchlist(update: Update, context: CallbackContext):
    """Handle the /watchlist command"""
```

These functions handle the various Telegram bot commands, processing user input and providing responses.

### Portfolio Management

```python
def show_portfolio(update: Update, context: CallbackContext):
    """Handle the /portfolio command"""
    
def add_to_portfolio(update: Update, context: CallbackContext):
    """Handle the /buy command"""
    
def remove_from_portfolio(update: Update, context: CallbackContext):
    """Handle the /sell command"""
```

These functions manage user portfolios, allowing users to track their investments.

### Bot Initialization and Main Functions

```python
def run_telegram_bot():
    """Initialize and run the Telegram bot"""
    
def test_mode():
    """Run in test mode for CLI testing"""
    
def main():
    """Main application entry point"""
```

These functions handle the application initialization and execution flow.

## Trading Strategies (strategies.py)

The trading strategies module defines various approaches to generating buy/sell signals based on technical indicators.

### Base Strategy Class

```python
class Strategy:
    """Base class for all trading strategies"""
    def __init__(self, name, description, parameters=None):
        """Initialize with name, description, and optional parameters"""
        
    def generate_signals(self, data, user_id, ticker, user_data=None):
        """Generate buy/sell signals - to be implemented by subclasses"""
        
    def get_description(self):
        """Return a description of the strategy with current parameters"""
```

This base class defines the interface for all trading strategies.

### Specific Strategy Implementations

#### RSI Strategy

```python
class RSIStrategy(Strategy):
    """RSI-based trading strategy"""
    def __init__(self, parameters=None):
        """Initialize with default parameters: 
        - take_profit: 3.0% (default)
        - stop_loss: 4.0% (default)
        - oversold: 30
        - overbought: 70
        - use_prediction: True
        """
        
    def generate_signals(self, data, user_id, ticker, user_data=None):
        """Generate signals based on RSI values"""
```

The RSI strategy generates signals based on Relative Strength Index (RSI) values, with configurable oversold and overbought thresholds.

#### Bollinger Bands Strategy

```python
class BollingerBandsStrategy(Strategy):
    """Bollinger Bands-based trading strategy"""
    def __init__(self, parameters=None):
        """Initialize with default parameters:
        - take_profit: 3.0% (default)
        - stop_loss: 4.0% (default)
        - use_prediction: True
        """
        
    def generate_signals(self, data, user_id, ticker, user_data=None):
        """Generate signals based on Bollinger Bands"""
```

The Bollinger Bands strategy generates signals based on price movements relative to the Bollinger Bands.

#### MACD Strategy

```python
class MACDStrategy(Strategy):
    """MACD-based trading strategy"""
    def __init__(self, parameters=None):
        """Initialize with default parameters:
        - take_profit: 3.0% (default)
        - stop_loss: 4.0% (default)
        - use_prediction: True
        """
        
    def generate_signals(self, data, user_id, ticker, user_data=None):
        """Generate signals based on MACD crossovers"""
```

The MACD strategy generates signals based on Moving Average Convergence Divergence (MACD) crossovers.

#### Combined Strategy

```python
class CombinedStrategy(Strategy):
    """Combined strategy using multiple indicators"""
    def __init__(self, parameters=None):
        """Initialize with default parameters:
        - take_profit: 3.0% (default)
        - stop_loss: 4.0% (default)
        - rsi_oversold: 30
        - rsi_overbought: 70
        - min_indicators: 2
        - use_prediction: True
        """
        
    def generate_signals(self, data, user_id, ticker, user_data=None):
        """Generate signals based on multiple indicators"""
```

The Combined Strategy requires multiple indicators to confirm a signal, providing stronger trading signals. It also adjusts profit/loss thresholds based on price forecasts:
- For uptrends: Standard thresholds (3% profit, 4% loss)
- For downtrends: Conservative thresholds (1% profit, 5% loss)

#### Gold Strategy

```python
class GoldStrategy(Strategy):
    """Specialized strategy for gold and precious metals"""
    def __init__(self, parameters=None):
        """Initialize with two parameter sets:
        
        Commodity Gold Parameters:
        - rsi_oversold: 35 (more aggressive than ETFs)
        - rsi_overbought: 65
        - bb_std_dev: 2.5 (wider bands for commodities)
        - take_profit: 2.0% (higher targets for commodities)
        - stop_loss: 2.5%
        - min_indicators: 2
        
        ETF Gold Parameters:
        - rsi_oversold: 40 (more conservative)
        - rsi_overbought: 60
        - bb_std_dev: 2.0
        - take_profit: 1.5% (conservative targets for ETFs)
        - stop_loss: 2.0%
        - min_indicators: 2
        """
        
    def generate_signals(self, data, user_id, ticker, user_data=None):
        """Generate buy/sell signals for gold assets with asset-specific parameters"""
```

The Gold Strategy is specialized for trading gold and precious metals with different parameter sets for commodities vs ETFs:
- Automatically detects asset type (gold_commodity, gold_etf, gold_miner)
- Uses more aggressive parameters for commodities due to higher volatility
- Uses more conservative parameters for ETFs which tend to be less volatile
- Provides clear differentiation in trading signals between commodities and ETFs

### Strategy Utility Functions

```python
def get_strategy(strategy_name):
    """Get a strategy by name"""
    
def update_strategy_params(strategy_name, parameters):
    """Update the parameters of a strategy"""
    
def get_available_strategies_info():
    """Get information about all available strategies"""
```

These utility functions provide access to the strategies and allow for parameter updates.

## Visualization (visualization.py)

The visualization module generates charts for technical analysis.

```python
def generate_chart(data, ticker, indicators=None, show_forecast=False, forecast_values=None):
    """Generate a technical analysis chart for a stock"""
    
def save_chart(data, ticker, filename=None, path='charts', **kwargs):
    """Generate and save a chart to disk"""
```

These functions create charts showing price data, technical indicators, and forecasted prices.

## Testing (test_stock.py)

The testing module provides command-line utilities for testing the application.

```python
def test_analyze():
    """Test the stock analysis functionality with a specific ticker"""
    
def test_portfolio():
    """Test the portfolio management functionality"""
    
def main():
    """Main test function"""
```

These functions allow for easy testing of the application's core functionality without needing the Telegram bot interface.

## CI/CD Pipeline (GitHub Actions)

The CI/CD pipeline is defined in `.github/workflows/stock-prophet-tests.yml` and has the following components:

### Workflow Triggers

```yaml
on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]
  workflow_dispatch:  # Allows manual triggering
  schedule:
    - cron: '0 0 * * 1-5'  # Run at midnight on weekdays
```

The workflow is triggered by:
- Pushes to main/master branches
- Pull requests to main/master branches
- Manual triggers via the GitHub Actions UI
- Scheduled runs at midnight on weekdays

### Environment Setup

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    
    # Add environment secrets and CI flag
    env:
      ALPHA_VANTAGE_API_KEY: ${{ secrets.ALPHA_VANTAGE_API_KEY }}
      TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
      CI: 'true'  # Mark as CI environment
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
```

This sets up an Ubuntu environment with Python 3.11 installed and configures environment variables including the CI flag.

### Dependency Installation

```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    if [ -f requirements-ci.txt ]; then 
      pip install -r requirements-ci.txt
    else 
      pip install -e .
    fi
```

This installs the necessary dependencies using either requirements-ci.txt or setup.py.

### CI Environment Compatibility

The application automatically detects when it's running in a CI environment through the `IS_CI_ENV` variable:

```python
# Example from main.py
import os

# Standardized CI environment detection
IS_CI_ENV = os.environ.get('CI', 'false').lower() == 'true'

# Conditional behavior based on environment
if IS_CI_ENV:
    # CI-optimized code path
    logging.info("Running in CI environment, optimizing operations")
else:
    # Normal operation code path
    pass
```

This pattern is applied consistently across all modules to ensure proper behavior in different environments.

### Test Execution

```yaml
- name: Run all tests with CI optimizations
  run: |
    echo "Running Stock Prophet test suite in CI mode"
    python test_stock.py
  
# If the main test passes, we're done - the test_stock.py file now handles all tests in CI mode
# But for backward compatibility and extra confidence, we'll run individual tests too
    
- name: Run additional tests for specific stocks
  if: success() || failure()  # Run even if previous step failed
  run: |
    echo "Testing Apple stock analysis"
    python test_stock.py AAPL
    echo "Testing NVIDIA stock analysis"
    python test_stock.py NVDA
  continue-on-error: true
    
- name: Test strategy functionality
  if: success() || failure()  # Run even if previous step failed
  run: |
    # This will test all available strategies
    for strategy in rsi bollinger macd combined; do
      echo "Testing $strategy strategy"
      python -c "import main; main.set_user_strategy('test_user', '$strategy'); print(f'Successfully set strategy to {main.get_user_strategy(\"test_user\")}');"
    done
  continue-on-error: true
    
- name: Run backtest test with shorter timeframe for CI
  if: success() || failure()  # Run even if previous step failed
  run: |
    echo "Testing backtesting functionality with AAPL (CI optimized)"
    python test_stock.py backtest AAPL combined 2024-02-14 2024-03-15 1d
  continue-on-error: true
```

This runs a series of tests with CI optimizations:
1. Main test suite with CI-specific optimizations
2. Individual stock analysis tests
3. Strategy functionality tests for all available strategies
4. Backtesting with a shorter timeframe for faster CI execution

## GitHub Repository Setup

The `github_setup.sh` script automates the GitHub repository setup process:

```bash
# Create a new GitHub repository
gh repo create "$repo_name" --$visibility --source=. --remote=origin

# Set up secrets for GitHub Actions
gh secret set GITHUB_TOKEN -b "$token"
gh secret set TELEGRAM_BOT_TOKEN -b "$token"

# Push code to GitHub
git add .
git commit -m "Initial commit for Stock Prophet"
git branch -M main
git push -u origin main
```

This script:
1. Creates a new GitHub repository with the specified name and visibility
2. Sets up necessary secrets for GitHub Actions
3. Initializes the repository and pushes the code

## Viewing GitHub Actions Logs

To view the logs for your GitHub Actions CI/CD pipeline:

1. Go to your GitHub repository in a web browser
2. Click on the "Actions" tab at the top of the page
3. You'll see a list of all workflow runs, with the most recent at the top
4. Click on any workflow run to see its details
5. In the workflow details page, you'll see each job in the workflow
6. Click on a job to expand it and see the logs for each step
7. You can download the complete logs by clicking the three dots (⋮) and selecting "Download log"

The logs will show:
- The output of each test run
- Any errors or warnings
- The time taken for each step
- The overall success or failure of the workflow

This provides valuable insights into your code's health and helps identify any issues that need to be addressed.