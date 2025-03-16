# GitHub Actions Guide for Stock Prophet

This guide explains how the CI/CD pipeline works for the Stock Prophet application using GitHub Actions.

## Overview

The Stock Prophet CI/CD pipeline automatically tests the application whenever code is pushed to the main branch, a pull request is created, or on a scheduled basis (every weekday at midnight).

## What's Being Tested

The pipeline tests the following aspects of the application:

1. **Stock Analysis** - Tests the core analysis functionality on multiple stocks (AAPL, NVDA)
2. **Portfolio Management** - Tests adding, viewing, and removing stocks from a portfolio
3. **Strategy Functions** - Tests all available trading strategies (RSI, Bollinger Bands, MACD, Combined)
4. **Backtesting** - Tests the backtesting functionality on historical data

## CI Optimizations

The pipeline includes several optimizations for running in a CI environment:

1. **API Call Skipping** - In CI mode, the application skips actual API calls to Alpha Vantage to avoid hitting rate limits
2. **Chart Generation** - Chart generation is simplified or skipped in CI mode to improve performance
3. **Error Handling** - Tests continue even if individual steps fail, with failures logged
4. **Date Range Adjustments** - Shorter date ranges are used for backtesting in CI mode
5. **Synthetic Test Data** - The watchlist is pre-populated with test data to ensure notification tests work

## How CI Detection Works

The application detects when it's running in a CI environment through:

1. The `CI=true` environment variable set in the GitHub Actions workflow
2. Each module checks for this variable and adjusts its behavior accordingly

## Setup Process

When the CI pipeline runs, these steps occur:

1. Python 3.11 is set up and dependencies are installed from `requirements-ci.txt`
2. The `github_setup.sh` script creates necessary directories and test data
3. All tests run in sequence, with failures logged but not stopping the pipeline
4. Test results are reported in the workflow logs

## Environment Variables

The following environment variables are used in the CI pipeline:

- `CI=true` - Marks the environment as CI
- `ALPHA_VANTAGE_API_KEY` - API key for Alpha Vantage (stored in GitHub Secrets)
- `TELEGRAM_BOT_TOKEN` - Telegram bot token (stored in GitHub Secrets)

## Adding New Tests

When adding new functionality to the application:

1. Ensure it has a CI-aware mode that doesn't rely on external APIs or slow operations
2. Add appropriate tests to `test_stock.py`
3. Update the GitHub Actions workflow if needed (`.github/workflows/stock-prophet-tests.yml`)

## Running Tests Locally in CI Mode

You can simulate the CI environment locally by setting the CI environment variable:

```bash
CI=true bash github_setup.sh
CI=true python test_stock.py
```

This is useful for debugging CI-related issues before pushing to GitHub.