#!/bin/bash
# Setup script for GitHub Actions CI/CD pipeline for Stock Prophet

echo "Starting CI/CD environment setup for Stock Prophet..."

# Create necessary directories (if they don't exist)
echo "Creating required directories..."
mkdir -p backtest_charts
mkdir -p charts
mkdir -p data

# Ensure required files exist
echo "Setting up required data files..."
echo "{}" > user_watchlists.json 
echo "{}" > user_portfolios.json

# Make necessary files executable
echo "Setting file permissions..."
chmod +x test_stock.py

# Add NVDA to test user's watchlist for watchlist testing
echo "Setting up test watchlist..."
echo '{"test_user": ["NVDA"]}' > user_watchlists.json

# Setup additional CI environment optimizations
echo "Setting up CI environment optimizations..."
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg  # Use non-interactive Matplotlib backend for CI

# Verify setup
echo "Verifying setup..."
echo "- Directories:"
ls -la | grep -E "backtest_charts|charts|data"
echo "- Files:"
ls -la | grep -E "user_watchlists.json|user_portfolios.json|test_stock.py"

echo "GitHub CI/CD environment setup complete."