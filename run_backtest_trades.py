#!/usr/bin/env python3
"""
Backtesting Script for Stock Prophet with More Aggressive Parameters
Generates actual trades for performance evaluation
"""

import logging
import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from backtesting import run_backtest
import strategies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def execute_trades(ticker, days=40):
    """
    Run a backtest with aggressive parameters to ensure trades are executed
    
    Args:
        ticker: Stock symbol to test
        days: Number of days to backtest
        
    Returns:
        Backtest results with trade data
    """
    # Calculate date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    # Create a custom strategy for backtesting with lower thresholds
    strategy_params = {
        'rsi_oversold': 45,          # Normal: 30, More aggressive: 45
        'rsi_overbought': 55,        # Normal: 70, More aggressive: 55
        'min_indicators': 1,         # Normal: 2, More aggressive: 1
        'take_profit': 1.0,          # Normal: 3.0, More aggressive: 1.0
        'stop_loss': 2.0,            # Normal: 4.0, More aggressive: 2.0
        'band_gap_percent': 0.1,     # Normal: 0.2, More aggressive: 0.1
        'signal_threshold': 0.005,   # Normal: 0.01, More aggressive: 0.005
        'use_prediction': False      # Turn off prediction to generate more signals
    }
    
    # Update combined strategy parameters for this backtest only
    original_params = strategies.get_strategy('combined').parameters.copy()
    strategies.update_strategy_params('combined', strategy_params)
    
    logger.info(f"Running backtest for {ticker} with aggressive parameters")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Strategy parameters: {strategy_params}")
    
    # Run the backtest with modified parameters
    backtest = run_backtest(ticker, 'combined', start_date, end_date, "1d")
    
    # Restore original parameters
    strategies.update_strategy_params('combined', original_params)
    
    # Calculate metrics and generate report
    backtest.calculate_metrics()
    report = backtest.generate_report()
    print(report)
    
    # Try to generate a chart
    try:
        chart_path = backtest.generate_chart()
        if chart_path:
            logger.info(f"Generated chart at {chart_path}")
    except Exception as e:
        logger.warning(f"Could not generate chart: {e}")
    
    # Print trade summary
    if backtest.trades:
        total_trades = len(backtest.trades)
        winning_trades = len([t for t in backtest.trades if t['profit_loss'] > 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        logger.info(f"=== TRADE SUMMARY ===")
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Winning trades: {winning_trades}")
        logger.info(f"Win rate: {win_rate:.2f}%")
        logger.info(f"Total return: {backtest.strategy_return:.2f}%")
        
        print("\nDetailed Trades:")
        for i, trade in enumerate(backtest.trades):
            profit_pct = (trade['exit_price'] / trade['entry_price'] - 1) * 100
            result = "PROFIT" if profit_pct > 0 else "LOSS"
            print(f"Trade {i+1}: {result} {profit_pct:.2f}% - Buy: {trade['entry_date'].strftime('%Y-%m-%d')} at ${trade['entry_price']:.2f}, Sell: {trade['exit_date'].strftime('%Y-%m-%d')} at ${trade['exit_price']:.2f}")
    else:
        logger.warning("No trades were executed even with aggressive parameters")
    
    return backtest

def main():
    """Main function to execute backtest trades"""
    # Test multiple tickers to find one that generates trades
    tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "META"]
    
    for ticker in tickers:
        print(f"\n=== EXECUTING TRADES FOR {ticker} ===\n")
        backtest = execute_trades(ticker)
        
        # If we found a ticker with trades, stop the loop
        if backtest.trades:
            logger.info(f"Successfully executed trades for {ticker}")
            break
    
    # If no trades were found for any ticker, try with even more aggressive parameters
    if not backtest.trades:
        logger.info("No trades found with initial parameters, trying with even more aggressive parameters")
        
        # Try with very aggressive parameters on a high-volatility stock
        super_aggressive_params = {
            'rsi_oversold': 49,
            'rsi_overbought': 51,
            'min_indicators': 1,
            'take_profit': 0.5,
            'stop_loss': 1.0,
            'band_gap_percent': 0.05,
            'signal_threshold': 0.001,
            'use_prediction': False
        }
        
        original_params = strategies.get_strategy('combined').parameters.copy()
        strategies.update_strategy_params('combined', super_aggressive_params)
        
        volatile_tickers = ["TSLA", "NVDA", "AMD"]
        for ticker in volatile_tickers:
            print(f"\n=== SUPER AGGRESSIVE TEST FOR {ticker} ===\n")
            backtest = execute_trades(ticker, days=60)  # Try longer timeframe
            if backtest.trades:
                break
        
        # Restore original parameters
        strategies.update_strategy_params('combined', original_params)

if __name__ == "__main__":
    main()