"""
Custom Backtest script for ZYXI, CRDF, CIFR, and ARVN
Focus on achieving 2% profit per trade with 60-day lookback period
"""

import os
import datetime
import pandas as pd
import numpy as np
from backtesting import run_backtest, BacktestResult, safe_float
import strategies
from visualization import save_chart
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Stocks to focus on
TARGET_STOCKS = ['ZYXI', 'CRDF', 'CIFR', 'ARVN']

# Strategy parameters optimized for 2% profit target
OPTIMIZED_PARAMETERS = {
    'rsi': {
        'overbought': 58,       # Lower threshold to identify overbought conditions earlier
        'oversold': 42,         # Higher threshold to identify oversold conditions earlier
        'take_profit': 2.0,     # Target 2% profit
        'stop_loss': 1.5        # Tighter stop loss to limit downside
    },
    'bollinger': {
        'std_dev': 1.8,         # Wider bands to catch more significant moves
        'take_profit': 2.0,     # Target 2% profit
        'stop_loss': 1.5        # Tighter stop loss to limit downside
    },
    'macd': {
        'fast_period': 8,       # Faster signal generation
        'slow_period': 17,      # Better for catching medium-term trends
        'signal_period': 7,     # Quicker signal confirmation
        'take_profit': 2.0,     # Target 2% profit
        'stop_loss': 1.5        # Tighter stop loss to limit downside
    },
    'combined': {
        'rsi_overbought': 58,   # Lower threshold
        'rsi_oversold': 42,     # Higher threshold
        'bb_std_dev': 1.8,      # Wider bands
        'macd_fast': 8,         # Faster MACD
        'macd_slow': 17,        # Medium-term trend
        'macd_signal': 7,       # Quicker signal
        'take_profit': 2.0,     # Target 2% profit
        'stop_loss': 1.5        # Tighter stop loss
    }
}

def apply_custom_strategy_parameters():
    """Apply custom parameters to strategies for better performance"""
    for strategy_name, params in OPTIMIZED_PARAMETERS.items():
        strategy = strategies.get_strategy(strategy_name)
        if strategy:
            strategy.parameters.update(params)
            logger.info(f"Applied custom parameters to {strategy_name} strategy: {params}")

def run_custom_backtests():
    """Run custom backtests for target stocks with optimized parameters"""
    # Calculate date range (past 60 days)
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=60)).strftime('%Y-%m-%d')
    
    # Apply custom parameters to strategies
    apply_custom_strategy_parameters()
    
    logger.info(f"Running 60-day backtests for {TARGET_STOCKS}")
    logger.info(f"Period: {start_date} to {end_date}, Timeframe: 1d")
    logger.info("Targeting 2% profit per trade with optimized parameters")
    logger.info("="*50)
    
    # Store profitable trades for reporting
    profitable_trades = {}
    
    # Run backtests for each stock and strategy
    for ticker in TARGET_STOCKS:
        profitable_trades[ticker] = []
        
        for strategy_name in ["combined", "rsi", "bollinger", "macd"]:
            logger.info(f"Testing {ticker} with {strategy_name} strategy...")
            
            # Run backtest
            result = run_backtest(ticker, strategy_name, start_date, end_date, "1d")
            if result is None:
                logger.error(f"Error backtesting {ticker} with {strategy_name}")
                continue
                
            # Calculate metrics
            result.calculate_metrics()
            
            # Filter profitable trades (>= 2%)
            profitable_trade_count = 0
            for trade in result.trades:
                profit_percent = (safe_float(trade['exit_price']) / safe_float(trade['entry_price']) - 1) * 100
                if profit_percent >= 2.0:
                    profitable_trade_count += 1
                    profitable_trades[ticker].append({
                        'strategy': strategy_name,
                        'entry_date': trade['entry_date'],
                        'exit_date': trade['exit_date'],
                        'entry_price': safe_float(trade['entry_price']),
                        'exit_price': safe_float(trade['exit_price']),
                        'profit_percent': profit_percent,
                        'profit_amount': safe_float(trade['profit_loss'])
                    })
            
            # Print results
            print(f"Backtesting Results for {ticker} - {strategy_name}:")
            print(f"Period: {start_date} to {end_date} (1d)")
            print(f"Total Trades: {len(result.trades)}")
            print(f"Profitable Trades (≥2%): {profitable_trade_count}")
            print(f"Strategy Return: {result.strategy_return:.2f}%")
            if hasattr(result, 'buy_and_hold_return'):
                print(f"Buy & Hold Return: {float(safe_float(result.buy_and_hold_return)):.2f}%")
                print(f"Outperformance: {result.strategy_return - float(safe_float(result.buy_and_hold_return)):.2f}%")
            print(f"Win Rate: {result.win_rate*100:.2f}%")
            print(f"Profit Factor: {result.profit_factor:.2f}")
            
            # Generate chart
            try:
                chart_path = f"backtest_charts/{ticker}_{strategy_name}_{start_date}_{end_date}.png"
                save_chart(result.data, ticker, chart_path)
                print(f"Chart saved to: {chart_path}")
            except Exception as e:
                logger.warning(f"Could not generate chart: {e}")
                
            print("-"*50)
    
    # Print summary of all profitable trades (>= 2%)
    print("\n=== Summary of Profitable Trades (≥2%) ===\n")
    for ticker in TARGET_STOCKS:
        if profitable_trades[ticker]:
            print(f"{ticker} - {len(profitable_trades[ticker])} profitable trades:")
            for trade in profitable_trades[ticker]:
                print(f"  {trade['strategy']}: {trade['entry_date'].strftime('%Y-%m-%d')} → {trade['exit_date'].strftime('%Y-%m-%d')}: ${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} (+{trade['profit_percent']:.2f}%)")
        else:
            print(f"{ticker}: No profitable trades above 2% threshold")
    
    # Return the profitable trades data for further analysis
    return profitable_trades
        
if __name__ == "__main__":
    profitable_trades = run_custom_backtests()