#!/usr/bin/env python3
"""
40-Day Backtesting Script for Stock Prophet
Tests trading strategies against 40 days of historical data to evaluate performance
"""

import logging
import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from backtesting import BacktestResult, run_backtest
import strategies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Safe numeric value function to handle both scalar and Series types
def safe_numeric_value(value):
    """Safely extract numeric value from pandas Series or scalar"""
    if isinstance(value, pd.Series):
        if len(value) > 0:
            return value.iloc[0]
        else:
            return 0.0
    return value if value is not None else 0.0

def run_40day_backtest(tickers=None, strategy_name="combined"):
    """
    Run backtests for the past 40 trading days and analyze performance.
    
    Args:
        tickers: List of stock symbols to test. If None, uses a default list.
        strategy_name: Name of the strategy to test.
        
    Returns:
        DataFrame with backtest results for each ticker.
    """
    if tickers is None:
        # Default ticker list - major tech and finance stocks
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "BAC", "V"]
    
    # Calculate date range for 40 trading days
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")  # Extra days to account for weekends/holidays
    
    results = []
    overall_win_rate = 0
    overall_total_trades = 0
    overall_winning_trades = 0
    
    # Create a directory for backtest charts if it doesn't exist
    os.makedirs("backtest_charts", exist_ok=True)
    
    for ticker in tickers:
        logger.info(f"Running 40-day backtest for {ticker} using {strategy_name} strategy")
        
        try:
            # Run the backtest
            backtest = run_backtest(ticker, strategy_name, start_date, end_date, "1d")
            
            # Calculate metrics
            backtest.calculate_metrics()
            
            # Generate chart
            chart_path = None
            try:
                chart_path = backtest.generate_chart()
                logger.info(f"Generated backtest chart for {ticker}: {chart_path}")
            except Exception as e:
                logger.warning(f"Failed to generate chart for {ticker}: {e}")

            # Safe extraction of metrics
            total_trades = safe_numeric_value(backtest.total_trades)
            winning_trades = safe_numeric_value(backtest.winning_trades)
            total_return = safe_numeric_value(backtest.total_return_pct)
            win_rate = safe_numeric_value(backtest.win_rate)
            profit_factor = safe_numeric_value(backtest.profit_factor)
            max_drawdown = safe_numeric_value(backtest.max_drawdown)
            avg_profit_pct = safe_numeric_value(backtest.avg_profit_pct)
            avg_loss_pct = safe_numeric_value(backtest.avg_loss_pct)
            
            # Update overall statistics if there were trades
            if total_trades > 0:
                overall_total_trades += total_trades
                overall_winning_trades += winning_trades
            
            # Add to results
            results.append({
                'Ticker': ticker,
                'Strategy': strategy_name,
                'Total Trades': total_trades,
                'Winning Trades': winning_trades,
                'Win Rate (%)': win_rate,
                'Total Return (%)': total_return,
                'Profit Factor': profit_factor,
                'Max Drawdown (%)': max_drawdown,
                'Avg Profit (%)': avg_profit_pct,
                'Avg Loss (%)': avg_loss_pct,
                'Chart Path': chart_path
            })
            
            # Output summary for this ticker
            logger.info(f"Backtest for {ticker} completed: {total_trades} trades, {win_rate:.2f}% win rate, {total_return:.2f}% return")
        
        except Exception as e:
            logger.error(f"Error backtesting {ticker}: {e}")
            results.append({
                'Ticker': ticker,
                'Strategy': strategy_name,
                'Total Trades': 0,
                'Winning Trades': 0,
                'Win Rate (%)': 0,
                'Total Return (%)': 0,
                'Profit Factor': 0,
                'Max Drawdown (%)': 0,
                'Avg Profit (%)': 0,
                'Avg Loss (%)': 0,
                'Chart Path': None,
                'Error': str(e)
            })
    
    # Calculate overall win rate
    if overall_total_trades > 0:
        overall_win_rate = (overall_winning_trades / overall_total_trades) * 100
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Output overall summary
    logger.info(f"===== 40-DAY BACKTEST SUMMARY =====")
    logger.info(f"Strategy: {strategy_name}")
    logger.info(f"Tickers tested: {len(tickers)}")
    logger.info(f"Overall win rate: {overall_win_rate:.2f}%")
    logger.info(f"Total trades executed: {overall_total_trades}")
    
    # Calculate average metrics across all tickers
    avg_metrics = results_df.mean(numeric_only=True)
    logger.info(f"Average win rate across tickers: {avg_metrics['Win Rate (%)']:.2f}%")
    logger.info(f"Average return across tickers: {avg_metrics['Total Return (%)']:.2f}%")
    
    # Save results to CSV
    csv_path = f"backtest_results_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved backtest results to {csv_path}")
    
    # Print detailed results for each ticker
    print("\n===== DETAILED BACKTEST RESULTS =====")
    for ticker_result in results:
        ticker = ticker_result['Ticker']
        trades = ticker_result['Total Trades']
        win_rate = ticker_result['Win Rate (%)']
        return_pct = ticker_result['Total Return (%)']
        
        if 'Error' in ticker_result and ticker_result['Error']:
            print(f"{ticker}: ERROR - {ticker_result['Error']}")
        else:
            print(f"{ticker}: {trades} trades, {win_rate:.2f}% win rate, {return_pct:.2f}% return")
    
    # Print overall results again for clarity
    print("\n===== OVERALL RESULTS =====")
    print(f"Strategy: {strategy_name}")
    print(f"Overall win rate: {overall_win_rate:.2f}%")
    print(f"Total trades executed: {overall_total_trades}")
    print(f"Average win rate across tickers: {avg_metrics['Win Rate (%)']:.2f}%")
    print(f"Average return across tickers: {avg_metrics['Total Return (%)']:.2f}%")
    print(f"Results saved to {csv_path}")
    
    return results_df

def main():
    """Main function to run backtests"""
    # Define list of tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
    
    # Define strategies to test
    strategies_to_test = ["combined"]  # You can add more: "rsi", "macd", "bollinger"
    
    # Run backtest for each strategy
    for strategy_name in strategies_to_test:
        run_40day_backtest(tickers, strategy_name)

if __name__ == "__main__":
    main()