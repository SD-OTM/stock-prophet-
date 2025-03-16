#!/usr/bin/env python3
"""
Extreme parameter backtest script for Stock Prophet
Tests very aggressive trading parameters on historical data
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from backtesting import run_backtest, BacktestResult
import json
from strategies import RSIStrategy, BollingerBandsStrategy, MACDStrategy, CombinedStrategy

# Define the stocks to test - focus on more volatile ones for extreme testing
STOCKS = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'PYPL']

# Define the time period for the backtest - use a longer period for more robust results
END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

# Define the timeframe
TIMEFRAME = '1d'  # Daily data

class ExtremeParameters:
    """Class to generate and test extreme parameters for trading strategies"""
    
    @staticmethod
    def get_rsi_extreme_params():
        """Generate extreme RSI parameters"""
        return [
            # Very tight RSI thresholds (for frequent trading)
            {"overbought": 60, "oversold": 40, "take_profit": 0.5, "stop_loss": 3.0},
            # Super-aggressive RSI thresholds
            {"overbought": 55, "oversold": 45, "take_profit": 0.3, "stop_loss": 1.0},
            # Ultra-narrow band RSI
            {"overbought": 52, "oversold": 48, "take_profit": 0.2, "stop_loss": 0.5}
        ]
    
    @staticmethod
    def get_bollinger_extreme_params():
        """Generate extreme Bollinger Bands parameters"""
        return [
            # Narrow bands for very frequent trading
            {"std_dev": 1.5, "take_profit": 0.5, "stop_loss": 2.0},
            # Ultra-narrow bands
            {"std_dev": 1.0, "take_profit": 0.3, "stop_loss": 1.0},
            # Extreme sensitivity
            {"std_dev": 0.8, "take_profit": 0.2, "stop_loss": 0.5}
        ]
    
    @staticmethod
    def get_macd_extreme_params():
        """Generate extreme MACD parameters"""
        return [
            # Very sensitive MACD settings
            {"fast_period": 6, "slow_period": 12, "signal_period": 3, "take_profit": 0.5, "stop_loss": 2.0},
            # Ultra-fast MACD
            {"fast_period": 3, "slow_period": 8, "signal_period": 2, "take_profit": 0.3, "stop_loss": 1.0},
            # Extreme short-term MACD
            {"fast_period": 2, "slow_period": 5, "signal_period": 2, "take_profit": 0.2, "stop_loss": 0.5}
        ]
    
    @staticmethod
    def get_combined_extreme_params():
        """Generate extreme combined strategy parameters"""
        return [
            # Very aggressive combo
            {
                "rsi_overbought": 60, "rsi_oversold": 40,
                "bb_std_dev": 1.5, 
                "macd_fast": 6, "macd_slow": 12, "macd_signal": 3,
                "take_profit": 0.5, "stop_loss": 2.0
            },
            # Ultra-aggressive combo
            {
                "rsi_overbought": 55, "rsi_oversold": 45,
                "bb_std_dev": 1.0,
                "macd_fast": 3, "macd_slow": 8, "macd_signal": 2,
                "take_profit": 0.3, "stop_loss": 1.0
            },
            # Extreme sensitivity combo
            {
                "rsi_overbought": 52, "rsi_oversold": 48,
                "bb_std_dev": 0.8,
                "macd_fast": 2, "macd_slow": 5, "macd_signal": 2,
                "take_profit": 0.2, "stop_loss": 0.5
            }
        ]

def run_extreme_backtest(ticker, strategy_name, params_set):
    """
    Run a backtest with extreme parameters
    
    Args:
        ticker: Stock symbol
        strategy_name: Strategy to test
        params_set: Dictionary of extreme parameters
        
    Returns:
        BacktestResult object
    """
    print(f"Testing {ticker} with {strategy_name} strategy using extreme parameters:")
    print(json.dumps(params_set, indent=2))
    
    # Initialize the appropriate strategy with extreme parameters
    if strategy_name == "rsi":
        strategy = RSIStrategy(params_set)
    elif strategy_name == "bollinger":
        strategy = BollingerBandsStrategy(params_set)
    elif strategy_name == "macd":
        strategy = MACDStrategy(params_set)
    elif strategy_name == "combined":
        strategy = CombinedStrategy(params_set)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Run the backtest with the strategy object
    result = run_backtest(ticker, strategy_name, START_DATE, END_DATE, TIMEFRAME)
    
    # Return the result
    return result

def run_all_extreme_backtests():
    """Run backtests with extreme parameters for all stocks"""
    print(f"=== Running Extreme Parameter Backtests ===")
    print(f"Period: {START_DATE} to {END_DATE}, Timeframe: {TIMEFRAME}")
    print(f"Stocks: {', '.join(STOCKS)}")
    print("=" * 50)
    
    # Create a results directory if it doesn't exist
    if not os.path.exists('backtest_results'):
        os.makedirs('backtest_results')
    
    # Prepare a DataFrame to store all results
    all_results = []
    
    # Get extreme parameters for each strategy
    rsi_params = ExtremeParameters.get_rsi_extreme_params()
    bollinger_params = ExtremeParameters.get_bollinger_extreme_params()
    macd_params = ExtremeParameters.get_macd_extreme_params()
    combined_params = ExtremeParameters.get_combined_extreme_params()
    
    # Loop through each stock
    for ticker in STOCKS:
        # Test with RSI extreme parameters
        for i, params in enumerate(rsi_params):
            try:
                result = run_extreme_backtest(ticker, "rsi", params)
                
                # Record results
                params_str = f"RSI {params['overbought']}/{params['oversold']}, TP:{params['take_profit']}%, SL:{params['stop_loss']}%"
                all_results.append({
                    'Ticker': ticker,
                    'Strategy': "rsi_extreme_" + str(i),
                    'Parameters': params_str,
                    'Initial Balance': result.initial_balance,
                    'Final Balance': result.final_balance,
                    'Total Return': result.total_return,
                    'Total Trades': result.total_trades,
                    'Win Rate': result.win_rate,
                    'Profit Factor': result.profit_factor,
                    'Max Drawdown': result.max_drawdown
                })
            except Exception as e:
                print(f"Error testing {ticker} with RSI extreme parameters: {e}")
        
        # Test with Bollinger Bands extreme parameters
        for i, params in enumerate(bollinger_params):
            try:
                result = run_extreme_backtest(ticker, "bollinger", params)
                
                # Record results
                params_str = f"BB STD:{params['std_dev']}, TP:{params['take_profit']}%, SL:{params['stop_loss']}%"
                all_results.append({
                    'Ticker': ticker,
                    'Strategy': "bollinger_extreme_" + str(i),
                    'Parameters': params_str,
                    'Initial Balance': result.initial_balance,
                    'Final Balance': result.final_balance,
                    'Total Return': result.total_return,
                    'Total Trades': result.total_trades,
                    'Win Rate': result.win_rate,
                    'Profit Factor': result.profit_factor,
                    'Max Drawdown': result.max_drawdown
                })
            except Exception as e:
                print(f"Error testing {ticker} with Bollinger extreme parameters: {e}")
        
        # Test with MACD extreme parameters
        for i, params in enumerate(macd_params):
            try:
                result = run_extreme_backtest(ticker, "macd", params)
                
                # Record results
                params_str = f"MACD {params['fast_period']}/{params['slow_period']}/{params['signal_period']}, TP:{params['take_profit']}%, SL:{params['stop_loss']}%"
                all_results.append({
                    'Ticker': ticker,
                    'Strategy': "macd_extreme_" + str(i),
                    'Parameters': params_str,
                    'Initial Balance': result.initial_balance,
                    'Final Balance': result.final_balance,
                    'Total Return': result.total_return,
                    'Total Trades': result.total_trades,
                    'Win Rate': result.win_rate,
                    'Profit Factor': result.profit_factor,
                    'Max Drawdown': result.max_drawdown
                })
            except Exception as e:
                print(f"Error testing {ticker} with MACD extreme parameters: {e}")
        
        # Test with Combined extreme parameters
        for i, params in enumerate(combined_params):
            try:
                result = run_extreme_backtest(ticker, "combined", params)
                
                # Record results
                params_str = f"Combined RSI:{params['rsi_overbought']}/{params['rsi_oversold']}, BB:{params['bb_std_dev']}, MACD:{params['macd_fast']}/{params['macd_slow']}, TP:{params['take_profit']}%, SL:{params['stop_loss']}%"
                all_results.append({
                    'Ticker': ticker,
                    'Strategy': "combined_extreme_" + str(i),
                    'Parameters': params_str,
                    'Initial Balance': result.initial_balance,
                    'Final Balance': result.final_balance,
                    'Total Return': result.total_return,
                    'Total Trades': result.total_trades,
                    'Win Rate': result.win_rate,
                    'Profit Factor': result.profit_factor,
                    'Max Drawdown': result.max_drawdown
                })
            except Exception as e:
                print(f"Error testing {ticker} with Combined extreme parameters: {e}")
    
    # Convert results to DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Sort by total return in descending order
        results_df = results_df.sort_values(by='Total Return', ascending=False)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for ticker in STOCKS:
            ticker_results = results_df[results_df['Ticker'] == ticker]
            if not ticker_results.empty:
                csv_path = f"backtest_results/{ticker}_extreme_backtest_{timestamp}.csv"
                ticker_results.to_csv(csv_path, index=False)
                print(f"Results for {ticker} saved to {csv_path}")
        
        # Save combined results
        csv_path = f"backtest_results_combined_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"All backtest results saved to {csv_path}")
        
        # Print top strategies for each stock
        print("\n=== Top Performing Extreme Strategies ===")
        for ticker in STOCKS:
            ticker_results = results_df[results_df['Ticker'] == ticker]
            if not ticker_results.empty:
                best_strategy = ticker_results.iloc[0]
                print(f"{ticker}: {best_strategy['Strategy']} - {best_strategy['Parameters']}")
                print(f"  Return: {best_strategy['Total Return']:.2f}%, Trades: {best_strategy['Total Trades']}, Win Rate: {best_strategy['Win Rate']:.2f}%")
    else:
        print("No backtest results were generated.")

if __name__ == "__main__":
    run_all_extreme_backtests()