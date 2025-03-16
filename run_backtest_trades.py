#!/usr/bin/env python3
"""
Run backtest trades script for Stock Prophet
Executes backtests with real trading simulations to evaluate strategies
"""

import os
import time
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from backtesting import run_backtest

# Define the stocks to test
STOCKS = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'PYPL', 'AMD']

# Define the strategies to test
STRATEGIES = ['rsi', 'macd', 'bollinger', 'combined']

# Define the time period for the backtest
END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

# Define the timeframe
TIMEFRAME = '1d'  # Daily data

def run_all_backtests():
    """Run backtests for all stocks and strategies"""
    print(f"=== Running Trading Backtests ===")
    print(f"Period: {START_DATE} to {END_DATE}, Timeframe: {TIMEFRAME}")
    print(f"Stocks: {', '.join(STOCKS)}")
    print(f"Strategies: {', '.join(STRATEGIES)}")
    print("=" * 50)

    # Create a results directory if it doesn't exist
    if not os.path.exists('backtest_results'):
        os.makedirs('backtest_results')

    # Create a directory for charts if it doesn't exist
    if not os.path.exists('backtest_charts'):
        os.makedirs('backtest_charts')

    # Prepare a DataFrame to store all results
    all_results = []

    # Loop through each stock and strategy
    for ticker in STOCKS:
        for strategy in STRATEGIES:
            try:
                print(f"Testing {ticker} with {strategy} strategy...")
                
                # Run the backtest
                result = run_backtest(ticker, strategy, START_DATE, END_DATE, TIMEFRAME)
                
                # Generate report
                report = result.generate_report()
                print(report)
                
                # Generate chart and save it
                try:
                    chart_path = result.generate_chart()
                    if chart_path:
                        print(f"Chart saved to {chart_path}")
                except Exception as chart_error:
                    print(f"Error generating chart: {chart_error}")
                
                # Add the results to our list
                all_results.append({
                    'Ticker': ticker,
                    'Strategy': strategy,
                    'Initial Balance': result.initial_balance,
                    'Final Balance': result.final_balance,
                    'Total Return': result.total_return,
                    'Buy & Hold Return': result.buy_hold_return,
                    'Outperformance': result.outperformance,
                    'Total Trades': result.total_trades,
                    'Win Rate': result.win_rate,
                    'Profit Factor': result.profit_factor,
                    'Max Drawdown': result.max_drawdown
                })
                
            except Exception as e:
                print(f"Error backtesting {ticker} with {strategy}: {e}")
                
            print("-" * 50)
    
    # Convert results to DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Save to CSV
        timestamp = int(time.time())
        csv_path = f"backtest_results_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"All backtest results saved to {csv_path}")
        
        # Generate a summary plot
        plt.figure(figsize=(10, 8))
        
        # Group by strategy and ticker
        strategy_returns = results_df.pivot(index='Ticker', columns='Strategy', values='Total Return')
        strategy_returns.plot(kind='bar', alpha=0.7)
        
        plt.title(f'Strategy Performance by Ticker ({START_DATE} to {END_DATE})')
        plt.ylabel('Total Return (%)')
        plt.xlabel('Ticker')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Strategy')
        plt.tight_layout()
        
        # Save the plot
        plot_path = f"backtest_charts/strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path)
        print(f"Strategy comparison chart saved to {plot_path}")
        
        # Display best performing strategy for each ticker
        print("\nBest Strategy by Ticker:")
        for ticker in STOCKS:
            ticker_data = results_df[results_df['Ticker'] == ticker]
            if not ticker_data.empty:
                best_strategy = ticker_data.loc[ticker_data['Total Return'].idxmax()]
                print(f"{ticker}: {best_strategy['Strategy']} ({best_strategy['Total Return']:.2f}%)")
    else:
        print("No backtest results were generated.")

if __name__ == "__main__":
    run_all_backtests()