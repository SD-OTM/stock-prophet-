"""
40-Day Backtest Analysis for Stock Prophet

This script runs a 40-day backtest on multiple stocks to evaluate trading strategy performance
and suggests parameter optimizations.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from backtesting import run_backtest
import strategies

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Stocks to test (major market stocks from different sectors)
TEST_STOCKS = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'JPM']

# Strategies to test
TEST_STRATEGIES = ['rsi', 'macd', 'bollinger', 'combined']

# Calculate the date range (40 days back from today)
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=40)).strftime('%Y-%m-%d')

def run_comprehensive_backtest():
    """Run comprehensive backtests on multiple stocks and strategies"""
    
    print(f"=== Running 40-Day Comprehensive Backtest Analysis ===")
    print(f"Period: {start_date} to {end_date}")
    print(f"Strategies: {', '.join(TEST_STRATEGIES)}")
    print(f"Stocks: {', '.join(TEST_STOCKS)}")
    print()
    
    # Store results for comparison
    results = []
    
    # Create directory for backtest charts if it doesn't exist
    os.makedirs('backtest_charts', exist_ok=True)
    
    # Test each stock with each strategy
    for ticker in TEST_STOCKS:
        for strategy_name in TEST_STRATEGIES:
            print(f"\nBacktesting {strategy_name.upper()} strategy on {ticker}...")
            
            # Run the backtest
            result = run_backtest(ticker, strategy_name, start_date, end_date, timeframe='1d')
            
            if result:
                # Store the results
                results.append({
                    'ticker': ticker,
                    'strategy': strategy_name,
                    'return': result.strategy_return,
                    'buy_hold_return': result.buy_and_hold_return if hasattr(result, 'buy_and_hold_return') and not isinstance(result.buy_and_hold_return, pd.Series) else 0,
                    'trades': len(result.trades),
                    'win_rate': result.win_rate if hasattr(result, 'win_rate') else 0,
                    'max_drawdown': result.max_drawdown
                })
                
                # Print results
                print(f"Strategy Return: {result.strategy_return:.2f}%")
                
                # Safe handling of buy_and_hold_return
                if hasattr(result, 'buy_and_hold_return'):
                    if isinstance(result.buy_and_hold_return, pd.Series):
                        print(f"Buy & Hold Return: Not calculated (data issue)")
                        print(f"Outperformance: N/A")
                    else:
                        print(f"Buy & Hold Return: {result.buy_and_hold_return:.2f}%")
                        print(f"Outperformance: {result.strategy_return - result.buy_and_hold_return:.2f}%")
                else:
                    print(f"Buy & Hold Return: Not calculated")
                    print(f"Outperformance: N/A")
                
                print(f"Total Trades: {len(result.trades)}")
                if hasattr(result, 'win_rate'):
                    print(f"Win Rate: {result.win_rate:.2f}%")
                print(f"Max Drawdown: {result.max_drawdown:.2f}%")
            
    if results:
        # Convert to DataFrame for analysis
        df_results = pd.DataFrame(results)
        
        # Analyze results by strategy
        print("\n=== Performance by Strategy ===")
        strategy_summary = df_results.groupby('strategy').agg({
            'return': 'mean',
            'buy_hold_return': 'mean',
            'trades': 'mean',
            'win_rate': 'mean',
            'max_drawdown': 'mean'
        })
        print(strategy_summary)
        
        # Find the best strategy overall
        best_strategy = strategy_summary['return'].idxmax()
        print(f"\nBest Overall Strategy: {best_strategy.upper()} with average return of {strategy_summary.loc[best_strategy, 'return']:.2f}%")
        
        # Analyze results by stock
        print("\n=== Performance by Stock ===")
        stock_summary = df_results.groupby('ticker').agg({
            'return': 'mean',
            'buy_hold_return': 'mean'
        })
        print(stock_summary)
        
        # Generate recommendations for parameter optimization
        print("\n=== Parameter Optimization Recommendations ===")
        
        # Check if combined strategy is performing well
        combined_perf = strategy_summary.loc['combined', 'return'] if 'combined' in strategy_summary.index else 0
        best_individual = strategy_summary.drop('combined', errors='ignore')['return'].max()
        
        if combined_perf < best_individual:
            print("- Consider increasing the weight of the better-performing individual strategies in the combined strategy")
        
        # Check if RSI strategy needs adjustment
        if 'rsi' in strategy_summary.index:
            rsi_perf = strategy_summary.loc['rsi', 'return']
            if rsi_perf < 0:
                print("- Consider adjusting RSI thresholds: decrease oversold (below 30) and increase overbought (above 70)")
            elif rsi_perf > 0 and rsi_perf < 5:
                print("- Consider fine-tuning RSI thresholds: test values around 25/75 for more conservative entry/exit")
        
        # Check if MACD strategy needs adjustment
        if 'macd' in strategy_summary.index:
            macd_perf = strategy_summary.loc['macd', 'return']
            if macd_perf < 0:
                print("- Consider increasing MACD signal threshold to reduce false signals")
            
        # Check if win rate is low across strategies
        avg_win_rate = strategy_summary['win_rate'].mean()
        if avg_win_rate < 50:
            print("- Consider improving stop loss mechanism: use trailing stops or tighter initial stops")
            print("- Review take profit levels: may need to be increased for better win rate")
        
        # Check if trading frequency is too low
        avg_trades = strategy_summary['trades'].mean()
        if avg_trades < 3:  # Less than 3 trades on average per strategy over 40 days
            print("- Trading frequency is low: consider decreasing thresholds for trade entry")
            print("- Test more aggressive parameter settings to increase trading opportunities")
        
        # Plot strategy performance comparison
        plt.figure(figsize=(10, 6))
        strategy_summary['return'].plot(kind='bar', color='skyblue')
        plt.title('Average Returns by Strategy (40-Day Backtest)')
        plt.ylabel('Return (%)')
        plt.xlabel('Strategy')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the chart
        chart_path = f'backtest_charts/strategy_comparison_{datetime.now().strftime("%Y%m%d")}.png'
        plt.savefig(chart_path)
        print(f"\nStrategy comparison chart saved to: {chart_path}")
        
        return df_results
    else:
        print("No valid backtest results were produced. Check logs for errors.")
        return None

if __name__ == "__main__":
    run_comprehensive_backtest()