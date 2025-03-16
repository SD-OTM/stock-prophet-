"""
40-Day Backtest Analysis for Stock Prophet (Fixed Version)

This script runs a 40-day backtest on select stocks to evaluate trading strategy performance
and provides parameter optimization recommendations.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Stocks to test (major market stocks from different sectors)
TEST_STOCKS = ['AAPL', 'MSFT', 'NVDA', 'TSLA']

# Strategy parameters to test
RSI_PARAMS = {
    'default': {'oversold': 30, 'overbought': 70, 'take_profit': 3.0, 'stop_loss': 4.0},
    'aggressive': {'oversold': 35, 'overbought': 65, 'take_profit': 2.0, 'stop_loss': 3.0},
    'conservative': {'oversold': 25, 'overbought': 75, 'take_profit': 4.0, 'stop_loss': 5.0}
}

MACD_PARAMS = {
    'default': {'signal_threshold': 0.01, 'take_profit': 3.0, 'stop_loss': 4.0},
    'sensitive': {'signal_threshold': 0.005, 'take_profit': 2.0, 'stop_loss': 3.0},
    'cautious': {'signal_threshold': 0.02, 'take_profit': 4.0, 'stop_loss': 5.0}
}

BB_PARAMS = {
    'default': {'band_gap_percent': 0.2, 'take_profit': 3.0, 'stop_loss': 4.0},
    'wide': {'band_gap_percent': 0.3, 'take_profit': 2.5, 'stop_loss': 3.5},
    'narrow': {'band_gap_percent': 0.1, 'take_profit': 3.5, 'stop_loss': 4.5}
}

# Calculate the date range (40 days back from today)
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=40)).strftime('%Y-%m-%d')

def safe_numeric_value(value):
    """Safely extract numeric value from pandas Series or scalar"""
    if isinstance(value, pd.Series):
        if len(value) > 0:
            return float(value.iloc[0])
        return float('nan')
    return float(value)

def safe_value_comparison(value1, value2, comparison='greater'):
    """Safely compare values that might be pandas Series"""
    v1 = safe_numeric_value(value1)
    v2 = safe_numeric_value(value2)
    
    if np.isnan(v1) or np.isnan(v2):
        return False
        
    if comparison == 'greater':
        return v1 > v2
    elif comparison == 'less':
        return v1 < v2
    elif comparison == 'greater_equal':
        return v1 >= v2
    elif comparison == 'less_equal':
        return v1 <= v2
    elif comparison == 'equal':
        return v1 == v2
    else:
        return False

def fetch_historical_data(ticker, start_date, end_date):
    """Fetch historical stock data for backtesting"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
        if data.empty:
            logger.error(f"No data retrieved for {ticker}")
            return None
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_rsi(data, window=14):
    """Calculate RSI indicator"""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands indicator"""
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    return upper_band, sma, lower_band

def run_rsi_backtest(data, params):
    """Run backtest with RSI strategy"""
    # Calculate indicators
    rsi = calculate_rsi(data)
    data['RSI'] = rsi
    
    # Initialize backtest values
    initial_balance = 10000.0
    balance = initial_balance
    position = None
    entry_price = 0
    trades = []
    
    # Run backtest
    for i in range(14, len(data)):  # Start after RSI has values
        curr_date = data.index[i]
        curr_price = safe_numeric_value(data['Close'].iloc[i])
        curr_rsi = safe_numeric_value(data['RSI'].iloc[i])
        
        # If we have a position, check for exit
        if position and entry_price > 0:
            # Calculate profit/loss
            pnl_pct = ((curr_price / entry_price) - 1) * 100
            
            # Exit conditions (take profit or stop loss)
            take_profit_hit = pnl_pct >= params['take_profit']
            stop_loss_hit = pnl_pct <= -params['stop_loss']
            overbought = curr_rsi >= params['overbought']
            
            if take_profit_hit or stop_loss_hit or overbought:
                # Exit position
                shares = position['shares']
                exit_value = shares * curr_price
                balance = exit_value
                
                # Record trade
                trade = {
                    'entry_date': position['date'],
                    'entry_price': position['price'],
                    'exit_date': curr_date,
                    'exit_price': curr_price,
                    'shares': shares,
                    'pnl': exit_value - position['value'],
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'Take Profit' if take_profit_hit else 
                                 'Stop Loss' if stop_loss_hit else 
                                 'RSI Overbought'
                }
                trades.append(trade)
                
                # Clear position
                position = None
        
        # If no position, check for entry
        elif curr_rsi <= params['oversold']:
            # Enter position
            shares = balance / curr_price
            position = {
                'date': curr_date,
                'price': curr_price,
                'shares': shares,
                'value': balance
            }
    
    # Calculate final balance if still in position
    if position:
        final_price = safe_numeric_value(data['Close'].iloc[-1])
        final_value = position['shares'] * final_price
        pnl_pct = ((final_price / position['price']) - 1) * 100
        
        # Record final trade
        trade = {
            'entry_date': position['date'],
            'entry_price': position['price'],
            'exit_date': data.index[-1],
            'exit_price': final_price,
            'shares': position['shares'],
            'pnl': final_value - position['value'],
            'pnl_pct': pnl_pct,
            'exit_reason': 'End of Backtest'
        }
        trades.append(trade)
        
        balance = final_value
    
    # Calculate buy & hold return
    initial_price = safe_numeric_value(data['Close'].iloc[14])
    final_price = safe_numeric_value(data['Close'].iloc[-1])
    buy_hold_return = ((final_price / initial_price) - 1) * 100
    
    # Calculate win rate and other metrics
    win_rate = 0
    if trades:
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = (winning_trades / len(trades)) * 100
    
    return {
        'final_balance': balance,
        'return_pct': ((balance / initial_balance) - 1) * 100,
        'buy_hold_return': buy_hold_return,
        'trade_count': len(trades),
        'win_rate': win_rate,
        'trades': trades
    }

def run_macd_backtest(data, params):
    """Run backtest with MACD strategy"""
    # Calculate indicators
    macd_line, signal_line, histogram = calculate_macd(data)
    data['MACD'] = macd_line
    data['MACD_Signal'] = signal_line
    data['MACD_Hist'] = histogram
    
    # Initialize backtest values
    initial_balance = 10000.0
    balance = initial_balance
    position = None
    entry_price = 0
    trades = []
    
    # Run backtest
    for i in range(26, len(data)):  # Start after MACD has values
        curr_date = data.index[i]
        curr_price = safe_numeric_value(data['Close'].iloc[i])
        curr_macd = safe_numeric_value(data['MACD'].iloc[i])
        curr_signal = safe_numeric_value(data['MACD_Signal'].iloc[i])
        
        # If we have a position, check for exit
        if position:
            # Calculate profit/loss
            pnl_pct = ((curr_price / entry_price) - 1) * 100
            
            # Exit conditions (take profit or stop loss or bearish crossover)
            take_profit_hit = pnl_pct >= params['take_profit']
            stop_loss_hit = pnl_pct <= -params['stop_loss']
            macd_bearish = (curr_macd < curr_signal) and (abs(curr_macd - curr_signal) > params['signal_threshold'])
            
            if take_profit_hit or stop_loss_hit or macd_bearish:
                # Exit position
                shares = position['shares']
                exit_value = shares * curr_price
                balance = exit_value
                
                # Record trade
                trade = {
                    'entry_date': position['date'],
                    'entry_price': position['price'],
                    'exit_date': curr_date,
                    'exit_price': curr_price,
                    'shares': shares,
                    'pnl': exit_value - position['value'],
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'Take Profit' if take_profit_hit else 
                                 'Stop Loss' if stop_loss_hit else 
                                 'MACD Bearish Crossover'
                }
                trades.append(trade)
                
                # Clear position
                position = None
        
        # If no position, check for entry
        else:
            # Check for bullish MACD crossover
            macd_bullish = (curr_macd > curr_signal) and (abs(curr_macd - curr_signal) > params['signal_threshold'])
            
            if macd_bullish:
                # Enter position
                shares = balance / curr_price
                position = {
                    'date': curr_date,
                    'price': curr_price,
                    'shares': shares,
                    'value': balance
                }
    
    # Calculate final balance if still in position
    if position:
        final_price = safe_numeric_value(data['Close'].iloc[-1])
        final_value = position['shares'] * final_price
        pnl_pct = ((final_price / position['price']) - 1) * 100
        
        # Record final trade
        trade = {
            'entry_date': position['date'],
            'entry_price': position['price'],
            'exit_date': data.index[-1],
            'exit_price': final_price,
            'shares': position['shares'],
            'pnl': final_value - position['value'],
            'pnl_pct': pnl_pct,
            'exit_reason': 'End of Backtest'
        }
        trades.append(trade)
        
        balance = final_value
    
    # Calculate buy & hold return
    initial_price = safe_numeric_value(data['Close'].iloc[26])
    final_price = safe_numeric_value(data['Close'].iloc[-1])
    buy_hold_return = ((final_price / initial_price) - 1) * 100
    
    # Calculate win rate and other metrics
    win_rate = 0
    if trades:
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = (winning_trades / len(trades)) * 100
    
    return {
        'final_balance': balance,
        'return_pct': ((balance / initial_balance) - 1) * 100,
        'buy_hold_return': buy_hold_return,
        'trade_count': len(trades),
        'win_rate': win_rate,
        'trades': trades
    }

def run_bb_backtest(data, params):
    """Run backtest with Bollinger Bands strategy"""
    # Calculate indicators
    upper_band, middle_band, lower_band = calculate_bollinger_bands(data)
    data['BB_upper'] = upper_band
    data['BB_middle'] = middle_band
    data['BB_lower'] = lower_band
    
    # Initialize backtest values
    initial_balance = 10000.0
    balance = initial_balance
    position = None
    entry_price = 0
    trades = []
    
    # Run backtest
    for i in range(20, len(data)):  # Start after BB has values
        curr_date = data.index[i]
        curr_price = safe_numeric_value(data['Close'].iloc[i])
        curr_upper = safe_numeric_value(data['BB_upper'].iloc[i])
        curr_lower = safe_numeric_value(data['BB_lower'].iloc[i])
        curr_middle = safe_numeric_value(data['BB_middle'].iloc[i])
        
        # If we have a position, check for exit
        if position:
            # Calculate profit/loss
            pnl_pct = ((curr_price / entry_price) - 1) * 100
            
            # Exit conditions (take profit, stop loss, or upper band touch)
            take_profit_hit = pnl_pct >= params['take_profit']
            stop_loss_hit = pnl_pct <= -params['stop_loss']
            upper_band_touch = curr_price >= curr_upper
            
            if take_profit_hit or stop_loss_hit or upper_band_touch:
                # Exit position
                shares = position['shares']
                exit_value = shares * curr_price
                balance = exit_value
                
                # Record trade
                trade = {
                    'entry_date': position['date'],
                    'entry_price': position['price'],
                    'exit_date': curr_date,
                    'exit_price': curr_price,
                    'shares': shares,
                    'pnl': exit_value - position['value'],
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'Take Profit' if take_profit_hit else 
                                 'Stop Loss' if stop_loss_hit else 
                                 'Upper BB Touch'
                }
                trades.append(trade)
                
                # Clear position
                position = None
        
        # If no position, check for entry
        else:
            # Check if price is near or below lower band and bands are wide enough
            price_near_lower = curr_price <= (curr_lower * 1.01)
            band_gap = (curr_upper - curr_lower) / curr_middle
            sufficient_volatility = band_gap > params['band_gap_percent']
            
            if price_near_lower and sufficient_volatility:
                # Enter position
                shares = balance / curr_price
                position = {
                    'date': curr_date,
                    'price': curr_price,
                    'shares': shares,
                    'value': balance
                }
    
    # Calculate final balance if still in position
    if position:
        final_price = safe_numeric_value(data['Close'].iloc[-1])
        final_value = position['shares'] * final_price
        pnl_pct = ((final_price / position['price']) - 1) * 100
        
        # Record final trade
        trade = {
            'entry_date': position['date'],
            'entry_price': position['price'],
            'exit_date': data.index[-1],
            'exit_price': final_price,
            'shares': position['shares'],
            'pnl': final_value - position['value'],
            'pnl_pct': pnl_pct,
            'exit_reason': 'End of Backtest'
        }
        trades.append(trade)
        
        balance = final_value
    
    # Calculate buy & hold return
    initial_price = safe_numeric_value(data['Close'].iloc[20])
    final_price = safe_numeric_value(data['Close'].iloc[-1])
    buy_hold_return = ((final_price / initial_price) - 1) * 100
    
    # Calculate win rate and other metrics
    win_rate = 0
    if trades:
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = (winning_trades / len(trades)) * 100
    
    return {
        'final_balance': balance,
        'return_pct': ((balance / initial_balance) - 1) * 100,
        'buy_hold_return': buy_hold_return,
        'trade_count': len(trades),
        'win_rate': win_rate,
        'trades': trades
    }

def create_strategy_summary_chart(df_results, save_path='backtest_charts'):
    """Create a chart comparing strategy performance"""
    try:
        # Make sure directory exists
        os.makedirs(save_path, exist_ok=True)
        
        # Group results by strategy
        strategy_summary = df_results.groupby('strategy').agg({
            'return': 'mean',
            'buy_hold_return': 'mean',
            'trades': 'mean',
            'win_rate': 'mean'
        })
        
        # Plot returns by strategy
        plt.figure(figsize=(12, 8))
        ax = strategy_summary[['return', 'buy_hold_return']].plot(kind='bar', color=['skyblue', 'lightgreen'])
        
        # Add a line for zero return
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3)
        
        # Add titles and labels
        plt.title('Average Returns by Strategy (40-Day Backtest)', fontsize=14)
        plt.ylabel('Return (%)', fontsize=12)
        plt.xlabel('Strategy', fontsize=12)
        plt.legend(['Strategy Return', 'Buy & Hold Return'], loc='best')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the chart
        chart_path = f'{save_path}/strategy_comparison_{datetime.now().strftime("%Y%m%d")}.png'
        plt.savefig(chart_path)
        plt.close()
        
        # Create another chart for win rates
        plt.figure(figsize=(12, 6))
        ax = strategy_summary[['win_rate']].plot(kind='bar', color='coral')
        
        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3)
        
        # Add titles and labels
        plt.title('Win Rates by Strategy (40-Day Backtest)', fontsize=14)
        plt.ylabel('Win Rate (%)', fontsize=12)
        plt.xlabel('Strategy', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the chart
        win_rate_chart_path = f'{save_path}/strategy_win_rates_{datetime.now().strftime("%Y%m%d")}.png'
        plt.savefig(win_rate_chart_path)
        plt.close()
        
        return chart_path
        
    except Exception as e:
        logger.error(f"Error creating strategy summary chart: {e}")
        return None

def run_comprehensive_backtest():
    """Run comprehensive backtests on multiple stocks and strategies"""
    
    print(f"=== Running 40-Day Comprehensive Backtest Analysis ===")
    print(f"Period: {start_date} to {end_date}")
    print(f"Testing RSI, MACD, and Bollinger Bands strategies with different parameters")
    print()
    
    # Store results for comparison
    results = []
    
    # Create directory for backtest charts if it doesn't exist
    os.makedirs('backtest_charts', exist_ok=True)
    
    # Test each stock with each strategy and parameter set
    for ticker in TEST_STOCKS:
        print(f"\nAnalyzing {ticker}...")
        
        # Fetch historical data
        data = fetch_historical_data(ticker, start_date, end_date)
        if data is None or data.empty:
            print(f"Skipping {ticker} due to data retrieval issues")
            continue
            
        print(f"Retrieved {len(data)} data points for {ticker}")
        
        # Test RSI strategy with different parameters
        for param_name, params in RSI_PARAMS.items():
            print(f"\nBacktesting RSI strategy ({param_name}) on {ticker}...")
            result = run_rsi_backtest(data.copy(), params)
            
            # Store results
            results.append({
                'ticker': ticker,
                'strategy': f'RSI-{param_name}',
                'return': result['return_pct'],
                'buy_hold_return': result['buy_hold_return'],
                'trades': result['trade_count'],
                'win_rate': result['win_rate'],
                'params': params
            })
            
            # Print results
            print(f"Strategy Return: {result['return_pct']:.2f}%")
            print(f"Buy & Hold Return: {result['buy_hold_return']:.2f}%")
            print(f"Outperformance: {result['return_pct'] - result['buy_hold_return']:.2f}%")
            print(f"Total Trades: {result['trade_count']}")
            print(f"Win Rate: {result['win_rate']:.2f}%")
            
        # Test MACD strategy with different parameters
        for param_name, params in MACD_PARAMS.items():
            print(f"\nBacktesting MACD strategy ({param_name}) on {ticker}...")
            result = run_macd_backtest(data.copy(), params)
            
            # Store results
            results.append({
                'ticker': ticker,
                'strategy': f'MACD-{param_name}',
                'return': result['return_pct'],
                'buy_hold_return': result['buy_hold_return'],
                'trades': result['trade_count'],
                'win_rate': result['win_rate'],
                'params': params
            })
            
            # Print results
            print(f"Strategy Return: {result['return_pct']:.2f}%")
            print(f"Buy & Hold Return: {result['buy_hold_return']:.2f}%")
            print(f"Outperformance: {result['return_pct'] - result['buy_hold_return']:.2f}%")
            print(f"Total Trades: {result['trade_count']}")
            print(f"Win Rate: {result['win_rate']:.2f}%")
            
        # Test Bollinger Bands strategy with different parameters
        for param_name, params in BB_PARAMS.items():
            print(f"\nBacktesting Bollinger Bands strategy ({param_name}) on {ticker}...")
            result = run_bb_backtest(data.copy(), params)
            
            # Store results
            results.append({
                'ticker': ticker,
                'strategy': f'BB-{param_name}',
                'return': result['return_pct'],
                'buy_hold_return': result['buy_hold_return'],
                'trades': result['trade_count'],
                'win_rate': result['win_rate'],
                'params': params
            })
            
            # Print results
            print(f"Strategy Return: {result['return_pct']:.2f}%")
            print(f"Buy & Hold Return: {result['buy_hold_return']:.2f}%")
            print(f"Outperformance: {result['return_pct'] - result['buy_hold_return']:.2f}%")
            print(f"Total Trades: {result['trade_count']}")
            print(f"Win Rate: {result['win_rate']:.2f}%")
    
    if results:
        # Convert to DataFrame for analysis
        df_results = pd.DataFrame(results)
        
        # Analyze results by strategy
        print("\n=== Performance by Strategy ===")
        strategy_summary = df_results.groupby('strategy').agg({
            'return': 'mean',
            'buy_hold_return': 'mean',
            'trades': 'mean',
            'win_rate': 'mean'
        })
        print(strategy_summary.round(2))
        
        # Find the best strategy overall
        best_strategy = strategy_summary['return'].idxmax()
        print(f"\nBest Overall Strategy: {best_strategy} with average return of {strategy_summary.loc[best_strategy, 'return']:.2f}%")
        
        # Create charts
        chart_path = create_strategy_summary_chart(df_results)
        if chart_path:
            print(f"\nStrategy comparison chart saved to: {chart_path}")
        
        # Generate parameter optimization recommendations
        print("\n=== Parameter Optimization Recommendations ===")
        
        # For each strategy type, find the best parameter set
        strategy_types = ['RSI', 'MACD', 'BB']
        for strategy_type in strategy_types:
            strategies_of_type = [s for s in df_results['strategy'] if s.startswith(f"{strategy_type}-")]
            if strategies_of_type:
                best_variant = df_results[df_results['strategy'].isin(strategies_of_type)].groupby('strategy')['return'].mean().idxmax()
                best_params = None
                
                # Find the parameters for the best variant
                for idx, row in df_results[df_results['strategy'] == best_variant].iterrows():
                    best_params = row['params']
                    break
                
                if best_params:
                    print(f"\nBest {strategy_type} Parameters:")
                    for param, value in best_params.items():
                        print(f"  {param}: {value}")
        
        # Find stocks that perform best with each strategy
        print("\n=== Best Strategy by Stock ===")
        for ticker in TEST_STOCKS:
            ticker_data = df_results[df_results['ticker'] == ticker]
            if not ticker_data.empty:
                best_strategy_idx = ticker_data['return'].idxmax()
                best_strategy_for_ticker = ticker_data.loc[best_strategy_idx, 'strategy']
                best_return = ticker_data.loc[best_strategy_idx, 'return']
                print(f"{ticker}: {best_strategy_for_ticker} (Return: {best_return:.2f}%)")
        
        return df_results
    else:
        print("No valid backtest results were produced. Check logs for errors.")
        return None

if __name__ == "__main__":
    run_comprehensive_backtest()