#!/usr/bin/env python3
"""
Extreme Backtesting Script for Stock Prophet
Uses ultra-aggressive parameters to force trade generation for analysis
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
from aggressive_strategies import AggressiveStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def force_trades_backtest(ticker, days=60, target_profit_pct=1.0, max_hold_periods=10):
    """
    Run a backtest with extremely aggressive parameters to force trades
    
    Args:
        ticker: Stock symbol to test
        days: Number of days to backtest
        target_profit_pct: Minimum profit target percentage (e.g., 1.0 for 1%)
        max_hold_periods: Maximum number of periods to hold before forced exit
                         (can be days or hours depending on interval)
        
    Returns:
        Dictionary with backtest results
    """
    # Calculate date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    # Create directory for results if needed
    os.makedirs("backtest_results", exist_ok=True)
    
    logger.info(f"Starting extreme backtest for {ticker} over {days} days")
    logger.info(f"Target profit: {target_profit_pct}%, max hold: {max_hold_periods} periods")
    logger.info(f"Period: {start_date} to {end_date}")
    
    # Get historical data directly using yfinance (can be daily or hourly)
    # For 10h trading, we'll still use daily data but will consider each point as 
    # a distinct trading opportunity and will force exits after max_hold_periods
    logger.info(f"Fetching historical data for {ticker}")
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    
    if len(data) < 10:
        logger.error(f"Not enough data points for {ticker}. Got {len(data)} points.")
        return {"ticker": ticker, "error": "Not enough data points", "trades": []}
    
    logger.info(f"Retrieved {len(data)} historical data points for {ticker}")
    
    # Calculate indicators
    data['RSI'] = calculate_rsi(data['Close'], 3)  # Very short RSI period
    calculate_bollinger_bands(data, 3, 2)  # Very short Bollinger Band period
    calculate_macd(data, 3, 7, 3)  # Very short MACD periods
    
    # Create strategy instance with 1% profit target
    strategy = AggressiveStrategy(parameters={
        'take_profit': target_profit_pct,  # User-defined profit target (default 1%)
        'stop_loss': target_profit_pct * 1.5,  # Stop loss 1.5x the profit target
        'rsi_oversold': 49,       # Barely below middle RSI (normal: 30)
        'rsi_overbought': 51,     # Barely above middle RSI (normal: 70)
        'min_indicators': 1,      # Only need one signal
        'band_gap_percent': 0.005, # Ultra-tiny band gap requirement
        'signal_threshold': 0.00001 # Practically no MACD threshold
    })
    
    # Initialize variables for the backtest
    initial_balance = 10000.0
    balance = initial_balance
    shares = 0
    trades = []
    user_data = {'backtest_user': {}}  # Tracks open positions
    entry_date = None
    entry_price = None
    
    # Initialize variables for tracking position holding time
    days_held = 0
    max_hold_days = max_hold_periods  # Maximum days to hold a position
    
    # Loop through the data to simulate trades
    for i in range(1, len(data)):
        current_date = data.index[i]
        current_price = data['Close'].iloc[i]
        
        # Get only the current point and the previous point for the strategy
        df_slice = data.iloc[i-1:i+1].copy()
        
        # If we're in a position, increment days held
        if shares > 0 and entry_date is not None:
            days_held += 1
        
        # Generate signals using our aggressive strategy
        signals = strategy.generate_signals(df_slice, 'backtest_user', ticker, user_data)
        
        # Create a forced sell signal if we've held the position too long
        force_sell = False
        sell_reason = ""
        
        # Check if we're in a position
        if shares > 0 and entry_date is not None and entry_price is not None:
            # Calculate current profit/loss
            if isinstance(current_price, pd.Series):
                current_price_val = current_price.iloc[0]
            else:
                current_price_val = current_price
                
            current_profit_pct = ((current_price_val / entry_price) - 1) * 100
            
            # Force sell conditions:
            # 1. Hit 1% profit target
            if current_profit_pct >= target_profit_pct:
                force_sell = True
                sell_reason = f"PROFIT TARGET ({target_profit_pct}%)"
            # 2. Held for max days
            elif days_held >= max_hold_days:
                force_sell = True
                sell_reason = f"MAX HOLD ({max_hold_days} days)"
            # 3. Stop loss at 1.5x target profit (downside)
            elif current_profit_pct <= -(target_profit_pct * 1.5):
                force_sell = True
                sell_reason = "STOP LOSS"
        
        # Process buy signals
        for signal in signals:
            if "Buy" in signal:
                # Handle buy signal
                if balance > 0:
                    # Calculate number of shares to buy (all available balance)
                    # Make sure current_price is a scalar, not a Series
                    if isinstance(current_price, pd.Series):
                        current_price = current_price.iloc[0]
                    
                    shares = balance / current_price
                    entry_price = current_price
                    entry_date = current_date
                    days_held = 0  # Reset days held counter
                    
                    logger.info(f"BUY: {shares:.2f} shares of {ticker} at ${entry_price:.2f} on {entry_date.strftime('%Y-%m-%d')}")
                    
                    # Record the trade but don't complete it yet
                    balance = 0  # Used all balance to buy
            
            elif "Sell" in signal and 'backtest_user' in user_data and ticker in user_data['backtest_user']:
                force_sell = True
                sell_reason = "TECHNICAL SIGNAL"
        
        # Handle selling (either from signal or forced)
        if force_sell and shares > 0 and entry_date is not None and entry_price is not None:
            # Make sure current_price is a scalar, not a Series
            if isinstance(current_price, pd.Series):
                current_price = current_price.iloc[0]
                
            exit_price = current_price
            profit_loss = (exit_price - entry_price) * shares
            profit_pct = ((exit_price / entry_price) - 1) * 100
            
            logger.info(f"SELL: {shares:.2f} shares of {ticker} at ${exit_price:.2f} on {current_date.strftime('%Y-%m-%d')}")
            logger.info(f"REASON: {sell_reason}")
            logger.info(f"P/L: ${profit_loss:.2f} ({profit_pct:.2f}%)")
            logger.info(f"Days held: {days_held}")
            
            # Record completed trade
            trades.append({
                'entry_date': entry_date,
                'entry_price': entry_price, 
                'exit_date': current_date,
                'exit_price': exit_price,
                'shares': shares,
                'profit_loss': profit_loss,
                'profit_pct': profit_pct,
                'days_held': days_held,
                'exit_reason': sell_reason
            })
            
            # Update balance
            balance = shares * exit_price
            shares = 0
            # Reset entry variables
            entry_date = None
            entry_price = None
            days_held = 0
    
    # Calculate final results
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t['profit_loss'] > 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    # Calculate final balance
    final_price = data['Close'].iloc[-1]
    if isinstance(final_price, pd.Series):
        final_price = final_price.iloc[0]
    
    final_balance = balance + (shares * final_price if shares > 0 else 0)
    total_return = ((final_balance / initial_balance) - 1) * 100
    
    # Handle case where there are still open positions
    if shares > 0:
        current_price = data['Close'].iloc[-1]
        # Make sure current_price is a scalar, not a Series
        if isinstance(current_price, pd.Series):
            current_price = current_price.iloc[0]
            
        profit_loss = (current_price - entry_price) * shares
        profit_pct = ((current_price / entry_price) - 1) * 100
        
        logger.info(f"OPEN POSITION: {shares:.2f} shares of {ticker} at ${current_price:.2f}")
        logger.info(f"Unrealized P/L: ${profit_loss:.2f} ({profit_pct:.2f}%)")
        
        # Add the open position to trades list as an incomplete trade
        trades.append({
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': data.index[-1],
            'exit_price': current_price,
            'shares': shares,
            'profit_loss': profit_loss,
            'profit_pct': profit_pct,
            'status': 'open'
        })
    
    # Generate report
    logger.info(f"=== BACKTEST RESULTS FOR {ticker} ===")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Initial Balance: ${initial_balance:.2f}")
    logger.info(f"Final Balance: ${final_balance:.2f}")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Win Rate: {win_rate:.2f}%")
    
    # Print detailed trade list
    print("\n=== DETAILED TRADE LIST ===")
    for i, trade in enumerate(trades, 1):
        status = trade.get('status', 'closed')
        result = "PROFIT" if trade['profit_loss'] > 0 else "LOSS"
        print(f"Trade {i}: {result} ${trade['profit_loss']:.2f} ({trade['profit_pct']:.2f}%)")
        print(f"  Buy: {trade['entry_date'].strftime('%Y-%m-%d')} at ${trade['entry_price']:.2f}")
        print(f"  Sell: {trade['exit_date'].strftime('%Y-%m-%d')} at ${trade['exit_price']:.2f}")
        if status == 'open':
            print(f"  STATUS: OPEN POSITION")
        print()
    
    # Save results to file
    results_file = f"backtest_results/{ticker}_extreme_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(trades).to_csv(results_file, index=False)
    logger.info(f"Saved detailed results to {results_file}")
    
    # Return summary
    return {
        "ticker": ticker,
        "period": f"{start_date} to {end_date}",
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "total_return_pct": total_return,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "win_rate": win_rate,
        "trades": trades
    }

def calculate_rsi(series, period=3):
    """Calculate RSI with very short period"""
    delta = series.diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(df, period=3, std_dev=2):
    """Calculate Bollinger Bands with very short period"""
    # Make sure we're dealing with a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Calculate middle band (simple moving average)
    middle_band = df['Close'].rolling(window=period).mean()
    df['BBM_3_2.0'] = middle_band
    
    # Calculate standard deviation
    rolling_std = df['Close'].rolling(window=period).std()
    
    # Calculate upper and lower bands
    df['BBU_3_2.0'] = middle_band + (std_dev * rolling_std)
    df['BBL_3_2.0'] = middle_band - (std_dev * rolling_std)
    
    return df

def calculate_macd(df, fast=3, slow=7, signal=3):
    """Calculate MACD with very short periods"""
    df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def main():
    """Main function to run extreme backtests"""
    # Expanded list of volatile tickers to find trades
    ticker_list = ["NVDA", "TSLA", "AMD", "AAPL", "META", "AMZN", "GOOGL", "MSFT", "NFLX", "PYPL"]
    
    # Set profit target and max holding period
    target_profit_pct = 1.0  # 1% minimum profit target
    max_hold_days = 5  # Maximum 5 trading days to hold position
    
    results = []
    for ticker in ticker_list:
        print(f"\n{'='*50}")
        print(f"RUNNING EXTREME BACKTEST FOR {ticker}")
        print(f"Target Profit: {target_profit_pct}%, Max Hold: {max_hold_days} days")
        print(f"{'='*50}\n")
        
        try:
            # Run backtest with our profit target and max hold days
            result = force_trades_backtest(ticker, target_profit_pct=target_profit_pct, max_hold_periods=max_hold_days)
            results.append(result)
            
            # Calculate how many completed trades (not open positions)
            completed_trades = len([t for t in result.get("trades", []) 
                                  if not t.get("status") == "open"])
            
            profitable_trades = len([t for t in result.get("trades", []) 
                                  if not t.get("status") == "open" and t.get("profit_pct", 0) >= 1.0])
            
            # If we got at least 2 completed profitable trades, consider it a success
            if profitable_trades >= 2:
                print(f"\nSUCCESS: Generated {profitable_trades} profitable trades (≥1%) for {ticker}!")
                print(f"Total Completed Trades: {completed_trades}")
                print(f"Win Rate: {result['win_rate']:.2f}%")
                print(f"Total Return: {result['total_return_pct']:.2f}%")
                
                # No need to test more tickers if we found one with good results
                break
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue
    
    # Print overall summary with 1% profit focus
    print("\n\n===== EXTREME BACKTEST SUMMARY =====")
    print(f"Target Profit: {target_profit_pct}%, Max Hold Period: {max_hold_days} days")
    print("\nRESULTS BY TICKER:")
    
    for result in results:
        ticker = result.get("ticker")
        all_trades = result.get("trades", [])
        total_trades = len(all_trades)
        
        # Get completed trades (not open positions)
        completed_trades = [t for t in all_trades if not t.get("status") == "open"]
        
        # Get profitable trades (≥1%)
        profitable_trades = [t for t in completed_trades if t.get("profit_pct", 0) >= target_profit_pct]
        
        # Calculate metrics
        num_completed = len(completed_trades)
        num_profitable = len(profitable_trades)
        
        # Calculate average profit on completed trades
        avg_profit = sum(t.get("profit_pct", 0) for t in completed_trades) / num_completed if num_completed > 0 else 0
        
        # Calculate profitable trade ratio
        profitable_ratio = (num_profitable / num_completed * 100) if num_completed > 0 else 0
        
        win_rate = result.get("win_rate", 0)
        return_pct = result.get("total_return_pct", 0)
        
        print(f"\n{ticker}:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Completed Trades: {num_completed}")
        print(f"  Profitable Trades (≥{target_profit_pct}%): {num_profitable}")
        print(f"  Profitable Trade Ratio: {profitable_ratio:.2f}%")
        print(f"  Avg Profit: {avg_profit:.2f}%")
        print(f"  Overall Return: {return_pct:.2f}%")

if __name__ == "__main__":
    main()