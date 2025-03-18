"""
Technical Indicators Module for Stock Prophet
Contains utility functions for calculating indicators and analyzing trends
"""

import pandas as pd 
import pandas_ta as ta
import numpy as np

def safe_float(value):
    """Safely convert a value to float, handling pandas Series objects"""
    try:
        # If it's a pandas Series, get the first value
        if hasattr(value, 'iloc'):
            return float(value.iloc[0])
        return float(value)
    except (ValueError, TypeError, IndexError):
        return 0.0

def calculate_indicators(data):
    """Calculate technical indicators for stock data"""
    if data is None or len(data) < 20:
        return data
        
    # Make a copy to avoid SettingWithCopyWarning
    df = data.copy()
    
    # Add RSI
    df.ta.rsi(length=14, append=True)
    
    # Add MACD
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    # Add Bollinger Bands
    df.ta.bbands(length=20, std=2, append=True)
    
    # Add Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Add ATR for volatility
    df.ta.atr(length=14, append=True)
    
    # Add stochastic oscillator
    df.ta.stoch(append=True)
    
    # Calculate price changes and volatility
    df['Daily_Change'] = df['Close'].pct_change() * 100
    
    # Fill NaN values from indicators
    df.fillna(0, inplace=True)
    
    # Calculate EMA crossovers
    if len(df) >= 50:  # Only if we have enough data
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['EMA_Crossover'] = np.where(df['EMA_9'] > df['EMA_21'], 1, -1)
    
    return df

def determine_trend(data):
    """Determine the current trend of a stock based on technical indicators"""
    if data is None or len(data) < 20:
        return "Unknown"
        
    # Get latest data point
    latest = data.iloc[-1]
    
    # Check for SMA alignment
    sma_alignment = 0
    if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
        if latest['SMA_20'] > latest['SMA_50']:
            sma_alignment += 1
        else:
            sma_alignment -= 1
            
    if 'SMA_50' in data.columns and 'SMA_200' in data.columns and len(data) >= 200:
        if latest['SMA_50'] > latest['SMA_200']:
            sma_alignment += 1
        else:
            sma_alignment -= 1
    
    # Check RSI
    rsi_signal = 0
    if 'RSI_14' in data.columns:
        rsi = latest['RSI_14']
        if rsi > 70:
            rsi_signal = -1  # Overbought
        elif rsi < 30:
            rsi_signal = 1   # Oversold
        elif rsi > 50:
            rsi_signal = 0.5  # Bullish but not overbought
        else:
            rsi_signal = -0.5  # Bearish but not oversold
    
    # Check MACD
    macd_signal = 0
    if 'MACD_12_26_9' in data.columns and 'MACDs_12_26_9' in data.columns:
        if latest['MACD_12_26_9'] > latest['MACDs_12_26_9']:
            macd_signal = 1
        else:
            macd_signal = -1
            
        # Add strength to the signal if MACD is positive or negative
        if latest['MACD_12_26_9'] > 0:
            macd_signal += 0.5
        else:
            macd_signal -= 0.5
    
    # Check price relative to Bollinger Bands
    bb_signal = 0
    if 'BBU_20_2.0' in data.columns and 'BBL_20_2.0' in data.columns:
        if latest['Close'] > latest['BBU_20_2.0']:
            bb_signal = -1  # Overbought
        elif latest['Close'] < latest['BBL_20_2.0']:
            bb_signal = 1   # Oversold
    
    # Get recent price action (last 5 days)
    recent_data = data.iloc[-5:] if len(data) >= 5 else data
    price_action = 1 if recent_data['Close'].iloc[-1] > recent_data['Close'].iloc[0] else -1
    
    # Calculate the overall trend score
    trend_score = sma_alignment + rsi_signal + macd_signal + bb_signal + price_action
    
    # Determine trend based on score
    if trend_score >= 3:
        return "Strong Uptrend"
    elif trend_score > 0:
        return "Uptrend"
    elif trend_score == 0:
        return "Sideways/Neutral"
    elif trend_score > -3:
        return "Downtrend"
    else:
        return "Strong Downtrend"