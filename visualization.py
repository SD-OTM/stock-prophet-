"""
Visualization module for Stock Prophet
Generates charts for technical indicators and stock performance
"""

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
import base64
from io import BytesIO

def generate_chart(data, ticker, indicators=None, show_forecast=False, forecast_values=None):
    """
    Generate a technical analysis chart for a stock
    
    Args:
        data: Pandas DataFrame with OHLC and indicator data
        ticker: Stock ticker symbol
        indicators: List of technical indicators to include
        show_forecast: Whether to show price forecast
        forecast_values: List of forecasted prices
        
    Returns:
        Base64 encoded string of the chart image
    """
    if indicators is None:
        indicators = ['RSI', 'EMA_9', 'EMA_21', 'BB_upper', 'BB_middle', 'BB_lower']
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(12, 10))
    
    # Define the layout based on which indicators are present
    has_macd = all(x in data.columns for x in ['MACD', 'MACD_Signal', 'MACD_Hist'])
    has_stoch = all(x in data.columns for x in ['Stoch_K', 'Stoch_D'])
    has_rsi = 'RSI' in data.columns
    has_adx = all(x in data.columns for x in ['ADX', 'DI+', 'DI-'])
    
    # Grid layout: determine how many indicator plots we need
    num_indicator_plots = sum([has_rsi, has_macd, has_stoch, has_adx])
    
    # Create the grid layout
    gs = plt.GridSpec(1 + num_indicator_plots, 1, height_ratios=[3] + [1] * num_indicator_plots)
    
    # Main price chart
    ax1 = plt.subplot(gs[0])
    
    # Price and volume
    ax1.plot(data.index, data['Close'], label='Close Price', color='black', linewidth=2)
    
    # Add EMA lines
    if 'EMA_9' in data.columns:
        ax1.plot(data.index, data['EMA_9'], label='EMA 9', color='blue', linewidth=1.5, alpha=0.8)
    if 'EMA_21' in data.columns:
        ax1.plot(data.index, data['EMA_21'], label='EMA 21', color='red', linewidth=1.5, alpha=0.8)
    if 'EMA_50' in data.columns:
        ax1.plot(data.index, data['EMA_50'], label='EMA 50', color='green', linewidth=1, alpha=0.7)
    if 'EMA_200' in data.columns:
        ax1.plot(data.index, data['EMA_200'], label='EMA 200', color='purple', linewidth=1, alpha=0.7)
    
    # Add Bollinger Bands
    if all(x in data.columns for x in ['BB_upper', 'BB_middle', 'BB_lower']):
        ax1.plot(data.index, data['BB_upper'], 'k--', alpha=0.5, linewidth=1)
        ax1.plot(data.index, data['BB_middle'], 'k-', alpha=0.5, linewidth=1)
        ax1.plot(data.index, data['BB_lower'], 'k--', alpha=0.5, linewidth=1)
        ax1.fill_between(data.index, data['BB_upper'], data['BB_lower'], alpha=0.1, color='gray')
    
    # Add forecast if requested
    if show_forecast and forecast_values:
        # Create forecast dates (continuing from last date in data)
        last_date = data.index[-1]
        
        # If datetime index, extend accordingly
        if isinstance(last_date, (datetime, np.datetime64)):
            # For hourly data, extend by hours
            forecast_dates = [last_date + np.timedelta64(i+1, 'h') for i in range(len(forecast_values))]
        else:
            # For numeric index, just extend by 1
            forecast_dates = [last_date + i + 1 for i in range(len(forecast_values))]
        
        # Plot forecast line
        ax1.plot(
            [data.index[-1]] + forecast_dates, 
            [data['Close'].iloc[-1]] + forecast_values, 
            'g--', 
            label='Forecast', 
            linewidth=2,
            alpha=0.8
        )
        
        # Add forecast points
        ax1.scatter(
            forecast_dates, 
            forecast_values, 
            color='green', 
            marker='o', 
            alpha=0.8
        )
        
        # Add forecast labels
        for i, (date, price) in enumerate(zip(forecast_dates, forecast_values)):
            ax1.annotate(
                f'{price:.2f}',
                (date, price),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8)
            )
    
    # Configure the main price chart
    ax1.set_title(f'{ticker} Technical Analysis', fontsize=16)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Format the date axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Create indicator subplots
    subplot_index = 1
    
    # RSI subplot
    if has_rsi:
        ax_rsi = plt.subplot(gs[subplot_index], sharex=ax1)
        ax_rsi.plot(data.index, data['RSI'], color='purple', linewidth=1.5)
        ax_rsi.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax_rsi.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax_rsi.fill_between(data.index, data['RSI'], 70, where=(data['RSI'] >= 70), color='red', alpha=0.3)
        ax_rsi.fill_between(data.index, data['RSI'], 30, where=(data['RSI'] <= 30), color='green', alpha=0.3)
        ax_rsi.set_ylabel('RSI', fontsize=10)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.grid(True, alpha=0.3)
        subplot_index += 1
    
    # MACD subplot
    if has_macd:
        ax_macd = plt.subplot(gs[subplot_index], sharex=ax1)
        ax_macd.plot(data.index, data['MACD'], label='MACD', color='blue', linewidth=1.5)
        ax_macd.plot(data.index, data['MACD_Signal'], label='Signal', color='red', linewidth=1.5)
        ax_macd.bar(data.index, data['MACD_Hist'], label='Histogram', color='gray', alpha=0.4, width=0.01)
        ax_macd.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax_macd.set_ylabel('MACD', fontsize=10)
        ax_macd.grid(True, alpha=0.3)
        ax_macd.legend(loc='upper left', fontsize=8)
        subplot_index += 1
    
    # Stochastic subplot
    if has_stoch:
        ax_stoch = plt.subplot(gs[subplot_index], sharex=ax1)
        ax_stoch.plot(data.index, data['Stoch_K'], label='%K', color='blue', linewidth=1.5)
        ax_stoch.plot(data.index, data['Stoch_D'], label='%D', color='red', linewidth=1.5)
        ax_stoch.axhline(80, color='red', linestyle='--', alpha=0.5)
        ax_stoch.axhline(20, color='green', linestyle='--', alpha=0.5)
        ax_stoch.fill_between(data.index, data['Stoch_K'], 80, where=(data['Stoch_K'] >= 80), color='red', alpha=0.3)
        ax_stoch.fill_between(data.index, data['Stoch_K'], 20, where=(data['Stoch_K'] <= 20), color='green', alpha=0.3)
        ax_stoch.set_ylabel('Stochastic', fontsize=10)
        ax_stoch.set_ylim(0, 100)
        ax_stoch.grid(True, alpha=0.3)
        ax_stoch.legend(loc='upper left', fontsize=8)
        subplot_index += 1
    
    # ADX subplot
    if has_adx:
        ax_adx = plt.subplot(gs[subplot_index], sharex=ax1)
        ax_adx.plot(data.index, data['ADX'], label='ADX', color='black', linewidth=1.5)
        ax_adx.plot(data.index, data['DI+'], label='+DI', color='green', linewidth=1.5)
        ax_adx.plot(data.index, data['DI-'], label='-DI', color='red', linewidth=1.5)
        ax_adx.axhline(25, color='gray', linestyle='--', alpha=0.5)
        ax_adx.set_ylabel('ADX', fontsize=10)
        ax_adx.grid(True, alpha=0.3)
        ax_adx.legend(loc='upper left', fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Add timestamp
    plt.figtext(
        0.01, 0.01, 
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
        fontsize=8, 
        color='gray'
    )
    
    # Save to BytesIO object
    img_data = BytesIO()
    plt.savefig(img_data, format='png', dpi=100)
    img_data.seek(0)
    
    # Encode to base64 for easy embedding in HTML or Telegram
    encoded = base64.b64encode(img_data.read()).decode('utf-8')
    
    # Close the plot to free memory
    plt.close(fig)
    
    return encoded

def save_chart(data, ticker, filename=None, path='charts', **kwargs):
    """
    Generate and save a chart to disk
    
    Args:
        data: Pandas DataFrame with OHLC and indicator data
        ticker: Stock ticker symbol
        filename: Name of the file to save
        path: Directory to save the chart
        **kwargs: Additional arguments for generate_chart
        
    Returns:
        Path to saved chart
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ticker}_{timestamp}.png"
    
    # Create directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Generate chart
    chart_data = generate_chart(data, ticker, **kwargs)
    
    # Decode and save
    chart_bytes = base64.b64decode(chart_data)
    
    file_path = os.path.join(path, filename)
    with open(file_path, 'wb') as f:
        f.write(chart_bytes)
    
    return file_path