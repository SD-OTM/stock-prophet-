"""
Backtesting module for Stock Prophet
Tests trading strategies against historical data to evaluate performance
"""

import logging
import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import strategies
import main
from io import BytesIO
import base64

def safe_float(value):
    """Safely convert a value to float, handling pandas Series objects"""
    if hasattr(value, 'item'):
        try:
            return float(value.item())
        except (ValueError, AttributeError):
            if hasattr(value, 'iloc') and len(value) > 0:
                return float(value.iloc[0])
            return 0.0
    return float(value)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if running in CI environment
IS_CI_ENV = os.environ.get('CI') == 'true'
if IS_CI_ENV:
    logger.info("Running backtesting in CI environment - some features will be modified for CI compatibility")

class BacktestResult:
    """Class to store and report backtesting results"""
    
    def __init__(self, ticker, strategy_name, start_date, end_date, timeframe):
        self.ticker = ticker
        self.strategy_name = strategy_name
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.trades = []
        self.initial_balance = 10000.0  # Default starting balance
        self.final_balance = self.initial_balance
        self.max_drawdown = 0.0
        self.profit_factor = 0.0
        self.win_rate = 0.0
        self.buy_and_hold_return = 0.0
        self.strategy_return = 0.0
        self.total_return = 0.0  # Add total_return attribute to fix the issue
        self.signals = []
        
    def add_trade(self, entry_date, entry_price, exit_date, exit_price, shares, profit_loss, trade_type):
        """Add a completed trade to the results"""
        self.trades.append({
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'shares': shares,
            'profit_loss': profit_loss,
            'trade_type': trade_type  # 'long' or 'short'
        })
        
    def add_signal(self, date, price, signal_type):
        """Add a buy/sell signal to the results"""
        self.signals.append({
            'date': date,
            'price': price,
            'type': signal_type  # 'buy' or 'sell'
        })
        
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            logger.warning("No trades executed during backtesting period")
            self.win_rate = 0.0
            self.profit_factor = 0.0
            self.max_drawdown = 0.0
            self.strategy_return = 0.0
            self.total_return = 0.0  # Set the total_return for compatibility
            return
            
        # Calculate profit metrics
        winning_trades = [t for t in self.trades if safe_float(t['profit_loss']) > 0]
        losing_trades = [t for t in self.trades if safe_float(t['profit_loss']) <= 0]
        
        total_profit = sum(safe_float(t['profit_loss']) for t in winning_trades)
        total_loss = abs(sum(safe_float(t['profit_loss']) for t in losing_trades))
        
        self.win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        self.profit_factor = total_profit / total_loss if safe_float(total_loss) > 0 else float('inf')
        
        # Calculate final balance
        self.final_balance = self.initial_balance + sum(safe_float(t['profit_loss']) for t in self.trades)
        self.strategy_return = (self.final_balance / self.initial_balance - 1) * 100.0
        self.total_return = self.strategy_return  # Set total_return same as strategy_return for compatibility
        
        # Calculate max drawdown
        balance_curve = [self.initial_balance]
        for trade in self.trades:
            balance_curve.append(balance_curve[-1] + safe_float(trade['profit_loss']))
            
        peak = self.initial_balance
        drawdowns = []
        
        for balance in balance_curve:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak * 100.0 if peak > 0 else 0
            drawdowns.append(drawdown)
            
        self.max_drawdown = max(drawdowns)
        
    def generate_report(self):
        """Generate a text report of the backtesting results"""
        self.calculate_metrics()
        
        report = [
            f"Backtesting Results for {self.ticker}",
            f"Strategy: {self.strategy_name}",
            f"Period: {self.start_date} to {self.end_date} ({self.timeframe})",
            f"",
            f"Performance Metrics:",
            f"Initial Balance: ${self.initial_balance:.2f}",
            f"Final Balance: ${self.final_balance:.2f}",
            f"Total Return: {self.strategy_return:.2f}%",
            f"Buy & Hold Return: {float(safe_float(self.buy_and_hold_return)):.2f}%",
            f"Outperformance: {self.strategy_return - float(safe_float(self.buy_and_hold_return)):.2f}%",
            f"",
            f"Trade Statistics:",
            f"Total Trades: {len(self.trades)}",
            f"Win Rate: {self.win_rate*100:.2f}%",
            f"Profit Factor: {self.profit_factor:.2f}",
            f"Max Drawdown: {self.max_drawdown:.2f}%",
            f"",
            f"Trade List:"
        ]
        
        for i, trade in enumerate(self.trades, 1):
            profit_loss_pct = (safe_float(trade['exit_price']) / safe_float(trade['entry_price']) - 1) * 100.0
            direction = "↑" if safe_float(profit_loss_pct) > 0 else "↓"
            # Format dates safely handling Series objects
            entry_date_str = trade['entry_date'].strftime('%Y-%m-%d %H:%M') if not isinstance(trade['entry_date'], pd.Series) else str(trade['entry_date'])
            exit_date_str = trade['exit_date'].strftime('%Y-%m-%d %H:%M') if not isinstance(trade['exit_date'], pd.Series) else str(trade['exit_date'])
            
            report.append(
                f"{i}. {entry_date_str} to {exit_date_str}: "
                f"${safe_float(trade['entry_price']):.2f} → ${safe_float(trade['exit_price']):.2f} {direction} "
                f"({profit_loss_pct:.2f}%) | P/L: ${safe_float(trade['profit_loss']):.2f}"
            )
            
        return "\n".join(report)
        
    def generate_chart(self):
        """Generate a chart of the backtesting results"""
        # Skip chart generation in CI environment to speed up tests
        if IS_CI_ENV:
            logger.info("Skipping chart generation in CI environment")
            return None
            
        if not self.trades or not self.signals:
            logger.warning("Not enough data to generate chart")
            return None
            
        # Create a figure with price chart and balance curve
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # Get historical data for the chart
        data = get_historical_data(self.ticker, self.start_date, self.end_date, self.timeframe)
        if data is None or len(data) == 0:
            logger.error(f"No historical data available for {self.ticker}")
            return None
            
        # Plot price chart
        ax1.plot(data.index, data['Close'], color='royalblue', linewidth=1.5)
        ax1.set_title(f"{self.ticker} Backtesting Results - {self.strategy_name}")
        ax1.set_ylabel('Price ($)')
        ax1.grid(True, alpha=0.3)
        
        # Plot buy signals
        buy_signals = [s for s in self.signals if s['type'] == 'buy']
        if buy_signals:
            buy_dates = [s['date'] for s in buy_signals]
            buy_prices = [s['price'] for s in buy_signals]
            ax1.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, alpha=0.7, label='Buy Signal')
        
        # Plot sell signals
        sell_signals = [s for s in self.signals if s['type'] == 'sell']
        if sell_signals:
            sell_dates = [s['date'] for s in sell_signals]
            sell_prices = [s['price'] for s in sell_signals]
            ax1.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, alpha=0.7, label='Sell Signal')
        
        ax1.legend()
        
        # Plot account balance
        balance_curve = [self.initial_balance]
        dates = [data.index[0]]
        
        for trade in self.trades:
            balance_curve.append(balance_curve[-1] + safe_float(trade['profit_loss']))
            dates.append(trade['exit_date'])
            
        ax2.plot(dates, balance_curve, color='green', linewidth=1.5)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Balance ($)')
        ax2.grid(True, alpha=0.3)
        
        # Format date axis
        plt.gcf().autofmt_xdate()
        
        # Save chart to a buffer
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        # Convert to base64 string
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return image_base64

def get_historical_data(ticker, start_date, end_date, interval='1d'):
    """
    Fetch historical stock data for backtesting
    
    Args:
        ticker: Stock symbol
        start_date: Start date for historical data
        end_date: End date for historical data
        interval: Data frequency ('1d', '1h', '1wk', '1mo')
        
    Returns:
        Pandas DataFrame with historical data
    """
    try:
        if interval == '1h':
            # For hourly data, we need to use intraday with limited history
            period = "7d" if (datetime.now() - datetime.strptime(start_date, '%Y-%m-%d')).days <= 7 else "60d"
            data = yf.download(ticker, period=period, interval=interval)
        else:
            # For daily, weekly, monthly data
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            
        if len(data) == 0:
            logger.warning(f"No data found for {ticker} from {start_date} to {end_date} at {interval} interval")
            return None
            
        logger.info(f"Retrieved {len(data)} historical data points for {ticker}")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return None

def run_backtest(ticker, strategy_name, start_date, end_date, timeframe='1d'):
    """
    Run a backtest of a trading strategy on historical data
    
    Args:
        ticker: Stock symbol to backtest
        strategy_name: Name of the strategy to test
        start_date: Start date for the backtest (YYYY-MM-DD)
        end_date: End date for the backtest (YYYY-MM-DD)
        timeframe: Data frequency ('1d', '1h', '1wk', '1mo')
        
    Returns:
        BacktestResult object with results
    """
    logger.info(f"Starting backtest for {ticker} using {strategy_name} strategy")
    logger.info(f"Period: {start_date} to {end_date}, Timeframe: {timeframe}")
    
    # Modify behavior for CI environment
    if IS_CI_ENV:
        logger.info("CI environment detected - will use simplified backtesting approach")
    
    # Get historical data
    data = get_historical_data(ticker, start_date, end_date, timeframe)
    if data is None or len(data) == 0:
        logger.error("Unable to run backtest: No data available")
        return None
        
    # Initialize results object
    result = BacktestResult(ticker, strategy_name, start_date, end_date, timeframe)
    
    # Get strategy instance
    strategy = strategies.get_strategy(strategy_name)
    if strategy is None:
        logger.error(f"Strategy '{strategy_name}' not found")
        return None
        
    # Calculate technical indicators
    data = main.calculate_indicators(data.copy())
    
    # Keep track of position
    in_position = False
    entry_price = 0.0
    entry_date = None
    position_type = None  # 'long' or 'short'
    position_size = 0
    
    # Set initial capital and position size
    available_capital = result.initial_balance
    risk_per_trade = 0.02  # 2% risk per trade
    
    # Calculate buy and hold return for comparison
    if len(data) >= 2:
        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]
        result.buy_and_hold_return = (end_price / start_price - 1) * 100.0
    
    # Simulate trading through the historical data
    for i in range(1, len(data)):
        current_date = data.index[i]
        current_price = data['Close'].iloc[i]
        
        # Skip the first few bars until we have enough data for indicators
        if i < 10:
            continue
            
        # Get a subset of data up to the current bar
        historical_data = data.iloc[:i+1]
        
        # Generate signals
        signals_list = strategy.generate_signals(historical_data, "backtest_user", ticker)
        
        # Check if signals returned is a list (new format) or dictionary (old format)
        if isinstance(signals_list, list):
            # Process the signal list
            buy_signal = False
            sell_signal = False
            
            # Scan through signals for buy/sell indicators
            for signal in signals_list:
                if "Buy" in signal or "buy" in signal or "🚀" in signal:
                    buy_signal = True
                elif "Sell" in signal or "sell" in signal or "📉" in signal:
                    sell_signal = True
            
            # Convert to dictionary format for backward compatibility
            signals = {
                'buy': buy_signal,
                'sell': sell_signal
            }
        else:
            # Use the signals dictionary directly if that's what was returned
            signals = signals_list or {}
        
        # Handle trading signals
        if not in_position and 'buy' in signals and signals['buy']:
            # Enter long position
            position_size = int((available_capital * risk_per_trade) / safe_float(current_price))
            if position_size > 0:
                entry_price = current_price
                entry_date = current_date
                in_position = True
                position_type = 'long'
                
                # Record signal
                result.add_signal(current_date, current_price, 'buy')
                logger.info(f"BUY signal at {current_date}: {ticker} @ ${safe_float(current_price):.2f}")
                
        elif not in_position and 'sell' in signals and signals['sell']:
            # Enter short position (if supported)
            position_size = int((available_capital * risk_per_trade) / safe_float(current_price))
            if position_size > 0:
                entry_price = current_price
                entry_date = current_date
                in_position = True
                position_type = 'short'
                
                # Record signal
                result.add_signal(current_date, current_price, 'sell')
                logger.info(f"SELL signal at {current_date}: {ticker} @ ${safe_float(current_price):.2f}")
                
        elif in_position:
            # Check for exit conditions
            take_profit = strategy.parameters.get('take_profit', 3.0) / 100.0
            stop_loss = strategy.parameters.get('stop_loss', 4.0) / 100.0
            
            if position_type == 'long':
                price_change = (current_price / entry_price - 1)
                
                # Check for exit signals
                exit_signal = False
                if 'sell' in signals and signals['sell']:
                    exit_signal = True
                if safe_float(price_change) >= take_profit:
                    exit_signal = True
                    logger.info(f"Take profit triggered: {safe_float(price_change*100):.2f}%")
                if safe_float(price_change) <= -stop_loss:
                    exit_signal = True
                    logger.info(f"Stop loss triggered: {safe_float(price_change*100):.2f}%")
                    
                if exit_signal:
                    # Close long position
                    profit_loss = position_size * (current_price - entry_price)
                    result.add_trade(entry_date, entry_price, current_date, current_price, 
                                     position_size, profit_loss, position_type)
                    result.add_signal(current_date, current_price, 'sell')
                    
                    logger.info(f"Closed LONG position at {current_date}: {ticker} @ ${safe_float(current_price):.2f}")
                    logger.info(f"P/L: ${safe_float(profit_loss):.2f} ({safe_float(price_change*100):.2f}%)")
                    
                    # Reset position
                    in_position = False
                    available_capital += profit_loss
                    
            elif position_type == 'short':
                price_change = (entry_price / current_price - 1)
                
                # Check for exit signals
                exit_signal = False
                if 'buy' in signals and signals['buy']:
                    exit_signal = True
                if safe_float(price_change) >= take_profit:
                    exit_signal = True
                    logger.info(f"Take profit triggered: {safe_float(price_change*100):.2f}%")
                if safe_float(price_change) <= -stop_loss:
                    exit_signal = True
                    logger.info(f"Stop loss triggered: {safe_float(price_change*100):.2f}%")
                    
                if exit_signal:
                    # Close short position
                    profit_loss = position_size * (entry_price - current_price)
                    result.add_trade(entry_date, entry_price, current_date, current_price, 
                                     position_size, profit_loss, position_type)
                    result.add_signal(current_date, current_price, 'buy')
                    
                    logger.info(f"Closed SHORT position at {current_date}: {ticker} @ ${safe_float(current_price):.2f}")
                    logger.info(f"P/L: ${safe_float(profit_loss):.2f} ({safe_float(price_change*100):.2f}%)")
                    
                    # Reset position
                    in_position = False
                    available_capital += profit_loss
    
    # Close any open positions at the end of the test period
    if in_position:
        final_price = data['Close'].iloc[-1]
        final_date = data.index[-1]
        
        if position_type == 'long':
            profit_loss = position_size * (final_price - entry_price)
        else:  # short
            profit_loss = position_size * (entry_price - final_price)
            
        result.add_trade(entry_date, entry_price, final_date, final_price, 
                         position_size, profit_loss, position_type)
        
        # Add an exit signal
        signal_type = 'sell' if position_type == 'long' else 'buy'
        result.add_signal(final_date, final_price, signal_type)
        
        logger.info(f"Closed position at end of test period: ${safe_float(final_price):.2f}")
        logger.info(f"P/L: ${safe_float(profit_loss):.2f}")
    
    # Calculate performance metrics
    result.calculate_metrics()
    
    logger.info(f"Backtest completed: {len(result.trades)} trades")
    logger.info(f"Final balance: ${result.final_balance:.2f} (Return: {result.strategy_return:.2f}%)")
    if hasattr(result, 'buy_and_hold_return') and not isinstance(result.buy_and_hold_return, pd.Series):
        logger.info(f"Buy & Hold Return: {result.buy_and_hold_return:.2f}%")
    else:
        logger.info("Buy & Hold Return: Not calculated")
    
    return result

def cmd_backtest(ticker, strategy_name="combined", start_date=None, end_date=None, timeframe="1d"):
    """Command-line interface for backtesting"""
    
    # Use default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        if timeframe == '1d':
            # Default to 1 year for daily data
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        elif timeframe == '1h':
            # Default to 7 days for hourly data (yfinance limit)
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        elif timeframe == '1wk':
            # Default to 3 years for weekly data
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        elif timeframe == '1mo':
            # Default to 5 years for monthly data
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    # In CI environment, use smaller date ranges to speed up tests
    if IS_CI_ENV:
        logger.info("CI environment detected - using smaller date range for faster testing")
        # Use shorter date range for quicker testing
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Ensure start_date is before end_date (use 30 days before end_date)
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        start_date = (end_date_obj - timedelta(days=30)).strftime('%Y-%m-%d')
        logger.info(f"Using CI-adjusted date range: {start_date} to {end_date}")
    
    # Run backtest
    result = run_backtest(ticker, strategy_name, start_date, end_date, timeframe)
    
    if result is not None:
        # Print report
        print(result.generate_report())
        
        # Generate and save chart
        chart_data = result.generate_chart()
        if chart_data:
            chart_dir = "backtest_charts"
            os.makedirs(chart_dir, exist_ok=True)
            
            chart_file = f"{chart_dir}/{ticker}_{strategy_name}_{start_date}_{end_date}.png"
            with open(chart_file, "wb") as f:
                f.write(base64.b64decode(chart_data))
                
            print(f"Chart saved to {chart_file}")
    
    return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python backtesting.py TICKER [STRATEGY] [START_DATE] [END_DATE] [TIMEFRAME]")
        print("Example: python backtesting.py AAPL rsi 2023-01-01 2023-12-31 1d")
        sys.exit(1)
    
    ticker = sys.argv[1]
    strategy = sys.argv[2] if len(sys.argv) > 2 else "combined"
    start = sys.argv[3] if len(sys.argv) > 3 else None
    end = sys.argv[4] if len(sys.argv) > 4 else None
    tf = sys.argv[5] if len(sys.argv) > 5 else "1d"
    
    cmd_backtest(ticker, strategy, start, end, tf)