import yfinance as yf
import pandas_ta as ta
from telegram import Update, InputFile, ReplyKeyboardMarkup, ReplyKeyboardRemove, KeyboardButton, ParseMode

# Handle different versions of python-telegram-bot
try:
    # Import from new version (v20+)
    from telegram.ext import Application, CommandHandler, CallbackContext, MessageHandler, filters, ConversationHandler
    # Create compatibility layer
    Filters = filters
except ImportError:
    # Import from older version (v13)
    from telegram.ext import Updater, CommandHandler, CallbackContext, MessageHandler, Filters, ConversationHandler
import logging
import os
import time
import threading
import json
from datetime import datetime
from collections import defaultdict
import sms_notifications
from sms_notifications import send_trading_signal_sms, send_price_alert_sms, send_portfolio_summary_sms
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import io
import base64
from PIL import Image

# Import visualization module
from visualization import generate_chart, save_chart

# Check if running in CI environment
IS_CI_ENV = os.environ.get('CI') == 'true'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import sentiment analysis module
from sentiment import get_sentiment_analysis

# Import strategies module
from strategies import get_strategy, update_strategy_params, get_available_strategies_info, AVAILABLE_STRATEGIES

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary to store user data (in memory) - user_id -> ticker -> buying_price
user_data = {}

# Dictionary to store user portfolio - user_id -> list of [ticker, price, quantity]
user_portfolios = defaultdict(list)

# Global dictionary to store user watchlists (user_id -> list of tickers)
user_watchlists = defaultdict(list)

# Dictionary to track when the last notification was sent for each user and ticker
last_notification_time = defaultdict(lambda: defaultdict(int))

# Files to persist user data
WATCHLIST_FILE = "user_watchlists.json"
PORTFOLIO_FILE = "user_portfolios.json"
PHONE_NUMBERS_FILE = "user_phone_numbers.json"

# Dictionary to store user phone numbers
user_phone_numbers = {}

# Conversation states for portfolio management
TICKER, PRICE, QUANTITY = range(3)

# Load watchlists from file if it exists
def load_watchlists():
    global user_watchlists
    try:
        if os.path.exists(WATCHLIST_FILE):
            with open(WATCHLIST_FILE, 'r') as f:
                watchlists = json.load(f)
                # Convert from regular dict to defaultdict
                user_watchlists = defaultdict(list)
                for user_id, tickers in watchlists.items():
                    user_watchlists[user_id] = tickers
            logger.info(f"Loaded watchlists for {len(user_watchlists)} users")
    except Exception as e:
        logger.error(f"Error loading watchlists: {e}")

# Save watchlists to file
def save_watchlists():
    try:
        with open(WATCHLIST_FILE, 'w') as f:
            # Convert defaultdict to regular dict for JSON serialization
            json.dump(dict(user_watchlists), f)
        logger.info(f"Saved watchlists for {len(user_watchlists)} users")
    except Exception as e:
        logger.error(f"Error saving watchlists: {e}")

# Load portfolios from file if it exists
def load_portfolios():
    global user_portfolios
    try:
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, 'r') as f:
                portfolios = json.load(f)
                # Convert from regular dict to defaultdict
                user_portfolios = defaultdict(list)
                for user_id, positions in portfolios.items():
                    user_portfolios[user_id] = positions
            logger.info(f"Loaded portfolios for {len(user_portfolios)} users")
    except Exception as e:
        logger.error(f"Error loading portfolios: {e}")

# Save portfolios to file
def save_portfolios():
    try:
        with open(PORTFOLIO_FILE, 'w') as f:
            # Convert defaultdict to regular dict for JSON serialization
            json.dump(dict(user_portfolios), f)
        logger.info(f"Saved portfolios for {len(user_portfolios)} users")
    except Exception as e:
        logger.error(f"Error saving portfolios: {e}")
        
# Load phone numbers from file if it exists
def load_phone_numbers():
    global user_phone_numbers
    try:
        if os.path.exists(PHONE_NUMBERS_FILE):
            with open(PHONE_NUMBERS_FILE, 'r') as f:
                user_phone_numbers = json.load(f)
            logger.info(f"Loaded phone numbers for {len(user_phone_numbers)} users")
    except Exception as e:
        logger.error(f"Error loading phone numbers: {e}")
        
# Save phone numbers to file
def save_phone_numbers():
    try:
        with open(PHONE_NUMBERS_FILE, 'w') as f:
            json.dump(user_phone_numbers, f)
        logger.info(f"Saved phone numbers for {len(user_phone_numbers)} users")
    except Exception as e:
        logger.error(f"Error saving phone numbers: {e}")

# Available timeframes for analysis
TIMEFRAMES = {
    "hourly": {"period": "1d", "interval": "1h", "description": "Hourly (last 24 hours)"},
    "daily": {"period": "1mo", "interval": "1d", "description": "Daily (last month)"},
    "weekly": {"period": "6mo", "interval": "1wk", "description": "Weekly (last 6 months)"},
    "monthly": {"period": "2y", "interval": "1mo", "description": "Monthly (last 2 years)"}
}

# Default timeframe
DEFAULT_TIMEFRAME = "hourly"

# Function to fetch stock data
def get_stock_data(ticker, period="1d", interval="1h", timeframe=None):
    try:
        # If timeframe is specified, use its settings
        if timeframe and timeframe in TIMEFRAMES:
            period = TIMEFRAMES[timeframe]["period"]
            interval = TIMEFRAMES[timeframe]["interval"]
        
        # Special handling for gold assets - use the same lists as in analyze_ticker function
        gold_commodities = ["XAU", "GC=F", "XAUUSD=X", "GC", "MGC=F"]  # Gold commodities
        gold_etfs = ["GLD", "IAU", "SGOL", "PHYS", "BAR"]  # Gold ETFs
        gold_miners = ["GOLD", "NEM", "AEM", "FNV", "WPM"]  # Gold mining companies
        
        is_gold_commodity = ticker.upper() in gold_commodities
        is_gold_etf = ticker.upper() in gold_etfs
        is_gold_miner = ticker.upper() in gold_miners
        
        logger.info(f"Fetching data for {ticker} with period={period}, interval={interval}")
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        # Check if data is empty
        if data.empty:
            logger.error(f"No data returned for ticker {ticker}")
            return None
            
        # Handle specific error cases
        if len(data) < 3:  # Need at least a few data points for indicators
            logger.warning(f"Insufficient data points for {ticker}: {len(data)}. Using a longer period.")
            # Try with a longer period
            data = stock.history(period="5d", interval=interval)
            if len(data) < 3:
                logger.error(f"Still insufficient data points for {ticker} after retry")
                return None
        
        # Tag with the appropriate asset type
        if is_gold_commodity:
            data.attrs['asset_type'] = 'gold_commodity'
            logger.info(f"Identified {ticker} as a gold commodity")
        elif is_gold_etf:
            data.attrs['asset_type'] = 'gold_etf'
            logger.info(f"Identified {ticker} as a gold ETF")
        elif is_gold_miner:
            data.attrs['asset_type'] = 'gold_miner'
            logger.info(f"Identified {ticker} as a gold mining stock")
        
        logger.info(f"Successfully fetched {len(data)} data points for {ticker}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None

# Function to calculate technical indicators
def calculate_indicators(data):
    try:
        # Check if we have enough data for the indicators
        if len(data) < 20:
            logger.warning(f"Not enough data points for reliable indicators: {len(data)} points, adjusting parameters")
            # Use smaller window sizes for limited data, but ensure they're viable
            # We need at least 2 data points for most indicators
            min_length = max(2, len(data) // 2)
            
            rsi_length = min(14, min_length)
            ema_short = min(9, min_length)
            ema_long = min(21, min_length)
            bb_length = min(20, min_length)
            # MACD needs at least fast+signal periods of data to work properly
            macd_fast = min(12, min_length // 2)
            macd_slow = min(26, min_length)
            macd_signal = min(9, min_length // 3)
            
            # Ensure we don't create invalid parameters
            if macd_slow <= macd_fast:
                macd_slow = macd_fast + 1
                
            stoch_k = min(14, min_length)
            stoch_d = min(3, max(2, min_length // 3))
            
            logger.info(f"Using adjusted parameters: RSI({rsi_length}), EMA({ema_short}/{ema_long}), BB({bb_length}), MACD({macd_fast}/{macd_slow}/{macd_signal}), Stoch({stoch_k}/{stoch_d})")
        else:
            rsi_length, ema_short, ema_long, bb_length = 14, 9, 21, 20
            macd_fast, macd_slow, macd_signal = 12, 26, 9
            stoch_k, stoch_d = 14, 3
        
        # Apply indicators only if we have enough data
        if len(data) >= 3:  # Absolute minimum for any technical indicator
            # RSI
            rsi = ta.rsi(data['Close'], length=rsi_length)
            if rsi is not None and not rsi.empty:
                data['RSI'] = rsi
            else:
                data['RSI'] = 50  # Neutral RSI

            # EMAs (short-term and long-term)
            ema9 = ta.ema(data['Close'], length=ema_short)
            if ema9 is not None and not ema9.empty:
                data['EMA_9'] = ema9
            else:
                data['EMA_9'] = data['Close']

            ema21 = ta.ema(data['Close'], length=ema_long)
            if ema21 is not None and not ema21.empty:
                data['EMA_21'] = ema21
            else:
                data['EMA_21'] = data['Close']
            
            # Only calculate longer EMAs if we have enough data
            if len(data) > 10:
                ema50 = ta.ema(data['Close'], length=min(50, len(data) // 2))
                if ema50 is not None and not ema50.empty:
                    data['EMA_50'] = ema50
                else:
                    data['EMA_50'] = data['Close']

            if len(data) > 20:
                ema200 = ta.ema(data['Close'], length=min(200, len(data) // 2))
                if ema200 is not None and not ema200.empty:
                    data['EMA_200'] = ema200
                else:
                    data['EMA_200'] = data['Close']
            
            # Bollinger Bands
            bbands = ta.bbands(data['Close'], length=bb_length)
            if bbands is not None and not bbands.empty:
                # The column names in the bbands dataframe will include the length parameter
                bb_upper_col = f"BBU_{bb_length}_2.0"
                bb_middle_col = f"BBM_{bb_length}_2.0"
                bb_lower_col = f"BBL_{bb_length}_2.0"
                
                # Log the available columns for debugging
                logger.info(f"Bollinger Bands columns: {bbands.columns.tolist()}")
                
                # Use dynamic column names based on the actual length used
                if bb_upper_col in bbands.columns and bb_middle_col in bbands.columns and bb_lower_col in bbands.columns:
                    data['BB_upper'] = bbands[bb_upper_col]
                    data['BB_middle'] = bbands[bb_middle_col]
                    data['BB_lower'] = bbands[bb_lower_col]
                else:
                    # If columns not found, provide fallback values
                    data['BB_upper'] = data['Close'] * 1.05  # 5% above close
                    data['BB_middle'] = data['Close']
                    data['BB_lower'] = data['Close'] * 0.95  # 5% below close
            else:
                # If bbands is None or empty, provide fallback values
                data['BB_upper'] = data['Close'] * 1.05  # 5% above close
                data['BB_middle'] = data['Close']
                data['BB_lower'] = data['Close'] * 0.95  # 5% below close
            
            # MACD - only if we have enough data points
            if len(data) > macd_slow:
                macd = ta.macd(data['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
                # Get the column names from the macd dataframe
                if macd is not None and not macd.empty and len(macd.columns) >= 3:
                    macd_col = macd.columns[0]  # MACD line
                    signal_col = macd.columns[1]  # Signal line
                    hist_col = macd.columns[2]  # Histogram
                    data['MACD'] = macd[macd_col]
                    data['MACD_Signal'] = macd[signal_col]
                    data['MACD_Hist'] = macd[hist_col]
                else:
                    # Fallback values
                    data['MACD'] = 0
                    data['MACD_Signal'] = 0
                    data['MACD_Hist'] = 0
            else:
                # Not enough data for MACD
                data['MACD'] = 0
                data['MACD_Signal'] = 0
                data['MACD_Hist'] = 0
            
            # Stochastic Oscillator
            if len(data) > stoch_k:
                stoch = ta.stoch(data['High'], data['Low'], data['Close'], k=stoch_k, d=stoch_d)
                if stoch is not None and not stoch.empty and len(stoch.columns) >= 2:
                    data['Stoch_K'] = stoch.iloc[:, 0]  # %K line
                    data['Stoch_D'] = stoch.iloc[:, 1]  # %D line
                else:
                    # Fallback values
                    data['Stoch_K'] = 50  # Neutral
                    data['Stoch_D'] = 50  # Neutral
            else:
                # Not enough data
                data['Stoch_K'] = 50  # Neutral
                data['Stoch_D'] = 50  # Neutral
            
            # Average Directional Index (ADX)
            # ADX requires more data to be reliable
            if len(data) >= 14:
                adx = ta.adx(data['High'], data['Low'], data['Close'], length=min(14, len(data) // 2))
                if adx is not None and not adx.empty and len(adx.columns) >= 3:
                    data['ADX'] = adx.iloc[:, 2]  # ADX line
                    data['DI+'] = adx.iloc[:, 0]  # +DI line
                    data['DI-'] = adx.iloc[:, 1]  # -DI line
                else:
                    # Fallback values
                    data['ADX'] = 20  # Neutral ADX
                    data['DI+'] = 20
                    data['DI-'] = 20
            else:
                # Not enough data
                data['ADX'] = 20  # Neutral ADX
                data['DI+'] = 20
                data['DI-'] = 20
        else:
            logger.warning(f"Too few data points ({len(data)}) to calculate any reliable indicators")
            # Create placeholder columns with the same values to avoid calculation errors
            data['RSI'] = 50  # Neutral RSI
            data['EMA_9'] = data['Close']
            data['EMA_21'] = data['Close']
            data['BB_upper'] = data['Close'] * 1.05  # 5% above close
            data['BB_middle'] = data['Close']
            data['BB_lower'] = data['Close'] * 0.95  # 5% below close
        
        # Fill any NaN values that might be present 
        data = data.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return data
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        # Instead of raising, provide a basic dataset with placeholders
        # to avoid crashing the application
        for col in ['RSI', 'EMA_9', 'EMA_21', 'BB_upper', 'BB_middle', 'BB_lower']:
            if col not in data.columns:
                if col == 'RSI':
                    data[col] = 50
                else:
                    data[col] = data['Close']
        
        return data

# Function to determine trend
def determine_trend(data):
    # Basic trend determination using short-term EMAs
    short_term_uptrend = data['EMA_9'].iloc[-1] > data['EMA_21'].iloc[-1]
    short_term_downtrend = data['EMA_9'].iloc[-1] < data['EMA_21'].iloc[-1]
    
    # Check for longer-term trends if data is available
    has_long_ema = all(col in data.columns for col in ['EMA_50', 'EMA_200'])
    long_term_uptrend = False
    long_term_downtrend = False
    
    if has_long_ema:
        long_term_uptrend = data['EMA_50'].iloc[-1] > data['EMA_200'].iloc[-1]
        long_term_downtrend = data['EMA_50'].iloc[-1] < data['EMA_200'].iloc[-1]
    
    # Check for MACD trend indications if available
    has_macd = all(col in data.columns for col in ['MACD', 'MACD_Signal'])
    macd_uptrend = False
    macd_downtrend = False
    
    if has_macd:
        macd_uptrend = data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1]
        macd_downtrend = data['MACD'].iloc[-1] < data['MACD_Signal'].iloc[-1]
    
    # ADX for trend strength if available
    has_adx = 'ADX' in data.columns
    strong_trend = False
    
    if has_adx:
        strong_trend = data['ADX'].iloc[-1] > 25
    
    # Determine the overall trend based on multiple indicators
    if short_term_uptrend and (not has_long_ema or long_term_uptrend) and (not has_macd or macd_uptrend):
        if has_adx and strong_trend:
            return "Strong Uptrend"
        return "Uptrend"
    elif short_term_downtrend and (not has_long_ema or long_term_downtrend) and (not has_macd or macd_downtrend):
        if has_adx and strong_trend:
            return "Strong Downtrend"
        return "Downtrend"
    elif has_long_ema:
        if long_term_uptrend and short_term_downtrend:
            return "Pullback in Uptrend"
        elif long_term_downtrend and short_term_uptrend:
            return "Relief Rally in Downtrend"
    
    # Default case
    return "Sideways/Neutral"

# Function to forecast the next 5 hours with enhanced accuracy
def forecast(data, steps=5):
    try:
        # Determine the best ARIMA parameters based on data characteristics
        from statsmodels.tsa.stattools import adfuller
        from statsmodels.tsa.arima.model import ARIMA
        import numpy as np
        
        # In CI environment, use simplified parameters to speed up testing
        if IS_CI_ENV:
            logger.info("CI environment detected, using simplified ARIMA(1,1,1) for faster testing")
            p, d, q = 1, 1, 1
        else:
            # Check for stationarity using Augmented Dickey-Fuller test
            result = adfuller(data['Close'].dropna())
            p_value = result[1]
            
            # Determine differencing parameter (d)
            if p_value < 0.05:
                # Already stationary
                d = 0
            else:
                # Need to difference
                d = 1
                
            # Determine autoregressive parameter (p) using simple autocorrelation estimation
            # More complex would be to use ACF and PACF plots, but this gives a reasonable estimate
            if len(data) > 20:
                # Compare correlation between original and lag
                orig = data['Close'][5:].values
                lagged = data['Close'][:-5].values
                correlation = np.corrcoef(orig, lagged)[0, 1]
                
                if abs(correlation) > 0.7:
                    p = 2  # Strong autocorrelation, use higher order
                else:
                    p = 1  # Weaker autocorrelation, use lower order
            else:
                p = 1  # Default for short series
                
            # Moving average component (q)
            q = 1  # Default
        
        logger.info(f"Using optimized ARIMA({p},{d},{q}) model for forecasting")
        
        # Fit the ARIMA model with optimized parameters
        model = ARIMA(data['Close'], order=(p, d, q))
        model_fit = model.fit()
        
        # Forecast the next 'steps' hours
        forecast_values = model_fit.forecast(steps=steps)
        forecast_list = forecast_values.tolist()
        
        # Simple forecast result with just the values
        return forecast_list
    except Exception as e:
        logger.error(f"Enhanced forecasting error: {e}")
        # Fallback to simple forecasting if enhanced method fails
        try:
            # Simple ARIMA(1,1,1) model as fallback
            model = ARIMA(data['Close'], order=(1, 1, 1))
            model_fit = model.fit()
            forecast_values = model_fit.forecast(steps=steps)
            return forecast_values.tolist()
        except Exception as e2:
            logger.error(f"Fallback forecasting error: {e2}")
            return None  # Return None if all forecasting fails

# Dictionary to store user strategies - user_id -> strategy_name
user_strategies = defaultdict(lambda: "combined")  # Default to combined strategy

# File to persist user strategies
STRATEGIES_FILE = "user_strategies.json"

# Function to load user strategies from file
def load_user_strategies():
    global user_strategies
    try:
        if os.path.exists(STRATEGIES_FILE):
            with open(STRATEGIES_FILE, 'r') as f:
                strategies = json.load(f)
                # Convert from regular dict to defaultdict
                user_strategies = defaultdict(lambda: "combined")
                for user_id, strategy in strategies.items():
                    user_strategies[user_id] = strategy
            logger.info(f"Loaded strategies for {len(user_strategies)} users")
    except Exception as e:
        logger.error(f"Error loading user strategies: {e}")

# Function to save user strategies to file
def save_user_strategies():
    try:
        with open(STRATEGIES_FILE, 'w') as f:
            # Convert defaultdict to regular dict for JSON serialization
            json.dump(dict(user_strategies), f)
        logger.info(f"Saved strategies preferences")
    except Exception as e:
        logger.error(f"Error saving user strategies: {e}")

# Function to generate entry/exit signals using selected strategy
def generate_signals(data, user_id, ticker):
    # Check if this is a gold-related asset
    gold_commodities = ["XAU", "GC=F", "SI=F", "GC", "SI", "HG=F", "HG", "MGC=F"]  # Gold/Silver/Copper commodities
    gold_etfs = ["GLD", "IAU", "GOLD", "SGOL", "PHYS", "BAR"]  # Gold ETFs
    
    is_gold_commodity = ticker.upper() in gold_commodities
    is_gold_etf = ticker.upper() in gold_etfs
    is_gold_asset = is_gold_commodity or is_gold_etf
    
    # Choose strategy based on asset type
    if is_gold_asset:
        strategy_name = "gold"
        strategy = get_strategy("gold")
        
        # Mark the data with detailed asset type for the strategy to use
        if is_gold_commodity:
            data.attrs['asset_type'] = 'gold_commodity'
            logger.info(f"Using Gold Strategy for {ticker} as it's a gold commodity")
        else:
            data.attrs['asset_type'] = 'gold_etf'
            logger.info(f"Using Gold Strategy for {ticker} as it's a gold-related ETF")
    else:
        # Get the user's preferred strategy for non-gold assets
        strategy_name = user_strategies[user_id]
        strategy = get_strategy(strategy_name)
        logger.info(f"Using {strategy.name} for user {user_id} on ticker {ticker}")
    
    # Generate signals using the selected strategy
    # Handle both old and new interface
    strategy_result = strategy.generate_signals(data, user_id, ticker, user_data)
    
    # Check if we got a tuple return (new interface) or list (old interface)
    if isinstance(strategy_result, tuple) and len(strategy_result) == 2:
        # New interface with (has_signals, signal_data)
        has_signals, signal_data = strategy_result
        
        # Convert signal data to readable signals
        signals = []
        if has_signals and signal_data.get('signal_type'):
            if signal_data.get('signal_type') == 'BUY':
                signals.append(f"🚀 Buy {ticker} at ${signal_data.get('price', data['Close'].iloc[-1]):.2f}")
                if 'indicators' in signal_data and is_gold_asset:
                    signals.append(f"📊 Gold-specific indicators detected: {signal_data['indicators']} bullish signals")
            elif signal_data.get('signal_type') == 'SELL':
                signals.append(f"📉 Sell {ticker} at ${signal_data.get('price', data['Close'].iloc[-1]):.2f}")
                if signal_data.get('reason') == 'take_profit' and 'profit_percent' in signal_data:
                    signals.append(f"💰 Take profit triggered at {signal_data['profit_percent']:.2f}%")
                elif signal_data.get('reason') == 'stop_loss' and 'loss_percent' in signal_data:
                    signals.append(f"🛑 Stop loss triggered at {signal_data['loss_percent']:.2f}%")
    else:
        # Old interface with just signals list
        signals = strategy_result
    
    # Return the signals
    return signals

# Function to set a user's preferred strategy
def set_user_strategy(user_id, strategy_name):
    if strategy_name in AVAILABLE_STRATEGIES:
        user_strategies[user_id] = strategy_name
        save_user_strategies()
        return True
    return False

# Function to get a user's preferred strategy
def get_user_strategy(user_id):
    return user_strategies[user_id]

# Function to handle the /start command
def start(update: Update, context: CallbackContext):
    user_name = update.message.from_user.first_name
    message_id = f"MSG-{int(time.time())}-START"
    welcome_message = f"ID: {message_id}\n*Welcome, Mr. Otmane!* I'm your Stock Prophet bot. Send me a stock ticker (e.g., *NVDA*) to get trading signals and forecasts."
    update.message.reply_text(welcome_message, parse_mode='Markdown')

# Function to analyze a stock ticker
def analyze_ticker(ticker, user_id="test_user", timeframe=None):
    try:
        # Generate a unique message ID (timestamp + ticker)
        message_id = f"MSG-{int(time.time())}-{ticker}"
        
        # If timeframe is specified, use it, otherwise use default
        if timeframe and timeframe in TIMEFRAMES:
            data = get_stock_data(ticker, timeframe=timeframe)
            timeframe_description = TIMEFRAMES[timeframe]["description"]
        else:
            data = get_stock_data(ticker)
            timeframe = DEFAULT_TIMEFRAME
            timeframe_description = TIMEFRAMES[DEFAULT_TIMEFRAME]["description"]
        
        # Check if we have valid data
        if data is None or data.empty:
            logger.error(f"No data returned for ticker {ticker}")
            return f"ID: {message_id}\nMr. Otmane, I couldn't retrieve data for {ticker}. The ticker may be invalid or there might be connection issues."
        
        # Log data shape for debugging
        logger.info(f"Retrieved data for {ticker} with shape: {data.shape}")
        
        try:
            # Get sentiment analysis for the ticker
            sentiment_summary, sentiment_data = get_sentiment_analysis(ticker)
            
            # Calculate technical indicators
            data = calculate_indicators(data)
            trend = determine_trend(data)
            
            # Get price forecast
            forecast_values = forecast(data)
            
            # Store the forecast values in the DataFrame's attrs (metadata)
            # This makes them accessible to strategy functions
            data.attrs['forecast_values'] = forecast_values
            
            # Check if this is a gold-related asset
            gold_commodities = ["XAU", "GC=F", "XAUUSD=X", "GC", "MGC=F"]  # Gold commodities
            gold_etfs = ["GLD", "IAU", "SGOL", "PHYS", "BAR"]  # Gold ETFs
            gold_miners = ["GOLD", "NEM", "AEM", "FNV", "WPM"]  # Gold mining companies
            
            is_gold_commodity = ticker.upper() in gold_commodities
            is_gold_etf = ticker.upper() in gold_etfs
            is_gold_miner = ticker.upper() in gold_miners
            is_gold_asset = is_gold_commodity or is_gold_etf  # Only use gold strategy for commodities and ETFs
            
            # Get the appropriate strategy (gold strategy for gold assets, user preference otherwise)
            if is_gold_asset:
                strategy_name = "gold"
                strategy = get_strategy("gold")
                
                # Mark the data with detailed asset type for the strategy to use
                if is_gold_commodity:
                    data.attrs['asset_type'] = 'gold_commodity'
                    logger.info(f"Using Gold Strategy for {ticker} as it's a gold commodity")
                else:
                    data.attrs['asset_type'] = 'gold_etf'
                    logger.info(f"Using Gold Strategy for {ticker} as it's a gold-related ETF")
            else:
                strategy_name = get_user_strategy(user_id)
                strategy = get_strategy(strategy_name)
            
            logger.info(f"Using {strategy.name} for user {user_id} on ticker {ticker}")
            
            # Generate signals with the selected strategy
            # Handle both old and new interface
            strategy_result = strategy.generate_signals(data, user_id, ticker)
            
            # Check if we got a tuple return (new interface) or list (old interface)
            if isinstance(strategy_result, tuple) and len(strategy_result) == 2:
                # New interface with (has_signals, signal_data)
                has_signals, signal_data = strategy_result
                
                # Convert signal data to readable signals
                signals = []
                if has_signals and signal_data.get('signal_type'):
                    if signal_data.get('signal_type') == 'BUY':
                        signals.append(f"🚀 Buy {ticker} at ${signal_data.get('price', data['Close'].iloc[-1]):.2f}")
                        if 'indicators' in signal_data and is_gold_asset:
                            signals.append(f"📊 Gold-specific indicators detected: {signal_data['indicators']} bullish signals")
                    elif signal_data.get('signal_type') == 'SELL':
                        signals.append(f"📉 Sell {ticker} at ${signal_data.get('price', data['Close'].iloc[-1]):.2f}")
                        if signal_data.get('reason') == 'take_profit' and 'profit_percent' in signal_data:
                            signals.append(f"💰 Take profit triggered at {signal_data['profit_percent']:.2f}%")
                        elif signal_data.get('reason') == 'stop_loss' and 'loss_percent' in signal_data:
                            signals.append(f"🛑 Stop loss triggered at {signal_data['loss_percent']:.2f}%")
            else:
                # Old interface with just signals list
                signals = strategy_result
            
            # Determine if price is forecasted to increase
            price_trend_up = False
            if forecast_values:
                current_price = data['Close'].iloc[-1]
                last_forecast = forecast_values[-1]
                price_trend_up = last_forecast > current_price
                price_change_pct = ((last_forecast / current_price) - 1) * 100
            
            # Add sentiment analysis to the response
            sentiment_summary, sentiment_data = get_sentiment_analysis(ticker)
            
            response = (
                f"ID: {message_id}\n"
                f"Mr. Otmane, here's your analysis for *{ticker}*:\n\n"
                f"{sentiment_summary}\n"
                f"📊 Trend: {trend}\n\n"
                f"📉 Technical Indicators:\n"
                f"RSI: {data['RSI'].iloc[-1]:.2f}\n"
                f"EMA (9): {data['EMA_9'].iloc[-1]:.2f}\n"
                f"EMA (21): {data['EMA_21'].iloc[-1]:.2f}\n"
            )
            
            # Add EMA 50/200 if we have enough data
            if 'EMA_50' in data.columns and 'EMA_200' in data.columns:
                response += f"EMA (50): {data['EMA_50'].iloc[-1]:.2f}\n"
                response += f"EMA (200): {data['EMA_200'].iloc[-1]:.2f}\n"
            
            response += f"Bollinger Bands (Upper/Middle/Lower): {data['BB_upper'].iloc[-1]:.2f}/{data['BB_middle'].iloc[-1]:.2f}/{data['BB_lower'].iloc[-1]:.2f}\n"
            
            # Add MACD if available
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                response += f"MACD: {data['MACD'].iloc[-1]:.4f} / Signal: {data['MACD_Signal'].iloc[-1]:.4f}\n"
            
            # Add Stochastic if available
            if 'Stoch_K' in data.columns and 'Stoch_D' in data.columns:
                response += f"Stochastic %K/%D: {data['Stoch_K'].iloc[-1]:.2f}/{data['Stoch_D'].iloc[-1]:.2f}\n"
            
            # Add ADX if available
            if 'ADX' in data.columns and 'DI+' in data.columns and 'DI-' in data.columns:
                response += f"ADX: {data['ADX'].iloc[-1]:.2f} (DI+: {data['DI+'].iloc[-1]:.2f} / DI-: {data['DI-'].iloc[-1]:.2f})\n"
            
            # Add price forecast if available
            if forecast_values:
                current_price = data['Close'].iloc[-1]
                last_forecast = forecast_values[-1]
                price_change_pct = ((last_forecast / current_price) - 1) * 100
                price_trend_up = last_forecast > current_price
                
                response += f"\n📈 Price Forecast (next 5 hours):\n"
                for i, price in enumerate(forecast_values):
                    response += f"{i+1}h: ${price:.2f}\n"
                
                # Add prediction-based trading guidance
                if price_trend_up:
                    response += f"\n📈 Price is predicted to rise by {price_change_pct:.2f}% over 5 hours.\n"
                    response += f"Standard trading thresholds applied (take profit: {strategy.parameters['take_profit']}%, stop loss: {strategy.parameters['stop_loss']}%).\n"
                else:
                    response += f"\n📉 Price is predicted to fall by {abs(price_change_pct):.2f}% over 5 hours.\n"
                    response += f"Conservative trading thresholds recommended (take profit: 1%, stop loss: 5%).\n"
            else:
                response += f"\n⚠️ Price forecast unavailable at the moment.\n"
            
            response += f"\n🚀 Trading Signals (using {strategy.name} Strategy):\n"
            
            # Add signals if any exist
            if signals:
                response += "\n".join(signals)
            else:
                response += "No strong signals at the moment."
            
            # Add advice on strategy
            if not signals:
                response += f"\n\nMr. Otmane, you might want to try a different strategy for {ticker}. Use /strategies to view options."
                
            return response
        except Exception as inner_e:
            logger.error(f"Error processing indicators for {ticker}: {inner_e}")
            return f"Mr. Otmane, there was an error processing indicators for {ticker}: {str(inner_e)}"
    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Mr. Otmane, I couldn't process your request for {ticker}. Error: {str(e)}"



# Function to handle the stock ticker input
def handle_ticker(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    
    # Extract ticker and timeframe from the message
    timeframe = None
    
    if update.message.text.startswith('/ticker'):
        message_id = f"MSG-{int(time.time())}-TICKER-ARGS"
        if not context.args:
            update.message.reply_text(f"ID: {message_id}\nMr. Otmane, please provide a ticker symbol. Example: /ticker AAPL")
            return
        ticker = context.args[0].upper()
        
        # Check if a timeframe was specified
        if len(context.args) > 1 and context.args[1].lower() in TIMEFRAMES:
            timeframe = context.args[1].lower()
    else:
        # Simple ticker message
        parts = update.message.text.split()
        ticker = parts[0].upper()
        
        # Check if a timeframe was specified
        if len(parts) > 1 and parts[1].lower() in TIMEFRAMES:
            timeframe = parts[1].lower()
    
    # Indicate that analysis is in progress
    message_id = f"MSG-{int(time.time())}-START"
    timeframe_text = f" with {TIMEFRAMES[timeframe]['description']}" if timeframe else ""
    update.message.reply_text(f"ID: {message_id}\nAnalyzing *{ticker}*{timeframe_text}... please wait, Mr. Otmane.")
    
    try:
        # Get stock data with specified timeframe
        if timeframe:
            data = get_stock_data(ticker, timeframe=timeframe)
        else:
            data = get_stock_data(ticker)
        
        # Check if we have valid data
        if data is None or data.empty:
            message_id = f"MSG-{int(time.time())}-ERROR"
            update.message.reply_text(f"ID: {message_id}\nMr. Otmane, I couldn't retrieve data for *{ticker}*. The ticker may be invalid or there might be connection issues.", parse_mode='Markdown')
            return
        
        # Calculate indicators
        data = calculate_indicators(data)
        
        # Get forecasts
        forecast_values = forecast(data)
        
        # Generate a chart
        try:
            # Create charts directory if it doesn't exist
            if not os.path.exists('charts'):
                os.makedirs('charts')
                
            # Generate chart with forecast
            chart_base64 = generate_chart(data, ticker, show_forecast=True, forecast_values=forecast_values)
            
            # Convert base64 to image file
            img_data = base64.b64decode(chart_base64)
            img_path = f"charts/{ticker}_{int(time.time())}.png"
            
            with open(img_path, 'wb') as f:
                f.write(img_data)
            
            # Send the chart to the user
            message_id = f"MSG-{int(time.time())}-CHART"
            with open(img_path, 'rb') as photo:
                update.message.reply_photo(
                    photo=photo,
                    caption=f"ID: {message_id}\nMr. Otmane, here's your technical analysis chart for *{ticker}*{timeframe_text}",
                    parse_mode='Markdown'
                )
            
            # Analyze the ticker - pass the user ID and timeframe
            response = analyze_ticker(ticker, user_id, timeframe)
            
            # Send the text analysis after the chart
            update.message.reply_text(response)
            
        except Exception as chart_error:
            logger.error(f"Error generating chart for {ticker}: {chart_error}")
            
            # If chart fails, still send the text analysis
            response = analyze_ticker(ticker, user_id, timeframe)
            update.message.reply_text(response)
            message_id = f"MSG-{int(time.time())}-CHART-ERROR"
            update.message.reply_text(f"ID: {message_id}\nMr. Otmane, I couldn't generate a chart visualization for *{ticker}* at this time.", parse_mode='Markdown')
    
    except Exception as e:
        logger.error(f"Error in handle_ticker: {e}")
        message_id = f"MSG-{int(time.time())}-ERROR"
        update.message.reply_text(f"ID: {message_id}\nMr. Otmane, there was an error analyzing *{ticker}*. Please try again later.", parse_mode='Markdown')

# Function to add a stock to the watchlist
def add_to_watchlist(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    
    # Check if ticker is provided
    message_id = f"MSG-{int(time.time())}-ADD-ARGS"
    if not context.args:
        update.message.reply_text(f"ID: {message_id}\nMr. Otmane, please provide a ticker symbol. Example: /add AAPL")
        return
    
    ticker = context.args[0].upper()
    
    # Special case for gold spot price
    gold_spot_tickers = ["XAU", "XAUUSD", "XAUUSD=X"]
    if ticker in gold_spot_tickers:
        # Use a consistent internal name for gold spot
        ticker = "XAUUSD"
        message_id = f"MSG-{int(time.time())}-ADD"
        if ticker not in user_watchlists[user_id]:
            user_watchlists[user_id].append(ticker)
            save_watchlists()
            update.message.reply_text(f"ID: {message_id}\nMr. Otmane, I've added Gold Spot (XAU/USD) to your watchlist! You'll receive regular updates on gold price every 30 minutes.")
        else:
            update.message.reply_text(f"ID: {message_id}\nMr. Otmane, Gold Spot (XAU/USD) is already in your watchlist.")
        return
    
    # Verify if the ticker exists by trying to fetch data
    data = get_stock_data(ticker)
    if data is None or data.empty:
        message_id = f"MSG-{int(time.time())}-ADD-INVALID"
        update.message.reply_text(f"ID: {message_id}\nMr. Otmane, I could not add *{ticker}* to your watchlist. The ticker may be invalid.", parse_mode='Markdown')
        return
    
    # Add to watchlist if not already there
    if ticker not in user_watchlists[user_id]:
        user_watchlists[user_id].append(ticker)
        save_watchlists()
        message_id = f"MSG-{int(time.time())}-ADD"
        update.message.reply_text(f"ID: {message_id}\nMr. Otmane, I've added *{ticker}* to your watchlist! You'll receive regular updates on this stock every 30 minutes.", parse_mode='Markdown')
    else:
        message_id = f"MSG-{int(time.time())}-EXISTS"
        update.message.reply_text(f"ID: {message_id}\nMr. Otmane, *{ticker}* is already in your watchlist.", parse_mode='Markdown')

# Function to remove a stock from the watchlist
def remove_from_watchlist(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    
    # Check if ticker is provided
    message_id = f"MSG-{int(time.time())}-REMOVE-ARGS"
    if not context.args:
        update.message.reply_text(f"ID: {message_id}\nMr. Otmane, please provide a ticker symbol. Example: /remove AAPL")
        return
    
    ticker = context.args[0].upper()
    
    # Handle special case for gold spot price
    gold_spot_tickers = ["XAU", "XAUUSD", "XAUUSD=X"]
    if ticker in gold_spot_tickers:
        # Check if the XAUUSD ticker is in the watchlist
        if "XAUUSD" in user_watchlists[user_id]:
            user_watchlists[user_id].remove("XAUUSD")
            save_watchlists()
            message_id = f"MSG-{int(time.time())}-REMOVE"
            update.message.reply_text(f"ID: {message_id}\nMr. Otmane, I've removed Gold Spot (XAU/USD) from your watchlist.", parse_mode='Markdown')
        else:
            message_id = f"MSG-{int(time.time())}-NOT-FOUND"
            update.message.reply_text(f"ID: {message_id}\nMr. Otmane, Gold Spot (XAU/USD) is not in your watchlist.", parse_mode='Markdown')
        return
    
    # Regular ticker handling
    # Remove from watchlist if present
    if ticker in user_watchlists[user_id]:
        user_watchlists[user_id].remove(ticker)
        save_watchlists()
        message_id = f"MSG-{int(time.time())}-REMOVE"
        update.message.reply_text(f"ID: {message_id}\nMr. Otmane, I've removed *{ticker}* from your watchlist.", parse_mode='Markdown')
    else:
        message_id = f"MSG-{int(time.time())}-NOT-FOUND"
        update.message.reply_text(f"ID: {message_id}\nMr. Otmane, *{ticker}* is not in your watchlist.", parse_mode='Markdown')

# Function to show the user's watchlist
def show_watchlist(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    
    # Display watchlist
    message_id = f"MSG-{int(time.time())}-WATCHLIST"
    if user_id in user_watchlists and user_watchlists[user_id]:
        formatted_tickers = []
        for ticker in user_watchlists[user_id]:
            # Special display for gold spot price
            if ticker == "XAUUSD":
                formatted_tickers.append("*Gold Spot (XAU/USD)*")
            else:
                formatted_tickers.append(f"*{ticker}*")
        tickers_str = ', '.join(formatted_tickers)
        update.message.reply_text(f"ID: {message_id}\nMr. Otmane, your watchlist: {tickers_str}\n\nYou'll receive automatic updates for these assets every 30 minutes.", parse_mode='Markdown')
    else:
        update.message.reply_text(f"ID: {message_id}\nMr. Otmane, your watchlist is empty. Add stocks with /add TICKER")

# Function to send automatic notifications for watchlist stocks
def send_watchlist_notifications(context: CallbackContext):
    now = int(time.time())  # Convert to integer to fix type error
    notification_interval = 30 * 60  # 30 minutes in seconds
    
    # For each user with a watchlist
    for user_id, tickers in user_watchlists.items():
        # For each ticker in the user's watchlist
        for ticker in tickers:
            # Check if it's time to send a notification (30 min interval)
            last_time = last_notification_time[user_id][ticker]
            if now - last_time >= notification_interval:
                try:
                    # Special case for gold spot price
                    if ticker == "XAUUSD":
                        # Import functions from gold spot module
                        from test_gold_spot import get_gold_spot_price, get_gold_stats
                        
                        # Get gold price
                        gold_price = get_gold_spot_price()
                        
                        if gold_price:
                            # Calculate gold investment stats
                            stats = get_gold_stats(gold_price)
                            
                            # Create message ID
                            message_id = f"MSG-{int(time.time())}-GOLD-SPOT"
                            
                            # Create formatted message
                            gold_response = f"ID: {message_id}\n"
                            gold_response += f"*Gold Spot Price (XAU/USD) Update*\n\n"
                            gold_response += f"💰 Current Price: *${gold_price:,.3f} USD* per troy ounce\n\n"
                            
                            # Add conversion rates
                            gold_response += f"*Conversion Rates:*\n"
                            gold_response += f"• 1 gram: ${stats['price_per_gram']:.2f}\n"
                            gold_response += f"• 1 kg: ${stats['price_per_kg']:.2f}\n\n"
                            
                            # Add trading information
                            gold_response += f"*Trading Information:*\n"
                            gold_response += f"• Take profit target (+1.5%): ${gold_price * 1.015:.2f}\n"
                            gold_response += f"• Stop loss level (-2.0%): ${gold_price * 0.98:.2f}\n\n"
                            
                            # Add ETF equivalent for reference
                            gold_response += f"*Market Equivalents:*\n"
                            gold_response += f"• GLD ETF equivalent: ~${gold_price/10:.2f} per share\n"
                            
                            # Send the message
                            context.bot.send_message(
                                chat_id=user_id, 
                                text=gold_response,
                                parse_mode='Markdown'
                            )
                            
                            # Update last notification time
                            last_notification_time[user_id][ticker] = now
                            logger.info(f"Sent Gold Spot price update to user {user_id}")
                            
                            # Check if user has a phone number for SMS gold price alerts
                            if user_id in user_phone_numbers:
                                phone_number = user_phone_numbers[user_id]
                                from sms_notifications import send_gold_price_sms
                                sms_sent = send_gold_price_sms(
                                    phone_number, 
                                    gold_price
                                )
                                if sms_sent:
                                    logger.info(f"Sent SMS gold price update to {user_id}")
                                else:
                                    logger.error(f"Failed to send SMS gold price update to {user_id}")
                            
                            continue  # Skip the regular stock analysis for gold spot
                    
                    # For regular tickers, get the stock data and analyze it
                    data = get_stock_data(ticker)
                    if data is None:
                        continue  # Skip if we can't get data
                    
                    # Calculate indicators
                    data = calculate_indicators(data)
                    
                    # Check if this is a gold-related asset
                    gold_commodities = ["XAU", "GC=F", "SI=F", "GC", "SI", "HG=F", "HG", "MGC=F"]  # Gold/Silver/Copper commodities
                    gold_etfs = ["GLD", "IAU", "GOLD", "SGOL", "PHYS", "BAR"]  # Gold ETFs
                    
                    is_gold_commodity = ticker.upper() in gold_commodities
                    is_gold_etf = ticker.upper() in gold_etfs
                    is_gold_asset = is_gold_commodity or is_gold_etf
                    
                    # Choose strategy based on asset type
                    if is_gold_asset:
                        strategy_name = "gold"
                        strategy = get_strategy("gold")
                        
                        # Mark the data with detailed asset type for the strategy to use
                        if is_gold_commodity:
                            data.attrs['asset_type'] = 'gold_commodity'
                            logger.info(f"Using Gold Strategy for {ticker} as it's a gold commodity")
                        else:
                            data.attrs['asset_type'] = 'gold_etf'
                            logger.info(f"Using Gold Strategy for {ticker} as it's a gold-related ETF")
                    else:
                        # Get the user's preferred strategy for non-gold assets
                        strategy_name = get_user_strategy(user_id)
                        strategy = get_strategy(strategy_name)
                        logger.info(f"Using {strategy.name} for user {user_id} on ticker {ticker}")
                    
                    # Generate signals for the ticker
                    # Handle both old and new interface
                    strategy_result = strategy.generate_signals(data, user_id, ticker)
                    
                    # Check if we got a tuple return (new interface) or list (old interface)
                    if isinstance(strategy_result, tuple) and len(strategy_result) == 2:
                        # New interface with (has_signals, signal_data)
                        has_signals, signal_data = strategy_result
                    else:
                        # Old interface with just signals list
                        signals = strategy_result
                        has_signals = bool(signals)  # True if any signals
                        signal_data = {}  # Empty dict for old interface
                    
                    # Get the analysis and send it - pass the user_id
                    analysis = analyze_ticker(ticker, user_id)
                    context.bot.send_message(chat_id=user_id, text=analysis)
                    
                    # Update the last notification time
                    last_notification_time[user_id][ticker] = now
                    logger.info(f"Sent automated update for {ticker} to user {user_id}")
                    
                    # Check if user has a phone number for SMS notifications
                    if user_id in user_phone_numbers and has_signals:
                        phone_number = user_phone_numbers[user_id]
                        current_price = data['Close'].iloc[-1]
                        
                        # Send SMS for buy/sell signals if there are any
                        if signal_data.get('signal_type') in ['BUY', 'SELL']:
                            sms_sent = send_trading_signal_sms(
                                phone_number,
                                ticker,
                                signal_data.get('signal_type'),
                                current_price,
                                strategy_name
                            )
                            if sms_sent:
                                logger.info(f"Sent SMS trading signal for {ticker} to {user_id}")
                            else:
                                logger.error(f"Failed to send SMS trading signal for {ticker} to {user_id}")
                        
                        # Send price alerts for significant price movements
                        elif signal_data.get('price_change', 0) >= 5.0:  # 5% or more price change
                            sms_sent = send_price_alert_sms(
                                phone_number,
                                ticker,
                                current_price,
                                signal_data.get('price_change', 0),
                                'SPIKE' if signal_data.get('price_change', 0) > 0 else 'DROP'
                            )
                            if sms_sent:
                                logger.info(f"Sent SMS price alert for {ticker} to {user_id}")
                            else:
                                logger.error(f"Failed to send SMS price alert for {ticker} to {user_id}")
                    
                except Exception as e:
                    logger.error(f"Error sending notification for {ticker} to user {user_id}: {e}")

# Function to show user's portfolio
def show_portfolio(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    
    message_id = f"MSG-{int(time.time())}-PORTFOLIO"
    if user_id not in user_portfolios or not user_portfolios[user_id]:
        update.message.reply_text(f"ID: {message_id}\nMr. Otmane, your portfolio is empty. Use /buy TICKER PRICE QUANTITY to add positions.")
        return
    
    total_value = 0
    total_cost = 0
    portfolio_text = "*Your Portfolio:*\n\n"
    
    for position in user_portfolios[user_id]:
        ticker, buy_price, quantity = position
        
        # Get current price
        try:
            data = get_stock_data(ticker)
            if data is None or data.empty:
                current_price = 0
                logger.error(f"Could not get current price for {ticker}")
            else:
                current_price = data['Close'].iloc[-1]
            
            position_value = current_price * quantity
            position_cost = float(buy_price) * float(quantity)
            profit_loss = position_value - position_cost
            profit_loss_pct = (profit_loss / position_cost) * 100 if position_cost > 0 else 0
            
            # Add to totals
            total_value += position_value
            total_cost += position_cost
            
            # Format the position information
            portfolio_text += f"*{ticker}*: {quantity} shares @ ${float(buy_price):.2f}\n"
            portfolio_text += f"Current Price: ${current_price:.2f}\n"
            portfolio_text += f"Value: ${position_value:.2f}\n"
            
            # Add profit/loss indicator
            if profit_loss >= 0:
                portfolio_text += f"P/L: +${profit_loss:.2f} (+{profit_loss_pct:.2f}%)\n\n"
            else:
                portfolio_text += f"P/L: -${abs(profit_loss):.2f} ({profit_loss_pct:.2f}%)\n\n"
                
        except Exception as e:
            logger.error(f"Error calculating portfolio value for {ticker}: {e}")
            portfolio_text += f"{ticker}: Error fetching current data\n\n"
    
    # Add portfolio summary
    total_profit_loss = total_value - total_cost
    total_profit_loss_pct = (total_profit_loss / total_cost) * 100 if total_cost > 0 else 0
    
    portfolio_text += "*Portfolio Summary:*\n"
    portfolio_text += f"Total Cost: ${total_cost:.2f}\n"
    portfolio_text += f"Total Value: ${total_value:.2f}\n"
    
    if total_profit_loss >= 0:
        portfolio_text += f"Total P/L: +${total_profit_loss:.2f} (+{total_profit_loss_pct:.2f}%)"
    else:
        portfolio_text += f"Total P/L: -${abs(total_profit_loss):.2f} ({total_profit_loss_pct:.2f}%)"
    
    portfolio_text = f"ID: {message_id}\n{portfolio_text}"
    update.message.reply_text(portfolio_text, parse_mode='Markdown')

# Function to add a stock to the portfolio
def add_to_portfolio(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    
    # Check if all arguments are provided (ticker, price, quantity)
    message_id = f"MSG-{int(time.time())}-BUY-ARGS"
    if len(context.args) < 3:
        update.message.reply_text(f"ID: {message_id}\nMr. Otmane, please provide ticker, price, and quantity. Example: /buy AAPL 150.00 10")
        return
    
    try:
        ticker = context.args[0].upper()
        price = float(context.args[1])
        quantity = float(context.args[2])
        
        # Verify the ticker exists
        data = get_stock_data(ticker)
        if data is None or data.empty:
            message_id = f"MSG-{int(time.time())}-BUY-INVALID"
            update.message.reply_text(f"ID: {message_id}\nMr. Otmane, I could not verify *{ticker}*. Please check the ticker symbol.", parse_mode='Markdown')
            return
        
        # Add to portfolio
        user_portfolios[user_id].append([ticker, price, quantity])
        save_portfolios()
        
        message_id = f"MSG-{int(time.time())}-BUY-SUCCESS"
        update.message.reply_text(f"ID: {message_id}\nMr. Otmane, I've added {quantity} shares of *{ticker}* at ${price:.2f} to your portfolio.", parse_mode='Markdown')
    
    except ValueError:
        message_id = f"MSG-{int(time.time())}-BUY-VALUE-ERROR"
        update.message.reply_text(f"ID: {message_id}\nMr. Otmane, the price or quantity you provided is invalid. Please use numbers only.")
    except Exception as e:
        logger.error(f"Error adding to portfolio: {e}")
        message_id = f"MSG-{int(time.time())}-BUY-ERROR"
        update.message.reply_text(f"ID: {message_id}\nError adding to portfolio: {str(e)}")

# Function to remove a stock from the portfolio
def remove_from_portfolio(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    
    # Check if ticker is provided
    message_id = f"MSG-{int(time.time())}-SELL-ARGS"
    if not context.args:
        update.message.reply_text(f"ID: {message_id}\nMr. Otmane, please provide a ticker symbol. Example: /sell AAPL")
        return
    
    ticker = context.args[0].upper()
    positions_to_remove = []
    
    # Find positions with this ticker
    if user_id in user_portfolios:
        for i, position in enumerate(user_portfolios[user_id]):
            if position[0] == ticker:
                positions_to_remove.append(i)
        
        # Remove positions in reverse order to avoid index issues
        for i in sorted(positions_to_remove, reverse=True):
            removed = user_portfolios[user_id].pop(i)
            message_id = f"MSG-{int(time.time())}-SELL-SUCCESS"
            update.message.reply_text(f"ID: {message_id}\nMr. Otmane, I've removed {removed[2]} shares of *{removed[0]}* from your portfolio.", parse_mode='Markdown')
        
        save_portfolios()
        
        # If no positions were found
        if not positions_to_remove:
            message_id = f"MSG-{int(time.time())}-SELL-NOT-FOUND"
            update.message.reply_text(f"ID: {message_id}\nMr. Otmane, no *{ticker}* positions were found in your portfolio.", parse_mode='Markdown')
    else:
        message_id = f"MSG-{int(time.time())}-SELL-EMPTY"
        update.message.reply_text(f"ID: {message_id}\nMr. Otmane, your portfolio is empty. Use /buy TICKER PRICE QUANTITY to add positions.")

# Function to get strategy information
def get_strategy_info(update: Update, context: CallbackContext):
    strategies_info = get_available_strategies_info()
    user_id = str(update.message.from_user.id)
    current_strategy = get_user_strategy(user_id)
    
    message_id = f"MSG-{int(time.time())}-STRATEGIES"
    response = f"ID: {message_id}\n*Available Trading Strategies:*\n\n"
    for name, description in strategies_info.items():
        if name == current_strategy:
            response += f"✅ *{name}* (ACTIVE): {description}\n\n"
        else:
            response += f"• *{name}*: {description}\n\n"
    
    response += "\nMr. Otmane, use /strategy STRATEGY_NAME to change your active strategy."
    update.message.reply_text(response, parse_mode='Markdown')

# Function to set strategy
def set_strategy(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    
    # Check if strategy name is provided
    message_id = f"MSG-{int(time.time())}-STRATEGY-ARGS"
    if not context.args:
        update.message.reply_text(f"ID: {message_id}\nMr. Otmane, please provide a strategy name. Example: /strategy rsi\nUse /strategies to see available options.")
        return
    
    strategy_name = context.args[0].lower()
    
    # Update the user's strategy
    if set_user_strategy(user_id, strategy_name):
        message_id = f"MSG-{int(time.time())}-STRATEGY-SUCCESS"
        update.message.reply_text(f"ID: {message_id}\nMr. Otmane, your trading strategy has been updated to: *{strategy_name}*", parse_mode='Markdown')
    else:
        message_id = f"MSG-{int(time.time())}-STRATEGY-INVALID"
        update.message.reply_text(f"ID: {message_id}\nMr. Otmane, sorry, '*{strategy_name}*' is not a valid strategy. Use /strategies to see available options.", parse_mode='Markdown')

# Function to display available commands
# Function to check gold spot price
def check_gold_price(update: Update, context: CallbackContext):
    """Command handler for /gold command to get the current gold spot price"""
    from test_gold_spot import get_gold_spot_price, get_gold_stats
    
    user_id = str(update.message.from_user.id)
    message_id = f"MSG-{int(time.time())}-GOLD"
    
    try:
        # Send initial message to user
        update.message.reply_text(f"ID: {message_id}\nMr. Otmane, I'm retrieving the current gold spot price for you...")
        
        # Get gold price (will use cached price if available and not expired)
        gold_price = get_gold_spot_price()
        
        if gold_price:
            # Calculate gold investment stats
            stats = get_gold_stats(gold_price)
            
            # Create formatted message
            gold_response = f"ID: {message_id}\n"
            gold_response += f"*Gold Spot Price (XAU/USD)*\n\n"
            gold_response += f"💰 Current Price: *${gold_price:,.3f} USD* per troy ounce\n\n"
            
            # Add conversion rates
            gold_response += f"*Conversion Rates:*\n"
            gold_response += f"• 1 gram: ${stats['price_per_gram']:.2f}\n"
            gold_response += f"• 1 kg: ${stats['price_per_kg']:.2f}\n\n"
            
            # Add trading information
            gold_response += f"*Trading Information:*\n"
            gold_response += f"• Take profit target (+1.5%): ${gold_price * 1.015:.2f}\n"
            gold_response += f"• Stop loss level (-2.0%): ${gold_price * 0.98:.2f}\n"
            gold_response += f"• Gold ETF (GLD) equivalent: ~${gold_price/10:.2f} per share\n\n"
            
            # Add additional note
            gold_response += "For technical analysis, you can use `/ticker GLD` for the Gold ETF."
            
            # Send the response
            update.message.reply_text(gold_response, parse_mode='Markdown')
            
            logger.info(f"Sent gold price information to user {user_id}")
        else:
            update.message.reply_text(f"ID: {message_id}\nMr. Otmane, I couldn't retrieve the current gold price. Please try again later or check if the Alpha Vantage API key is properly set up.")
    
    except Exception as e:
        logger.error(f"Error checking gold price: {e}")
        update.message.reply_text(f"ID: {message_id}\nMr. Otmane, I encountered an error while retrieving the gold price: {str(e)}")

# Function to set user's phone number for SMS alerts
def set_phone_number(update: Update, context: CallbackContext):
    """Command handler for /sms command to set user's phone number for SMS alerts"""
    user = update.effective_user
    user_id = str(user.id)
    message_id = f"MSG-{int(time.time())}-SMS"
    
    # Check if the user provided a phone number
    if not context.args or len(context.args) != 1:
        update.message.reply_text(
            f"ID: {message_id}\nMr. Otmane, please provide a valid phone number in international format.\n"
            f"Example: /sms +1234567890", 
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Get the phone number from the command
    phone_number = context.args[0].strip()
    
    # Simple validation - must start with + and have at least 10 digits
    if not (phone_number.startswith('+') and len(phone_number) >= 10 and phone_number[1:].isdigit()):
        update.message.reply_text(
            f"ID: {message_id}\nMr. Otmane, the phone number format is invalid.\n"
            f"Please use international format starting with + (e.g., +1234567890).",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Save the phone number
    user_phone_numbers[user_id] = phone_number
    save_phone_numbers()
    
    # Test if Twilio is configured
    twilio_ready = sms_notifications.is_twilio_configured()
    
    if twilio_ready:
        # Send a test message if possible
        test_sent = sms_notifications.test_sms_service(phone_number)
        if test_sent:
            success_message = (
                f"ID: {message_id}\nMr. Otmane, your phone number *{phone_number}* has been saved successfully.\n"
                f"A test message has been sent to verify the service."
            )
        else:
            success_message = (
                f"ID: {message_id}\nMr. Otmane, your phone number *{phone_number}* has been saved successfully.\n"
                f"However, there was an issue sending the test SMS. Your settings are saved, but SMS delivery may not work."
            )
    else:
        success_message = (
            f"ID: {message_id}\nMr. Otmane, your phone number *{phone_number}* has been saved successfully.\n"
            f"However, SMS notifications are not available because Twilio is not fully configured."
        )
    
    update.message.reply_text(success_message, parse_mode=ParseMode.MARKDOWN)

def help_command(update: Update, context: CallbackContext):
    # Create a list of timeframes for the help text
    timeframe_help = ""
    for name, details in TIMEFRAMES.items():
        timeframe_help += f"• {name} - {details['description']}\n"
    
    message_id = f"MSG-{int(time.time())}-HELP"
    help_text = f"""
ID: {message_id}
*Welcome, Mr. Otmane!*

*Stock Prophet Bot Commands*

/start - Start the bot and get a welcome message
/help - Show this help message
/ticker SYMBOL [TIMEFRAME] - Analyze a stock ticker (e.g., /ticker AAPL daily)
/gold - Get current gold spot price (XAU/USD)

*Timeframe Options:*
{timeframe_help}
Example: AAPL daily - Analyzes Apple stock with daily data

*Watchlist Commands:*
/add SYMBOL - Add a stock to your watchlist
/remove SYMBOL - Remove a stock from your watchlist
/watchlist - View your current watchlist

*Portfolio Commands:*
/portfolio - Show your current portfolio and performance
/buy TICKER PRICE QUANTITY - Add a stock to your portfolio
/sell TICKER - Remove a stock from your portfolio

*Strategy Commands:*
/strategies - View all available trading strategies
/strategy NAME - Set your preferred trading strategy (e.g., /strategy macd)

*Gold Trading:*
The bot will automatically use a specialized Gold Strategy for gold-related assets (GLD, IAU, GOLD, etc.)

*SMS Notifications:*
/sms PHONE_NUMBER - Set your phone number for SMS alerts (e.g., /sms +1234567890)

*Backtest Command:*
/backtest TICKER STRATEGY START_DATE END_DATE TIMEFRAME - Test a strategy on historical data

You can also just send a ticker symbol directly without the /ticker command.

*About Stock Prophet*
This bot analyzes stocks using technical indicators and helps you make informed trading decisions.

*Technical Indicators Used:*
• RSI (Relative Strength Index)
• EMAs (9, 21, 50, 200 periods)
• Bollinger Bands
• MACD (Moving Average Convergence Divergence)
• Stochastic Oscillator
• ADX (Average Directional Index)

*Price Forecasting:*
• Uses adaptive ARIMA model for accurate predictions
• Generates visual chart with price projections
• Forecasts price movement for the next 4 hours
• Automatic updates every 30 minutes for your watchlist stocks

*Trading Strategies:*
• RSI Strategy - Uses oversold/overbought conditions
• Bollinger Bands Strategy - Mean reversion trading
• MACD Strategy - Trend following with crossovers
• Combined Strategy - Multiple indicators for stronger confirmation

The bot uses a configurable take profit / stop loss when positions are opened.
    """
    update.message.reply_text(help_text, parse_mode='Markdown')

# Function to set bot commands for the Telegram bot
def set_bot_commands(updater):
    """Register commands with Telegram to show in the UI when typing /"""
    try:
        from telegram import BotCommand
        commands = [
            BotCommand("start", "Start the bot and get a welcome message"),
            BotCommand("help", "Show help with all available commands"),
            BotCommand("ticker", "Analyze a stock ticker (e.g., /ticker AAPL)"),
            BotCommand("gold", "Get current gold spot price (XAU/USD)"),
            BotCommand("add", "Add a stock to your watchlist"),
            BotCommand("remove", "Remove a stock from your watchlist"),
            BotCommand("watchlist", "View your current watchlist"),
            BotCommand("portfolio", "Show your current portfolio and performance"),
            BotCommand("buy", "Add a stock to your portfolio (e.g., /buy AAPL 150.00 10)"),
            BotCommand("sell", "Remove a stock from your portfolio"),
            BotCommand("strategies", "View all available trading strategies"),
            BotCommand("strategy", "Set your preferred trading strategy"),
            BotCommand("sms", "Set your phone number for SMS alerts (e.g., /sms +1234567890)")
        ]
        
        updater.bot.set_my_commands(commands)
        logger.info("Successfully registered commands with Telegram")
        return True
    except Exception as cmd_error:
        logger.error(f"Error registering commands: {cmd_error}")
        return False

# Main function to run the bot
def run_telegram_bot(is_heroku=False, port=None, url=None):
    try:
        # Check for environment variable first
        import os
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
        
        if token == "YOUR_TELEGRAM_BOT_TOKEN":
            print("Warning: Using placeholder token. Bot will not connect to Telegram.")
            print("Set the TELEGRAM_BOT_TOKEN environment variable to use a real bot.")
            # Instead of failing, we'll use the test_mode
            return False
            
        # Load any saved watchlists, portfolios, and strategies
        load_watchlists()
        load_portfolios()
        load_user_strategies()
        load_phone_numbers()
            
        updater = Updater(token)
        dispatcher = updater.dispatcher
        job_queue = updater.job_queue

        # Add command handlers
        dispatcher.add_handler(CommandHandler("start", start))
        dispatcher.add_handler(CommandHandler("help", help_command))
        dispatcher.add_handler(CommandHandler("ticker", handle_ticker))
        
        # Add gold price command handler
        dispatcher.add_handler(CommandHandler("gold", check_gold_price))
        
        # Add watchlist command handlers
        dispatcher.add_handler(CommandHandler("add", add_to_watchlist))
        dispatcher.add_handler(CommandHandler("remove", remove_from_watchlist))
        dispatcher.add_handler(CommandHandler("watchlist", show_watchlist))
        
        # Add portfolio command handlers
        dispatcher.add_handler(CommandHandler("portfolio", show_portfolio))
        dispatcher.add_handler(CommandHandler("buy", add_to_portfolio))
        dispatcher.add_handler(CommandHandler("sell", remove_from_portfolio))
        
        # Add strategy command handlers
        dispatcher.add_handler(CommandHandler("strategies", get_strategy_info))
        dispatcher.add_handler(CommandHandler("strategy", set_strategy))
        
        # Add SMS notifications command handler
        dispatcher.add_handler(CommandHandler("sms", set_phone_number))
        
        # Add a handler for direct ticker input (no command)
        dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_ticker))
        
        # Set up automatic notifications job (runs every 30 seconds to check tickers)
        job_queue.run_repeating(send_watchlist_notifications, interval=30, first=30)
        logger.info("Set up automatic notifications job (checking every 30 seconds)")

        # Set bot commands - this will make commands appear in the Telegram UI when typing /
        set_bot_commands(updater)

        # Start the bot
        print("Starting Telegram bot with token:", token[:5] + "..." + token[-5:])
        
        if is_heroku and url:
            # Running on Heroku with webhook
            print(f"Setting up webhook on {url}")
            updater.start_webhook(listen="0.0.0.0",
                                 port=int(port),
                                 url_path=token,
                                 webhook_url=f"{url}/{token}")
            print(f"Bot is running on Heroku with webhook mode! URL: {url}")
        else:
            # Standard polling mode
            print("Bot is ready! Try sending it a message or the /start command in Telegram.")
            updater.start_polling()
            
        updater.idle()
        return True
    except Exception as e:
        print(f"Error starting Telegram bot: {e}")
        return False

# Function to run in test mode
def test_mode():
    print("Running in test mode without Telegram bot")
    print("Enter a stock ticker (e.g., AAPL, MSFT, GOOG) or 'quit' to exit:")
    user_id = "test_user"  # Default test user ID
    
    while True:
        ticker = input("> ").strip().upper()
        if ticker == "QUIT":
            break
        
        if ticker:
            print("\nAnalyzing ticker:", ticker)
            result = analyze_ticker(ticker, user_id)
            print("\n" + result + "\n")

# Main entry point
def main():
    # Load user phone numbers for SMS notifications
    load_phone_numbers()
    
    # Load watchlists and portfolios
    load_watchlists()
    load_portfolios()
    
    # Try to run the Telegram bot
    if not run_telegram_bot():
        # If Telegram bot fails, run in test mode
        test_mode()

if __name__ == "__main__":
    main()

