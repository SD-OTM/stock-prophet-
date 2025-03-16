import yfinance as yf
import pandas_ta as ta
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext, MessageHandler, Filters, ConversationHandler
import logging
import os
import time
import threading
import json
from datetime import datetime
from collections import defaultdict
from statsmodels.tsa.arima.model import ARIMA

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

# Function to fetch stock data
def get_stock_data(ticker, period="1d", interval="1h"):
    try:
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
            data['RSI'] = ta.rsi(data['Close'], length=rsi_length)
            
            # EMAs (short-term and long-term)
            data['EMA_9'] = ta.ema(data['Close'], length=ema_short)
            data['EMA_21'] = ta.ema(data['Close'], length=ema_long)
            
            # Only calculate longer EMAs if we have enough data
            if len(data) > 10:
                data['EMA_50'] = ta.ema(data['Close'], length=min(50, len(data) // 2))
            if len(data) > 20:
                data['EMA_200'] = ta.ema(data['Close'], length=min(200, len(data) // 2))
            
            # Bollinger Bands
            bbands = ta.bbands(data['Close'], length=bb_length)
            if not bbands.empty:
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
            
            # MACD - only if we have enough data points
            if len(data) > macd_slow:
                macd = ta.macd(data['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
                # Get the column names from the macd dataframe
                if not macd.empty and len(macd.columns) >= 3:
                    macd_col = macd.columns[0]  # MACD line
                    signal_col = macd.columns[1]  # Signal line
                    hist_col = macd.columns[2]  # Histogram
                    data['MACD'] = macd[macd_col]
                    data['MACD_Signal'] = macd[signal_col]
                    data['MACD_Hist'] = macd[hist_col]
            
            # Stochastic Oscillator
            if len(data) > stoch_k:
                stoch = ta.stoch(data['High'], data['Low'], data['Close'], k=stoch_k, d=stoch_d)
                if not stoch.empty and len(stoch.columns) >= 2:
                    data['Stoch_K'] = stoch.iloc[:, 0]  # %K line
                    data['Stoch_D'] = stoch.iloc[:, 1]  # %D line
            
            # Average Directional Index (ADX)
            # ADX requires more data to be reliable
            if len(data) >= 14:
                adx = ta.adx(data['High'], data['Low'], data['Close'], length=min(14, len(data) // 2))
                if not adx.empty and len(adx.columns) >= 3:
                    data['ADX'] = adx.iloc[:, 2]  # ADX line
                    data['DI+'] = adx.iloc[:, 0]  # +DI line
                    data['DI-'] = adx.iloc[:, 1]  # -DI line
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

# Function to forecast the next 5 hours
def forecast(data, steps=5):
    try:
        # Fit an ARIMA model to the closing prices
        model = ARIMA(data['Close'], order=(1, 1, 1))  # ARIMA(1,1,1) model
        model_fit = model.fit()
        # Forecast the next 'steps' hours
        forecast_values = model_fit.forecast(steps=steps)
        return forecast_values.tolist()  # Convert to a list for easier formatting
    except Exception as e:
        logger.error(f"Forecasting error: {e}")
        return None  # Return None if forecasting fails

# Function to generate entry/exit signals
def generate_signals(data, user_id, ticker):
    latest = data.iloc[-1]
    signals = []
    
    # Check if we have a complete dataset with all indicators
    has_macd = all(col in data.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist'])
    has_stoch = all(col in data.columns for col in ['Stoch_K', 'Stoch_D'])
    has_adx = all(col in data.columns for col in ['ADX', 'DI+', 'DI-'])
    has_ema_long = all(col in data.columns for col in ['EMA_50', 'EMA_200'])

    # Check if the user has an open position for this ticker
    if user_id in user_data and ticker in user_data[user_id]:
        buying_price = user_data[user_id][ticker]
        
        # Sell Signal (Take Profit or Stop Loss)
        if (latest['Close'] >= buying_price * 1.02) or (latest['Close'] <= buying_price * 0.98):  # 2% profit or 2% loss
            signals.append(f"📉 Sell {ticker} at {latest['Close']:.2f} (Take profit or stop loss triggered).")
            del user_data[user_id][ticker]  # Close the position
        
        # Additional sell signals based on technical indicators
        elif latest['RSI'] > 70:  # Overbought
            signals.append(f"📉 Consider selling {ticker} at {latest['Close']:.2f} (RSI indicates overbought conditions).")
        
        elif has_macd and latest['MACD'] < latest['MACD_Signal'] and latest['MACD_Hist'] < 0:  # MACD bearish crossover
            signals.append(f"📉 Consider selling {ticker} at {latest['Close']:.2f} (MACD bearish crossover).")
        
        elif has_stoch and latest['Stoch_K'] > 80 and latest['Stoch_D'] > 80:  # Stochastic overbought
            signals.append(f"📉 Consider selling {ticker} at {latest['Close']:.2f} (Stochastic indicates overbought conditions).")
    
    else:
        # Original buy signal
        rsi_oversold = latest['RSI'] < 30
        price_above_bb_lower = latest['Close'] > latest['BB_lower']
        short_term_uptrend = latest['EMA_9'] > latest['EMA_21']
        
        # Additional buy signals
        golden_cross = has_ema_long and latest['EMA_50'] > latest['EMA_200']
        macd_bullish = has_macd and latest['MACD'] > latest['MACD_Signal'] and latest['MACD_Hist'] > 0
        stoch_bullish = has_stoch and latest['Stoch_K'] < 20 and latest['Stoch_D'] < 20 and latest['Stoch_K'] > latest['Stoch_D']
        adx_strong_trend = has_adx and latest['ADX'] > 25 and latest['DI+'] > latest['DI-']
        
        # Combine signals for stronger buy recommendation
        # Classic RSI + Bollinger Bands + EMA strategy
        if rsi_oversold and price_above_bb_lower and short_term_uptrend:
            signals.append(f"🚀 Buy {ticker} at {latest['Close']:.2f} (Oversold, price above lower Bollinger Band, and uptrend).")
            # Store the buying price
            if user_id not in user_data:
                user_data[user_id] = {}
            user_data[user_id][ticker] = latest['Close']
        
        # MACD + Stochastic + ADX strategy
        elif has_macd and has_stoch and has_adx and macd_bullish and stoch_bullish and adx_strong_trend:
            signals.append(f"🚀 Buy {ticker} at {latest['Close']:.2f} (MACD bullish, Stochastic oversold with bullish crossover, strong trend).")
            # Store the buying price
            if user_id not in user_data:
                user_data[user_id] = {}
            user_data[user_id][ticker] = latest['Close']
        
        # Golden cross (longer-term bullish signal)
        elif golden_cross and short_term_uptrend and price_above_bb_lower:
            signals.append(f"🚀 Buy {ticker} at {latest['Close']:.2f} (Golden cross detected, price in uptrend and above lower Bollinger Band).")
            # Store the buying price
            if user_id not in user_data:
                user_data[user_id] = {}
            user_data[user_id][ticker] = latest['Close']
    
    return signals

# Function to handle the /start command
def start(update: Update, context: CallbackContext):
    update.message.reply_text("Welcome! Send me a stock ticker (e.g., NVDA) to get trading signals.")

# Function to analyze a stock ticker
def analyze_ticker(ticker):
    try:
        data = get_stock_data(ticker)
        
        # Check if we have valid data
        if data is None or data.empty:
            logger.error(f"No data returned for ticker {ticker}")
            return f"Sorry, I couldn't retrieve data for {ticker}. The ticker may be invalid or there might be connection issues."
        
        # Log data shape for debugging
        logger.info(f"Retrieved data for {ticker} with shape: {data.shape}")
        
        try:
            data = calculate_indicators(data)
            trend = determine_trend(data)
            signals = generate_signals(data, "test_user", ticker)
            
            # Get price forecast
            forecast_values = forecast(data)

            response = (
                f"📊 Trend for {ticker}: {trend}\n\n"
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
                response += f"\n📈 Price Forecast for {ticker} (next 5 hours):\n"
                for i, price in enumerate(forecast_values):
                    response += f"{i+1}h: {price:.2f}\n"
            else:
                response += f"\n⚠️ Price forecast unavailable at the moment.\n"
            
            response += "\n🚀 Trading Signals:\n"
            
            # Add signals if any exist
            if signals:
                response += "\n".join(signals)
            else:
                response += "No strong signals at the moment."
                
            return response
        except Exception as inner_e:
            logger.error(f"Error processing indicators for {ticker}: {inner_e}")
            return f"Error processing indicators for {ticker}: {str(inner_e)}"
    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Sorry, I couldn't process your request for {ticker}. Error: {str(e)}"



# Function to handle the stock ticker input
def handle_ticker(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    
    # Extract ticker from the message
    if update.message.text.startswith('/ticker'):
        if not context.args:
            update.message.reply_text("Please provide a ticker symbol. Example: /ticker AAPL")
            return
        ticker = context.args[0].upper()
    else:
        ticker = update.message.text.upper()
    
    # Analyze the ticker
    response = analyze_ticker(ticker)
    update.message.reply_text(response)

# Function to add a stock to the watchlist
def add_to_watchlist(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    
    # Check if ticker is provided
    if not context.args:
        update.message.reply_text("Please provide a ticker symbol. Example: /add AAPL")
        return
    
    ticker = context.args[0].upper()
    
    # Verify if the ticker exists by trying to fetch data
    data = get_stock_data(ticker)
    if data is None or data.empty:
        update.message.reply_text(f"Could not add {ticker} to watchlist. The ticker may be invalid.")
        return
    
    # Add to watchlist if not already there
    if ticker not in user_watchlists[user_id]:
        user_watchlists[user_id].append(ticker)
        save_watchlists()
        update.message.reply_text(f"Added {ticker} to your watchlist! You'll receive regular updates on this stock.")
    else:
        update.message.reply_text(f"{ticker} is already in your watchlist.")

# Function to remove a stock from the watchlist
def remove_from_watchlist(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    
    # Check if ticker is provided
    if not context.args:
        update.message.reply_text("Please provide a ticker symbol. Example: /remove AAPL")
        return
    
    ticker = context.args[0].upper()
    
    # Remove from watchlist if present
    if ticker in user_watchlists[user_id]:
        user_watchlists[user_id].remove(ticker)
        save_watchlists()
        update.message.reply_text(f"Removed {ticker} from your watchlist.")
    else:
        update.message.reply_text(f"{ticker} is not in your watchlist.")

# Function to show the user's watchlist
def show_watchlist(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    
    # Display watchlist
    if user_id in user_watchlists and user_watchlists[user_id]:
        tickers = ', '.join(user_watchlists[user_id])
        update.message.reply_text(f"Your watchlist: {tickers}\n\nYou'll receive automatic forecasts for these stocks every 30 minutes.")
    else:
        update.message.reply_text("Your watchlist is empty. Add stocks with /add TICKER")

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
                    # Get the analysis and send it
                    analysis = analyze_ticker(ticker)
                    context.bot.send_message(chat_id=user_id, text=analysis)
                    
                    # Update the last notification time
                    last_notification_time[user_id][ticker] = now
                    logger.info(f"Sent automated update for {ticker} to user {user_id}")
                except Exception as e:
                    logger.error(f"Error sending notification for {ticker} to user {user_id}: {e}")

# Function to show user's portfolio
def show_portfolio(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    
    if user_id not in user_portfolios or not user_portfolios[user_id]:
        update.message.reply_text("Your portfolio is empty. Use /buy TICKER PRICE QUANTITY to add positions.")
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
    
    update.message.reply_text(portfolio_text, parse_mode='Markdown')

# Function to add a stock to the portfolio
def add_to_portfolio(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    
    # Check if all arguments are provided (ticker, price, quantity)
    if len(context.args) < 3:
        update.message.reply_text("Please provide ticker, price, and quantity. Example: /buy AAPL 150.00 10")
        return
    
    try:
        ticker = context.args[0].upper()
        price = float(context.args[1])
        quantity = float(context.args[2])
        
        # Verify the ticker exists
        data = get_stock_data(ticker)
        if data is None or data.empty:
            update.message.reply_text(f"Could not verify {ticker}. Please check the ticker symbol.")
            return
        
        # Add to portfolio
        user_portfolios[user_id].append([ticker, price, quantity])
        save_portfolios()
        
        update.message.reply_text(f"Added {quantity} shares of {ticker} at ${price:.2f} to your portfolio.")
    
    except ValueError:
        update.message.reply_text("Invalid price or quantity. Please use numbers only.")
    except Exception as e:
        logger.error(f"Error adding to portfolio: {e}")
        update.message.reply_text(f"Error adding to portfolio: {str(e)}")

# Function to remove a stock from the portfolio
def remove_from_portfolio(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    
    # Check if ticker is provided
    if not context.args:
        update.message.reply_text("Please provide a ticker symbol. Example: /sell AAPL")
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
            update.message.reply_text(f"Removed {removed[2]} shares of {removed[0]} from your portfolio.")
        
        save_portfolios()
        
        # If no positions were found
        if not positions_to_remove:
            update.message.reply_text(f"No {ticker} positions found in your portfolio.")
    else:
        update.message.reply_text("Your portfolio is empty.")

# Function to display available commands
def help_command(update: Update, context: CallbackContext):
    help_text = """
*Stock Prophet Bot Commands*

/start - Start the bot and get a welcome message
/help - Show this help message
/ticker SYMBOL - Analyze a stock ticker (e.g., /ticker AAPL)

*Watchlist Commands:*
/add SYMBOL - Add a stock to your watchlist
/remove SYMBOL - Remove a stock from your watchlist
/watchlist - View your current watchlist

*Portfolio Commands:*
/portfolio - Show your current portfolio and performance
/buy TICKER PRICE QUANTITY - Add a stock to your portfolio
/sell TICKER - Remove a stock from your portfolio

You can also just send a ticker symbol directly without the /ticker command.

*About Stock Prophet*
This bot analyzes stocks using technical indicators and helps you make trading decisions.

*Technical Indicators Used:*
• RSI (Relative Strength Index)
• EMAs (9, 21, 50, 200 periods)
• Bollinger Bands
• MACD (Moving Average Convergence Divergence)
• Stochastic Oscillator
• ADX (Average Directional Index)

*Price Forecasting:*
• Uses ARIMA (Auto-Regressive Integrated Moving Average) model
• Forecasts price movement for the next 4 hours
• Automatic updates every 30 minutes for your watchlist stocks

*Trading Strategies:*
• Classic RSI + Bollinger Bands + EMA strategy
• MACD + Stochastic + ADX strategy
• Golden/Death Cross with confirmation

The bot uses a 2% take profit / stop loss when positions are opened.
    """
    update.message.reply_text(help_text, parse_mode='Markdown')

# Main function to run the bot
def run_telegram_bot():
    try:
        # Check for environment variable first
        import os
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
        
        if token == "YOUR_TELEGRAM_BOT_TOKEN":
            print("Warning: Using placeholder token. Bot will not connect to Telegram.")
            print("Set the TELEGRAM_BOT_TOKEN environment variable to use a real bot.")
            # Instead of failing, we'll use the test_mode
            return False
            
        # Load any saved watchlists and portfolios
        load_watchlists()
        load_portfolios()
            
        updater = Updater(token)
        dispatcher = updater.dispatcher
        job_queue = updater.job_queue

        # Add command handlers
        dispatcher.add_handler(CommandHandler("start", start))
        dispatcher.add_handler(CommandHandler("help", help_command))
        dispatcher.add_handler(CommandHandler("ticker", handle_ticker))
        
        # Add watchlist command handlers
        dispatcher.add_handler(CommandHandler("add", add_to_watchlist))
        dispatcher.add_handler(CommandHandler("remove", remove_from_watchlist))
        dispatcher.add_handler(CommandHandler("watchlist", show_watchlist))
        
        # Add portfolio command handlers
        dispatcher.add_handler(CommandHandler("portfolio", show_portfolio))
        dispatcher.add_handler(CommandHandler("buy", add_to_portfolio))
        dispatcher.add_handler(CommandHandler("sell", remove_from_portfolio))
        
        # Add a handler for direct ticker input (no command)
        dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_ticker))
        
        # Set up automatic notifications job (runs every 30 seconds to check tickers)
        job_queue.run_repeating(send_watchlist_notifications, interval=30, first=30)
        logger.info("Set up automatic notifications job (checking every 30 seconds)")

        # Start the bot
        print("Starting Telegram bot with token:", token[:5] + "..." + token[-5:])
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
    
    while True:
        ticker = input("> ").strip().upper()
        if ticker == "QUIT":
            break
        
        if ticker:
            print("\nAnalyzing ticker:", ticker)
            result = analyze_ticker(ticker)
            print("\n" + result + "\n")

# Main entry point
def main():
    # Try to run the Telegram bot
    if not run_telegram_bot():
        # If Telegram bot fails, run in test mode
        test_mode()

if __name__ == "__main__":
    main()

