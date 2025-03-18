"""
Daily Market Scanner for Stock Prophet
Identifies best stock picks for same-day trading opportunities
Shows performance summary including total profit
"""

import os
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import logging
import strategies
import json
# Import from the new indicators module to avoid circular imports
from indicators import calculate_indicators, determine_trend, safe_float

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TARGET_STOCKS based on our custom backtest results (default)
DEFAULT_TARGET_STOCKS = ['ZYXI', 'CRDF', 'CIFR', 'ARVN']

# Path to user watchlists file
WATCHLIST_FILE = "user_watchlists.json"

def get_focus_stocks_from_watchlist(user_id=None):
    """
    Get focus stocks from user's watchlist if available,
    otherwise return default target stocks
    
    Args:
        user_id: Optional user ID to get specific user's watchlist
        
    Returns:
        List of focus stock tickers
    """
    try:
        if os.path.exists(WATCHLIST_FILE):
            with open(WATCHLIST_FILE, 'r') as f:
                watchlists = json.load(f)
                
                # If user_id is provided, get that specific user's watchlist
                if user_id and str(user_id) in watchlists:
                    return watchlists[str(user_id)]
                    
                # If no specific user_id or user has no watchlist, 
                # use the first non-empty watchlist we find
                for uid, tickers in watchlists.items():
                    if tickers:
                        return tickers
        
        # If no watchlists found or all empty, return default targets
        return DEFAULT_TARGET_STOCKS
    except Exception as e:
        logger.error(f"Error loading focus stocks from watchlist: {e}")
        return DEFAULT_TARGET_STOCKS

# Get target stocks (either from watchlist or default)
TARGET_STOCKS = get_focus_stocks_from_watchlist()

# Market sector ETFs to gauge sector strength
SECTOR_ETFS = [
    'XLK',  # Technology
    'XLF',  # Financial
    'XLV',  # Healthcare
    'XLE',  # Energy
    'XLI',  # Industrial
    'XLY',  # Consumer Discretionary
    'XLP',  # Consumer Staples
    'XLRE', # Real Estate
    'XLU',  # Utilities
    'XLB',  # Materials
    'XLC'   # Communication Services
]

# Market index ETFs
INDEX_ETFS = [
    'SPY',  # S&P 500
    'QQQ',  # Nasdaq 100
    'DIA',  # Dow Jones
    'IWM',  # Russell 2000
    'VTI'   # Total Market
]

# Commodity-related stocks and ETFs
COMMODITY_TICKERS = [
    'GLD',  # Gold
    'SLV',  # Silver
    'USO',  # Oil
    'UNG',  # Natural Gas
    'BTC-USD',  # Bitcoin
    'ETH-USD'   # Ethereum
]

# High liquidity growth stocks
GROWTH_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',  # Large caps
    'AMD', 'PYPL', 'SHOP', 'SQ', 'SNOW', 'RBLX',              # Mid-caps
    'NET', 'DDOG', 'CRWD', 'ZS', 'DOCN', 'MDB'                # Software/Cloud
]

# Momentum stocks (frequently volatile with high trading volume)
MOMENTUM_STOCKS = [
    'GME', 'AMC', 'BBBY', 'PLTR', 'SOFI', 'LCID', 
    'NIO', 'DKNG', 'PLUG', 'MARA', 'RIOT', 'COIN'
]

# Compile all stocks to scan
ADDITIONAL_STOCKS = GROWTH_STOCKS + MOMENTUM_STOCKS + SECTOR_ETFS + INDEX_ETFS + COMMODITY_TICKERS

# Different timeframes for intraday analysis
INTRADAY_TIMEFRAMES = ["1h", "30m", "15m"]

class DailyPickScanner:
    """Scanner for finding best daily trading opportunities with 2%+ profit potential"""
    
    def __init__(self, tickers=None, additional_tickers=None, timeframes=None, market_scan_limit=50):
        """Initialize scanner with target stocks and parameters"""
        self.tickers = tickers or TARGET_STOCKS
        self.additional_tickers = additional_tickers or ADDITIONAL_STOCKS
        self.timeframes = timeframes or INTRADAY_TIMEFRAMES
        self.all_tickers = self.tickers.copy()  # First add our focused tickers
        
        # Add market sector ETFs for market analysis
        self.all_tickers.extend(SECTOR_ETFS)
        
        # Add index ETFs for market trend analysis
        self.all_tickers.extend(INDEX_ETFS)
        
        # Add major growth stocks 
        self.all_tickers.extend(GROWTH_STOCKS[:15])
        
        # Add momentum stocks
        self.all_tickers.extend(MOMENTUM_STOCKS[:10])
        
        # Add crypto and commodities
        self.all_tickers.extend(['GLD', 'BTC-USD', 'ETH-USD'])
        
        # Add other additional tickers if provided but limit total
        if additional_tickers:
            remaining_slots = market_scan_limit - len(self.all_tickers)
            if remaining_slots > 0:
                # Prioritize user-provided tickers
                self.all_tickers.extend(additional_tickers[:remaining_slots])
        
        # Remove duplicates while preserving order
        self.all_tickers = list(dict.fromkeys(self.all_tickers))
        
        # Ensure we don't exceed the scan limit
        if len(self.all_tickers) > market_scan_limit:
            self.all_tickers = self.all_tickers[:market_scan_limit]
            
        logger.info(f"Scanning {len(self.all_tickers)} tickers for trading opportunities")
            
        self.results = {}
        self.trade_history = []
        self.total_profit = 0.0
        self.winning_trades = 0
        self.total_trades = 0
        self.profitable_trades = []
        
        # Get combined strategy with custom parameters
        self.strategy = strategies.get_strategy("combined")
        
        # Override strategy parameters for better intraday performance
        self.strategy.parameters.update({
            'rsi_overbought': 60,
            'rsi_oversold': 40,
            'bb_std_dev': 1.5,
            'macd_fast': 6, 
            'macd_slow': 14,
            'macd_signal': 6,
            'take_profit': 2.0,  # Minimum target of 2% profit
            'stop_loss': 1.5     # Tighter stop loss to limit downside
        })
    
    def scan_opportunities(self):
        """Scan all tickers for trading opportunities"""
        logger.info(f"Scanning {len(self.all_tickers)} stocks for daily trading opportunities...")
        
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Track tickers with bullish signals
        bullish_tickers = []
        
        # Process each ticker
        for ticker in self.all_tickers:
            try:
                # Get daily data for initial screening
                daily_data = self._get_stock_data(ticker, period="5d", interval="1d")
                if daily_data is None or len(daily_data) < 3:
                    logger.warning(f"Not enough daily data for {ticker}")
                    continue
                    
                # Calculate daily indicators
                daily_data = calculate_indicators(daily_data)
                
                # Determine trend
                trend = determine_trend(daily_data)
                
                # Check for bullish signals using combined strategy
                try:
                    has_signals, signals = self.strategy.generate_signals(daily_data, "scanner", ticker)
                except Exception as e:
                    logger.warning(f"Error generating signals for {ticker}: {e}")
                    has_signals = False
                    signals = {}
                
                # Get intraday data for more precise entry/exit
                intraday_signals = []
                for timeframe in self.timeframes:
                    try:
                        # Use our improved method to handle MultiIndex
                        intraday_data = self._get_stock_data(ticker, period="2d", interval=timeframe)
                        if intraday_data is not None and len(intraday_data) > 10:
                            intraday_data = calculate_indicators(intraday_data)
                            try:
                                has_intraday_signals, intraday_signal_data = self.strategy.generate_signals(
                                    intraday_data, "scanner", ticker
                                )
                            except Exception as e:
                                logger.warning(f"Error generating intraday signals for {ticker}: {e}")
                                has_intraday_signals = False
                                intraday_signal_data = {}
                                
                            if has_intraday_signals and 'buy' in intraday_signal_data and intraday_signal_data['buy']:
                                # Get the latest timestamp safely
                                latest_timestamp = intraday_data.index[-1]
                                
                                # Format the time portion properly
                                if isinstance(latest_timestamp, pd.Timestamp):
                                    signal_time = latest_timestamp.strftime('%H:%M')
                                else:
                                    # Handle other index types by converting to Timestamp first
                                    signal_time = pd.Timestamp(latest_timestamp).strftime('%H:%M')
                                    
                                intraday_signals.append({
                                    'timeframe': timeframe,
                                    'signal_time': signal_time,
                                    'price': safe_float(intraday_data['Close'].iloc[-1])
                                })
                    except Exception as e:
                        logger.error(f"Error processing intraday data for {ticker} ({timeframe}): {e}")
                
                # Check current price and daily price target (2%+ gain)
                current_price = safe_float(daily_data['Close'].iloc[-1])
                price_target = current_price * 1.02  # Minimum 2% profit target
                stop_loss = current_price * 0.985  # 1.5% stop loss
                
                # Store results
                self.results[ticker] = {
                    'current_price': current_price,
                    'price_target': price_target,
                    'stop_loss': stop_loss,
                    'trend': trend,
                    'daily_signal': has_signals and 'buy' in signals and signals['buy'],
                    'intraday_signals': intraday_signals,
                    'rsi': daily_data['RSI_14'].iloc[-1] if 'RSI_14' in daily_data else None,
                    'macd': daily_data['MACD_12_26_9'].iloc[-1] if 'MACD_12_26_9' in daily_data else None,
                    'signal': daily_data['MACDs_12_26_9'].iloc[-1] if 'MACDs_12_26_9' in daily_data else None,
                    'bb_upper': daily_data['BBU_20_2.0'].iloc[-1] if 'BBU_20_2.0' in daily_data else None,
                    'bb_middle': daily_data['BBM_20_2.0'].iloc[-1] if 'BBM_20_2.0' in daily_data else None,
                    'bb_lower': daily_data['BBL_20_2.0'].iloc[-1] if 'BBL_20_2.0' in daily_data else None,
                    'volume': daily_data['Volume'].iloc[-1],
                    'score': 0  # Will be calculated later
                }
                
                # Calculate score based on technical indicators and signals
                score = 0
                
                # Add points for trend
                if trend == "Strong Uptrend":
                    score += 3
                elif trend == "Uptrend":
                    score += 2
                elif trend == "Sideways/Neutral":
                    score += 0
                elif trend == "Downtrend":
                    score -= 1
                elif trend == "Strong Downtrend":
                    score -= 2
                
                # Add points for daily signal
                if self.results[ticker]['daily_signal']:
                    score += 2
                    bullish_tickers.append(ticker)
                
                # Add points for intraday signals
                score += len(intraday_signals)
                
                # Add points for RSI
                rsi = self.results[ticker]['rsi']
                if rsi is not None:
                    if 40 <= rsi <= 50:  # Oversold but starting to recover
                        score += 1
                    elif 50 <= rsi <= 60:  # Bullish momentum without being overbought
                        score += 2
                    elif rsi < 30:  # Extremely oversold
                        score += 0.5
                
                # Add points for MACD crossover
                macd = self.results[ticker]['macd']
                signal = self.results[ticker]['signal']
                if macd is not None and signal is not None:
                    if macd > signal and macd > 0:  # Bullish crossover
                        score += 2
                    elif macd > signal and macd < 0:  # Bullish crossover but still negative
                        score += 1
                
                # Add points for Bollinger Band position
                price = self.results[ticker]['current_price']
                bb_lower = self.results[ticker]['bb_lower']
                bb_middle = self.results[ticker]['bb_middle']
                bb_upper = self.results[ticker]['bb_upper']
                if bb_lower is not None and bb_middle is not None and bb_upper is not None:
                    if price < bb_lower:  # Oversold
                        score += 1
                    elif bb_lower < price < bb_middle:  # Room to grow
                        score += 1.5
                
                # Store the final score
                self.results[ticker]['score'] = score
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
    
    def _get_stock_data(self, ticker, period="5d", interval="1d"):
        """Get stock data using yfinance with improved MultiIndex handling"""
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if data.empty or len(data) == 0:
                logger.warning(f"No data retrieved for {ticker}")
                return None
                
            # Handle MultiIndex if present (yfinance API change)
            if isinstance(data.index, pd.MultiIndex):
                logger.info(f"Converting MultiIndex to DatetimeIndex for {ticker} (interval: {interval})")
                
                # For intraday data, combine date and time for proper datetime index
                if interval in ["1h", "30m", "15m", "5m", "1m"]:
                    if data.index.nlevels >= 2:
                        # Create proper datetime objects from the MultiIndex levels
                        try:
                            datetime_index = pd.to_datetime(
                                data.index.get_level_values(0).strftime('%Y-%m-%d') + ' ' + 
                                data.index.get_level_values(1).strftime('%H:%M:%S')
                            )
                        except Exception as e:
                            logger.warning(f"Error creating datetime index: {e}, falling back to first level only")
                            datetime_index = data.index.get_level_values(0)
                    else:
                        # If it's a single level index for some reason
                        datetime_index = data.index.get_level_values(0)
                else:
                    # For daily data, just use the first level
                    datetime_index = data.index.get_level_values(0)
                
                # Create a new DataFrame with the DatetimeIndex
                new_data = pd.DataFrame(index=datetime_index)
                
                # Copy all columns individually
                for col in data.columns:
                    new_data[col] = data[col].values
                
                data = new_data
            
            # Additionally ensure all datetime indexes are properly converted to DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
                
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def get_top_picks(self, limit=5):
        """Get top stock picks based on score"""
        if not self.results:
            return []
            
        # Sort by score in descending order
        sorted_results = sorted(
            [(ticker, data) for ticker, data in self.results.items()],
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        return sorted_results[:limit]
    
    def simulate_trades(self, days_back=30, interval="1d"):
        """Simulate trades for the past X days to calculate profit"""
        logger.info(f"Simulating trades for the past {days_back} days with interval {interval}...")
        
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # Track trades with 2%+ profit
        self.profitable_trades = []
        
        # For better market analysis, include top 10 most liquid stocks too
        test_tickers = self.tickers + list(set(GROWTH_STOCKS[:10]) - set(self.tickers))
        
        for ticker in test_tickers:
            try:
                # Use our improved method directly instead of downloading again
                data = self._get_stock_data(ticker, period=f"{days_back}d", interval=interval)
                if data is None or len(data) < 5:
                    logger.warning(f"Not enough historical data for {ticker}")
                    continue
                
                # Ensure datetime index (already handled in _get_stock_data, but double-check)
                if not isinstance(data.index, pd.DatetimeIndex):
                    logger.warning(f"Converting index to DatetimeIndex for {ticker} (simulation)")
                    data.index = pd.to_datetime(data.index)
                
                # Calculate indicators
                data = calculate_indicators(data)
                
                # Apply trading strategy
                in_position = False
                entry_price = 0
                entry_date = None
                
                for i in range(1, len(data)):
                    current_date = data.index[i]
                    current_price = data['Close'].iloc[i]
                    
                    # Skip the first few bars until we have enough data for indicators
                    if i < 5:
                        continue
                        
                    # Get subset of data up to the current point
                    historical_data = data.iloc[:i+1]
                    
                    # Generate signals
                    has_signals, signals = self.strategy.generate_signals(historical_data, "simulator", ticker)
                    
                    if not in_position and has_signals and 'buy' in signals and signals['buy']:
                        # Enter position
                        entry_price = current_price
                        entry_date = current_date
                        in_position = True
                        # Handle MultiIndex or standard DatetimeIndex for entry_date
                        if isinstance(entry_date, pd.Timestamp):
                            entry_date_str = entry_date.strftime('%Y-%m-%d')
                        else:
                            # Handle MultiIndex case
                            entry_date_str = pd.Timestamp(entry_date).strftime('%Y-%m-%d')
                        logger.info(f"BUY signal: {ticker} @ ${entry_price:.2f} on {entry_date_str}")
                    
                    elif in_position:
                        # Check for exit conditions
                        take_profit = self.strategy.parameters.get('take_profit', 2.0) / 100.0
                        stop_loss = self.strategy.parameters.get('stop_loss', 1.5) / 100.0
                        
                        # Calculate price change
                        price_change = (current_price / entry_price - 1)
                        
                        # Check for exit signals
                        exit_signal = False
                        exit_reason = ""
                        
                        if has_signals and 'sell' in signals and signals['sell']:
                            exit_signal = True
                            exit_reason = "Sell signal"
                        elif price_change >= take_profit:
                            exit_signal = True
                            exit_reason = f"Take profit: +{price_change*100:.2f}%"
                        elif price_change <= -stop_loss:
                            exit_signal = True
                            exit_reason = f"Stop loss: {price_change*100:.2f}%"
                        
                        if exit_signal:
                            # Exit position
                            exit_price = current_price
                            exit_date = current_date
                            profit_pct = price_change * 100
                            
                            # Record trade
                            trade_info = {
                                'ticker': ticker,
                                'entry_date': entry_date,
                                'exit_date': exit_date,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'profit_pct': profit_pct,
                                'reason': exit_reason,
                                'days_held': (exit_date - entry_date).days
                            }
                            
                            self.trade_history.append(trade_info)
                            
                            # Update statistics
                            self.total_profit += profit_pct
                            self.total_trades += 1
                            if profit_pct > 0:
                                self.winning_trades += 1
                                
                                # Track trades with 2%+ profit
                                if profit_pct >= 2.0:
                                    self.profitable_trades.append(trade_info)
                            
                            # Handle MultiIndex or standard DatetimeIndex for exit_date
                            if isinstance(exit_date, pd.Timestamp):
                                exit_date_str = exit_date.strftime('%Y-%m-%d')
                            else:
                                # Handle MultiIndex case
                                exit_date_str = pd.Timestamp(exit_date).strftime('%Y-%m-%d')
                            logger.info(f"SELL signal: {ticker} @ ${exit_price:.2f} on {exit_date_str} - {exit_reason}")
                            
                            # Reset position
                            in_position = False
            
            except Exception as e:
                logger.error(f"Error simulating trades for {ticker}: {e}")
        
        # Calculate additional metrics
        if self.profitable_trades:
            self.avg_2pct_profit = sum(t['profit_pct'] for t in self.profitable_trades) / len(self.profitable_trades)
            self.count_2pct_profit = len(self.profitable_trades)
            self.avg_hold_time = sum(t['days_held'] for t in self.profitable_trades) / len(self.profitable_trades)
        else:
            self.avg_2pct_profit = 0
            self.count_2pct_profit = 0
            self.avg_hold_time = 0
    
    def format_results(self):
        """Format scanner results for display"""
        if not self.results:
            return "No results available. Run scan_opportunities() first."
        
        # Get top 10 picks across the entire market
        top_picks = self.get_top_picks(limit=10)
        if not top_picks:
            return "No suitable picks found today."
        
        result_str = "=== TODAY'S TOP 10 MARKET PICKS FOR SAME-DAY TRADING (2%+ PROFIT TARGET) ===\n\n"
        
        # Group stocks by category for better visualization
        target_tickers = set(self.tickers)
        high_score_picks = [p for p in top_picks if p[1]['score'] >= 5.0]
        
        # Add market analysis section
        result_str += "🔍 MARKET ANALYSIS:\n"
        
        # Get sector ETF data from results
        sector_data = {ticker: data for ticker, data in self.results.items() if ticker in SECTOR_ETFS}
        bullish_sectors = [ticker for ticker, data in sector_data.items() if data['trend'] in ["Uptrend", "Strong Uptrend"]]
        
        # Get index ETF data
        index_data = {ticker: data for ticker, data in self.results.items() if ticker in INDEX_ETFS}
        market_bullish = sum(1 for _, data in index_data.items() if data['trend'] in ["Uptrend", "Strong Uptrend"])
        market_bearish = sum(1 for _, data in index_data.items() if data['trend'] in ["Downtrend", "Strong Downtrend"])
        
        # Add market overview
        if market_bullish > market_bearish:
            result_str += "• Overall Market: BULLISH 📈\n"
        elif market_bearish > market_bullish:
            result_str += "• Overall Market: BEARISH 📉\n"
        else:
            result_str += "• Overall Market: NEUTRAL ↔️\n"
        
        # Add strongest sectors
        if bullish_sectors:
            result_str += f"• Strongest Sectors: {', '.join(bullish_sectors)}\n"
        
        # Add cryptocurrency section if we have data
        crypto_data = {ticker: data for ticker, data in self.results.items() if ticker in ['BTC-USD', 'ETH-USD']}
        if crypto_data:
            crypto_bullish = any(data['trend'] in ["Uptrend", "Strong Uptrend"] for _, data in crypto_data.items())
            result_str += f"• Crypto: {'BULLISH 📈' if crypto_bullish else 'BEARISH/NEUTRAL 📉'}\n"
        
        # Add commodity section
        gold_data = self.results.get('GLD', None)
        if gold_data:
            gold_trend = gold_data['trend']
            result_str += f"• Gold: {gold_trend} {'📈' if 'Uptrend' in gold_trend else '📉' if 'Downtrend' in gold_trend else '↔️'}\n"
        
        result_str += "\n🔝 TOP TRADING OPPORTUNITIES:\n"
        
        # List our top picks with relevant details
        for i, (ticker, data) in enumerate(top_picks, 1):
            price = data['current_price']
            target = data['price_target']
            stop = data['stop_loss']
            profit_potential = ((target / price) - 1) * 100
            
            # Determine category
            category = ""
            if ticker in target_tickers:
                category = "⭐ FOCUS"
            elif ticker in GROWTH_STOCKS:
                category = "💹 GROWTH"
            elif ticker in MOMENTUM_STOCKS:
                category = "🚀 MOMENTUM"
            elif ticker in SECTOR_ETFS:
                category = "📊 SECTOR"
            elif ticker in INDEX_ETFS:
                category = "📈 INDEX"
            elif ticker in COMMODITY_TICKERS:
                category = "💰 COMMODITY"
            
            result_str += f"{i}. {ticker} {category}: ${price:.2f}\n"
            result_str += f"   Target: ${target:.2f} (+{profit_potential:.2f}%)\n"
            result_str += f"   Stop Loss: ${stop:.2f}\n"
            result_str += f"   Trend: {data['trend']}\n"
            result_str += f"   Signal: {'BUY ✅' if data['daily_signal'] else 'NEUTRAL/SELL ❌'}\n"
            
            # Add intraday signals if available
            if data['intraday_signals']:
                result_str += "   Intraday Signals:\n"
                for signal in data['intraday_signals']:
                    result_str += f"     • {signal['timeframe']} @ {signal['signal_time']} - ${signal['price']:.2f}\n"
            
            result_str += f"   Score: {data['score']:.1f}/10\n\n"
        
        # Add trading performance summary
        result_str += "=== TRADING PERFORMANCE SUMMARY ===\n"
        result_str += f"Total Trades: {self.total_trades}\n"
        result_str += f"Winning Trades: {self.winning_trades} ({(self.winning_trades/self.total_trades)*100:.1f}% win rate)\n" if self.total_trades > 0 else "Winning Trades: 0 (0% win rate)\n"
        result_str += f"Total Profit: {self.total_profit:.2f}%\n"
        result_str += f"Average Profit per Trade: {self.total_profit/self.total_trades:.2f}%\n" if self.total_trades > 0 else "Average Profit per Trade: 0.00%\n"
        
        # Add 2%+ profit statistics
        if hasattr(self, 'profitable_trades') and self.profitable_trades:
            result_str += f"\n=== 2%+ PROFIT OPPORTUNITIES ===\n"
            result_str += f"Number of 2%+ Profit Trades: {self.count_2pct_profit}\n"
            result_str += f"Average Profit for 2%+ Trades: {self.avg_2pct_profit:.2f}%\n"
            result_str += f"Average Hold Time: {self.avg_hold_time:.1f} days\n\n"
            
            # Add details of the most recent profitable trades (up to 5)
            result_str += "Recent Profitable Trades (2%+):\n"
            recent_profitable = sorted(self.profitable_trades, key=lambda x: x['exit_date'], reverse=True)[:5]
            
            for i, trade in enumerate(recent_profitable, 1):
                # Handle MultiIndex or standard DatetimeIndex for entry_date and exit_date
                if isinstance(trade['entry_date'], pd.Timestamp):
                    entry_date_str = trade['entry_date'].strftime('%Y-%m-%d')
                else:
                    # Handle MultiIndex case
                    entry_date_str = pd.Timestamp(trade['entry_date']).strftime('%Y-%m-%d')
                    
                if isinstance(trade['exit_date'], pd.Timestamp):
                    exit_date_str = trade['exit_date'].strftime('%Y-%m-%d')
                else:
                    # Handle MultiIndex case
                    exit_date_str = pd.Timestamp(trade['exit_date']).strftime('%Y-%m-%d')
                    
                days = trade['days_held']
                result_str += f"{i}. {trade['ticker']}: ${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} "
                result_str += f"(+{trade['profit_pct']:.2f}%) in {days} {'day' if days == 1 else 'days'}\n"
                result_str += f"   Period: {entry_date_str} to {exit_date_str}\n"
        
        # Add timestamp
        import datetime
        result_str += f"\n\nLast Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return result_str

    def generate_command_response(self, user_id=None):
        """Generate response for the /picks command with personalization"""
        # First ensure we have scanned for opportunities
        if not self.results:
            self.scan_opportunities()
        
        # Simulate past trades to get performance
        self.simulate_trades(days_back=30)
        
        # Get formatted results
        results = self.format_results()
        
        # Add personalization if using watchlist
        if user_id and self.tickers != DEFAULT_TARGET_STOCKS:
            # Prepend a personalized message
            personalized_intro = (
                "Mr. Otmane, here are your personalized daily stock picks based on your watchlist:\n\n"
            )
            return personalized_intro + results
        
        return results

def daily_picks_command(user_id=None):
    """Handle /picks command to get daily stock picks using user's watchlist if available"""
    try:
        # Get focus stocks from user's watchlist if available
        focus_stocks = get_focus_stocks_from_watchlist(user_id)
        logger.info(f"Using focus stocks for user {user_id}: {focus_stocks}")
        
        # Create scanner with user's focus stocks
        scanner = DailyPickScanner(tickers=focus_stocks)
        scanner.scan_opportunities()
        scanner.simulate_trades()
        
        # Pass user_id to generate personalized response
        return scanner.generate_command_response(user_id=user_id)
    except Exception as e:
        logger.error(f"Error in daily_picks_command: {e}")
        return f"Error generating daily picks: {str(e)}"

if __name__ == "__main__":
    # Test the scanner
    scanner = DailyPickScanner()
    scanner.scan_opportunities()
    scanner.simulate_trades()
    print(scanner.format_results())