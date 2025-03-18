"""
Trading strategy module for Stock Prophet application
Defines different trading strategies with customizable parameters
"""

import logging
import numpy as np
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)

def safe_float(value):
    """Safely convert a value to float, handling pandas Series objects"""
    if hasattr(value, 'item'):
        return float(value.item())
    return float(value)

class Strategy:
    """Base class for all trading strategies"""
    def __init__(self, name, description, parameters=None):
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        
    def generate_signals(self, data, user_id, ticker, user_data=None):
        """
        Generate buy/sell signals - to be implemented by subclasses
        
        Returns:
            tuple: (has_signals, signal_data) where:
                - has_signals is a boolean indicating if any signals were generated
                - signal_data is a dictionary with signal information (signal_type, price, etc.)
        """
        raise NotImplementedError("Subclasses must implement generate_signals()")

    def get_description(self):
        """Return a description of the strategy with current parameters"""
        param_desc = ', '.join([f"{k}: {v}" for k, v in self.parameters.items()])
        return f"{self.name}: {self.description} [Parameters: {param_desc}]"


class RSIStrategy(Strategy):
    """RSI-based trading strategy"""
    def __init__(self, parameters=None):
        default_params = {
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            'take_profit': 3.0,  # Percentage - Default 3%
            'stop_loss': 4.0,    # Percentage - Default 4%
            'use_prediction': True  # Consider prediction for TP/SL
        }
        
        # Override defaults with provided parameters
        if parameters:
            default_params.update(parameters)
            
        super().__init__(
            name="RSI Strategy",
            description="Uses RSI to identify oversold and overbought conditions",
            parameters=default_params
        )
    
    def generate_signals(self, data, user_id, ticker, user_data=None):
        if user_data is None:
            user_data = {}
            
        latest = data.iloc[-1]
        signals = []
        
        # Check for forecast values if we're using prediction-based TP/SL
        price_trend_up = True
        
        # Check if there are forecast columns included in the data
        if self.parameters['use_prediction'] and 'forecast_values' in data.attrs:
            forecasts = data.attrs['forecast_values']
            if forecasts:
                # Check if the forecasted prices show an uptrend (last forecast > current price)
                current_price = safe_float(latest['Close'])
                last_forecast = safe_float(forecasts[-1])
                price_trend_up = last_forecast > current_price
                
                if price_trend_up:
                    logger.info(f"Detected upward price trend for {ticker}. Applying standard take profit/stop loss.")
                else:
                    logger.info(f"No upward price trend detected for {ticker}. Adjusting to conservative take profit/stop loss.")
        
        # Check if the user has an open position for this ticker
        if user_id in user_data and ticker in user_data[user_id]:
            buying_price = safe_float(user_data[user_id][ticker])
            close_price = safe_float(latest['Close'])
            
            # Sell Signal (Take Profit or Stop Loss)
            # Only apply the full parameters if price trend is up based on prediction
            if price_trend_up:
                take_profit_price = buying_price * (1 + self.parameters['take_profit']/100)
                stop_loss_price = buying_price * (1 - self.parameters['stop_loss']/100)
            else:
                # For downward trends, use more conservative thresholds (1% profit, 5% loss)
                take_profit_price = buying_price * 1.01  # 1% take profit
                stop_loss_price = buying_price * 0.95   # 5% stop loss
            
            if (close_price >= take_profit_price):
                profit_pct = ((close_price / buying_price) - 1) * 100
                signals.append(f"📉 Sell {ticker} at {close_price:.2f} (Take profit triggered at {profit_pct:.2f}%).")
                del user_data[user_id][ticker]  # Close the position
            elif (close_price <= stop_loss_price):
                loss_pct = (1 - (close_price / buying_price)) * 100
                signals.append(f"📉 Sell {ticker} at {close_price:.2f} (Stop loss triggered at {loss_pct:.2f}%).")
                del user_data[user_id][ticker]  # Close the position
            
            # Additional sell signals based on technical indicators
            elif safe_float(latest['RSI']) > self.parameters['overbought_threshold']:  # Overbought
                signals.append(f"📉 Consider selling {ticker} at {close_price:.2f} (RSI {safe_float(latest['RSI']):.2f} indicates overbought conditions).")
        
        else:
            # Original buy signal, but only generate if price is forecasted to rise
            close_price = safe_float(latest['Close'])
            rsi_oversold = safe_float(latest['RSI']) < self.parameters['oversold_threshold']
            
            if rsi_oversold and (not self.parameters['use_prediction'] or price_trend_up):
                signals.append(f"🚀 Buy {ticker} at {close_price:.2f} (RSI {safe_float(latest['RSI']):.2f} indicates oversold conditions).")
                # Store the buying price
                if user_id not in user_data:
                    user_data[user_id] = {}
                user_data[user_id][ticker] = close_price
            elif rsi_oversold and self.parameters['use_prediction'] and not price_trend_up:
                signals.append(f"⚠️ RSI indicates oversold conditions for {ticker}, but price is forecasted to decrease. Consider waiting.")
        
        return signals


class BollingerBandsStrategy(Strategy):
    """Bollinger Bands-based trading strategy"""
    def __init__(self, parameters=None):
        default_params = {
            'take_profit': 3.0,  # Percentage - Default 3%
            'stop_loss': 4.0,    # Percentage - Default 4%
            'band_gap_percent': 0.2,  # Minimum gap between bands as percentage
            'use_prediction': True  # Consider prediction for TP/SL
        }
        
        # Override defaults with provided parameters
        if parameters:
            default_params.update(parameters)
            
        super().__init__(
            name="Bollinger Bands Strategy",
            description="Uses Bollinger Bands for mean reversion trading",
            parameters=default_params
        )
    
    def generate_signals(self, data, user_id, ticker, user_data=None):
        if user_data is None:
            user_data = {}
            
        latest = data.iloc[-1]
        signals = []
        
        # Check for forecast values if we're using prediction-based TP/SL
        price_trend_up = True
        
        # Check if there are forecast columns included in the data
        if self.parameters['use_prediction'] and 'forecast_values' in data.attrs:
            forecasts = data.attrs['forecast_values']
            if forecasts:
                # Check if the forecasted prices show an uptrend (last forecast > current price)
                current_price = safe_float(latest['Close'])
                last_forecast = safe_float(forecasts[-1])
                price_trend_up = last_forecast > current_price
                
                if price_trend_up:
                    logger.info(f"Detected upward price trend for {ticker}. Applying standard take profit/stop loss.")
                else:
                    logger.info(f"No upward price trend detected for {ticker}. Adjusting to conservative take profit/stop loss.")
        
        # Check if we have Bollinger Bands data
        if not all(band in data.columns for band in ['BB_upper', 'BB_middle', 'BB_lower']):
            return []
        
        # Check if the user has an open position for this ticker
        if user_id in user_data and ticker in user_data[user_id]:
            buying_price = safe_float(user_data[user_id][ticker])
            close_price = safe_float(latest['Close'])
            
            # Sell Signal (Take Profit or Stop Loss)
            # Only apply the full parameters if price trend is up based on prediction
            if price_trend_up:
                take_profit_price = buying_price * (1 + self.parameters['take_profit']/100)
                stop_loss_price = buying_price * (1 - self.parameters['stop_loss']/100)
            else:
                # For downward trends, use more conservative thresholds (1% profit, 5% loss)
                take_profit_price = buying_price * 1.01  # 1% take profit
                stop_loss_price = buying_price * 0.95   # 5% stop loss
            
            if (close_price >= take_profit_price):
                profit_pct = ((close_price / buying_price) - 1) * 100
                signals.append(f"📉 Sell {ticker} at {close_price:.2f} (Take profit triggered at {profit_pct:.2f}%).")
                del user_data[user_id][ticker]  # Close the position
            elif (close_price <= stop_loss_price):
                loss_pct = (1 - (close_price / buying_price)) * 100
                signals.append(f"📉 Sell {ticker} at {close_price:.2f} (Stop loss triggered at {loss_pct:.2f}%).")
                del user_data[user_id][ticker]  # Close the position
            
            # Additional sell signals - close when price hits upper band
            elif safe_float(latest['Close']) >= safe_float(latest['BB_upper']):
                signals.append(f"📉 Consider selling {ticker} at {close_price:.2f} (Price hit upper Bollinger Band).")
        
        else:
            # Buy signal when price is near or below lower band
            close_price = safe_float(latest['Close'])
            bb_lower = safe_float(latest['BB_lower'])
            price_near_lower_band = close_price <= bb_lower * 1.01
            
            # Check if bands are wide enough to indicate volatility
            band_gap = (safe_float(latest['BB_upper']) - safe_float(latest['BB_lower'])) / safe_float(latest['BB_middle'])
            sufficient_volatility = band_gap > self.parameters['band_gap_percent']
            
            # Only generate buy signal if not using prediction or price trend is up
            if price_near_lower_band and sufficient_volatility and (not self.parameters['use_prediction'] or price_trend_up):
                signals.append(f"🚀 Buy {ticker} at {close_price:.2f} (Price at lower Bollinger Band with sufficient volatility).")
                # Store the buying price
                if user_id not in user_data:
                    user_data[user_id] = {}
                user_data[user_id][ticker] = close_price
            elif price_near_lower_band and sufficient_volatility and self.parameters['use_prediction'] and not price_trend_up:
                signals.append(f"⚠️ Price near lower Bollinger Band for {ticker}, but price is forecasted to decrease. Consider waiting.")
        
        return signals


class MACDStrategy(Strategy):
    """MACD-based trading strategy"""
    def __init__(self, parameters=None):
        default_params = {
            'take_profit': 3.0,  # Percentage - Default 3%
            'stop_loss': 4.0,    # Percentage - Default 4%
            'signal_threshold': 0.01,  # Minimum MACD-Signal difference
            'use_prediction': True  # Consider prediction for TP/SL
        }
        
        # Override defaults with provided parameters
        if parameters:
            default_params.update(parameters)
            
        super().__init__(
            name="MACD Strategy",
            description="Uses MACD crossovers for trend identification",
            parameters=default_params
        )
    
    def generate_signals(self, data, user_id, ticker, user_data=None):
        if user_data is None:
            user_data = {}
            
        latest = data.iloc[-1]
        signals = []
        
        # Check for forecast values if we're using prediction-based TP/SL
        price_trend_up = True
        
        # Check if there are forecast columns included in the data
        if self.parameters['use_prediction'] and 'forecast_values' in data.attrs:
            forecasts = data.attrs['forecast_values']
            if forecasts:
                # Check if the forecasted prices show an uptrend (last forecast > current price)
                current_price = safe_float(latest['Close'])
                last_forecast = safe_float(forecasts[-1])
                price_trend_up = last_forecast > current_price
                
                if price_trend_up:
                    logger.info(f"Detected upward price trend for {ticker}. Applying standard take profit/stop loss.")
                else:
                    logger.info(f"No upward price trend detected for {ticker}. Adjusting to conservative take profit/stop loss.")
        
        # Check if we have MACD data
        if not all(col in data.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
            return []
        
        # Check if the user has an open position for this ticker
        if user_id in user_data and ticker in user_data[user_id]:
            buying_price = safe_float(user_data[user_id][ticker])
            close_price = safe_float(latest['Close'])
            
            # Sell Signal (Take Profit or Stop Loss)
            # Only apply the full parameters if price trend is up based on prediction
            if price_trend_up:
                take_profit_price = buying_price * (1 + self.parameters['take_profit']/100)
                stop_loss_price = buying_price * (1 - self.parameters['stop_loss']/100)
            else:
                # For downward trends, use more conservative thresholds (1% profit, 5% loss)
                take_profit_price = buying_price * 1.01  # 1% take profit
                stop_loss_price = buying_price * 0.95   # 5% stop loss
            
            if (close_price >= take_profit_price):
                profit_pct = ((close_price / buying_price) - 1) * 100
                signals.append(f"📉 Sell {ticker} at {close_price:.2f} (Take profit triggered at {profit_pct:.2f}%).")
                del user_data[user_id][ticker]  # Close the position
            elif (close_price <= stop_loss_price):
                loss_pct = (1 - (close_price / buying_price)) * 100
                signals.append(f"📉 Sell {ticker} at {close_price:.2f} (Stop loss triggered at {loss_pct:.2f}%).")
                del user_data[user_id][ticker]  # Close the position
            
            # Additional sell signals based on MACD crossover (bearish)
            elif (safe_float(latest['MACD']) < safe_float(latest['MACD_Signal'])) and abs(safe_float(latest['MACD']) - safe_float(latest['MACD_Signal'])) > self.parameters['signal_threshold']:
                signals.append(f"📉 Consider selling {ticker} at {close_price:.2f} (MACD bearish crossover).")
        
        else:
            # Buy signal on MACD crossover (bullish)
            close_price = safe_float(latest['Close'])
            macd_bullish = (safe_float(latest['MACD']) > safe_float(latest['MACD_Signal'])) and abs(safe_float(latest['MACD']) - safe_float(latest['MACD_Signal'])) > self.parameters['signal_threshold']
            
            # Only generate buy signal if not using prediction or price trend is up
            if macd_bullish and (not self.parameters['use_prediction'] or price_trend_up):
                signals.append(f"🚀 Buy {ticker} at {close_price:.2f} (MACD bullish crossover).")
                # Store the buying price
                if user_id not in user_data:
                    user_data[user_id] = {}
                user_data[user_id][ticker] = close_price
            elif macd_bullish and self.parameters['use_prediction'] and not price_trend_up:
                signals.append(f"⚠️ MACD shows bullish crossover for {ticker}, but price is forecasted to decrease. Consider waiting.")
        
        return signals


class CombinedStrategy(Strategy):
    """Combined strategy using multiple indicators"""
    def __init__(self, parameters=None):
        default_params = {
            'take_profit': 2.0,  # Optimized based on 60-day backtest (avg 3-7% profit)
            'stop_loss': 1.5,    # Tighter stop loss to preserve capital
            'rsi_oversold': 45,  # Optimized RSI threshold based on successful trades
            'rsi_overbought': 55, # Optimized RSI threshold based on successful trades
            'min_indicators': 1,  # Only one indicator needed to confirm (optimized)
            'use_prediction': True,  # Use prediction to filter out poor trade setups
            'price_percent_trigger': 0.5  # Trigger trades on 0.5% price movements
        }
        
        # Override defaults with provided parameters
        if parameters:
            default_params.update(parameters)
            
        super().__init__(
            name="Combined Strategy",
            description="Uses multiple indicators for stronger confirmation",
            parameters=default_params
        )
    
    def generate_signals(self, data, user_id, ticker, user_data=None):
        if user_data is None:
            user_data = {}
            
        latest = data.iloc[-1]
        signals = []
        
        # Check for forecast values if we're using prediction-based TP/SL
        price_trend_up = True
        
        # Check if there are forecast columns included in the data
        if self.parameters['use_prediction'] and 'forecast_values' in data.attrs:
            forecasts = data.attrs['forecast_values']
            if forecasts:
                # Check if the forecasted prices show an uptrend (last forecast > current price)
                current_price = latest['Close']
                last_forecast = forecasts[-1]
                price_trend_up = last_forecast > current_price
                
                if price_trend_up:
                    logger.info(f"Detected upward price trend for {ticker}. Applying standard take profit/stop loss.")
                else:
                    logger.info(f"No upward price trend detected for {ticker}. Adjusting to conservative take profit/stop loss.")
        
        # Check for available indicators
        has_macd = all(col in data.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist'])
        has_rsi = 'RSI' in data.columns
        has_bb = all(col in data.columns for col in ['BB_upper', 'BB_middle', 'BB_lower'])
        has_ema = all(col in data.columns for col in ['EMA_9', 'EMA_21'])
        
        # Check if the user has an open position for this ticker
        if user_id in user_data and ticker in user_data[user_id]:
            buying_price = user_data[user_id][ticker]
            
            # Sell Signal (Take Profit or Stop Loss)
            # Only apply the full parameters if price trend is up based on prediction
            if price_trend_up:
                take_profit_price = buying_price * (1 + self.parameters['take_profit']/100)
                stop_loss_price = buying_price * (1 - self.parameters['stop_loss']/100)
            else:
                # For downward trends, use more conservative thresholds (1% profit, 5% loss)
                take_profit_price = buying_price * 1.01  # 1% take profit
                stop_loss_price = buying_price * 0.95   # 5% stop loss
            
            if (latest['Close'] >= take_profit_price):
                profit_pct = ((latest['Close'] / buying_price) - 1) * 100
                signals.append(f"📉 Sell {ticker} at {latest['Close']:.2f} (Take profit triggered at {profit_pct:.2f}%).")
                del user_data[user_id][ticker]  # Close the position
            elif (latest['Close'] <= stop_loss_price):
                loss_pct = (1 - (latest['Close'] / buying_price)) * 100
                signals.append(f"📉 Sell {ticker} at {latest['Close']:.2f} (Stop loss triggered at {loss_pct:.2f}%).")
                del user_data[user_id][ticker]  # Close the position
            else:
                # Count bearish indicators
                bearish_indicators = 0
                
                # RSI Overbought
                if has_rsi and latest['RSI'] is not None and latest['RSI'].item() > self.parameters['rsi_overbought']:
                    bearish_indicators += 1
                
                # MACD Bearish Crossover
                if has_macd and latest['MACD'] is not None and latest['MACD_Signal'] is not None and latest['MACD'].item() < latest['MACD_Signal'].item():
                    bearish_indicators += 1
                
                # Price at/above upper Bollinger Band
                if has_bb and latest['Close'] is not None and latest['BB_upper'] is not None and latest['Close'].item() >= latest['BB_upper'].item():
                    bearish_indicators += 1
                
                # Short-term EMA crosses below long-term
                if has_ema and latest['EMA_9'] is not None and latest['EMA_21'] is not None and latest['EMA_9'].item() < latest['EMA_21'].item():
                    bearish_indicators += 1
                
                # If enough bearish indicators, suggest selling
                if bearish_indicators >= self.parameters['min_indicators']:
                    signals.append(f"📉 Consider selling {ticker} at {latest['Close']:.2f} ({bearish_indicators} bearish indicators detected).")
        
        else:
            # Count bullish indicators
            bullish_indicators = 0
            
            # RSI Oversold - now with higher threshold for easier triggering
            if has_rsi and latest['RSI'] is not None and latest['RSI'].item() < self.parameters['rsi_oversold']:
                bullish_indicators += 1
            
            # Force a bullish indicator if RSI is really low (extreme oversold)
            if has_rsi and latest['RSI'] is not None and latest['RSI'].item() < 30:
                bullish_indicators += 1  # Add an extra count for extreme oversold
            
            # MACD Bullish Crossover
            if has_macd and latest['MACD'] is not None and latest['MACD_Signal'] is not None and latest['MACD'].item() > latest['MACD_Signal'].item():
                bullish_indicators += 1
            
            # Price at/below lower Bollinger Band
            if has_bb and latest['Close'] is not None and latest['BB_lower'] is not None and latest['Close'].item() <= latest['BB_lower'].item() * 1.01:  # Allow 1% above lower band
                bullish_indicators += 1
            
            # Short-term EMA crosses above long-term
            if has_ema and latest['EMA_9'] is not None and latest['EMA_21'] is not None and latest['EMA_9'].item() > latest['EMA_21'].item() * 0.99:  # Allow 1% tolerance
                bullish_indicators += 1
                
            # Check for recent price drop as a buy signal (mean reversion)
            if len(data) > 2:
                prev_close = data['Close'].iloc[-2]
                current_close = latest['Close'].item() if hasattr(latest['Close'], 'item') else latest['Close']
                prev_close = prev_close.item() if hasattr(prev_close, 'item') else prev_close
                if float(current_close) < float(prev_close):
                    # Price dropped from previous bar
                    percent_drop = (float(prev_close) - float(current_close)) / float(prev_close) * 100
                    if percent_drop > self.parameters.get('price_percent_trigger', 0.5):
                        # Sharp price drop can be a buy opportunity
                        bullish_indicators += 1
            
            # If enough bullish indicators, suggest buying
            # Only generate buy signal if not using prediction or price trend is up
            if bullish_indicators >= self.parameters['min_indicators'] and (not self.parameters['use_prediction'] or price_trend_up):
                current_price = float(latest['Close'].item() if hasattr(latest['Close'], 'item') else latest['Close'])
                signals.append(f"🚀 Buy {ticker} at {current_price:.2f} ({bullish_indicators} bullish indicators detected).")
                # Store the buying price
                if user_id not in user_data:
                    user_data[user_id] = {}
                user_data[user_id][ticker] = latest['Close']
            elif bullish_indicators >= self.parameters['min_indicators'] and self.parameters['use_prediction'] and not price_trend_up:
                signals.append(f"⚠️ Found {bullish_indicators} bullish indicators for {ticker}, but price is forecasted to decrease. Consider waiting.")
        
        return signals


class GoldStrategy(Strategy):
    """Specialized strategy for gold and precious metals"""
    def __init__(self, parameters=None):
        # Define separate default parameters for commodity gold vs ETF gold
        commodity_params = {
            'rsi_oversold': 35,  # Commodities can be more volatile than ETFs
            'rsi_overbought': 65,
            'bb_std_dev': 2.5,   # Wider bands for commodities
            'take_profit': 2.0,  # Higher targets for commodities
            'stop_loss': 2.5,    # Higher risk for commodities
            'min_indicators': 2,  # Minimum number of indicators confirming for a signal
            'use_prediction': True
        }
        
        etf_params = {
            'rsi_oversold': 40,  # Gold ETFs tend to be less volatile
            'rsi_overbought': 60,
            'bb_std_dev': 2.0,   # Gold ETFs often respond well to Bollinger Bands
            'take_profit': 1.5,  # More conservative targets for ETFs
            'stop_loss': 2.0,    # Lower risk for ETFs
            'min_indicators': 2,
            'use_prediction': True
        }
        
        # Start with ETF parameters as default
        default_params = etf_params.copy()
        
        # Override defaults with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Store both parameter sets for later use
        self.commodity_params = commodity_params
        self.etf_params = etf_params
            
        super().__init__(
            name="Gold Strategy",
            description="Specialized strategy for gold and precious metals with optimized parameters for both commodities and ETFs",
            parameters=default_params
        )
    
    def generate_signals(self, data, user_id, ticker, user_data=None):
        """Generate buy/sell signals for gold assets"""
        if user_data is None:
            user_data = {}
            
        latest = data.iloc[-1]
        signals = []
        
        # Determine if this is a gold commodity or ETF and use the right parameters
        is_commodity = False
        if 'asset_type' in data.attrs:
            if data.attrs['asset_type'] == 'gold_commodity':
                # Use commodity-specific parameters
                is_commodity = True
                active_params = self.commodity_params.copy()
                logger.info(f"Using gold commodity parameters for {ticker}")
            elif data.attrs['asset_type'] == 'gold_etf':
                # Use ETF-specific parameters
                active_params = self.etf_params.copy()
                logger.info(f"Using gold ETF parameters for {ticker}")
            else:
                # Default to current parameters
                active_params = self.parameters
        else:
            # Default to current parameters if asset_type not specified
            active_params = self.parameters
        
        # Check for forecast values
        price_trend_up = True
        if active_params['use_prediction'] and 'forecast_values' in data.attrs:
            forecasts = data.attrs['forecast_values']
            if forecasts:
                current_price = latest['Close']
                last_forecast = forecasts[-1]
                price_trend_up = last_forecast > current_price
                
                if price_trend_up:
                    logger.info(f"Detected upward price trend for {ticker}. Applying standard take profit/stop loss.")
                else:
                    logger.info(f"No upward price trend detected for {ticker}. Adjusting to conservative take profit/stop loss.")
        
        # Check for available indicators
        has_rsi = 'RSI' in data.columns
        has_bb = all(col in data.columns for col in ['BB_upper', 'BB_middle', 'BB_lower'])
        has_ema = all(col in data.columns for col in ['EMA_9', 'EMA_21'])
        
        # Check if the user has an open position for this ticker
        if user_id in user_data and ticker in user_data[user_id]:
            buying_price = user_data[user_id][ticker]
            
            # Sell Signal (Take Profit or Stop Loss)
            take_profit_price = buying_price * (1 + active_params['take_profit']/100)
            stop_loss_price = buying_price * (1 - active_params['stop_loss']/100)
            
            if (latest['Close'] >= take_profit_price):
                profit_pct = ((latest['Close'] / buying_price) - 1) * 100
                signals.append(f"📉 Sell {ticker} at {latest['Close']:.2f} (Take profit triggered at {profit_pct:.2f}%).")
                
                # Add metadata for notifications
                signal_data = {
                    'signal_type': 'SELL',
                    'reason': 'take_profit',
                    'price': float(latest['Close']),
                    'profit_percent': float(profit_pct),
                    'ticker': ticker
                }
                
                del user_data[user_id][ticker]  # Close the position
                return True, signal_data
                
            elif (latest['Close'] <= stop_loss_price):
                loss_pct = (1 - (latest['Close'] / buying_price)) * 100
                signals.append(f"📉 Sell {ticker} at {latest['Close']:.2f} (Stop loss triggered at {loss_pct:.2f}%).")
                
                # Add metadata for notifications
                signal_data = {
                    'signal_type': 'SELL',
                    'reason': 'stop_loss',
                    'price': float(latest['Close']),
                    'loss_percent': float(loss_pct),
                    'ticker': ticker
                }
                
                del user_data[user_id][ticker]  # Close the position
                return True, signal_data
            
            # Count bearish indicators
            bearish_indicators = 0
            
            # RSI Overbought
            if has_rsi and latest['RSI'] is not None and latest['RSI'].item() > active_params['rsi_overbought']:
                bearish_indicators += 1
            
            # Price at/above upper Bollinger Band
            if has_bb and latest['Close'] is not None and latest['BB_upper'] is not None and latest['Close'].item() >= latest['BB_upper'].item():
                bearish_indicators += 1
            
            # If enough bearish indicators, suggest selling
            if bearish_indicators >= active_params['min_indicators']:
                signals.append(f"📉 Consider selling {ticker} at {latest['Close']:.2f} ({bearish_indicators} bearish indicators detected).")
                
                # Add metadata for notifications
                signal_data = {
                    'signal_type': 'SELL',
                    'reason': 'bearish_indicators',
                    'price': float(latest['Close']),
                    'indicators': bearish_indicators,
                    'ticker': ticker
                }
                
                return True, signal_data
        
        else:
            # Count bullish indicators
            bullish_indicators = 0
            
            # RSI Oversold
            if has_rsi and latest['RSI'] is not None and latest['RSI'].item() < active_params['rsi_oversold']:
                bullish_indicators += 1
            
            # Price at/below lower Bollinger Band
            if has_bb and latest['Close'] is not None and latest['BB_lower'] is not None and latest['Close'].item() <= latest['BB_lower'].item():
                bullish_indicators += 1
            
            # Short-term EMA crosses above long-term
            if has_ema and latest['EMA_9'] is not None and latest['EMA_21'] is not None and latest['EMA_9'].item() > latest['EMA_21'].item():
                bullish_indicators += 1
            
            # If enough bullish indicators, suggest buying
            if bullish_indicators >= active_params['min_indicators'] and (not active_params['use_prediction'] or price_trend_up):
                # Add commodity/ETF specific message
                asset_type_msg = "gold commodity" if is_commodity else "gold ETF"
                current_price = float(latest['Close'].item() if hasattr(latest['Close'], 'item') else latest['Close'])
                signals.append(f"🚀 Buy {ticker} at {current_price:.2f} ({bullish_indicators} bullish indicators detected for {asset_type_msg}).")
                
                # Store the buying price
                if user_id not in user_data:
                    user_data[user_id] = {}
                user_data[user_id][ticker] = latest['Close']
                
                # Add metadata for notifications
                signal_data = {
                    'signal_type': 'BUY',
                    'reason': 'bullish_indicators',
                    'price': float(latest['Close']),
                    'indicators': bullish_indicators,
                    'ticker': ticker,
                    'is_commodity': is_commodity
                }
                
                return True, signal_data
                
            elif bullish_indicators >= active_params['min_indicators'] and active_params['use_prediction'] and not price_trend_up:
                signals.append(f"⚠️ Found {bullish_indicators} bullish indicators for {ticker}, but price is forecasted to decrease. Consider waiting.")
        
        # If no signals were generated
        return False, {}

# Dictionary of available strategies
AVAILABLE_STRATEGIES = {
    'rsi': RSIStrategy(),
    'bollinger': BollingerBandsStrategy(),
    'macd': MACDStrategy(),
    'combined': CombinedStrategy(),
    'gold': GoldStrategy()
}


def get_strategy(strategy_name):
    """Get a strategy by name"""
    if strategy_name in AVAILABLE_STRATEGIES:
        return AVAILABLE_STRATEGIES[strategy_name]
    return AVAILABLE_STRATEGIES['combined']  # Default to combined


def update_strategy_params(strategy_name, parameters):
    """Update the parameters of a strategy"""
    if strategy_name in AVAILABLE_STRATEGIES:
        for key, value in parameters.items():
            if key in AVAILABLE_STRATEGIES[strategy_name].parameters:
                AVAILABLE_STRATEGIES[strategy_name].parameters[key] = value
        return True
    return False


def get_available_strategies_info():
    """Get information about all available strategies"""
    return {name: strategy.get_description() for name, strategy in AVAILABLE_STRATEGIES.items()}