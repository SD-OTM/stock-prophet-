"""
Aggressive trading strategies module for backtesting
Defines extremely sensitive trading strategies to generate signals
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Safe numeric value function to handle both scalar and Series types
def safe_numeric_value(value):
    """Safely extract numeric value from pandas Series or scalar"""
    if isinstance(value, pd.Series):
        if len(value) > 0:
            return value.iloc[0]
        else:
            return 0.0
    return value if value is not None else 0.0

class AggressiveStrategy:
    """Ultra-sensitive strategy for backtesting purposes"""
    def __init__(self, name="Aggressive Combined", parameters=None):
        # Default parameters set for maximum sensitivity
        default_params = {
            'take_profit': 0.1,       # Tiny profit target (0.1%)
            'stop_loss': 0.3,         # Tiny stop loss (0.3%)
            'rsi_oversold': 49,       # Almost middle, barely oversold
            'rsi_overbought': 51,     # Almost middle, barely overbought
            'min_indicators': 1,      # Only 1 indicator needed for signal
            'band_gap_percent': 0.005, # Ultra-tiny band gap required
            'signal_threshold': 0.00001 # Practically no MACD threshold
        }
        
        # Override defaults with provided parameters
        if parameters:
            default_params.update(parameters)
            
        self.name = name
        self.description = "Ultra-sensitive combined strategy for backtesting"
        self.parameters = default_params
        
    def generate_signals(self, data, user_id, ticker, user_data=None):
        """Generate ultra-sensitive buy/sell signals for backtesting"""
        if user_data is None:
            user_data = {}
            
        # Make sure we have at least some data
        if len(data) < 2:
            return []
            
        signals = []
        latest = data.iloc[-1]
        
        # Check if the user has an open position for this ticker
        has_position = False
        if user_id in user_data and ticker in user_data[user_id]:
            has_position = True
            buying_price = user_data[user_id][ticker]
            
            # More aggressive, smaller profit and loss targets
            take_profit_price = buying_price * (1 + self.parameters['take_profit']/100)
            stop_loss_price = buying_price * (1 - self.parameters['stop_loss']/100)
            
            current_price = safe_numeric_value(latest['Close'])
            
            # Sell signals - take profit or stop loss
            if current_price >= take_profit_price:
                profit_pct = ((current_price / buying_price) - 1) * 100
                signals.append(f"📉 Sell {ticker} at {current_price:.2f} (Take profit triggered at {profit_pct:.2f}%).")
                del user_data[user_id][ticker]  # Close the position
            elif current_price <= stop_loss_price:
                loss_pct = (1 - (current_price / buying_price)) * 100
                signals.append(f"📉 Sell {ticker} at {current_price:.2f} (Stop loss triggered at {loss_pct:.2f}%).")
                del user_data[user_id][ticker]  # Close the position
                
            # Additional conditions for selling
            elif 'RSI' in latest and safe_numeric_value(latest['RSI']) > self.parameters['rsi_overbought']:
                signals.append(f"📉 Sell {ticker} at {current_price:.2f} (RSI overbought at {safe_numeric_value(latest['RSI']):.2f}).")
                del user_data[user_id][ticker]  # Close the position
                
            # MACD bearish crossover - check if MACD is below Signal with minimal threshold
            elif all(col in latest for col in ['MACD', 'MACD_Signal']):
                macd = safe_numeric_value(latest['MACD'])
                signal = safe_numeric_value(latest['MACD_Signal'])
                if macd < signal and abs(macd - signal) > self.parameters['signal_threshold']:
                    signals.append(f"📉 Sell {ticker} at {current_price:.2f} (MACD bearish at {macd:.4f}).")
                    del user_data[user_id][ticker]  # Close the position
                
        else:
            # Buy signals - much more aggressive
            indicator_signals = 0
            buy_reasons = []
            current_price = safe_numeric_value(latest['Close'])
            
            # RSI - check for "oversold" with higher threshold
            if 'RSI' in latest:
                rsi = safe_numeric_value(latest['RSI'])
                if rsi < self.parameters['rsi_oversold']:
                    indicator_signals += 1
                    buy_reasons.append(f"RSI {rsi:.2f} < {self.parameters['rsi_oversold']}")
            
            # Bollinger Bands - price near lower band
            if all(band in data.columns for band in ['BBL_3_2.0', 'BBM_3_2.0', 'BBU_3_2.0']):
                lower_band = safe_numeric_value(latest['BBL_3_2.0'])
                middle_band = safe_numeric_value(latest['BBM_3_2.0'])
                upper_band = safe_numeric_value(latest['BBU_3_2.0'])
                
                # Price near lower band
                if current_price <= lower_band * 1.01:
                    indicator_signals += 1
                    buy_reasons.append(f"Price near lower BB ({current_price:.2f} <= {lower_band:.2f}*1.01)")
                
                # Check if bands are wide enough to indicate volatility
                band_gap = (upper_band - lower_band) / middle_band if middle_band else 0
                if band_gap > self.parameters['band_gap_percent']:
                    indicator_signals += 1
                    buy_reasons.append(f"BB gap {band_gap:.4f} > {self.parameters['band_gap_percent']}")
            
            # MACD - bullish crossover with minimal threshold
            if all(col in latest for col in ['MACD', 'MACD_Signal']):
                macd = safe_numeric_value(latest['MACD'])
                signal = safe_numeric_value(latest['MACD_Signal'])
                if macd > signal and abs(macd - signal) > self.parameters['signal_threshold']:
                    indicator_signals += 1
                    buy_reasons.append(f"MACD bullish ({macd:.4f} > {signal:.4f})")
            
            # Check if we have enough indicators confirming the signal
            if indicator_signals >= self.parameters['min_indicators']:
                reasons = ", ".join(buy_reasons)
                signals.append(f"🚀 Buy {ticker} at {current_price:.2f} (Signals: {reasons}).")
                
                # Store the buying price
                if user_id not in user_data:
                    user_data[user_id] = {}
                user_data[user_id][ticker] = current_price
        
        return signals