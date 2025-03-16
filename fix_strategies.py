"""
Fix for pandas Series handling in strategies.py

This script patches the strategy classes to handle pandas Series objects correctly
to avoid "The truth value of a Series is ambiguous" errors during backtesting.
"""

import logging
import re
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_numeric_value(value):
    """Safely extract numeric value from pandas Series or scalar"""
    if isinstance(value, pd.Series):
        if len(value) > 0:
            return float(value.iloc[0])
        return 0.0
    return float(value)

def is_series_comparison(line):
    """Check if a line likely contains a pandas Series comparison"""
    comparison_patterns = [
        r'.*latest\[.*\].*[<>=]+',  # Check for patterns like latest['RSI'] < threshold
        r'.*forecast.*>.*price',     # Check for forecast > price patterns
        r'.*price.*>.*forecast',     # Check for price > forecast patterns
        r'.*\bif\b.*\(.*\).*:',      # Check for if statements with conditions
    ]
    
    return any(re.search(pattern, line) for pattern in comparison_patterns)

def fix_strategies_file():
    """Read and fix the strategies.py file"""
    try:
        with open('strategies.py', 'r') as f:
            content = f.read()
        
        # Add safe_numeric_value function to the file after imports
        import_section_end = content.find('logger = logging.getLogger(__name__)') + len('logger = logging.getLogger(__name__)')
        fixed_content = content[:import_section_end] + '\n\n'
        fixed_content += """def safe_numeric_value(value):
    \"\"\"Safely extract numeric value from pandas Series or scalar\"\"\"
    if isinstance(value, pd.Series):
        if len(value) > 0:
            return float(value.iloc[0])
        return 0.0
    return float(value)
"""
        fixed_content += content[import_section_end:]
        
        # Fix potential pandas Series comparisons
        lines = fixed_content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            fixed_line = line
            
            # Skip modifying comment lines and whitespace
            if line.strip().startswith('#') or not line.strip():
                fixed_lines.append(line)
                continue
                
            # Fix price_trend_up assignments
            if 'price_trend_up = last_forecast > current_price' in line:
                indent = line[:line.find('price_trend_up')]
                fixed_line = indent + 'price_trend_up = safe_numeric_value(last_forecast) > safe_numeric_value(current_price)'
            
            # Fix RSI comparisons
            elif 'rsi_oversold = latest[\'RSI\'] <' in line:
                indent = line[:line.find('rsi_oversold')]
                fixed_line = indent + 'rsi_oversold = safe_numeric_value(latest[\'RSI\']) < safe_numeric_value(self.parameters[\'oversold_threshold\'])'
            elif 'latest[\'RSI\'] > self.parameters[\'overbought_threshold\']' in line:
                indent = line[:line.find('elif') if 'elif' in line else line.find('if')]
                comparison = 'safe_numeric_value(latest[\'RSI\']) > safe_numeric_value(self.parameters[\'overbought_threshold\'])'
                fixed_line = line.replace('latest[\'RSI\'] > self.parameters[\'overbought_threshold\']', comparison)
            
            # Fix Close price comparisons
            elif 'latest[\'Close\'] >= take_profit_price' in line:
                fixed_line = line.replace('latest[\'Close\'] >= take_profit_price', 
                                        'safe_numeric_value(latest[\'Close\']) >= take_profit_price')
            elif 'latest[\'Close\'] <= stop_loss_price' in line:
                fixed_line = line.replace('latest[\'Close\'] <= stop_loss_price', 
                                        'safe_numeric_value(latest[\'Close\']) <= stop_loss_price')
            elif 'latest[\'Close\'] >= latest[\'BB_upper\']' in line:
                fixed_line = line.replace('latest[\'Close\'] >= latest[\'BB_upper\']', 
                                        'safe_numeric_value(latest[\'Close\']) >= safe_numeric_value(latest[\'BB_upper\'])')
            
            # Fix price_near_lower_band
            elif 'price_near_lower_band = latest[\'Close\'] <= latest[\'BB_lower\']' in line:
                indent = line[:line.find('price_near_lower_band')]
                fixed_line = indent + 'price_near_lower_band = safe_numeric_value(latest[\'Close\']) <= safe_numeric_value(latest[\'BB_lower\']) * 1.01'
            
            # Fix MACD comparisons
            elif 'macd_bullish = (latest[\'MACD\'] > latest[\'MACD_Signal\'])' in line:
                indent = line[:line.find('macd_bullish')]
                fixed_line = indent + 'macd_bullish = (safe_numeric_value(latest[\'MACD\']) > safe_numeric_value(latest[\'MACD_Signal\'])) and '
                fixed_line += 'abs(safe_numeric_value(latest[\'MACD\']) - safe_numeric_value(latest[\'MACD_Signal\'])) > self.parameters[\'signal_threshold\']'
            elif '(latest[\'MACD\'] < latest[\'MACD_Signal\'])' in line:
                comparison = '(safe_numeric_value(latest[\'MACD\']) < safe_numeric_value(latest[\'MACD_Signal\']))'
                fixed_line = line.replace('(latest[\'MACD\'] < latest[\'MACD_Signal\'])', comparison)
                fixed_line = fixed_line.replace('abs(latest[\'MACD\'] - latest[\'MACD_Signal\'])', 
                                             'abs(safe_numeric_value(latest[\'MACD\']) - safe_numeric_value(latest[\'MACD_Signal\']))')
            
            fixed_lines.append(fixed_line)
        
        # Write the fixed content back to a new file
        with open('strategies_fixed.py', 'w') as f:
            f.write('\n'.join(fixed_lines))
        
        logger.info(f"Fixed strategies file created at strategies_fixed.py. Apply this fix to make backtesting work properly.")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing strategies file: {e}")
        return False

if __name__ == "__main__":
    print("Applying fixes to handle pandas Series properly in strategies.py...")
    if fix_strategies_file():
        print("✅ Successfully created strategies_fixed.py")
        print("To use the fixed file, run: mv strategies_fixed.py strategies.py")
    else:
        print("❌ Failed to fix strategies.py")