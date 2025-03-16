#!/usr/bin/env python3

import os
import sys
sys.path.append('.')
from main import get_enhanced_gold_analysis

# Test enhanced gold analysis
gold_price, gld_data, indicators, forecast_values, signals, trend = get_enhanced_gold_analysis()

print('\n=== ENHANCED GOLD ANALYSIS ===\n')
print(f'Gold Price: ${gold_price:.2f}')
print(f'Trend: {trend}')

print('\n=== TECHNICAL INDICATORS ===\n')
if indicators is not None:
    for key in ['RSI', 'EMA_9', 'EMA_21', 'MACD', 'MACD_Signal', 'STOCHk_3_2_2', 'STOCHd_3_2_2']:
        if key in indicators:
            print(f'{key}: {indicators[key]:.4f}')

print('\n=== PRICE FORECAST ===\n')
if forecast_values:
    for i, val in enumerate(forecast_values[:5]):
        print(f'Hour {i+1}: ${val*10:.2f}')

print('\n=== TRADING SIGNALS ===\n')
for signal in signals:
    print(signal)
if not signals:
    print('No strong signals detected')

print('\n=== END OF TEST ===\n')