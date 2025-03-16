#!/usr/bin/env python3
"""
Test script for gold price analysis and SMS simulation without Twilio
"""

import os
import argparse
from datetime import datetime

from sms_notifications import is_twilio_configured
from test_gold_spot import get_gold_spot_price, get_gold_stats
from sms_notifications import send_gold_price_sms

def format_message(gold_price, price_change=0.75):
    """Format a gold price message for display"""
    # Calculate ETF price (approximately 1/10 of spot price)
    gld_etf_price = gold_price / 10
    
    # Calculate trading levels
    take_profit_price = gold_price * 1.015  # Default 1.5% take profit
    stop_loss_price = gold_price * 0.98    # Default 2% stop loss
    
    # Calculate derived values
    gram_price = gold_price / 31.1035  # Troy ounce to grams
    kg_price = gram_price * 1000  # Gram to kg
    
    # Direction indicators based on price change
    if price_change > 0:
        direction = "📈"
        change_text = f"+{price_change:.2f}%"
    elif price_change < 0:
        direction = "📉"
        change_text = f"{price_change:.2f}%"
    else:
        direction = "➡️"
        change_text = "Unchanged"
    
    # Format the date and time
    current_time = datetime.now().strftime("%b %d, %Y at %H:%M")
    
    # Build the message for display
    message = []
    message.append(f"GOLD SPOT PRICE UPDATE {direction}")
    message.append(f"Date: {current_time}")
    message.append("")
    message.append(f"XAU/USD: ${gold_price:.2f} per troy ounce")
    message.append(f"24h Change: {change_text}")
    message.append(f"GLD ETF: ${gld_etf_price:.2f}")
    message.append("")
    message.append(f"Gold Weight Conversions:")
    message.append(f"1 gram: ${gram_price:.2f}")
    message.append(f"1 kg: ${kg_price:.2f}")
    message.append("")
    message.append(f"Trading Levels (Gold Strategy):")
    message.append(f"Take Profit: ${take_profit_price:.2f}")
    message.append(f"Stop Loss: ${stop_loss_price:.2f}")
    message.append("")
    message.append(f"Mr. Otmane, use these price levels in your gold trading strategy!")
    
    return "\n".join(message)

def main():
    """Main function to demonstrate gold price analysis and messaging without Twilio"""
    parser = argparse.ArgumentParser(description='Test gold price analysis and messaging without Twilio')
    parser.add_argument('--phone', '-p', help='Phone number to simulate sending SMS to (in E.164 format, e.g., +1234567890)')
    parser.add_argument('--ci', '-c', action='store_true', help='Run in CI mode with fixed values')
    
    args = parser.parse_args()
    
    # Set CI environment variable if --ci flag is provided
    if args.ci:
        os.environ['CI'] = 'true'
        os.environ['TEST_MODE'] = 'true'
        if 'GOLD_PRICE' not in os.environ:
            os.environ['GOLD_PRICE'] = '2984.91'
    
    # Set default phone for simulation if not provided
    phone = args.phone or "+12345678900"
    
    # Print the status of Twilio configuration
    if is_twilio_configured():
        print("Twilio is configured. Real SMS could be sent if requested.")
    else:
        print("Twilio is not configured. Running in simulation mode.")
        print("SMS messages will be simulated but not actually sent.")
    
    print("\n=== GETTING GOLD SPOT PRICE ===")
    
    # Get the gold price
    try:
        gold_price = get_gold_spot_price()
        if gold_price:
            print(f"Successfully retrieved gold price: ${gold_price:.2f} per troy ounce")
            
            # Calculate our own gold statistics
            gram_price = gold_price / 31.1035  # Troy ounce to grams
            kg_price = gram_price * 1000  # Gram to kg
            take_profit = gold_price * 1.015  # +1.5%
            stop_loss = gold_price * 0.98  # -2%
            
            print(f"1 gram cost: ${gram_price:.2f}")
            print(f"1 kg cost: ${kg_price:.2f}")
            print(f"Take profit level: ${take_profit:.2f}")
            print(f"Stop loss level: ${stop_loss:.2f}")
            
            # Calculate ETF price (approximately 1/10 of spot price)
            gld_etf_price = gold_price / 10
            print(f"Approximate GLD ETF price: ${gld_etf_price:.2f}")
            
            # Format a full message for display
            print("\n=== FORMATTED MESSAGE ===")
            msg = format_message(gold_price)
            print(msg)
            
            # Simulate sending SMS
            print("\n=== SIMULATING SMS DELIVERY ===")
            print(f"Simulating sending gold price SMS to {phone}...")
            
            # For demo purposes, assume a price change of +0.75%
            result = send_gold_price_sms(phone, gold_price, gld_etf_price, 0.75)
            
            if result:
                if is_twilio_configured():
                    print("SMS sent successfully!")
                else:
                    print("SMS simulation successful! (Twilio not configured - no real SMS was sent)")
            else:
                print("Failed to simulate SMS. Check the logs for details.")
        else:
            print("Error: Failed to retrieve gold price.")
            return
    except Exception as e:
        print(f"Error: {e}")
        return
    
    print("\n=== TEST COMPLETED SUCCESSFULLY ===")
    
if __name__ == "__main__":
    main()