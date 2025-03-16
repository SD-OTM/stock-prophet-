#!/usr/bin/env python3
"""
Test script for SMS notifications in Stock Prophet
"""

import os
import argparse
from sms_notifications import (
    test_sms_service, 
    send_trading_signal_sms, 
    send_price_alert_sms,
    send_gold_price_sms,
    test_gold_sms_service
)

def main():
    """Main function to test SMS notifications"""
    parser = argparse.ArgumentParser(description='Test SMS notifications for Stock Prophet')
    parser.add_argument('--phone', '-p', help='Phone number to send test SMS to (in E.164 format, e.g., +1234567890)')
    parser.add_argument('--type', '-t', choices=['test', 'signal', 'price', 'gold', 'gold_test'], default='test',
                        help='Type of test message to send (default: test)')
    
    args = parser.parse_args()
    
    # Check if required environment variables are set
    if not all([os.environ.get("TWILIO_ACCOUNT_SID"), 
                os.environ.get("TWILIO_AUTH_TOKEN"), 
                os.environ.get("TWILIO_PHONE_NUMBER")]):
        print("Error: Twilio environment variables are not set.")
        print("Please set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER.")
        return
    
    # If phone number is not provided, try to use a default one
    phone = args.phone
    if not phone:
        print("No phone number provided. Checking if there's a default in environment...")
        phone = os.environ.get("DEFAULT_PHONE_NUMBER")
        
    if not phone:
        print("Error: No phone number provided. Use --phone option.")
        return
    
    print(f"Sending {args.type} SMS to {phone}...")
    
    # Send the appropriate type of test message
    result = False
    if args.type == 'test':
        result = test_sms_service(phone)
    elif args.type == 'signal':
        result = send_trading_signal_sms(phone, "AAPL", "BUY", 180.75, "Combined Strategy")
    elif args.type == 'price':
        result = send_price_alert_sms(phone, "TSLA", 850.50, 5.23, "SPIKE")
    elif args.type == 'gold':
        # Get the gold price from test_gold_spot.py
        try:
            from test_gold_spot import get_gold_spot_price
            gold_price = get_gold_spot_price()
            if gold_price:
                gld_etf_price = gold_price / 10  # Approximate GLD ETF price
                result = send_gold_price_sms(phone, gold_price, gld_etf_price, 0.75)
            else:
                print("Error: Failed to retrieve gold price.")
                return
        except Exception as e:
            print(f"Error retrieving gold price: {e}")
            return
    elif args.type == 'gold_test':
        result = test_gold_sms_service(phone)
    
    if result:
        print("SMS sent successfully!")
    else:
        print("Failed to send SMS. Check the logs for details.")

if __name__ == "__main__":
    main()