#!/usr/bin/env python3
"""
Test script for gold price SMS notifications in Stock Prophet
"""

import os
import argparse
from sms_notifications import send_gold_price_sms, test_gold_sms_service
from test_gold_spot import get_gold_spot_price

def main():
    """Main function to test gold price SMS notifications"""
    parser = argparse.ArgumentParser(description='Test gold price SMS notifications for Stock Prophet')
    parser.add_argument('--phone', '-p', help='Phone number to send test SMS to (in E.164 format, e.g., +1234567890)')
    parser.add_argument('--test-mode', '-t', action='store_true', help='Use test mode with sample data instead of real gold price')
    
    args = parser.parse_args()
    
    # Check if Twilio is configured
    from sms_notifications import is_twilio_configured
    
    if not is_twilio_configured():
        print("Twilio is not configured. Running in simulation mode.")
        print("SMS messages will be simulated but not actually sent.")
        print("To enable real SMS, configure the following environment variables:")
        print("TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER")
    
    # If phone number is not provided, use a test number
    phone = args.phone
    if not phone:
        print("No phone number provided. Checking if there's a default in environment...")
        phone = os.environ.get("DEFAULT_PHONE_NUMBER")
        
    if not phone:
        print("No phone number provided. Using a test number for simulation.")
        phone = "+12345678900"  # Test phone number for simulation mode
    
    # Get real gold price first to display the accurate data in both cases
    try:
        real_gold_price = get_gold_spot_price()
        if not real_gold_price:
            print("Warning: Failed to retrieve real gold price, using sample data.")
            real_gold_price = 2984.91  # Use sample data as fallback
    except Exception as e:
        print(f"Warning: Error retrieving gold price: {e}")
        real_gold_price = 2984.91  # Use sample data as fallback

    # Calculate ETF price
    real_gld_etf_price = real_gold_price / 10  # Approximate GLD ETF price

    # Run the appropriate test
    if args.test_mode:
        print(f"Sending test gold price SMS to {phone}...")
        result = test_gold_sms_service(phone)
        # Override variables for display
        gold_price = real_gold_price
        gld_etf_price = real_gld_etf_price
    else:
        print(f"Sending real gold price SMS to {phone}...")
        # For demo purposes, assume a price change of +0.75%
        result = send_gold_price_sms(phone, real_gold_price, real_gld_etf_price, 0.75)
        # Set variables for display
        gold_price = real_gold_price
        gld_etf_price = real_gld_etf_price
    
    # Display simulated or real SMS result
    if is_twilio_configured():
        if result:
            print("SMS sent successfully!")
        else:
            print("Failed to send SMS. Check the logs for details.")
    else:
        print("SMS simulation successful! (Twilio not configured - no real SMS was sent)")
        
    # Always show message preview, whether real or simulated
    print("\nMessage content preview:")
    print("=================================")
    print(f"GOLD SPOT PRICE UPDATE 📈")
    if 'gold_price' in locals():
        print(f"XAU/USD: ${gold_price:.2f} per troy ounce")
    else:
        print("XAU/USD: $2,984.91 per troy ounce (sample data)")
    print(f"24h Change: +0.75%")
    if 'gld_etf_price' in locals() and gld_etf_price:
        print(f"GLD ETF: ${gld_etf_price:.2f}")
    else:
        print("GLD ETF: $298.49 (sample data)")
    print("=================================")

if __name__ == "__main__":
    main()