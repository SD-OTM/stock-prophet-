#!/usr/bin/env python3
"""
SMS Notification module for Stock Prophet
Sends SMS alerts for important trading signals using Twilio
"""

import os
import logging
import time
from datetime import datetime
from twilio.rest import Client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Twilio configuration from environment variables
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")

def is_twilio_configured():
    """Check if Twilio credentials are properly configured"""
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
        logger.warning("Twilio credentials not fully configured. SMS notifications will not be sent.")
        return False
    return True

def send_sms(to_phone_number, message):
    """
    Send an SMS notification using Twilio
    
    Args:
        to_phone_number: Recipient's phone number in E.164 format (+1XXXXXXXXXX)
        message: Text message to send
        
    Returns:
        Boolean indicating success or failure
    """
    message_id = f"SMS-{int(time.time())}"
    
    # Check if Twilio is configured
    if not is_twilio_configured():
        logger.error(f"ID: {message_id} - Cannot send SMS. Twilio not configured.")
        return False
    
    try:
        # Initialize Twilio client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Send the SMS
        twilio_message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to_phone_number
        )
        
        logger.info(f"ID: {message_id} - SMS sent successfully to {to_phone_number}, SID: {twilio_message.sid}")
        return True
    
    except Exception as e:
        logger.error(f"ID: {message_id} - Failed to send SMS to {to_phone_number}: {str(e)}")
        return False

def send_trading_signal_sms(user_phone, ticker, signal_type, price, strategy_name):
    """
    Send a formatted trading signal SMS alert
    
    Args:
        user_phone: User's phone number
        ticker: Stock symbol
        signal_type: 'BUY' or 'SELL'
        price: Current stock price
        strategy_name: Name of the strategy generating the signal
        
    Returns:
        Boolean indicating success or failure
    """
    # Format the message
    if signal_type.upper() == "BUY":
        emoji = "🚀"
        action = "BUY"
    else:
        emoji = "💰"
        action = "SELL"
    
    message = f"STOCK PROPHET ALERT {emoji}\n"
    message += f"{action} SIGNAL: {ticker} at ${price:.2f}\n"
    message += f"Strategy: {strategy_name}\n"
    message += f"Mr. Otmane, take action now to maximize your profits!"
    
    # Send the SMS
    return send_sms(user_phone, message)

def send_price_alert_sms(user_phone, ticker, price, change_percent, alert_type):
    """
    Send a price movement alert SMS
    
    Args:
        user_phone: User's phone number
        ticker: Stock symbol
        price: Current stock price
        change_percent: Percentage change that triggered the alert
        alert_type: 'SPIKE' or 'DROP'
        
    Returns:
        Boolean indicating success or failure
    """
    # Format the message
    if alert_type.upper() == "SPIKE":
        emoji = "📈"
        movement = "SPIKE"
    else:
        emoji = "📉"
        movement = "DROP"
    
    message = f"STOCK PROPHET PRICE ALERT {emoji}\n"
    message += f"{ticker} {movement}: ${price:.2f}\n"
    message += f"Change: {change_percent:.2f}%\n"
    message += f"Mr. Otmane, consider reviewing your position!"
    
    # Send the SMS
    return send_sms(user_phone, message)

def send_portfolio_summary_sms(user_phone, total_value, total_profit_loss, profit_loss_percent):
    """
    Send a portfolio performance summary SMS
    
    Args:
        user_phone: User's phone number
        total_value: Current portfolio value
        total_profit_loss: Total profit/loss amount
        profit_loss_percent: Profit/loss percentage
        
    Returns:
        Boolean indicating success or failure
    """
    # Format the message with appropriate emoji based on performance
    if total_profit_loss >= 0:
        emoji = "🟢"
        profit_text = f"+${total_profit_loss:.2f} (+{profit_loss_percent:.2f}%)"
    else:
        emoji = "🔴"
        profit_text = f"-${abs(total_profit_loss):.2f} ({profit_loss_percent:.2f}%)"
    
    message = f"STOCK PROPHET DAILY SUMMARY {emoji}\n"
    message += f"Portfolio Value: ${total_value:.2f}\n"
    message += f"Performance: {profit_text}\n"
    message += f"Mr. Otmane, your portfolio's daily update is ready."
    
    # Send the SMS
    return send_sms(user_phone, message)

def send_gold_price_sms(user_phone, gold_price, gold_etf_price=None, price_change=None):
    """
    Send a gold spot price update SMS
    
    Args:
        user_phone: User's phone number
        gold_price: Current gold spot price per troy ounce
        gold_etf_price: Optional GLD ETF price for comparison
        price_change: Optional price change from previous day (percentage)
        
    Returns:
        Boolean indicating success or failure
    """
    # Format the date and time
    current_time = datetime.now().strftime("%b %d, %Y at %H:%M")
    
    # Direction indicators
    if price_change:
        if price_change > 0:
            direction = "📈"
            change_text = f"+{price_change:.2f}%"
        elif price_change < 0:
            direction = "📉"
            change_text = f"{price_change:.2f}%"
        else:
            direction = "➡️"
            change_text = "Unchanged"
    else:
        direction = "🔄"
        change_text = "N/A"
    
    # Calculate trading levels (for gold strategy)
    take_profit_price = gold_price * 1.015  # Default 1.5% take profit
    stop_loss_price = gold_price * 0.98    # Default 2% stop loss
    
    # Build the message
    message = f"GOLD SPOT PRICE UPDATE {direction}\n"
    message += f"Date: {current_time}\n\n"
    message += f"XAU/USD: ${gold_price:.2f} per troy ounce\n"
    message += f"24h Change: {change_text}\n"
    
    # Include ETF price if available
    if gold_etf_price:
        message += f"GLD ETF: ${gold_etf_price:.2f}\n"
    
    # Add trading information
    message += f"\nTrading Levels (Gold Strategy):\n"
    message += f"Take Profit: ${take_profit_price:.2f}\n"
    message += f"Stop Loss: ${stop_loss_price:.2f}\n\n"
    
    # Message for Mr. Otmane
    message += f"Mr. Otmane, use these price levels in your gold trading strategy!"
    
    # Send the SMS
    return send_sms(user_phone, message)

# Function to test SMS functionality
def test_sms_service(test_phone_number):
    """
    Send a test SMS to verify the service is working
    
    Args:
        test_phone_number: Phone number to send test message to
        
    Returns:
        Boolean indicating success or failure
    """
    test_message = "This is a test message from Stock Prophet. Mr. Otmane, your SMS notification service is working correctly!"
    
    logger.info(f"Sending test SMS to {test_phone_number}")
    return send_sms(test_phone_number, test_message)

def test_gold_sms_service(test_phone_number):
    """
    Send a test gold price SMS alert
    
    Args:
        test_phone_number: Phone number to send test message to
        
    Returns:
        Boolean indicating success or failure
    """
    # Use the current market gold price
    gold_price = 2984.91  # Current market price provided by user
    gld_etf_price = 298.49  # GLD ETF is approximately 1/10 of gold price
    
    logger.info(f"Sending test gold price SMS to {test_phone_number}")
    return send_gold_price_sms(test_phone_number, gold_price, gld_etf_price, 0.75)