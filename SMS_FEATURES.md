# SMS Notification Features in Stock Prophet

## Overview

This document describes the SMS notification system in Stock Prophet and how it operates without requiring Twilio installation or credentials.

## SMS Modules

The SMS notification system uses the following components:

1. **sms_notifications.py** - Core module for SMS functionality
2. **test_gold_sms.py** - Test script for gold price SMS alerts
3. **test_gold_without_twilio.py** - Test script that demonstrates SMS simulation

## How the System Works

### Conditional Imports

The system uses conditional imports to make Twilio optional:

```python
try:
    from twilio.rest import Client
    TWILIO_IMPORT_SUCCESS = True
except ImportError:
    TWILIO_IMPORT_SUCCESS = False
    Client = None  # Define a placeholder
```

### Simulation Mode

When Twilio is not installed or not configured, the system automatically falls back to simulation mode:

```python
def is_twilio_configured():
    """Check if Twilio credentials are properly configured"""
    # Always return False if running in CI environment
    if os.environ.get('CI') == 'true':
        logger.info("Running in CI environment - SMS simulation mode activated")
        return False
    
    # Check if Twilio was successfully imported
    if not TWILIO_IMPORT_SUCCESS:
        logger.warning("Twilio package not installed. SMS notifications will not be sent.")
        return False
        
    # Check if credentials are available
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
        logger.warning("Twilio credentials not fully configured. SMS notifications will not be sent.")
        return False
    return True
```

### Simulated SMS Display

When in simulation mode, SMS messages are formatted and displayed as debug output:

```
=== SIMULATED SMS MESSAGE ===
TO: +12345678900
FROM: STOCK-PROPHET
TIME: 2025-03-16 20:04:33

GOLD SPOT PRICE UPDATE 📈
Date: Mar 16, 2025 at 20:04
XAU/USD: $2984.91 per troy ounce
24h Change: +0.75%
GLD ETF: $298.49

Trading Levels (Gold Strategy):
Take Profit: $3029.68
Stop Loss: $2925.21

Mr. Otmane, use these price levels in your gold trading strategy!
=== END OF MESSAGE ===
```

## SMS Types

The system supports several types of SMS messages:

1. **Trading Signals** - Buy/sell signals from trading strategies
2. **Price Alerts** - Notifications about significant price movements
3. **Portfolio Summaries** - Daily portfolio performance reports
4. **Gold Price Alerts** - Specialized alerts for gold price movements

## Gold-specific SMS Features

The gold price SMS notifications provide:

- Current gold spot price (XAU/USD)
- 24-hour price change percentage
- GLD ETF equivalent price
- Weight conversions (gram, kilogram)
- Trading levels (take profit, stop loss)

## Testing SMS Features

You can test the SMS features without Twilio:

```bash
# Run the test gold SMS script in simulation mode
python test_gold_without_twilio.py

# Run in CI simulation mode
CI=true python test_gold_without_twilio.py
```

## Adding New SMS Features

When adding new SMS notification types:

1. Create a new function in `sms_notifications.py`
2. Format the message appropriately
3. Call the `send_sms()` function with the message
4. Add a test function that can run in simulation mode

## Production Use

For production use, you need to set up Twilio and configure:

1. `TWILIO_ACCOUNT_SID`
2. `TWILIO_AUTH_TOKEN` 
3. `TWILIO_PHONE_NUMBER`

Once configured, the system will send real SMS messages instead of simulating them.