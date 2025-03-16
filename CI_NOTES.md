# Stock Prophet CI/CD Setup Notes

## Overview

This document outlines how the CI/CD pipeline is set up for Stock Prophet, particularly focusing on how tests run without requiring external service credentials like Twilio.

## CI Environment Variables

In CI environments, the following environment variables are set:

- `CI=true` - Indicates the code is running in a CI environment
- `TEST_MODE=true` - Additional flag to enable test-specific behaviors
- `GOLD_PRICE=2984.91` - Fixed gold price for consistent testing
- `ALPHA_VANTAGE_API_KEY` - API key for Alpha Vantage (from GitHub Secrets)
- `TELEGRAM_BOT_TOKEN` - Token for Telegram Bot (from GitHub Secrets)

## Conditional Module Dependencies

To ensure the codebase can run in CI environments without all optional dependencies installed:

1. The code uses conditional imports for modules like Twilio that might not be installed:
   ```python
   try:
       from twilio.rest import Client
       TWILIO_IMPORT_SUCCESS = True
   except ImportError:
       TWILIO_IMPORT_SUCCESS = False
       Client = None  # Define a placeholder
   ```

2. All functionality that depends on these optional modules checks if the import was successful:
   ```python
   # Check if Twilio was successfully imported
   if not TWILIO_IMPORT_SUCCESS:
       logger.warning("Twilio package not installed. SMS notifications will not be sent.")
       return False
   ```

3. This approach allows the codebase to run without Twilio installed, while still providing full functionality when it is available.

## SMS Simulation Mode

The system is designed to run without Twilio credentials by using a "simulation mode" for SMS messages. This allows tests to verify SMS generation logic without actually sending messages.

### How SMS Simulation Works

1. The `is_twilio_configured()` function in `sms_notifications.py` checks for Twilio credentials, CI environment, and successful imports.
2. In CI mode, it always returns `False`, forcing simulation mode.
3. SMS messages are still generated and logged, but not sent to Twilio.
4. A visual representation is printed to the console for easier debugging.

### Testing SMS Features

Test the SMS functionality with:

```bash
# Run with CI simulation mode
python test_gold_without_twilio.py --ci

# Or set environment variables manually
TEST_MODE=true GOLD_PRICE=2984.91 python test_gold_without_twilio.py
```

## Gold Price Handling in Tests

To ensure consistent test results for gold-related features:

1. In CI environments, a fixed gold price (2984.91) is used instead of calling the API.
2. This price can be overridden by setting the `GOLD_PRICE` environment variable.
3. The test code in `test_gold_spot.py` detects CI/test environments and uses this fixed price.

## GitHub Actions Workflow

The GitHub Actions workflow `.github/workflows/stock-prophet-tests.yml` runs multiple tests:

1. Basic stock analysis
2. Gold-specific features
3. Strategy testing
4. Backtesting

All tests use the simulation modes when running in CI to avoid external API dependencies where possible.

## Adding New Tests

When adding new tests that require external services:

1. Add CI environment detection
2. Provide simulation modes for external services
3. Use fixed test data in CI environments
4. Update the workflow file to include your new test

## Troubleshooting CI Issues

If tests are failing in CI but working locally:

1. Check if your test is properly detecting the CI environment
2. Ensure simulation modes are triggered correctly
3. Verify that test data paths are correctly resolved in CI environment

Remember that CI runs with a clean environment each time, so any cached data or state from previous runs won't be available.