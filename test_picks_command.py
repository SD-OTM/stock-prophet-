#!/usr/bin/env python3

"""
Test script to simulate the /picks command
For testing MultiIndex handling in daily_pick_scanner.py
"""

import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the daily_picks_command function
try:
    from daily_pick_scanner import daily_picks_command
    logger.info("Successfully imported daily_picks_command")
except ImportError as e:
    logger.error(f"Error importing daily_picks_command: {e}")
    sys.exit(1)

def main():
    """Main function to test picks command"""
    logger.info("Starting test_picks_command.py")
    
    # Simulate user ID (None for default watchlist)
    user_id = "test_user"  # Could also use 'None' for non-personalized response
    
    try:
        # Call the function and print results
        result = daily_picks_command(user_id=user_id)
        
        # Print first 1000 characters for preview
        logger.info(f"Result preview: {result[:1000]}...")
        logger.info(f"Total result length: {len(result)} characters")
        
        # Print full result
        print("\n" + "="*80)
        print("PICKS COMMAND RESULT:")
        print("="*80)
        print(result)
        print("="*80 + "\n")
        
        logger.info("Successfully completed test")
        return 0
    except Exception as e:
        logger.error(f"Error testing daily_picks_command: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())