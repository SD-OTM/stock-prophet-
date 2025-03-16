import os
import sys
import json
import time
from main import (analyze_ticker, user_portfolios, add_to_portfolio, 
                 remove_from_portfolio, show_portfolio, save_portfolios)

def test_analyze():
    """
    Test the stock analysis functionality with a specific ticker
    """
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = "AAPL"  # Default ticker
    
    print(f"Analyzing stock: {ticker}")
    result = analyze_ticker(ticker)
    print("\nResults:")
    print(result)

def test_portfolio():
    """
    Test the portfolio management functionality
    """
    test_user_id = "test_user_123"
    test_ticker = "AAPL"
    test_price = 200.00
    test_quantity = 10
    
    # Create a mock Update and Context object for testing
    class MockUpdate:
        class MockMessage:
            class MockUser:
                def __init__(self, user_id):
                    self.id = user_id
                    
            def __init__(self, user_id):
                self.from_user = self.MockUser(user_id)
                
            def reply_text(self, text, parse_mode=None):
                print(f"[Bot] {text}")
                
        def __init__(self, user_id):
            self.message = self.MockMessage(user_id)
    
    class MockContext:
        def __init__(self, args):
            self.args = args
    
    # Test adding a stock to portfolio
    print("\n=== Testing Portfolio Management ===")
    print(f"\n1. Adding {test_quantity} shares of {test_ticker} at ${test_price}")
    
    mock_update = MockUpdate(test_user_id)
    mock_context = MockContext([test_ticker, str(test_price), str(test_quantity)])
    add_to_portfolio(mock_update, mock_context)
    
    # Test showing portfolio
    print("\n2. Viewing portfolio")
    mock_update = MockUpdate(test_user_id)
    mock_context = MockContext([])
    show_portfolio(mock_update, mock_context)
    
    # Test removing a stock from portfolio
    print("\n3. Removing stock from portfolio")
    mock_update = MockUpdate(test_user_id)
    mock_context = MockContext([test_ticker])
    remove_from_portfolio(mock_update, mock_context)
    
    # Verify portfolio is empty
    print("\n4. Verifying portfolio is empty")
    mock_update = MockUpdate(test_user_id)
    mock_context = MockContext([])
    show_portfolio(mock_update, mock_context)
    
    # Clean up test data
    if test_user_id in user_portfolios:
        del user_portfolios[test_user_id]
        save_portfolios()
    
    print("\n=== Portfolio Management Test Complete ===")

def main():
    """
    Main test function
    """
    if len(sys.argv) > 1 and sys.argv[1].lower() == "portfolio":
        test_portfolio()
    else:
        test_analyze()

if __name__ == "__main__":
    main()