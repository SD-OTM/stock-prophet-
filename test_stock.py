import os
import sys
import json
import time
import pandas as pd
from main import (analyze_ticker, user_portfolios, add_to_portfolio,
                 remove_from_portfolio, show_portfolio, save_portfolios)
from backtesting import cmd_backtest

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

def test_backtest():
    """
    Test the backtesting functionality with a specific ticker
    """
    if len(sys.argv) < 3:
        print("Usage for backtesting: python test_stock.py backtest TICKER [STRATEGY] [START_DATE] [END_DATE] [TIMEFRAME]")
        print("Example: python test_stock.py backtest AAPL rsi 2024-01-01 2024-03-15 1d")
        return
        
    ticker = sys.argv[2].upper()
    strategy = sys.argv[3] if len(sys.argv) > 3 else "combined"
    start_date = sys.argv[4] if len(sys.argv) > 4 else None
    end_date = sys.argv[5] if len(sys.argv) > 5 else None
    timeframe = sys.argv[6] if len(sys.argv) > 6 else "1d"
    
    print(f"\n=== Backtesting {strategy.upper()} strategy on {ticker} ===")
    print(f"Period: {start_date or 'Default'} to {end_date or 'Default'}, Timeframe: {timeframe}")
    
    result = cmd_backtest(ticker, strategy, start_date, end_date, timeframe)
    
    if result:
        print("\nBacktest completed successfully.")
        print(f"Strategy Return: {result.strategy_return:.2f}%")
        
        # Safe handling of buy_and_hold_return
        if hasattr(result, 'buy_and_hold_return'):
            if isinstance(result.buy_and_hold_return, pd.Series):
                print(f"Buy & Hold Return: Not calculated (data issue)")
                # Use just the strategy return for outperformance
                print(f"Outperformance: N/A")
            else:
                print(f"Buy & Hold Return: {result.buy_and_hold_return:.2f}%")
                print(f"Outperformance: {result.strategy_return - result.buy_and_hold_return:.2f}%")
        else:
            print(f"Buy & Hold Return: Not calculated")
            print(f"Outperformance: N/A")
        print(f"Total Trades: {len(result.trades)}")
        
        # Check if chart was generated
        chart_file = f"backtest_charts/{ticker}_{strategy}_{start_date or 'auto'}_{end_date or 'auto'}.png"
        if os.path.exists(chart_file):
            print(f"Chart saved to: {chart_file}")
    else:
        print("Backtesting failed. See logs for details.")
    
    print("\n=== Backtesting Test Complete ===")

def main():
    """
    Main test function
    """
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "portfolio":
            test_portfolio()
        elif command == "backtest":
            test_backtest()
        else:
            test_analyze()
    else:
        test_analyze()

if __name__ == "__main__":
    main()