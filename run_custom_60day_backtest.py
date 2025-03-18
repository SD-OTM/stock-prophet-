"""
Script to run 60-day backtest focusing on ZYXI, CRDF, CIFR, and ARVN
with optimized parameters targeting 2% profit per trade
"""

import logging
from custom_backtest import run_custom_backtests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to run the 60-day custom backtest"""
    logger.info("Starting 60-day custom backtest with 2% profit target")
    profitable_trades = run_custom_backtests()
    
    # Analyze results
    for ticker, trades in profitable_trades.items():
        if trades:
            avg_profit = sum(trade['profit_percent'] for trade in trades) / len(trades)
            logger.info(f"{ticker}: {len(trades)} trades with average profit of {avg_profit:.2f}%")
        else:
            logger.info(f"{ticker}: No profitable trades above 2% threshold")
    
    logger.info("60-day custom backtest completed")
    
if __name__ == "__main__":
    main()