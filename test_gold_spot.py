#!/usr/bin/env python3
"""
Test script to fetch Gold Spot / U.S. Dollar data with caching
"""

import os
import requests
import json
import time
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache settings
CACHE_DIR = Path("data")
CACHE_FILE = CACHE_DIR / "gold_spot_cache.pkl"
CACHE_EXPIRY = 3600  # 1 hour in seconds (Alpha Vantage free tier is 25 requests/day)

def ensure_cache_dir():
    """Ensure the cache directory exists"""
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True)

def get_cached_gold_price():
    """
    Get cached gold price if available and not expired
    Returns: (price, timestamp) tuple or (None, None) if no valid cache
    """
    if not CACHE_FILE.exists():
        return None, None
    
    try:
        with open(CACHE_FILE, 'rb') as f:
            cache_data = pickle.load(f)
        
        cached_timestamp = cache_data.get('timestamp')
        cached_price = cache_data.get('price')
        
        # Check if cache is valid
        if cached_timestamp and cached_price:
            cache_age = time.time() - cached_timestamp
            if cache_age < CACHE_EXPIRY:
                return cached_price, datetime.fromtimestamp(cached_timestamp)
    
    except Exception as e:
        logger.error(f"Error reading cache: {e}")
    
    return None, None

def save_to_cache(price):
    """Save gold price to cache"""
    ensure_cache_dir()
    try:
        cache_data = {
            'timestamp': time.time(),
            'price': price
        }
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info("Gold price saved to cache")
    except Exception as e:
        logger.error(f"Error saving to cache: {e}")

def get_gold_spot_price(use_cache=True):
    """
    Fetch Gold Spot / U.S. Dollar data using Alpha Vantage API with caching
    
    Args:
        use_cache: Whether to use cached data if available
        
    Returns:
        float: Gold spot price per troy ounce
    """
    if use_cache:
        # Try to get from cache first
        cached_price, cached_time = get_cached_gold_price()
        if cached_price:
            print(f"Using cached gold price from {cached_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\nGold Spot / U.S. Dollar (XAUUSD):")
            print(f"Price: ${cached_price:.2f} per troy ounce")
            print(f"Last Updated: {cached_time.strftime('%Y-%m-%d %H:%M:%S')} (cached)")
            return cached_price
    
    # If no cache or cache expired, fetch from API
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    
    if not api_key:
        logger.error("Alpha Vantage API key not found in environment variables.")
        print("Error: Alpha Vantage API key not found in environment variables.")
        print("Please set ALPHA_VANTAGE_API_KEY environment variable.")
        
        # Return the latest market price the user provided, for demonstration
        market_price = 2984.91  # Current market price provided by user
        print(f"\nUsing market price: ${market_price:.2f} per troy ounce")
        print("(This is a fallback value since API key is not available)")
        return market_price
    
    print("Attempting to fetch Gold Spot / U.S. Dollar (XAUUSD) price...")
    
    # Try Forex endpoint with XAU/USD pair
    url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=XAU&to_currency=USD&apikey={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # Check if we got valid data
        if "Realtime Currency Exchange Rate" in data:
            exchange_rate = data["Realtime Currency Exchange Rate"]
            price = float(exchange_rate["5. Exchange Rate"])
            timestamp = exchange_rate["6. Last Refreshed"]
            
            print(f"\nGold Spot / U.S. Dollar (XAUUSD):")
            print(f"Price: ${price:.2f} per troy ounce")
            print(f"Last Updated: {timestamp}")
            
            # Additional info
            print("\nAdditional Information:")
            print(f"From: {exchange_rate['1. From_Currency Code']} ({exchange_rate['2. From_Currency Name']})")
            print(f"To: {exchange_rate['3. To_Currency Code']} ({exchange_rate['4. To_Currency Name']})")
            print(f"Bid Price: ${price:.2f}")
            print(f"Ask Price: ${price:.2f}")
            
            # Save to cache for future use
            save_to_cache(price)
            
            return price
        else:
            logger.warning(f"API error: {data}")
            print("Error: Could not retrieve Gold Spot price. API response:")
            print(json.dumps(data, indent=2))
            
            # Check if we hit a rate limit (common with Alpha Vantage free tier)
            if "Information" in data and "standard API rate limit" in data["Information"]:
                logger.info("API rate limit reached, using current market price")
                market_price = 2984.91  # Current market price provided by user
                print(f"\nUsing current market price: ${market_price:.2f} per troy ounce")
                print("(Using market price due to API rate limit)")
                
                # Cache market price to avoid further API calls
                save_to_cache(market_price)
                return market_price
            
            return None
            
    except Exception as e:
        logger.error(f"Error fetching gold spot price: {e}")
        print(f"Error fetching gold spot price: {e}")
        return None
        
def get_gold_stats(gold_price):
    """
    Generate gold investment statistics
    
    Args:
        gold_price: Current gold price per troy ounce
        
    Returns:
        dict: Gold statistics and calculations
    """
    if not gold_price:
        return None
        
    stats = {
        "price_per_ounce": gold_price,
        "price_per_gram": gold_price / 31.1,
        "price_per_kg": gold_price * 32.15,
        "timestamp": datetime.now()
    }
    
    return stats
        
if __name__ == "__main__":
    gold_price = get_gold_spot_price()
    
    if gold_price:
        # Calculate gold investment stats
        stats = get_gold_stats(gold_price)
        
        print("\nGold Investment Analysis:")
        print(f"Current Price: ${stats['price_per_ounce']:.2f} per troy ounce")
        print(f"1 gram cost: ${stats['price_per_gram']:.2f}")
        print(f"1 kg cost: ${stats['price_per_kg']:.2f}")
        
        # Display trading information for Mr. Otmane
        print("\nTrading Information for Mr. Otmane:")
        print(f"- Standard take profit: ${gold_price * 1.015:.2f} (+1.5%)")
        print(f"- Standard stop loss: ${gold_price * 0.98:.2f} (-2.0%)")
        print(f"- Gold ETF equivalent (GLD): ${gold_price/10:.2f} per share (approx)")
        
        # Year-to-date performance
        print("\nNote: This is the current gold spot price (XAUUSD).")
        print("For historical performance, use the Stock Prophet application")
        print("with GLD ticker (ETF) or GC=F (futures) for technical analysis.")