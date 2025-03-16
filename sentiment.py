"""
Sentiment Analysis module for Stock Prophet
Analyzes news and social media sentiment related to stocks
"""

import logging
import os
import json
import requests
from datetime import datetime, timedelta
import pandas as pd
import re
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Alpha Vantage API for News Sentiment
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")

# Check if running in CI environment
CI_ENV = os.environ.get('CI') == 'true'

# Sentiment score ranges
SENTIMENT_RANGES = {
    "Very Negative": (-1.0, -0.6),
    "Negative": (-0.6, -0.2),
    "Neutral": (-0.2, 0.2),
    "Positive": (0.2, 0.6),
    "Very Positive": (0.6, 1.0)
}

class SentimentAnalysis:
    """Class for analyzing sentiment from news and social media"""
    
    @staticmethod
    def get_news_sentiment(ticker, days=3):
        """
        Get news sentiment for a ticker using Alpha Vantage API
        
        Args:
            ticker: Stock symbol
            days: Number of days to look back for news
            
        Returns:
            Dictionary with sentiment data including scores and news items
        """
        # In CI environment, always return neutral sentiment to avoid API issues
        if CI_ENV:
            logger.info("Running in CI environment, skipping actual API call")
            return {"sentiment_score": 0, "sentiment_label": "Neutral", "news_count": 0, "news_items": []}
            
        if not ALPHA_VANTAGE_API_KEY:
            logger.warning("Alpha Vantage API key not set. Cannot retrieve news sentiment.")
            return {"sentiment_score": 0, "sentiment_label": "Neutral", "news_count": 0, "news_items": []}
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            time_from = start_date.strftime('%Y%m%dT0000')
            time_to = end_date.strftime('%Y%m%dT2359')
            
            # Alpha Vantage News API endpoint
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&time_from={time_from}&time_to={time_to}&apikey={ALPHA_VANTAGE_API_KEY}"
            
            response = requests.get(url)
            data = response.json()
            
            if "Error Message" in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return {"sentiment_score": 0, "sentiment_label": "Neutral", "news_count": 0, "news_items": []}
            
            # Extract sentiment data
            news_items = []
            total_sentiment = 0
            relevant_news_count = 0
            
            if "feed" in data:
                for article in data["feed"]:
                    # Check if the article is relevant to the ticker
                    is_relevant = False
                    article_sentiment = 0
                    
                    if "ticker_sentiment" in article:
                        for ticker_data in article["ticker_sentiment"]:
                            if ticker_data["ticker"] == ticker:
                                is_relevant = True
                                article_sentiment = float(ticker_data["ticker_sentiment_score"])
                                break
                    
                    if is_relevant:
                        relevant_news_count += 1
                        total_sentiment += article_sentiment
                        
                        # Get article details
                        news_items.append({
                            "title": article.get("title", "No Title"),
                            "time_published": article.get("time_published", ""),
                            "url": article.get("url", ""),
                            "sentiment_score": article_sentiment,
                            "sentiment_label": SentimentAnalysis.get_sentiment_label(article_sentiment)
                        })
            
            # Calculate average sentiment
            avg_sentiment = total_sentiment / relevant_news_count if relevant_news_count > 0 else 0
            sentiment_label = SentimentAnalysis.get_sentiment_label(avg_sentiment)
            
            logger.info(f"Retrieved {relevant_news_count} news items for {ticker} with average sentiment: {avg_sentiment:.2f} ({sentiment_label})")
            
            return {
                "sentiment_score": avg_sentiment,
                "sentiment_label": sentiment_label,
                "news_count": relevant_news_count,
                "news_items": news_items
            }
            
        except Exception as e:
            logger.error(f"Error fetching news sentiment for {ticker}: {e}")
            return {"sentiment_score": 0, "sentiment_label": "Neutral", "news_count": 0, "news_items": []}
    
    @staticmethod
    def get_sentiment_label(score):
        """
        Convert a sentiment score to a label
        
        Args:
            score: Sentiment score from -1.0 to 1.0
            
        Returns:
            String label describing the sentiment
        """
        for label, (lower, upper) in SENTIMENT_RANGES.items():
            if lower <= score < upper:
                return label
        
        # Default fallback
        return "Neutral"
    
    @staticmethod
    def get_sentiment_emoji(label):
        """
        Get an emoji representing the sentiment label
        
        Args:
            label: Sentiment label
            
        Returns:
            Emoji string
        """
        emoji_map = {
            "Very Negative": "😡",
            "Negative": "😕",
            "Neutral": "😐",
            "Positive": "😊",
            "Very Positive": "🥳"
        }
        
        return emoji_map.get(label, "😐")
    
    @staticmethod
    def generate_sentiment_summary(ticker, sentiment_data):
        """
        Generate a text summary of sentiment analysis
        
        Args:
            ticker: Stock symbol
            sentiment_data: Dictionary with sentiment data
            
        Returns:
            String with formatted summary
        """
        score = sentiment_data["sentiment_score"]
        label = sentiment_data["sentiment_label"]
        news_count = sentiment_data["news_count"]
        emoji = SentimentAnalysis.get_sentiment_emoji(label)
        
        summary = f"📰 *Sentiment Analysis for {ticker}*: {emoji}\n\n"
        
        if news_count == 0:
            summary += "No recent news found for this ticker.\n"
        else:
            summary += f"Based on {news_count} recent news articles:\n"
            summary += f"• Overall sentiment: {label} ({score:.2f})\n\n"
            
            # Add trading implications based on sentiment
            if score < -0.6:
                summary += "Trading implications: Extremely bearish news sentiment. Consider caution with new long positions.\n\n"
            elif score < -0.2:
                summary += "Trading implications: Bearish news sentiment. Watch for potential support levels.\n\n"
            elif score < 0.2:
                summary += "Trading implications: Neutral news sentiment. Rely more on technical indicators.\n\n"
            elif score < 0.6:
                summary += "Trading implications: Bullish news sentiment. Supports technical bullish signals.\n\n"
            else:
                summary += "Trading implications: Extremely bullish news sentiment. Watch for potential overextension.\n\n"
            
            # Include top 3 most recent news items
            if sentiment_data["news_items"]:
                summary += "*Recent News Headlines:*\n"
                for i, item in enumerate(sentiment_data["news_items"][:3]):
                    # Format the published time
                    if "time_published" in item and item["time_published"]:
                        try:
                            # Parse the time format (usually YYYYMMDDTHHMM)
                            time_str = item["time_published"]
                            if len(time_str) >= 8:
                                date_part = time_str[:8]
                                time_part = time_str[9:13] if 'T' in time_str else ""
                                
                                year = date_part[:4]
                                month = date_part[4:6]
                                day = date_part[6:8]
                                
                                hour = time_part[:2] if time_part else "00"
                                minute = time_part[2:4] if len(time_part) >= 4 else "00"
                                
                                formatted_time = f"{year}-{month}-{day} {hour}:{minute}"
                            else:
                                formatted_time = time_str
                        except:
                            formatted_time = item["time_published"]
                    else:
                        formatted_time = "Unknown time"
                    
                    # Add sentiment emoji for each headline
                    news_emoji = SentimentAnalysis.get_sentiment_emoji(item["sentiment_label"])
                    
                    summary += f"{i+1}. {news_emoji} {item['title']}\n"
                    summary += f"   ({formatted_time} | {item['sentiment_label']})\n"
        
        return summary

def get_sentiment_analysis(ticker):
    """
    Get sentiment analysis for a ticker
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Text summary of sentiment analysis
    """
    sentiment = SentimentAnalysis()
    sentiment_data = sentiment.get_news_sentiment(ticker)
    summary = sentiment.generate_sentiment_summary(ticker, sentiment_data)
    return summary, sentiment_data

if __name__ == "__main__":
    # Simple test for the sentiment analysis
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python sentiment.py TICKER")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    print(f"Getting sentiment analysis for {ticker}...")
    
    summary, _ = get_sentiment_analysis(ticker)
    print(summary)