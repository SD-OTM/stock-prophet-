import os
import main
from main import run_telegram_bot
import patch_pandas_ta

if __name__ == "__main__":
    # Apply the patch for pandas_ta
    patch_pandas_ta.patch_pandas_ta()
    
    # Check if running on Railway/Heroku (PORT env variable is set)
    port = int(os.environ.get("PORT", 8443))
    
    # Get the app URL (to set webhook)
    app_url = os.environ.get("APP_URL", "")
    
    # Load environment variables and data
    main.load_watchlists()
    main.load_portfolios()
    main.load_user_strategies()
    main.load_phone_numbers()
    
    # Run the Telegram bot
    run_telegram_bot(is_heroku=True, port=port, url=app_url)
    
    print("Stock Prophet is running on Railway/Heroku!")