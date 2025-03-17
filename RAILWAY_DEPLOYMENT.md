# Stock Prophet Railway Deployment Guide

This guide provides instructions for deploying the Stock Prophet application on Railway.app platform.

## Prerequisites

Before deploying to Railway, you'll need:

1. A [Railway.app](https://railway.app/) account
2. Your Telegram Bot Token from BotFather
3. An Alpha Vantage API key (for sentiment analysis and gold spot price)
4. Optionally, Twilio credentials for SMS notifications

## Deployment Steps

### 1. Fork or Clone the Repository

Ensure you have this repository available in your GitHub account.

### 2. Connect to Railway

1. Login to Railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Connect your GitHub account if not already connected
5. Select this repository
6. Choose the **railway-deployment** branch

### 3. Configure Environment Variables

In your Railway.app project dashboard, go to the "Variables" tab and add the following environment variables:

Required Variables:
- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token from BotFather
- `ALPHA_VANTAGE_API_KEY`: Your Alpha Vantage API key
- `APP_URL`: The URL of your Railway app (will be provided by Railway after initial deployment)

Optional Variables (for SMS functionality):
- `TWILIO_ACCOUNT_SID`: Your Twilio account SID
- `TWILIO_AUTH_TOKEN`: Your Twilio auth token
- `TWILIO_PHONE_NUMBER`: Your Twilio phone number

### 4. Configure Railway Service

After connecting your repository, you'll need to make some adjustments to the service settings:

1. Go to the "Settings" tab of your service
2. Under "Build & Deploy", make sure:
   - Builder: NIXPACKS
   - Start Command: `python heroku_main.py`
   - Restart Policy: ON_FAILURE
3. Important: Under "Files" make sure to rename `railway_requirements.txt` to `requirements.txt` in the Railway file system

### 5. Deploy

1. Railway will automatically start deploying your application once the repository is connected and settings are configured
2. Wait for the build and deployment to complete
3. Once deployed, copy the deployment URL (e.g., https://stock-prophet-production.up.railway.app/)
4. Update the `APP_URL` environment variable with this URL
5. Railway will automatically redeploy with the updated variable

### 6. Verify Deployment

1. Open Telegram and navigate to your bot
2. Type `/start` to begin interacting with your bot
3. If everything is configured correctly, you should receive a welcome message

## Troubleshooting

- **Webhook Issues**: If the bot isn't responding, check the logs in Railway to ensure the webhook is properly set up
- **Environment Variables**: Double-check all environment variables are correctly set
- **Logs**: Use the "Logs" section in your Railway dashboard to identify any errors

## Data Persistence

Railway provides ephemeral storage, which means that any data saved locally in the application will be lost when the application restarts. For production use, consider connecting a database service through Railway to store user data persistently.

## Maintenance

Railway automatically rebuilds and deploys your application when changes are pushed to the connected GitHub repository branch.