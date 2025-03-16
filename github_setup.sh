#!/bin/bash
# GitHub Repository Setup Script for Stock Prophet
# This script helps initialize a GitHub repository for Stock Prophet

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI not found. Please install it to use this script."
    echo "Visit: https://cli.github.com/"
    exit 1
fi

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo "Git not found. Please install Git to use this script."
    exit 1
fi

# Check if user is authenticated with GitHub
if ! gh auth status &> /dev/null; then
    echo "Please authenticate with GitHub first:"
    gh auth login
fi

# Create a new repository on GitHub
echo "Creating a new GitHub repository for Stock Prophet..."
read -p "Repository name (default: stock-prophet): " repo_name
repo_name=${repo_name:-stock-prophet}

read -p "Repository visibility (public/private) [private]: " visibility
visibility=${visibility:-private}

echo "Creating repository: $repo_name ($visibility)"
gh repo create "$repo_name" --$visibility --source=. --remote=origin

# Initialize Git repository if not already done
if [ ! -d .git ]; then
    git init
fi

# Check for GitHub token as environment variable
if [ -z "$GITHUB_TOKEN" ]; then
    echo "Warning: GITHUB_TOKEN environment variable not found."
    echo "It's recommended to set up this token for GitHub Actions."
    read -p "Would you like to add it now? (y/n) [y]: " add_token
    add_token=${add_token:-y}
    
    if [ "$add_token" = "y" ]; then
        read -p "Enter your GitHub token: " token
        gh secret set GITHUB_TOKEN -b "$token"
    fi
fi

# Check for Telegram token
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "Warning: TELEGRAM_BOT_TOKEN environment variable not found."
    echo "This token is necessary for the Telegram bot functionality."
    read -p "Would you like to add it now? (y/n) [y]: " add_telegram
    add_telegram=${add_telegram:-y}
    
    if [ "$add_telegram" = "y" ]; then
        read -p "Enter your Telegram bot token: " token
        gh secret set TELEGRAM_BOT_TOKEN -b "$token"
    fi
fi

# Add, commit, and push changes
git add .
git commit -m "Initial commit for Stock Prophet"
git branch -M main
git push -u origin main

echo "Repository setup complete!"
echo "GitHub Actions workflows are now configured and will run automatically on pushes and PRs."
echo "Visit your repository at: https://github.com/$(gh api user | jq -r '.login')/$repo_name"