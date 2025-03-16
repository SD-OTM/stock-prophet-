# Viewing GitHub Actions Logs: A Visual Guide

This guide demonstrates how to view the logs for your GitHub Actions CI/CD pipeline.

## Step 1: Navigate to your GitHub repository

Go to your GitHub repository in a web browser.

```
   https://github.com/[your-username]/stock-prophet
   
   +-------------------------------+
   |  GitHub - Stock Prophet       |
   +-------------------------------+
   |  Code  Issues  Pull Requests  |
   |  Actions  Projects  Wiki      |
   |                               |
   |  <your code will be here>     |
   |                               |
   +-------------------------------+
```

## Step 2: Access the Actions tab

Click on the "Actions" tab at the top of the repository page.

```
   +-------------------------------+
   |  GitHub - Stock Prophet       |
   +-------------------------------+
   |  Code  Issues  Pull Requests  |
----> Actions  Projects  Wiki      |
   |                               |
   +-------------------------------+
```

## Step 3: View workflow runs

You'll see a list of all workflow runs, with the most recent at the top.

```
   +-------------------------------+
   |  Actions - Stock Prophet      |
   +-------------------------------+
   |  All workflows  ▼             |
   |                               |
   |  ✅ Stock Prophet Tests       |
   |  3 minutes ago · main         |
   |                               |
   |  ✅ Stock Prophet Tests       |
   |  2 hours ago · main           |
   |                               |
   |  ❌ Stock Prophet Tests       |
   |  Yesterday · pull/42          |
   |                               |
   +-------------------------------+
```

## Step 4: Select a workflow run

Click on a workflow run to view its details.

```
   +-------------------------------+
   |  Run - Stock Prophet Tests    |
   +-------------------------------+
   |  ✅ Stock Prophet Tests       |
   |  3 minutes ago · main         |
   |                               |
   |  ✅ test (3m 24s)             |
   |                               |
   |  Summary  Jobs  Artifacts     |
   +-------------------------------+
```

## Step 5: View job details

Click on a job (like "test") to expand it and see the steps.

```
   +-------------------------------+
   |  Job - test                   |
   +-------------------------------+
   |  ✅ Set up job (14s)          |
   |  ✅ Checkout repository (5s)  |
   |  ✅ Set up Python (8s)        |
   |  ✅ Install dependencies (45s)|
   |  ✅ Run stock analysis tests  |
   |     (1m 32s)                  |
   |  ✅ Run portfolio test (18s)  |
   |  ✅ Test strategy             |
   |     functionality (22s)       |
   +-------------------------------+
```

## Step 6: View step logs

Click on any step to view its detailed logs.

```
   +-------------------------------+
   |  Run stock analysis tests     |
   +-------------------------------+
   |  Testing Apple stock analysis |
   |  Analyzing stock: AAPL        |
   |  2025-03-16 05:16:49,354 -    |
   |  main - INFO - Fetching data  |
   |  for AAPL with period=1d,     |
   |  interval=1h                  |
   |  ...                          |
   |                               |
   |  Testing Tesla stock analysis |
   |  Analyzing stock: TSLA        |
   |  ...                          |
   |                               |
   |  Testing NVIDIA stock analysis|
   |  Analyzing stock: NVDA        |
   |  ...                          |
   +-------------------------------+
```

## Step 7: Download complete logs (optional)

You can download the complete logs by clicking the three dots (⋮) and selecting "Download log".

```
   +-------------------------------+
   |  Run stock analysis tests     |
   +-------------------------------+
   |  ⋮ ▼                          |
   |  |---------------|            |
   |  | Download log  |            |
   |  |---------------|            |
   |                               |
   |  Testing Apple stock analysis |
   |  ...                          |
   +-------------------------------+
```

## Understanding the Logs

The logs provide valuable information about your CI/CD pipeline:

1. **Environment Setup**: Information about the Python environment and installed dependencies
2. **Test Execution**: Output of each test run for different stocks
3. **Errors and Warnings**: Any issues encountered during testing
4. **Performance Metrics**: Time taken for each step
5. **Overall Status**: Success or failure of the workflow

## Troubleshooting Common Issues

### Workflow Failures

If your workflow fails, look for error messages in red in the logs:

```
   +-------------------------------+
   |  Run stock analysis tests     |
   +-------------------------------+
   |  Testing Apple stock analysis |
   |  Analyzing stock: AAPL        |
   |                               |
   |  ❌ Error: ImportError: No     |
   |  module named 'yfinance'      |
   |                               |
   |  Command failed with exit     |
   |  code 1                       |
   +-------------------------------+
```

### Missing Secrets

If your workflow needs access to secrets (like TELEGRAM_BOT_TOKEN), ensure they're properly set up in your repository settings.

### API Rate Limits

Yahoo Finance and other APIs may have rate limits. If you see errors related to this, you might need to add delays between API calls or use authentication.

## Automated Notifications

You can set up notifications for workflow status:

1. Go to your GitHub profile settings
2. Navigate to "Notifications"
3. Under "GitHub Actions", choose your preferred notification method

With these settings, you'll receive alerts when workflows fail, helping you respond quickly to issues.