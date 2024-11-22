# Stock Price Analysis and Prediction

This project integrates data science, machine learning, and natural language processing (NLP) to analyze and predict stock prices. It provides a comprehensive toolkit for users to understand market trends, predict future stock prices, and make informed investment decisions.

## Key Features:

1. **Historical Stock Data Fetching**: 
   - Uses the `yfinance` library to fetch historical data for any stock ticker over a specified period.

2. **Data Processing**:
   - Calculates essential financial metrics such as 20-day and 50-day moving averages, daily returns, and percentage changes.
   - Processes the data to ensure it's ready for machine learning tasks.

3. **Predictive Modeling**:
   - Employs a `RandomForestRegressor` model to predict the next day’s closing price based on historical trends and computed features.
   - The model is saved and reused to avoid retraining for the same stock ticker.

4. **Actionable Recommendations**:
   - Compares the predicted price with the current price to provide actionable investment advice (Buy, Sell, or Hold).

5. **LLM-Based Insights**:
   - Uses a pre-trained Hugging Face model to analyze processed stock data and generate human-readable investment insights.
   - Summarizes key statistics like closing prices and moving averages to assist in decision-making.

6. **User Interaction**:
   - A command-line interface (CLI) allows users to input stock tickers and specify the analysis period.

This project is designed for finance enthusiasts, data scientists, and investors who want to leverage AI tools to gain deeper insights into stock market trends and make data-driven decisions.
