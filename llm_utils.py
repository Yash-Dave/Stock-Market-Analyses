import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from transformers import pipeline
import os
import joblib  # To save and load the model

# Ensure the models folder exists
if not os.path.exists("models"):
    os.makedirs("models")


def llm_stock_analysis(stock_ticker, processed_data):
    """
    Use a pre-trained LLM to analyze stock data and provide investment insights.
    """
    print("Generating LLM-based analysis...")

    # Initialize the Hugging Face pipeline (using GPT-2 as an example)
    model = pipeline("text-generation", model="distilgpt2")

    # Create a summary prompt with the stock's key statistics
    latest_close = processed_data['Close'].iloc[-1]
    ma20 = processed_data['MA20'].iloc[-1]
    ma50 = processed_data['MA50'].iloc[-1]
    percent_change = processed_data['Percent Change'].iloc[-1]

    prompt = (f"The latest closing price is {latest_close:.2f}. \n\n"
              f"The 20-day moving average is {ma20:.2f}, and the 50-day moving average is {ma50:.2f}. \n\n"
              f"The daily percent change is {percent_change:.2f}%. Based on this information, \n\n")
    # Generate a response
    response = model(prompt, max_length=150, num_return_sequences=1)
    return response[0]['generated_text']


def prepare_features(data):
    """
    Add features for predictive modeling and split the data into training and testing sets.
    """
    print("Preparing features for prediction...")

    # Create lag features
    data['Prev_Close'] = data['Close'].shift(1)
    data['Price_Change'] = data['Close'] - data['Prev_Close']
    data['Percent_Change'] = (data['Price_Change'] / data['Prev_Close']) * 100  # Correct


    # Drop rows with missing values
    data.dropna(inplace=True)

    # Define features and target
    features = ['MA20', 'MA50', 'Prev_Close', 'Price_Change', 'Percent_Change']
    target = 'Close'

    # Split data into training and testing sets
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Features prepared successfully.")
    return X_train, X_test, y_train, y_test


def train_predictive_model(X_train, y_train, X_test, y_test, model_path):
    """
    Train a Random Forest model to predict the next day's stock price.
    """
    print("Training the predictive model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Predict on the test set
    predictions = model.predict(X_test)

    # Evaluate model performance
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model trained. Mean Absolute Error: {mae:.2f}")

    return model


def load_model(model_path):
    """
    Load a trained model from the models folder.
    """
    print(f"Loading model from {model_path}...")
    return joblib.load(model_path)


def predict_next_day(model, recent_data):
    """
    Predict the next day's stock price using the trained model and recent data.
    """
    print("Predicting the next day's stock price...")
    next_day_features = recent_data[['MA20', 'MA50', 'Prev_Close', 'Price_Change', 'Percent_Change']].iloc[-1].values.reshape(1, -1)
    next_day_prediction = model.predict(next_day_features)[0]
    return next_day_prediction

def make_recommendation(current_price, predicted_price):
    """
    Recommend Buy, Hold, or Sell based on the current price and predicted price.
    """
    change_percentage = ((predicted_price - current_price) / current_price) * 100

    if change_percentage > 1:
        return "Buy", change_percentage
    elif change_percentage < -1:
        return "Sell", change_percentage
    else:
        return "Hold", change_percentage


def fetch_stock_data(ticker, years):
    """
    Fetch historical stock data for a given ticker based on duration in years.
    """
    print(f"Fetching data for {ticker} over the past {years} years...")
    end_date = datetime.now()  # Today's date
    start_date = end_date - timedelta(days=years * 365)  # Approximation for 'years'
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    print(f"Data for {ticker} fetched successfully.")
    return data


def process_stock_data(data):
    """
    Process stock data to add moving averages, daily returns, and percentage changes.
    """
    print("Processing data...")
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['Daily Returns'] = data['Close'].pct_change()
    data['Percent Change'] = data['Daily Returns'] * 100
    data.dropna(inplace=True)  # Drop rows with missing values due to rolling computations
    print("Data processed successfully.")
    return data


def analyze_stock_data(data):
    """
    Analyze the processed stock data and print key statistics and insights.
    """
    print("\n--- Stock Data Analysis ---")
    print(f"Max Close Price: {data['Close'].max():.2f}")
    print(f"Min Close Price: {data['Close'].min():.2f}")
    print(f"Mean Close Price: {data['Close'].mean():.2f}")
    print(f"Standard Deviation of Close Price: {data['Close'].std():.2f}")
    print(f"Latest Close Price: {data['Close'].iloc[-1]:.2f}")
    print(f"20-day Moving Average: {data['MA20'].iloc[-1]:.2f}")
    print(f"50-day Moving Average: {data['MA50'].iloc[-1]:.2f}")


if __name__ == "__main__":
    # Take input from the user
    ticker = input("Enter the stock ticker (e.g., AAPL for Apple): ").strip().upper()
    years = int(input("Enter the duration in years to analyze the stock data: ").strip())

    # File to save the trained model
    model_path = os.path.join("models", f"{ticker}_model.pkl")

    # Fetch and process stock data
    stock_data = fetch_stock_data(ticker, years)
    processed_data = process_stock_data(stock_data)

    # Prepare features and train-test split
    X_train, X_test, y_train, y_test = prepare_features(processed_data)

    # Train the prediction model
    if not os.path.exists(model_path):
        prediction_model = train_predictive_model(X_train, y_train, X_test, y_test, model_path)
    else:
        prediction_model = load_model(model_path)

    # Predict the next day's stock price
    next_day_prediction = predict_next_day(prediction_model, processed_data)
    print(f"\nPredicted next day's closing price: {next_day_prediction:.2f}")

    # Analyze stock data
    analyze_stock_data(processed_data)

    # Use LLM for investment analysis
    llm_analysis = llm_stock_analysis(ticker, processed_data)
    print("\n--- LLM Investment Analysis ---")
    print(llm_analysis)
