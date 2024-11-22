import streamlit as st
from llm_utils import (
    fetch_stock_data, process_stock_data, prepare_features,
    train_predictive_model, predict_next_day, llm_stock_analysis,save_model, load_model,
)
import os

def main():
    st.title("Stock Analysis and Prediction")
    st.write("An interactive tool for stock analysis and insights.")

    # Sidebar for inputs
    st.sidebar.header("Input Parameters")
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL").strip().upper()
    years = st.sidebar.slider("Select Duration (Years):", 1, 10, 5)

    # Buttons for actions
    if st.sidebar.button("Analyze"):
        # Fetch and process stock data
        stock_data = fetch_stock_data(ticker, years)
        processed_data = process_stock_data(stock_data)

        # Prepare features and train model
        X_train, X_test, y_train, y_test = prepare_features(processed_data)
        prediction_model = train_predictive_model(X_train, y_train, X_test, y_test)

        # Predict next day's stock price
        next_day_prediction = predict_next_day(prediction_model, processed_data)
        st.subheader("Next Day Prediction")
        st.write(f"Predicted next day's closing price: ${next_day_prediction:.2f}")

        # LLM investment analysis
        llm_analysis = llm_stock_analysis(ticker, processed_data)
        st.subheader("LLM Investment Analysis")
        st.write(llm_analysis)

        # Save model
        model_path = f"models/{ticker}_model.pkl"
        save_model(prediction_model, model_path)

if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")
    main()
