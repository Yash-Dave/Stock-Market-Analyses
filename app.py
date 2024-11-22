import streamlit as st
import os
from llm_utils import (
    fetch_stock_data,
    process_stock_data,
    prepare_features,
    train_predictive_model,
    load_model,
    predict_next_day,
    analyze_stock_data,
    llm_stock_analysis,
    make_recommendation,
)
import matplotlib.pyplot as plt

# Ensure models folder exists
if not os.path.exists("models"):
    os.makedirs("models")

# App Title
st.title("Stock Analysis and Prediction App")

# Sidebar
st.sidebar.header("User Input Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL").upper()
years = st.sidebar.slider("Select Analysis Period (Years)", min_value=1, max_value=20, value=5)

# Main App Logic
if st.button("Analyze Stock"):
    try:
        # Fetch and process data
        stock_data = fetch_stock_data(ticker, years)
        if stock_data.empty:
            st.error(f"No data found for {ticker}. Please check the ticker.")
            st.stop()
        processed_data = process_stock_data(stock_data)

        # Train or load model
        model_path = os.path.join("models", f"{ticker}_model.pkl")
        X_train, X_test, y_train, y_test = prepare_features(processed_data)
        if not os.path.exists(model_path):
            prediction_model = train_predictive_model(X_train, y_train, X_test, y_test, model_path)
        else:
            prediction_model = load_model(model_path)

        # Prediction and Recommendation
        predicted_price = predict_next_day(prediction_model, processed_data)
        current_price = processed_data['Close'].iloc[-1]
        recommendation, percent_change = make_recommendation(current_price, predicted_price)
        # LLM Analysis
        llm_analysis = llm_stock_analysis(ticker, processed_data)

        st.subheader("Prediction and Recommendation")
        st.write(f"Predicted Next Day Price: ${predicted_price:.2f}")
        st.write(f"Recommendation: **{recommendation}** ({percent_change:.2f}% change)")
        with st.expander("Investment Analysis"):
            st.markdown(llm_analysis)

        st.write("Processed DataFrame Preview")
        st.dataframe(data.head())
        # Display stock data
        st.subheader(f"Stock Data for {ticker}")
        st.dataframe(processed_data)


         # Plot stock data
        def plot_price_data(data):
            plt.figure(figsize=(10, 6))
            plt.plot(data['Close'], label="Close Price", color='blue')
            plt.plot(data['MA20'], label="20-Day MA", color='orange')
            plt.plot(data['MA50'], label="50-Day MA", color='green')
            plt.legend()
            plt.title(f"{ticker} Stock Prices")
            plt.xlabel("Date")
            plt.ylabel("Price")
            st.pyplot(plt)

        plot_price_data(processed_data)

    

        # Download Processed Data
        csv = processed_data.to_csv().encode('utf-8')
        st.download_button(
            "Download Processed Data as CSV",
            data=csv,
            file_name=f"{ticker}_processed_data.csv",
            mime='text/csv',
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
