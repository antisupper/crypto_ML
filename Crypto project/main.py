#import
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

#load/clean data
def load_data():
    df = pd.read_csv("Cryptocurrency Historical prices/coin_Bitcoin.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.drop(["SNo", "Name", "Symbol", "Volume", "Marketcap"], axis=1, inplace=True)
    df = df[["Date", "Close"]]
    return df

#Plot1
def plot_price(df):
    plt.figure(figsize=(12,8))
    plt.plot(df["Date"],df["Close"], marker="o", linestyle="-",color="red")
    plt.title("plot 1: prices")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.xticks(rotation=50)
    plt.tight_layout()
    plt.show()

#previouse close data
def create_previous_close(df):
    df["Previous_Close"] = df["Close"].shift(1)
    df = df.dropna()
    return df

#split data
def split_data(df):
    x = df[["Previous_Close"]]
    y = df["Close"]
    return train_test_split(x, y, test_size=0.2, random_state=42)

#Scale Features (Clean the Data)
def scale_features(x_trains, x_test):
    cleaner = StandardScaler()
    x_trains_cleaned = cleaner.fit_transform(x_trains)
    x_test_cleaned = cleaner.transform(x_test)
    return x_trains_cleaned, x_test_cleaned,cleaner

#Train the Model
def train_model(x_trains_cleaned, y_trains):
    model = LinearRegression()
    model.fit(x_trains_cleaned, y_trains)
    return model

#Test for accuracy
def test_agent(model, x_test_cleaned, y_test):
    prediction = model.predict(x_test_cleaned)
    mse = mean_squared_error(y_test, prediction)
    print("Mean square error:", mse)
    return prediction

#Plot2
def plot_predictions(x_test, y_test, prediction):
    plt.figure(figsize=(12,8))
    plt.scatter(x_test, y_test, color="lightcoral", label="Actual prices")
    plt.plot(x_test, prediction, color="paleturquoise", label="Predicted prices")
    plt.title("plot 2: prediction")
    plt.xlabel("previous day close price")
    plt.ylabel("Next day close price")
    plt.legend()
    plt.grid(True)
    plt.show()

#plot3
def predict_future_prices(model, scaler, df):
    future_dates = pd.date_range(start=df['Date'].max(), end='2025-01-01', freq='D')
    future_features = pd.DataFrame({'Previous_Close': df['Close'].iloc[-1]}, index=future_dates)
    future_features_scaled = scaler.transform(future_features)
    future_prices = model.predict(future_features_scaled)
    return future_dates, future_prices

# Plot future predictions
def plot_future_predictions(df, model, scaler, future_dates, future_prices, X_train_scaled):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Date'], df['Close'], color='black', label='Actual Prices')
    predicted_train_dates = df['Date'][:len(X_train_scaled)]
    plt.plot(predicted_train_dates, model.predict(X_train_scaled), color='blue', linewidth=3, label='Predicted Prices (Training)')
    plt.plot(future_dates, future_prices, color='green', linestyle='--', linewidth=2, label='Future Prediction')
    plt.title('Linear Regression Model: Bitcoin Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Control Function
def manager():
    df = load_data()
    plot_price(df)
    df = create_previous_close(df)
    x_trains, x_test, y_trains, y_test = split_data(df)
    x_trains_cleaned, x_test_cleaned, cleaner = scale_features(x_trains, x_test)
    model = train_model(x_trains_cleaned, x_trains)
    prediction = test_agent(model, x_test_cleaned, y_test)
    plot_predictions(x_test, y_test, prediction)
    # Phase 3
    # Predict future prices
    future_dates, future_prices = predict_future_prices(model, cleaner, df)

    # Plot future predictions
    plot_future_predictions(df, model, cleaner, future_dates, future_prices, x_trains_cleaned)

if __name__ == "__main__":
    manager()