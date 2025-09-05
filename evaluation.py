#Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os 

#Importing the Trained Models for Predictions 
from src.lstm import lstm_train_predict
from src.randomForest import randomForest_train_predict
from src.svr import svr_train_predict

#1) IMPORTING DATA ----------------------------------------------------------------------
data = pd.read_csv('data/stocks_data.csv')

df = pd.DataFrame(data)

#2) EVALUATION FUNCTION -----------------------------------------------------------------
def evaluate_model(model, stock, Y_test, Y_predictions):
    print(f"{model} evaluation for {stock}:")

    mse = mean_squared_error(Y_test, Y_predictions)
    r2 = r2_score(Y_test, Y_predictions)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return r2, mse

#3) PREDICTING STOCK --------------------------------------------------------------------
stocks = data['Name'].unique() #Getting all different stocks in the Data
features = ['open', 'high', 'low', 'volume', 'return', 'rolling_mean', 'rolling_std']
target = 'close'

#Dictionary to store the results for each model
results = {'lstm_r2' : [], 'lstm_mse' : [],
           'randomForest_r2' : [], 'randomForest_mse' : [],
           'svr_r2' : [], 'svr_mse' : []} 

for stock in stocks: #Iterating over each stock to be predicted by each model

    # ---- Stock Data Preparation ----
    stock_data = data[data['Name'] == stock] #Getting the data for the stock

    X = stock_data[features] #Getting the features
    y = stock_data[target] #Getting the target
    dates = stock_data['date'] #Getting the dates for each stock

    X_train, X_test, Y_train, Y_test, dates_train, dates_test = train_test_split(X, y, dates, test_size=0.2, random_state= 42)

    # Sort test data by date
    X_test_sorted = X_test.sort_index()
    y_test_sorted = Y_test.sort_index()
    dates_test_sorted = pd.to_datetime(dates_test.sort_index())

    # ---- LSTM Model ----
    Y_predictions = lstm_train_predict(X_train, Y_train, X_test_sorted)

    r2, mse = evaluate_model('LSTM', stock, y_test_sorted, Y_predictions)

    results['lstm_r2'].append(r2)
    results['lstm_mse'].append(mse)

    plt.figure(figsize=(10, 6))
    plt.plot(dates_test_sorted, y_test_sorted.values, label='True Values', color='green')
    plt.plot(dates_test_sorted, Y_predictions, label='LSTM Predictions', color='purple')
    plt.title(f'{stock} - LSTM')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.savefig(os.path.join('result/lstm_plots', f'{stock}_lstm.png'))
    plt.close()

    # ---- Random Forest Model ----
    Y_predictions = randomForest_train_predict(X_train, Y_train, X_test_sorted)

    r2, mse = evaluate_model('Random Forest', stock, y_test_sorted, Y_predictions)

    results['randomForest_r2'].append(r2)
    results['randomForest_mse'].append(mse)

    plt.figure(figsize=(10, 6))
    plt.plot(dates_test_sorted, y_test_sorted.values, label='True Values', color='green')
    plt.plot(dates_test_sorted, Y_predictions, label='Random Forest Predictions', color='purple')
    plt.title(f'{stock} - Random Forest')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.savefig(os.path.join('result/randomForest_plots', f'{stock}_randomForest.png'))
    plt.close()

    # ---- SVR (Support Vector Regressor) Model ----
    Y_predictions = svr_train_predict(X_train, Y_train, X_test_sorted)

    r2, mse = evaluate_model('SVR', stock, y_test_sorted, Y_predictions)

    results['svr_r2'].append(r2)
    results['svr_mse'].append(mse)

    plt.figure(figsize=(10, 6))
    plt.plot(dates_test_sorted, y_test_sorted.values, label='True Values', color='green')
    plt.plot(dates_test_sorted, Y_predictions, label='SVR Predictions', color='purple')
    plt.title(f'{stock} - SVR')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.savefig(os.path.join('result/svr_plots', f'{stock}_svr.png'))
    plt.close()

#4) PRINTING RESULTS --------------------------------------------------------------------
print("Results:")

# LSTM Average Results 
lstm_r2 = np.mean(results['lstm_r2'])
lstm_mse = np.mean(results['lstm_mse'])
print(f"LSTM Average R2 Score: {lstm_r2:.4f}")
print(f"LSTM Average Mean Squared Error: {lstm_mse:.4f}")

# Random Forest Average Results
randomForest_r2 = np.mean(results['randomForest_r2'])
randomForest_mse = np.mean(results['randomForest_mse'])
print(f"Random Forest Average R2 Score: {randomForest_r2:.4f}")
print(f"Random Forest Average Mean Squared Error: {randomForest_mse:.4f}")

# SVR Average Results
svr_r2 = np.mean(results['svr_r2'])
svr_mse = np.mean(results['svr_mse'])
print(f"SVR Average R2 Score: {svr_r2:.4f}")
print(f"SVR Average Mean Squared Error: {svr_mse:.4f}")