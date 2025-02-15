import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import load_model

def testing(x_test, y_test, df, data, lstm, test_size, filepath='checkpoints'):
    """Tests the LSTM model and calculates performance metrics."""

    # Load the entire model from the .keras file
    try:
        lstm = load_model(filepath + '/lstm_model.keras')
    except OSError as e:
        print(f"Error loading model: {e}")
        return None, None, None, None  

    y_pred = lstm.predict(x_test)
    mse = np.sqrt(mean_squared_error(y_test, y_pred))

    pattern = pd.DataFrame(y_pred, columns=['predicted'])
    pattern['actual'] = y_test

    signal_actual = [0]
    for i in range(pattern.shape[0] - 1):
        if pattern['actual'].iloc[i + 1] > pattern['actual'].iloc[i]:
            signal_actual.append(1)
        else:
            signal_actual.append(0)

    signal_predicted = [0]
    for i in range(pattern.shape[0] - 1):
        if pattern['predicted'].iloc[i + 1] > pattern['predicted'].iloc[i]:
            signal_predicted.append(1)
        else:
            signal_predicted.append(0)

    pattern['signal_actual'] = np.array(signal_actual)
    pattern['signal_predicted'] = np.array(signal_predicted)

    acc = accuracy_score(signal_actual, signal_predicted)

    Final_df = data[-test_size:][['Date', 'Close']]
    Final_df.reset_index(level=0, inplace=True)
    Final_df.drop(['index'], axis=1, inplace=True)
    Final_df['signal_actual'] = pattern['signal_actual']
    Final_df['signal_predicted'] = pattern['signal_predicted']
    Final_df['Close_returns'] = 0

    for i in range(1, len(Final_df)):
        close_vals = ((Final_df['Close'].iloc[i] / Final_df['Close'].iloc[i - 1]) - 1)
        Final_df['Close_returns'].iloc[i] = close_vals

    val = 100
    Final_df['NAV'] = 0
    Final_df['NAV'].iloc[0] = val

    for i in range(1, Final_df.shape[0]):
        val = val * (1 + Final_df['Close_returns'].iloc[i - 1])
        Final_df['NAV'].iloc[i] = val

    val = 100
    Final_df['NAV_strategy'] = 0
    Final_df['NAV_strategy'].iloc[0] = val

    for i in range(1, Final_df.shape[0]):
        val = val * (1 + (Final_df['Close_returns'].iloc[i] * Final_df['signal_predicted'].iloc[i - 1]))
        Final_df['NAV_strategy'].iloc[i] = val

    simple_returns = (Final_df['NAV'].iloc[len(Final_df) - 1] - Final_df['NAV'].iloc[0]) / 100
    strategy_returns = (Final_df['NAV_strategy'].iloc[len(Final_df) - 1] - Final_df['NAV_strategy'].iloc[0]) / 100

    return mse, acc, simple_returns, strategy_returns
