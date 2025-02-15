import yfinance as yf
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler

def get_data(ticker, start_date='2016-01-01', end_date='2025-02-01'):
    """Downloads historical stock data from Yahoo Finance."""
    tickers = [ticker]
    data = yf.download(tickers, start=start_date, end=end_date, group_by='tickers')

    # Flatten Multi-Level Columns (Crucial Fix)
    data.columns = [col[1] if isinstance(col, tuple) else col for col in data.columns]

    return data

def tech(data):
    """Adds technical indicators to the DataFrame."""
    data.index = pd.to_datetime(data.index)

    # Ensure correct column names are passed to ta.add_all_ta_features
    try:
        df = ta.add_all_ta_features(
            data,
            open="Open",  # Correct column name
            high="High",  # Correct column name
            low="Low",  # Correct column name
            close="Close",  # Correct column name
            volume="Volume",  # Correct column name
            fillna=True
        )
    except KeyError as e:
        print(f"Error adding technical indicators: {e}")
        return None  

    df.drop(['Open', 'High', 'Low', 'Volume'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(level=0, inplace=True)
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    g = df.shape[1]
    df = scaler.fit_transform(data.iloc[:, 1:g])
    df = pd.DataFrame(df)
    df.columns = data.columns[1:]
    return df
