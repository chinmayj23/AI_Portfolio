import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.model_selection import train_test_split
from src.data import get_data, tech
from src.utils import pc
from src.models import lstm_model
from src.testing import testing
from src.portfolio import new_data, expected_returns, display_simulated_ef_with_random
import numpy as np
from tensorflow.keras.models import load_model

# Define tickers
# tickers=['ULTRACEMCO.NS','GRASIM.NS','SHREECEM.NS','JSWSTEEL.NS','SBILIFE.NS','TATACONSUM.NS','ICICIBANK.NS','SBIN.NS','INDUSINDBK.NS','AXISBANK.NS','UPL.NS','KOTAKBANK.NS','MARUTI.NS','RELIANCE.NS','DRREDDY.NS','TECHM.NS','ONGC.NS','TATAMOTORS.NS','WIPRO.NS','HCLTECH.NS','BAJAJFINSV.NS','BAJFINANCE.NS','LT.NS','HEROMOTOCO.NS','NESTLEIND.NS','HDFCLIFE.NS','SUNPHARMA.NS','BRITANNIA.NS','M&M.NS','TITAN.NS','NTPC.NS','CIPLA.NS','EICHERMOT.NS','ITC.NS','DIVISLAB.NS','IOC.NS','COALINDIA.NS','TCS.NS', 'ASIANPAINT.NS','HINDUNILVR.NS','POWERGRID.NS','HDFC.NS','HINDALCO.NS','BAJAJ-AUTO.NS','HDFCBANK.NS','INFY.NS','TATASTEEL.NS','BHARTIARTL.NS','BPCL.NS','ADANIPORTS.NS']
tickers=['ULTRACEMCO.NS','GRASIM.NS','SHREECEM.NS','JSWSTEEL.NS','SBILIFE.NS','TATACONSUM.NS','ICICIBANK.NS','SBIN.NS']
def data_preparation(df, lag, test_size):
    """Prepares the data for LSTM training."""
    data = df.copy()
    data = np.array(data)
    total_size = len(data)
    train_size = total_size - test_size
    X = []
    Y = []
    for i in range(lag, len(data)):
        X.append(data[i - lag:i, :])
        Y.append(data[i, 0])
    X = np.array(X)
    Y = np.array(Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, shuffle=False)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)  # 20% validation
    return x_train, x_validation, x_test, y_train, y_validation, y_test

def main():
    """Main function to execute the trading strategy."""
    results=pd.DataFrame(columns=['stock','buy&hold returns','strategy returns','pattern accuracy','Scaled MSE'])
    stocks=[]
    bnh=[]
    strat=[]
    accu=[]
    Mse=[]
    load_model_flag = True  # Set to True to load a pre-trained model. Change to False for training
    model_path = 'checkpoints/lstm_model.keras'
    for i in tickers:
        print(f"Processing ticker: {i}")
        data=get_data(i)
        df=tech(data)
        df=pc(df)
        x_train,x_validation,x_test,y_train,y_validation,y_test=data_preparation(df,1,365)
        if load_model_flag and os.path.exists(model_path):
            print(f"Loading pre-trained model from {model_path}")
            lstm = load_model(model_path)
            res = None  # No training history to store when loading
        else:
            print(f"Training model for {i}")
            res, lstm=lstm_model(x_train,x_validation,y_train,y_validation,1)

        mse,acc,simple_returns,strategy_returns=testing(x_test,y_test,df,data,lstm,365)
        stocks.append(i)
        bnh.append(simple_returns)
        strat.append(strategy_returns)
        accu.append(acc)
        Mse.append(mse)
    results['stock']=stocks
    results['buy&hold returns']=bnh
    results['strategy returns']=strat
    results['pattern accuracy']=accu
    results['Scaled MSE']=Mse
    lstm.summary()

    new_tickers, df_use = new_data(results)
    er = expected_returns(new_tickers, results)
    cov_matrix = df_use.pct_change(1).cov()
    max_sharpe_allocation, min_vol_allocation, rp, sdp = display_simulated_ef_with_random(er, cov_matrix, num_portfolios=1000, risk_free_rate=0.02, df_use=df_use)

    # Save results to Excel (you might want to adjust the file path)
    results.to_excel('results.xlsx', index=False)  

    # Save allocations to Excel
    max_sharpe_allocation.to_excel('alloc_maxsharpe.xlsx') 
    min_vol_allocation.to_excel('alloc_minvol.xlsx') 

    print("Results and allocations saved to Excel files.")

if __name__ == "__main__":
    main()
