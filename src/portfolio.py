import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
#from pypfopt.cla import CLA
#from pypfopt.plotting import Plotting
#from matplotlib.ticker import FuncFormatter
from src.data import get_data

def new_data(results):
    """Filters stocks based on strategy performance and retrieves price data."""
    new_tickers = []
    for i in range(len(results)):
        if ((results['strategy returns'].loc[i] > 0) and (results['strategy returns'].loc[i] > results['buy&hold returns'].loc[i])):
            new_tickers.append(results['stock'].loc[i])

    price_data = []
    for j in new_tickers:
        data_new = get_data(j)
        price_data.append(data_new['Close'])

    df_stocks = pd.concat(price_data, axis=1)
    df_stocks.columns = new_tickers
    df_use = df_stocks[-365:]
    df_port = df_use.pct_change(1)
    cov = df_port.cov()
    return new_tickers, df_use

def expected_returns(new_tickers, results):
    """Extracts expected returns for the selected stocks from the results DataFrame."""
    er = []
    for i in new_tickers:
        for j in range(len(results)):
            if (results['stock'].loc[j] == i):
                er.append(results['strategy returns'].loc[j])
    er = pd.Series(er)
    return er

def portfolio_annualized_performance(weights, mean_returns, cov_matrix):
    """Calculates the annualized volatility and returns of a portfolio."""
    returns = np.sum(mean_returns*weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(365)
    return std, returns

def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    """Generates random portfolios and calculates their Sharpe ratios."""
    num = len(mean_returns)  # Use length of mean_returns instead of df_use.columns
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(num)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualized_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record

def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, df_use):
    """Simulates portfolio optimization and displays results."""
    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)

    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=df_use.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index=df_use.columns, columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    print("-" * 80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualized Return:", round(rp, 2))
    print("Annualized Volatility:", round(sdp, 2))
    print("\n")
    print(max_sharpe_allocation)
    print("-" * 80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualized Return:", round(rp_min, 2))
    print("Annualized Volatility:", round(sdp_min, 2))
    print("\n")
    print(min_vol_allocation)

    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp, rp, marker='x', color='r', s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min, rp_min, marker='*', color='g', s=500, label='Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('Volatility')
    plt.ylabel('Returns')
    plt.legend(labelspacing=0.8)
    plt.savefig('ef.png')  #removed /content/drive/MyDrive/

    return max_sharpe_allocation, min_vol_allocation, rp, sdp
