import pandas as pd
import numpy as np
import streamlit as st

from sklearn.cluster import KMeans

from datetime import datetime
import matplotlib.pyplot as plt

import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models 
from pypfopt import expected_returns
from pypfopt. discrete_allocation import DiscreteAllocation, get_latest_prices


def plot_clusters(kmeans_input):
    labels, X = kmeans_input

    # Create a scatter plot for clusters
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
    plt.title('K-Means Plot')
    plt.xlabel('Returns')
    plt.ylabel('Variances')
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    st.pyplot(fig)

def plot_stock_portfolio(my_stocks):
    # Create a line plot for stock/portfolio
    fig, ax = plt.subplots()
    for c in my_stocks.columns.values:
        ax.plot(my_stocks[c], label=c)
    plt.title('Portfolio Adj. Close Price History')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Adj. Prize', fontsize=18)
    plt.legend(my_stocks.columns.values, loc='upper left')
    st.pyplot(fig)

def generate_portfolio(port_value, option):
    # Initialize empty lists and variables
    assets = []
    index = ""

    plots = {'kmeans_plot':None, 'portfolio_plot':None}

    try:
        # Define assets and index based on the selected option
        if (option==1):
          index='^NSEMDCP50'
          assets = ['ABB.NS', 'AUBANK.NS', 'ABBOTINDIA.NS', 'ALKEM.NS', 'ASHOKLEY.NS', 'ASTRAL.NS', 'AUROPHARMA.NS', 'BALKRISIND.NS', 'BATAINDIA.NS', 'BHARATFORG.NS', 'CANBK.NS', 'COFORGE.NS', 'CONCOR.NS', 'CUMMINSIND.NS', 'ESCORTS.NS', 'FEDERALBNK.NS', 'GODREJPROP.NS', 'GUJGASLTD.NS', 'HINDPETRO.NS', 'HONAUT.NS', 'IDFCFIRSTB.NS', 'INDHOTEL.NS', 'JINDALSTEL.NS', 'JUBLFOOD.NS', 'LTTS.NS', 'LICHSGFIN.NS', 'LUPIN.NS', 'MRF.NS', 'M&MFIN.NS', 'MFSL.NS', 'OBEROIRLTY.NS', 'OFSS.NS', 'PAGEIND.NS', 'PERSISTENT.NS', 'PETRONET.NS', 'POLYCAB.NS', 'PFC.NS', 'PNB.NS', 'RECLTD.NS', 'SAIL.NS', 'TVSMOTOR.NS', 'TATACOMM.NS', 'TORNTPOWER.NS', 'TRENT.NS', 'UBL.NS', 'IDEA.NS', 'VOLTAS.NS', 'ZEEL.NS', 'ZYDUSLIFE.NS']
    
        if (option==2):
          index='^NSEI'
          assets = ['ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'ITC.NS', 'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'UPL.NS', 'ULTRACEMCO.NS', 'WIPRO.NS']
          
        assets.append(index)
    
        stockStartDate='2013-01-01'
    
        today = datetime.today().strftime('%Y-%m-%d')
    
        # Download stock data using Yahoo Finance
        df = yf.download(
            tickers=assets,
            start=stockStartDate,
            end=today
        )
        
    except:
        raise Exception("Error in downloading stocks. Some stocks might be outdated/unavailabile. Feel free to rasie a PR to fix it.")
        
    # Drop NaN values and focus on 'Adj Close' columns
    df = df.dropna(axis=1)
    df = df.iloc[:, df.columns.get_level_values(0) == 'Adj Close']
    df.columns = df.columns.droplevel(0)

    # Calculate daily returns, mean annual returns, and annual return variance
    daily_returns = df.pct_change()
    annual_mean_returns = daily_returns.mean() * 250
    annual_return_variance = daily_returns.var() * 250

    # Create a DataFrame for storing returns and variances
    df2 = pd.DataFrame(df.columns, columns=['Stock'])
    df2['Variances'] = annual_return_variance.values
    df2['Returns'] = annual_mean_returns.values

    # Clustering based on returns and variances
    X = df2[['Returns', 'Variances']].values
    inertia_list = [] # within-cluster sum of squares
    for k in range(2, 16):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        inertia_list.append(kmeans.inertia_)
    
    # Calculating Optimal Clusters
    diff = [inertia_list[i] - inertia_list[i - 1] for i in range(1, len(inertia_list))]
    optimal_clusters = diff.index(min(diff)) + 3
    print("Number of Clusters = ", optimal_clusters)
    
    # Clustering
    kmeans = KMeans(n_clusters=optimal_clusters).fit(X)
    labels = kmeans.labels_
    df2['Cluster_Labels'] = labels

    plots['kmeans_plot'] = [labels, X] # Adding K-means plot  

    # Display stocks grouped by cluster
    for i in range(0, 3):
        symbol = df2[df2['Cluster_Labels'] == i].head()
        print(symbol[['Stock', 'Cluster_Labels']])

    # Filtering stocks from a particular cluster
    finallist = df2[df2['Cluster_Labels'] != 0].reset_index(drop=False)
    finallist = [i[0] for i in finallist[['Stock']].values.tolist()]
    weights = np.array([1 / len(finallist) for i in range(len(finallist))])

    # Download filtered stock data
    df = yf.download(
        tickers=finallist,
        start=stockStartDate,
        end=today
    )

    df = df.iloc[:, df.columns.get_level_values(0) == 'Adj Close']
    df.columns = df.columns.droplevel(0)

    plots['portfolio_plot'] = df # add to plot var

    # Calculate returns and covariance matrix
    returns = df.pct_change()
    cov_matrix_annual = returns.cov() * 250

    # Calculate portfolio metrics
    port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
    port_volatility = np.sqrt(port_variance)
    portfolioSimpleAnnualReturn = np.sum(returns.mean() * weights) * 250

    # Log portfolio statistics
    percent_var = str(round(port_variance, 2) * 100) + '%'
    percent_vol = str(round(port_volatility, 2) * 100) + '%'
    percent_ret = str(round(portfolioSimpleAnnualReturn, 2) * 100) + '%'
    
    print('Expected annual return' + percent_ret)
    print('Annual volatility/risk' + percent_vol)
    print('Annual Variance' + percent_var)

    portfolio_stats = {
        'expected-annual-return': percent_ret,
        'annual-volatility-risk': percent_vol,
        'annual-variance': percent_var
    }

    # Calculate expected returns and covariance matrix for portfolio optimization
    mu = expected_returns.mean_historical_return(df)
    s = risk_models.sample_cov(df)
    ef = EfficientFrontier(mu, s)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    print(cleaned_weights)
    ef.portfolio_performance(verbose=True)

    # Calculate and display discrete allocation of portfolio
    latest_prices = get_latest_prices(df)
    weights = cleaned_weights
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=port_value)
    allocation, leftover = da.lp_portfolio()
    print('Discrete allocation:', allocation)
    print('Funds remaining: {:.2f}'.format(leftover))
    print(latest_prices)
    return {'allocation': allocation, 'stats': portfolio_stats, 'plots': plots}

