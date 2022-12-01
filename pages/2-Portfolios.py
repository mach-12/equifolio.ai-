import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn import preprocessing
import sklearn.cluster as cluster
import sklearn.metrics as metrics
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models 
from pypfopt import expected_returns
from pypfopt. discrete_allocation import DiscreteAllocation, get_latest_prices


def generate_portfolio(port_value, option):
  assets = []
  index= ""

  if (option==1):
    index='^NSEMDCP50'
    assets = ['ABB.NS', 'AUBANK.NS', 'ABBOTINDIA.NS', 'ALKEM.NS', 'ASHOKLEY.NS', 'ASTRAL.NS', 'AUROPHARMA.NS', 'BALKRISIND.NS', 'BATAINDIA.NS', 'BHARATFORG.NS', 'CANBK.NS', 'COFORGE.NS', 'CONCOR.NS', 'CUMMINSIND.NS', 'ESCORTS.NS', 'FEDERALBNK.NS', 'GODREJPROP.NS', 'GUJGASLTD.NS', 'HINDPETRO.NS', 'HONAUT.NS', 'IDFCFIRSTB.NS', 'INDHOTEL.NS', 'JINDALSTEL.NS', 'JUBLFOOD.NS', 'LTTS.NS', 'LICHSGFIN.NS', 'LUPIN.NS', 'MRF.NS', 'M&MFIN.NS', 'MFSL.NS', 'OBEROIRLTY.NS', 'OFSS.NS', 'PAGEIND.NS', 'PERSISTENT.NS', 'PETRONET.NS', 'POLYCAB.NS', 'PFC.NS', 'PNB.NS', 'RECLTD.NS', 'SRTRANSFIN.NS', 'SAIL.NS', 'TVSMOTOR.NS', 'TATACOMM.NS', 'TORNTPOWER.NS', 'TRENT.NS', 'UBL.NS', 'IDEA.NS', 'VOLTAS.NS', 'ZEEL.NS', 'ZYDUSLIFE.NS']

  if (option==2):
    index='^NSEI'
    assets = ['ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'HDFC.NS', 'ICICIBANK.NS', 'ITC.NS', 'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'UPL.NS', 'ULTRACEMCO.NS', 'WIPRO.NS']
  
  # assets = ['AEGISCHEM.NS', 'AETHER.NS', 'ALLCARGO.NS', 'ALOKINDS.NS', 'AMARAJABAT.NS', 'AMBER.NS', 'ANGELONE.NS', 'ANURAS.NS', 'APOLLOTYRE.NS', 'BSE.NS', 'BAJAJELEC.NS', 'BALAMINES.NS', 'BALRAMCHIN.NS', 'MAHABANK.NS', 'BDL.NS', 'BIRLACORPN.NS', 'BSOFT.NS', 'BORORENEW.NS', 'BRIGADE.NS', 'BCG.NS', 'MAPMYINDIA.NS', 'CESC.NS', 'CAMPUS.NS', 'CANFINHOME.NS', 'CARBORUNIV.NS', 'CDSL.NS', 'CENTURYTEX.NS', 'CHAMBLFERT.NS', 'CHEMPLASTS.NS', 'CUB.NS', 'CAMS.NS', 'CREDITACC.NS', 'CYIENT.NS', 'DCMSHRIRAM.NS', 'DEEPAKFERT.NS', 'DELTACORP.NS', 'EIDPARRY.NS', 'EASEMYTRIP.NS', 'ELGIEQUIP.NS', 'EXIDEIND.NS', 'FINEORG.NS', 'FSL.NS', 'GLENMARK.NS', 'GRANULES.NS', 'GRAPHITE.NS', 'GNFC.NS', 'HFCL.NS', 'HINDCOPPER.NS', 'IDBI.NS', 'IDFC.NS', 'IIFL.NS', 'IRB.NS', 'IBULHSGFIN.NS', 'INDIGOPNTS.NS', 'INTELLECT.NS', 'JBCHEPHARM.NS', 'JSL.NS', 'JUBLINGREA.NS', 'JUSTDIAL.NS', 'KEI.NS', 'KPITTECH.NS', 'KEC.NS', 'LATENTVIEW.NS', 'LXCHEM.NS', 'LUXIND.NS', 'MMTC.NS', 'MGL.NS', 'MANAPPURAM.NS', 'MRPL.NS', 'MASTEK.NS', 'MEDPLUS.NS', 'METROBRAND.NS', 'METROPOLIS.NS', 'MCX.NS', 'NBCC.NS', 'NLCINDIA.NS', 'NETWORK18.NS', 'PVR.NS', 'POLYPLEX.NS', 'PRAJIND.NS', 'QUESS.NS', 'RBLBANK.NS', 'RADICO.NS', 'RAIN.NS', 'REDINGTON.NS', 'ROUTE.NS', 'SAPPHIRE.NS', 'RENUKA.NS', 'SOBHA.NS', 'STLTECH.NS', 'SUNTECK.NS', 'SUZLON.NS', 'TV18BRDCST.NS', 'TANLA.NS', 'TRIVENI.NS', 'UTIAMC.NS', 'VIPIND.NS', 'VTL.NS', 'WELSPUNIND.NS']
  
  assets.append(index)

  stockStartDate='2013-01-01'

  today = datetime.today().strftime('%Y-%m-%d')


  df = yf.download(
          tickers = assets,       # tickers list or string as well
          start = stockStartDate,  
          end = today    # optional, default is '1mo'
  )    

  #  2.  Select 'Close' (price at market close) column only
  df = df.iloc[:, df.columns.get_level_values(0)=='Adj Close']


  #  3.  Remove the dataframe multi-index
  df.columns = df.columns.droplevel(0)  # multi-index

  daily_returns = df.pct_change()
  annual_mean_returns = daily_returns.mean()*250
  annual_return_variance = daily_returns.var()*250

  df2 = pd.DataFrame(df.columns, columns=['Stock'])
  df2['Variances'] = annual_return_variance.values
  df2['Returns'] = annual_mean_returns.values


  #Use the Elbow method to determine the number of clusters to use to group the stocks
  #Get and store the annual returns and annual variances
  X = df2[ ['Returns', 'Variances' ]].values
  inertia_list = []
  for k in range (2,16):
    #Create and train the model
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertia_list.append (kmeans. inertia_)
    #Plot the data plt.plot (range(2,16), inertia _list)
  plt.plot (range (2,16), inertia_list) 
  plt.title( 'Elbow Curve') 
  plt.xlabel('Number of Clusters') 
  plt.ylabel ('Inertia or sum square error')
  # plt.show

  #Get and show the labels / groups
  kmeans = KMeans(n_clusters=3).fit(X)
  labels = kmeans.labels_


  df2['Cluster_Labels'] = labels


  #Plot and show the different clusters
  plt.scatter(X[:, 0], X[:, 1], c = labels, cmap='rainbow') 
  plt.title('K-Means Plot' ) 
  plt.xlabel('Returns') 
  plt.ylabel('Variances')
  plt.legend(labels)
  # plt.show()

  for i in range(0,3):
    symbol=df2[df2['Cluster_Labels']==i].head()
    print(symbol[['Stock','Cluster_Labels']])

  finallist = df2[df2['Cluster_Labels'] != 0].reset_index(drop=False)

  finallist = [i[0] for i in finallist[['Stock']].values.tolist()]

  weights=np.array([1/len(finallist) for i in range(len(finallist))])

  df = yf.download(
          tickers = finallist,       # tickers list or string as well
          start = stockStartDate,  
          end = today    # optional, default is '1mo'
  )    

  #  2.  Select 'Close' (price at market close) column only
  df = df.iloc[:, df.columns.get_level_values(0)=='Adj Close']


  #  3.  Remove the dataframe multi-index
  df.columns = df.columns.droplevel(0)  # multi-index



  #Visually show stock/portfolio
  title='Portfolio Adj. Close Price History'
  #Get the stocks
  my_stocks = df
  for c in my_stocks.columns.values:
    plt.plot(my_stocks[c],label=c)
  plt.title(title)
  plt.xlabel('Date', fontsize =18)
  plt.ylabel('Adj. Prize', fontsize=18)
  plt.legend(my_stocks.columns.values, loc='upper left')
  # plt.show()

  #Show the daily simple returns
  returns = df.pct_change()


  #CoVariance Matrix
  cov_matrix_annual = returns.cov()*250


  port_variance = np.dot(weights.T,np.dot(cov_matrix_annual,weights))


  port_volatility=np.sqrt(port_variance)

  #Calculating the anual portfolio return
  portfolioSimpleAnnualReturn = np.sum(returns.mean()*weights) * 250


  percent_var = str(round(port_variance,2)*100)+'%'
  percent_vol = str(round(port_volatility,2)*100)+'%'
  percent_ret = str(round(portfolioSimpleAnnualReturn,2)*100)+'%'
  print('Expected annual return'+percent_ret)
  print('Annual volatility/risk'+percent_vol)
  print('Annual Variance'+percent_var)


  #Portfolio Optimization 
  #Calculate the expected returns and the annualised sample covariance matrix of asset returns
  mu = expected_returns. mean_historical_return(df)
  s = risk_models.sample_cov(df)
  #optimize for max sharpe ratio
  ef = EfficientFrontier (mu, s)
  weights = ef.max_sharpe()
  cleaned_weights = ef.clean_weights()
  print(cleaned_weights)
  ef.portfolio_performance(verbose=True)

  latest_prices = get_latest_prices(df)
  weights = cleaned_weights
  da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = port_value) # Should be taken by users
  allocation, leftover = da.lp_portfolio()
  print('Discrete allocation:', allocation)
  print('Funds remaining: {:.2f}'.format(leftover))
  return allocation


st.set_page_config(
    page_title="EQUIFOLIO.AI"
)


col1, col2 = st.columns([0.6, 1], gap='small')

with col1:
    st.image('https://github.com/mach-12/equifolio.ai-/blob/main/data/img/equifolio_logo.png?raw=True', width = 220)
with col2:
    st.title("EquiFolio.ai")
    st.header("Your AI stock portfolio")

st.markdown("---")

st.title("Portfolios")

st.header('🗠 Deep Blue')
st.markdown('Low Risk Low Return, Blue Chip stocks')
with st.expander("More info"):
    st.write("""
        This low-risk portfolio carry minimal risk and a stable return assurance. These portfolios are always a step ahead of inflation. Choosing the best low risk stocks can help stabilise the risk-reward ratio in an investor’s portfolio.
    """)
    port_val1 = st.number_input("Enter investment amount: ", min_value=0, max_value=100000000, step=10000)
    if st.button('Generate Portfolio'):
        port = generate_portfolio(port_val1, 2)
        # st.write()
        st.write(port)
    st.image("https://github.com/mach-12/equifolio.ai-/blob/main/data/img/deep_blue.png?raw=True")

st.header('🗠 Dynamic Green')
st.markdown('Moderate Risk Moderate Return')
with st.expander("More info"):
    st.write("""
        This moderate-risk portfolio exposes investors’ capital to only average levels of risk. The portfolio invests capital in varied equities to maintain reasonable market risks against inflation-adjusted returns.
    """)
    
    port_val2 = st.number_input("Enter investment amount:", min_value=0, max_value=100000000, step=10000)
    if st.button('Generate Portfolio '):
        port = generate_portfolio(port_val2, 2)
        # st.write()
        st.write(port)

    st.image("https://github.com/mach-12/equifolio.ai-/blob/main/data/img/dynamic_green.png?raw=True")