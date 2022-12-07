import streamlit as st
import pandas as pd
import numpy as np
import cufflinks as cf
import yfinance as yf
import datetime

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

# Sidebar
with st.container():
  st.subheader('Query parameters')
  start_date = st.date_input("Start date", datetime.date(2010, 1, 1))
  today = str(datetime.date.today()).split('-')
  end_date = st.date_input("End date", datetime.date(int(today[0]), int(today[1]), int(today[2])))

# Retrieving tickers data
  ticker_list = pd.read_csv('https://github.com/mach-12/equifolio.ai-/blob/main/data/img/Stocks.csv?raw=True')
  tickerSymbol = st.selectbox('Stock ticker', ticker_list) # Select ticker symbol

if st.button('Fetch'):
  tickerData = yf.Ticker(tickerSymbol) # Get ticker data
  tickerDf = tickerData.history(period='1d', start=start_date, end=end_date) #get the historical prices for this ticker
  
  # Ticker information
  string_logo = '<img src=%s>' % tickerData.info['logo_url']
  st.markdown(string_logo, unsafe_allow_html=True)
  
  string_name = tickerData.info['longName']
  st.header('**%s**' % string_name)
  
  string_summary = tickerData.info['longBusinessSummary']
  st.info(string_summary)
  
  # Ticker data
  st.header('**Ticker data**')
  st.write(tickerDf)
  
  # Bollinger bands
  st.header('**Bollinger Band Chart**')
  qf=cf.QuantFig(tickerDf,title='First Quant Figure',legend='top',name='GS')
  qf.add_bollinger_bands()
  fig = qf.iplot(asFigure=True)
  st.plotly_chart(fig)


####
#st.write('---')
#st.write(tickerData.info)