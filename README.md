# equifolio.ai
#### Video Demo:  https://youtu.be/sa3kFrvjVvY
#### Description: Equity Portfolio Optimization and generation  

# How to run the project
Commands to run this project:
~ pipenv shell
~ streamlit run 1-EquiFolio.py

Incase a dependency is missing, try:
~ pipenv install

# Requirements
streamlit 
pandas 
scikit-learn 
numpy 
black 
yfinance 
matplotlib 
streamlit-option-menu 
pyportfolioopt 
cufflinks

# Background
Public investment instocks and mutual funds has seen a steady rise in India.
There is a need for a consumer product which can help people make diversified portfolios.

In this project, we have developed a product which enables users naïve to the stock market,to create portfolios with good long time returns based on the user's needs. 

Historical stock data of 10 year period was taken from the NSE NIFTY indices. 

Using ML models, We predict one-year ahead fund return and performance.
We construct a time series of monthly returns and then rank the top stocks. 

The ranking also considers the fundamentals of the stocks: P/E Ratio, Revenue Growth.

The total portfolio value is then distributed in a way as to optimise the reutrn of the portfolio.
This is done based on Risk, Annual Variance, and Annual Return.

# Project Structure
The proeject consists of Four Pages:
1) Homepage
2) Portfolios
3) Stock Explorer
4) Pricing

# 1) Homepage
We descibe the viability of our project and explain it to a new user visiting the website.
There is a investment return calculator included which demonstrates the advantage of equity stock investing over bank fixed deposits.
The 4% withdrawl rule is also applied, yet the returns are soaringly good on investing on the index.

# 2) Portfolios
This is the main part of the project. I have created two portfolios according to risk levels:
1) Deep Blue
2) Dynamic Green

1) Deep Blue
This low-risk portfolio carry minimal risk and a stable return assurance. These portfolios are always a step ahead of inflation.
Choosing the best low risk stocks can help stabilise the risk-reward ratio in an investor’s portfolio.

2) Dynamic Green
This moderate-risk portfolio exposes investors’ capital to only average levels of risk.
The portfolio invests capital in varied equities to maintain reasonable market risks against inflation-adjusted returns.

I take user input of the amount of capital they want to invest.

I picked NSE stock data using the Yahoo Finance API and then passed them into my ML model.
- The ML model consists of a clsutering algorithm that takes the Volatility and Returns of stocks and groups them into different clusters.
- I then pick stocks form each cluster in a way to maximise the Sharpie Ratio
- Weights are assigned to the selected stocks.
- Then by using the current market price, we buy stocks based on the amount available for spending.
- The result is obtained stocks and share amount to buy.

# 3) Stock Explorer
The stock Explorer can display various data for a selected stock ticker.
Selected the range of dates and stock symbol.
You can see the company logo, descrption, Daily stock data of High-Low, and Bolinger band stock charts.

# 4) Pricing
The final product will charge a one time fee to generate a portfolio. The fee is described in this page.

