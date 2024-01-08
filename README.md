# EquiFolio.ai: 📈 Equity Portfolio Optimization and Generation  
![image](https://github.com/mach-12/equifolio.ai-/assets/89384052/ed53b9ed-bf29-4e08-9349-b3267556aa28)

## ⭐⭐ Winning Project at First Project Showcase, Benentt University (Runner-Up)

### Deployed Project: https://equifolio-ai.streamlit.app/

Developed by [Mann Dharmesh Acharya](https://github.com/mach-12) and [Arpan Sethi](https://github.com/arpansethi30) 



### Video Demo
https://github.com/mach-12/equifolio.ai-/assets/89384052/6bf234a4-c13a-47d4-868a-8c03cfbdcb44


## Table of Contents
- [Background](#background)
- [ML Algorithm](#ml-algorithm)
- [Instructions to Run](#instructions-to-run)
- [Requirements](#requirements)
- [Description of Project](#description-of-project)
- [Contributions and Attritions](#contributions-and-attritions)

## Background
📊 Public investment in stocks and mutual funds has been steadily rising in India. This project addresses the need for a product that assists users in creating diversified portfol
ios, especially those new to the stock market. We've developed a tool that uses historical stock data from the NSE NIFTY indices to enable users to construct portfolios with potentially good long-term returns.


## ML Algorithm

- **Input** is investment amount and the Stock Index of choice. 10 year historical stock data is retrieved using  `yfinance`.

- Calculating returns and variances, it utilizes **K-means clustering** to group stocks, determining an optimal number of clusters and filtering a portfolio based on these clusters.

- We then compute portfolio metrics like **variance**, **volatility**, and expected annual return for the filtered portfolio. Further optimization is performed using the `Efficient Frontier` module to maximize the **Sharpe ratio** and find optimal weights for the selected stocks.
  
- Finally, using these optimized weights and current stock prices, we perform a **discrete allocation**, determining the allocation of funds across these stocks for a specified total portfolio value.
  
- The **Output** includes the allocation details, portfolio statistics, and visual plots illustrating the clustering and portfolio's adjusted close prices.
  
## Instructions to Run
To run this project, follow these commands:

- Install Requirements
```bash
pip install pipenv
pipenv install -r requirements.txt
```

- Run project
```bash
pipenv shell
streamlit run 1-EquiFolio.py
```

## Requirements

```bash
pandas
numpy
yfinance
cufflinks
matplotlib
scikit-learn
pyportfolioopt
streamlit
streamlit-option-menu
black
pexpect
```

## Description of Project
The project comprises four primary pages:

### 1) Homepage

- Provides an overview of the project's viability. Features an investment return calculator illustrating the advantages of equity stock investment compared to bank fixed deposits. Incorporates the 4% withdrawal rule, showcasing substantial returns on index fund investments.

### 2) Portfolios
   
- This section encompasses two distinct portfolios based on risk levels: **Deep Blue** and **Dynamic Green** which pick stocks from different Indices.

 ### 3) Stock Explorer
   
- The Stock Explorer segment offers various data representations for a selected stock ticker. Users can specify date ranges and stock symbols to view: Company description, Daily stock data, including High-Low values and Bollinger band stock charts

 ### 4) Pricing

- How much it cost to create a portfolio. Giving a business perspective to the project.


## Contributions and Attritions 
This project is freely available for use in any manner. 

Contributions are welcome and encouraged!

Feel free to utilize this project for your needs, and if used or distributed, kindly attribute it this way:

```
EquiFolio.ai by Mann Acharya
Repository: https://github.com/mach-12/equifolio.ai-
```
