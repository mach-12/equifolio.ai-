import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from PIL import Image


def get_total_return(income, expenses):
  investment_rate = 0.17  #NIFTY 50 returns of last 5 years divided by 50
  investment = income-expenses
  annual_return = investment*investment_rate
  year=2022
  withdrawal_rate = 0.04 # 4% percent rule

  Year = []
  Yearly_Income = []
  Yearly_Expenses = []
  Yearly_Invested_Amount = []
  Total_Invested_Amount = []
  Annual_Returns = []

  Year.append(year)
  Yearly_Income.append(income)
  Yearly_Expenses.append(expenses)
  Yearly_Invested_Amount.append(investment-expenses)
  Total_Invested_Amount.append(investment)
  Annual_Returns.append(annual_return)

  #Loop for n years
  invested_years = 10
  for i in range(0,invested_years-1):
    investment = (income-expenses) + annual_return + investment 
    annual_return = investment *investment_rate
    Year.append(year+i+1)
    Yearly_Income.append(income)
    Yearly_Expenses.append(expenses)
    Yearly_Invested_Amount.append(investment-expenses)
    Total_Invested_Amount.append(investment)
    Annual_Returns.append(annual_return)

  df = pd.DataFrame()
  df['Year'] = Year
  df['Yearly_Income'] = Yearly_Income
  df['Yearly_Expenses'] = Yearly_Expenses
  df['Yearly_Invested_Amount'] = Yearly_Expenses
  df['Total_Invested_Amount'] = Total_Invested_Amount
  df['Annual_Returns'] = Annual_Returns
  Yearly_Withdrawal_Amount = np.array(Total_Invested_Amount)*withdrawal_rate
  df['Yearly_Withdrawal_Amount'] = Yearly_Withdrawal_Amount
  fin_ret = (df.Total_Invested_Amount[9] + df.Annual_Returns[9])
  fin_ret -= fin_ret*0.10
  return fin_ret

def indian_comma(num):
    x = str(num)
    if (len(x) < 3):
        return x
    x = x[::-1]
    res = ""
    count = 0
    for i in x:
        res += i
        count += 1
        if count == 3:
            res+=','
        if count % 2 != 0 and count > 4 and count < len(x):
            res+=','
    return "₹ " + res[::-1]

st.set_page_config(
    page_title="EQUIFOLIO.AI"
)

logo_img = Image.open('data\img\equifolio_logo.png')

col1, col2 = st.columns([0.6, 1], gap='small')

with col1:
    st.image(logo_img, width = 220)
with col2:
    st.title("EquiFolio.ai")
    st.header("Your AI stock portfolio")

st.markdown("---")

col3, col4 = st.columns([1.3, 1], gap='large')

with col3:    
    st.subheader('“Never depend on a single income. Make an investment to create a second source.”  -Warren Buffett')
    st.title('Want to invest in stocks?')
    st.write("""We often think of value investing as more art than science but some of its greatest practitioners display quant-like discipline. AI has much to offer this field for those who are open to the possibilities.""")

    st.markdown('')

    st.info("""
    ⚡Completely Calm Investing
    """)
    st.markdown('')
    st.info("""
    ⚡We evolve as markets change
    """)
    st.markdown('')
    st.info("""
    ⚡Generate Portfolio
    """)
    


    

with col4:
    st.header('Return calculator')
    st.caption('See how stock investment compounds')
    an_income = st.number_input("Enter your annual income (INR)", min_value=0, max_value=100000000, step=10000)
    an_exp = st.number_input("Enter your annual expenses (INR)", min_value=0, max_value=100000000, step=10000)
    # st.text("NIFTY annual return rate - 17%")
    st.info("""
    NIFTY annual return rate **17%**   
    """)
    st.info(
    """
    Withdrawl rate **4%** *(By the 4% rule)*   
    """)
    final_return = round(get_total_return(an_income, an_exp))
    st.markdown("---")
    st.metric("Return (INR)", indian_comma(final_return))

