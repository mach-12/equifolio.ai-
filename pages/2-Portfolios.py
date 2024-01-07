# Imports
import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px

from backend.optimizer import generate_portfolio, plot_clusters, plot_stock_portfolio

def display_portfolio(data):
    # Convert allocation data to a DataFrame for easier manipulation
    allocation_df = pd.DataFrame(data['allocation'].items(), columns=['Stock', 'Allocation'])
    allocation_df['Allocation'] = allocation_df['Allocation'].astype(int)
    
    # Create an interactive pie chart
    fig = px.pie(allocation_df, values='Allocation', names='Stock', title='Allocation', 
                 labels={'Stock': 'Stocks'}, 
                 hover_data={'Allocation': ':.1f%'},
    )
    
    # Display the interactive chart
    st.plotly_chart(fig)

    # Create a DataFrame for the table
    stats_df = pd.DataFrame(data['stats'].items(), columns=['Statistic', 'Value'])

    col3, col4 = st.columns([1, 1], gap='small')

    with col3:
      st.subheader('Statistics')
      st.write(stats_df.set_index('Statistic'))
   
    with col4:
        st.subheader('Stock Allocation')
        st.write(allocation_df)

    st.header("Technical Plots")
    plot_clusters(data['plots']['kmeans_plot'])
    plot_stock_portfolio(data['plots']['portfolio_plot'])

      
        
 
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
    port_val1 = st.number_input("Enter investment amount: ", min_value=5000, max_value=100000000, step=10000)
    if st.button('Generate Portfolio'):
        port = generate_portfolio(port_val1, 2)
        display_portfolio(port)
    st.image("https://github.com/mach-12/equifolio.ai-/blob/main/data/img/deep_blue.png?raw=True")

st.header('🗠 Dynamic Green')
st.markdown('Moderate Risk Moderate Return')
with st.expander("More info"):
    st.write("""
        This moderate-risk portfolio exposes investors’ capital to only average levels of risk. The portfolio invests capital in varied equities to maintain reasonable market risks against inflation-adjusted returns.
    """)
    
    port_val2 = st.number_input("Enter investment amount:", min_value=5000, max_value=100000000, step=10000)
    if st.button('Generate Portfolio '):

        port = generate_portfolio(port_val2, 1)
        display_portfolio(port)

    st.image("https://github.com/mach-12/equifolio.ai-/blob/main/data/img/dynamic_green.png?raw=True")