import streamlit as st
import pandas as pd
import numpy as np


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

st.title("Pricing")
st.image("https://github.com/mach-12/equifolio.ai-/blob/main/data/img/deep_blue.png?raw=True", 'Low Risk, Low Return')
st.header('Deep Blue: ₹2500 per portfolio')
st.markdown("---")
st.image("https://github.com/mach-12/equifolio.ai-/blob/main/data/img/dynamic_green.png?raw=True", 'Moderate Risk, Moderate Return')
st.header('Dynamic Green: ₹4000 per portfolio')