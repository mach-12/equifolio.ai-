import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from PIL import Image

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

st.title("Pricing")
st.image("data\img\deep_blue.png", 'Low Risk, Low Return')
st.header('Deep Blue: ₹2500 per portfolio')
st.markdown("---")
st.image("data\img\dynamic_green.png", 'Moderate Risk, Moderate Return')
st.header('Dynamic Green: ₹4000 per portfolio')