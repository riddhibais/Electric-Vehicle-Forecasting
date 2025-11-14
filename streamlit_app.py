# streamlit_app.py (Home Page)

import streamlit as st

# Set Theme and Layout 
st.set_page_config(
    page_title="EV Forecast: Green Skills Project", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("🏡 Welcome to the Smart EV Forecasting System")
st.subheader("A Green Skills Project")
st.markdown("---")

st.info("""
    **Project Goal:** To accurately predict Electric Vehicle energy consumption and range using machine learning, 
    and provide actionable Green Driving recommendations (CO2 Emission Offset & Eco Mode Comparison) 
    for maximum efficiency.
    
    👈 Please select a page from the sidebar to begin:
    * **Dashboard:** Use the ML model for real-time predictions and green analysis.
    * **Smart Assistant:** Chat with the AI for doubts and nearest charging station information.
""")



