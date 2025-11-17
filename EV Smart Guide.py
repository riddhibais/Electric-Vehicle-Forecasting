# Streamlit_App.py (Home Page)

import streamlit as st

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="EV Forecast: Green Skills Project",
    page_icon="⚡", # Adding an icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom Styling (Light Green Aesthetic) ---
st.markdown("""
<style>
/* --- APP BACKGROUND: LIGHT MINT GREEN --- */
.stApp {
    background-color: #f0fff0; /* Lightest Mint Green/Halka Hare rang */
}

/* Titles: Dark Green */
h1 {color: #00796b; text-align: center; padding-top: 20px;} 
/* Subheaders: Bright Green */
h2 {
    color: #4CAF50;
    border-bottom: 2px solid #e6ffe6;
    padding-bottom: 5px;
}
/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #e6ffe6; /* Slightly darker green for contrast */
}

/* Info box styling (Light green background) */
div.stAlert.st-info {
    background-color: #f0fff0;
    border-left: 5px solid #00796b;
    color: #00796b;
}

</style>
""", unsafe_allow_html=True) 

# --- 3. Main Content ---
st.title("🏡 Welcome to the Smart EV Forecasting System")
st.subheader("A Green Skills Project")
st.markdown("---")

st.info("""
    **Project Goal:** To accurately predict Electric Vehicle energy consumption and range using machine learning, 
    and provide actionable Green Driving recommendations (CO2 Emission Offset & Eco Mode Comparison) 
    for maximum efficiency.
    
    👈 Please select a page from the sidebar to begin:
    * **Range Predictor:** Use the ML model for real-time predictions and green analysis.
    * **Smart Assistant:** Chat with the AI for doubts and nearest charging station information.
""")

st.markdown("---")

st.subheader("Developed by")
st.markdown("### [Riddhi Bais]")



