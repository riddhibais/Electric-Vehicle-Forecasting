# pages/1_Dashboard.py

import streamlit as st
import pandas as pd
import common_functions as cf # Model, constants, and functions are here

# --- Page Setup ---
st.set_page_config(layout="wide")

# --- Load Model ---
model = cf.download_file_from_drive()

st.title("⚡ EV Energy Forecasting & Green Driving Dashboard")
st.markdown("---")

# --- Default/Current Scenario Values ---
# Using realistic defaults for the dashboard display
DEFAULT_SPEED = 60.0
DEFAULT_TEMP = 25.0
DEFAULT_SOC = 75.0
DEFAULT_SLOPE = 0.0
DEFAULT_MODE = 2 # Urban
DEFAULT_ROAD = 2 # Urban
DEFAULT_TRAFFIC = 2 # Moderate

# --- Scenario Input ---
with st.container():
    st.header("1. Current Driving Scenario")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_speed = st.slider("Current Speed (km/h)", 30.0, 120.0, DEFAULT_SPEED, step=5.0)
    
    with col2:
        current_soc = st.slider("Battery State of Charge (%)", 10.0, 100.0, DEFAULT_SOC, step=5.0)
        
    with col3:
        current_slope = st.slider("Road Slope (%)", -10.0, 10.0, DEFAULT_SLOPE, step=1.0)
        
    col4, col5, col6 = st.columns(3)
    
    with col4:
        current_road = st.selectbox("Road Type", options=[1, 2, 3], format_func=lambda x: {1: "Highway", 2: "Urban", 3: "Rural"}.get(x), index=1)
    
    with col5:
        current_mode = st.selectbox("Driving Mode", options=[1, 2, 3], format_func=lambda x: {1: "Eco", 2: "Normal", 3: "Sport"}.get(x), index=1)
    
    with col6:
        current_temp = st.number_input("Ambient Temperature (°C)", 0.0, 50.0, DEFAULT_TEMP)

st.markdown("---")

# ==============================================================================
# 2. PREDICTION & METRICS
# ==============================================================================

st.header("2. Forecasting Results & Green Metrics")

if model is None:
    st.error("Model could not be loaded. Please check model file availability.")
    st.stop()
else:
    # Prepare input data dictionary
    input_data = cf.prepare_input(
        speed=current_speed,
        temp=current_temp,
        mode=current_mode,
        road=current_road,
        traffic=DEFAULT_TRAFFIC,
        slope=current_slope,
        battery_state=current_soc
    )
    
    # Run Prediction
    consumption_current = cf.predict_energy_consumption_local(input_data, model)
    
    # Calculate Range and CO2 Metrics
    if consumption_current > 0:
        predicted_range_current, co2_saved_kg = cf.calculate_range_metrics(consumption_current, current_soc)
    else:
        predicted_range_current = 0.0
        co2_saved_kg = 0.0
        
    
    colA, colB, colC = st.columns(3)
    
    with colA:
        st.metric(
            label="Energy Consumption (kWh/km)", 
            value=f"{consumption_current:.3f}", 
            delta_color="inverse", 
            help="Predicted energy required to travel one kilometer. Lower is better."
        )

    with colB:
        st.metric(
            label="Predicted Remaining Range (km)",
            value=f"{predicted_range_current:.0f}",
            help=f"Remaining range based on current SOC ({current_soc}%) and consumption."
        )

    with colC:
        st.metric(
            label="Estimated CO2 Offset (kg)",
            value=f"{co2_saved_kg:.1f}",
            help="Estimated CO2 saving compared to a standard petrol car over the predicted range."
        )

st.markdown("---")
