# pages/1_Dashboard.py

import streamlit as st
import common_functions as cf
from streamlit_extras.switch_page_button import switch_page # Library for page navigation

# --- Load Model ---
model = cf.download_file_from_drive()

st.title("🔋 EV Range & Green Impact Predictor")
st.markdown("---") 

# Image add karna (This is optional, remove if still causing issues)
# st.image("https://images.unsplash.com/photo-1549419137-b6f38059c381?q=80&w=1770&auto=format&fit=crop", caption="Green Energy: Driving the future with Electric Vehicles.") 

st.header("EV Range Prediction & Green Driving Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    current_soc = st.slider("Current Battery State (SOC) %", 10, 100, 75)
    temp = st.slider("Outside Temperature (°C)", -5.0, 45.0, 25.0)
    slope = st.slider("Road Slope (%)", -5.0, 5.0, 0.0) 

with col2:
    speed = st.slider("Average Speed (km/h)", 20, 120, 60)
    mode_options = {"Eco": 1, "Normal": 2, "Sport": 3}
    driving_mode_name = st.selectbox("Driving Mode", list(mode_options.keys()))
    driving_mode = mode_options[driving_mode_name]

with col3:
    road_options = {"Highway": 1, "Urban": 2, "Rural": 3}
    road_type_name = st.selectbox("Road Type", list(road_options.keys()))
    road_type = road_options[road_type_name]
    
    traffic_options = {"Low": 1, "Medium": 2, "High": 3}
    traffic_condition_name = st.selectbox("Traffic Condition", list(traffic_options.keys()))
    traffic_condition = traffic_options[traffic_condition_name]
    
# --- Prediction Button and Logic (Green Skills Included) ---
if st.button("Predict Range & Green Impact", key='predict_btn', use_container_width=True):
    
    if model is not None:
        with st.spinner('Calculating prediction from ML Model...'):
            
            # 1. Current Mode Prediction 
            input_data_dict = cf.prepare_input(speed, temp, driving_mode, road_type, traffic_condition, slope, current_soc)
            consumption_current = cf.predict_energy_consumption_local(input_data_dict, model)
            predicted_range_current, co2_saved_kg = cf.calculate_range_metrics(consumption_current, current_soc)

            st.success("✅ Prediction Successful!")
            
            colA, colB, colC, colD = st.columns(4)
            
            # Display Current Metrics
            colA.metric("Predicted Consumption", f"{consumption_current:.4f} kWh/km")
            colB.metric("Predicted Range", f"{predicted_range_current:.0f} km")
            
            # Display Green Skill 1: Emission Offset
            colC.metric("Emission Offset (CO2 Saved)", f"{co2_saved_kg:.1f} kg", "Green Skill")
            
            st.subheader("💡 Analysis")
            st.info(f"On a **{road_type_name}** road in **{driving_mode_name} Mode**, your vehicle can travel approximately **{predicted_range_current:.0f} km** while saving **{co2_saved_kg:.1f} kg** of CO2 emissions compared to a fossil fuel car.")
            
            # 2. Green Skill 2: Eco Mode Comparison
            if driving_mode != 1:
                
                input_data_dict_eco = cf.prepare_input(speed, temp, 1, road_type, traffic_condition, slope, current_soc)
                consumption_eco = cf.predict_energy_consumption_local(input
