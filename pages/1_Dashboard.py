# pages/1_Dashboard.py

# --- 1. IMPORTS 
import streamlit as st
import common_functions as cf
import pandas as pd 

# --- 2. MODEL LOADING ---
model = cf.download_file_from_drive()

# --- 3. PAGE CONFIGURATION (Simple Configuration) ---
st.set_page_config(
    page_title="EV Range Prediction Dashboard",
    page_icon="📈",
    layout="wide", 
)

# --- 4. TITLE (Simple Title) ---
st.title("📈 EV Range Prediction Dashboard")
st.markdown("---") 


st.markdown("""
<style>
/* --- APP BACKGROUND: EK DUM LIGHT GREEN --- */
.stApp {
    background-color: #f0fff0; /* Lightest Mint Green/Halka Hare rang */
}

/* Main Title: Dark Green */
h1 {color: #00796b; text-align: center; font-weight: 700;} 
/* Subheaders: Bright Green */
h3 {color: #4CAF50;}

/* Sidebar background ko bhi halka green touch */
[data-testid="stSidebar"] {
    background-color: #e6ffe6; /* Thoda zyada green taaki alag dikhe */
}

/* Success/Info Boxes ko Green Skill look dena */
div.stAlert.st-success { 
    background-color: #e6ffe6; /* Light Green */
    border-left: 5px solid #4CAF50;
    color: #00796b; 
}
/* Info boxes ko bhi halka green touch */
div.stAlert.st-info {
    background-color: #f0fff0;
    border-left: 5px solid #00796b;
}
/* Button ko thoda solid green color */
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True) 



# --- 5. SIDEBAR METRICS ---
st.sidebar.header("📊 Model Performance (RFR)")
st.sidebar.metric("R² Score (Accuracy)", "0.9997", "Excellent")
st.sidebar.metric("Mean Absolute Error (MAE)", "0.0076 kWh", "Very Low")
st.sidebar.subheader("Driving Mode Mapping")
st.sidebar.markdown("1: Eco | 2: Normal | 3: Sport")


st.header("EV Range Prediction & Green Driving Analysis")

# --- 6. USER INPUTS ---
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
    
# --- 7. PREDICTION LOGIC ---
if st.button("Predict Range & Green Impact", key='predict_btn', use_container_width=True):
    
    if model is not None:
        with st.spinner('Calculating prediction from ML Model...'):
            
            # 1. Current Mode Prediction
            input_data_dict = cf.prepare_input(speed, temp, driving_mode, road_type, traffic_condition, slope, current_soc)
            consumption_current = cf.predict_energy_consumption_local(input_data_dict, model)
            
            # Green Skills Logic (Call from common_functions)
            predicted_range_current, co2_saved_kg = cf.calculate_range_metrics(consumption_current, current_soc)

            st.success("✅ Prediction Successful!")
            
            st.markdown("---")
            st.subheader("📊 Your Predicted Journey Metrics")

            colA, colB, colC, colD = st.columns(4)
            
            # Display Current Metrics
            colA.metric("Predicted Consumption", f"{consumption_current:.4f} kWh/km")
            
            # 🟢 CHANGE 1: Added "Approx." word
            colB.metric("Approx. Predicted Range", f"{predicted_range_current:.0f} km")
            
            # Display Green Skill 1: Emission Offset
            colC.metric("Emission Offset (CO2 Saved)", f"{co2_saved_kg:.1f} kg", "🔥 Green Impact!")
            
            # Display Driving Efficiency 
            colD.metric("Driving Efficiency", f"{100 - (consumption_current * 100):.1f} %", "High Score!")
            
            st.markdown("---")
            st.subheader("💡 Green Skill Analysis (Approximate Values)")

            # 🟢 CHANGE 2: Added 60 kWh EV model disclaimer and "approximately"
            st.info(
                f"⚠️ **Note:** These calculations are based on a generic **60 kWh Long-Range EV** (Approx. 400-450 km full range), as energy consumption varies significantly between car models.\n\n"
                f"On a *{road_type_name}* road in *{driving_mode_name} Mode*, this vehicle can travel **approximately** **{predicted_range_current:.0f} km** while saving **{co2_saved_kg:.1f} kg** of CO2 emissions compared to a fossil fuel car."
            )
            
            # 2. Green Skill 2: Eco Mode Comparison
            if driving_mode != 1:
                
                input_data_dict_eco = cf.prepare_input(speed, temp, 1, road_type, traffic_condition, slope, current_soc)
                consumption_eco = cf.predict_energy_consumption_local(input_data_dict_eco, model)
                predicted_range_eco, _ = cf.calculate_range_metrics(consumption_eco, current_soc)
                
                range_diff = predicted_range_eco - predicted_range_current
                
                st.markdown("---")
                st.subheader("🌳 Eco Mode Benefit (Maximize Range)")
                col_eco1, col_eco2 = st.columns(2)
                
                # 🟢 CHANGE 3: Added "Approx." word
                col_eco1.metric("Approx. Range in Eco Mode", f"{predicted_range_eco:.0f} km")
                col_eco2.metric("Range Gain vs Current Mode", f"{range_diff:.0f} km", f"{range_diff:.0f} km Gain!")
                
                if range_diff > 0:
                    st.success(f"By switching to *Eco Mode* (Green Skill), you can gain **approximately** **{range_diff:.0f} km** of extra range, making your trip significantly more efficient!")
                else:
                    st.warning("Eco Mode did not provide significant gain, likely due to low speed or low battery state.")
            
    else:
        st.error("Model not loaded. Please ensure the model file is accessible.")
