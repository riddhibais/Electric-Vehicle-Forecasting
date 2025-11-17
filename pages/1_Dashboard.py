# pages/1_Dashboard.py

# --- 1. IMPORTS (Saare imports top par) ---
import streamlit as st
import common_functions as cf
import pandas as pd # Agar zaroorat padi toh

# --- 2. MODEL LOADING ---
# Model ko load karna
model = cf.download_file_from_drive()

# --- 3. PAGE CONFIGURATION AND AESTHETICS (Green Skill Theme) ---
st.set_page_config(
    page_title="Green Skill Dashboard",
    page_icon="🍃", # Green Leaf icon
    layout="wide", 
)

st.markdown("""
<style>
/* Main Title: Dark Green */
h1 {color: #00796b; text-align: center; font-weight: 700;} 
/* Subheaders: Bright Green */
h3 {color: #4CAF50;}

/* Info Boxes: Light green background and dark border for Green Skill focus */
div.stAlert.st-success { 
    background-color: #e6ffe6; /* Very light green */
    border-left: 5px solid #4CAF50;
    color: #00796b; /* Dark text for readability */
}

/* Green Skill message ko highlight karna */
.stMarkdown p strong { 
    color: #00796b !important;
}

/* Sidebar background ko bhi thoda aesthetic look */
.css-1d3s5yz {
    background-color: #f0f2f6; 
}
</style>
""", unsafe_allow_html=True) 

# --- 4. MAIN TITLE ---
st.title("🌱 EV Range Prediction Dashboard: Green Skill Score")
st.markdown("---") 

# --- 5. SIDEBAR METRICS ---
st.sidebar.header("📊 Model Performance (RFR)")
st.sidebar.metric("R² Score (Accuracy)", "0.9997", "Excellent")
st.sidebar.metric("Mean Absolute Error (MAE)", "0.0076 kWh", "Very Low")
st.sidebar.subheader("Driving Mode Mapping")
st.sidebar.markdown("1: Eco | 2: Normal | 3: Sport")


st.header("⚡ EV Range Prediction & Green Driving Analysis")

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

            # Success message will now have the custom green background
            st.success("✅ Prediction Successful!")
            
            st.markdown("---")
            st.subheader("📊 Your Predicted Journey Metrics")

            colA, colB, colC, colD = st.columns(4)
            
            # Display Current Metrics
            colA.metric("Predicted Consumption", f"{consumption_current:.4f} kWh/km")
            colB.metric("Predicted Range", f"**{predicted_range_current:.0f} km**")
            
            # Display Green Skill 1: Emission Offset (Using Custom Styling)
            colC.metric("Emission Offset (CO2 Saved)", f"{co2_saved_kg:.1f} kg", "🔥
