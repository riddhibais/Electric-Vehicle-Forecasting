import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import gdown 
import os 



DRIVE_FILE_ID = '11DRnNwkkYM9OxZELxU93B0pvFjLQYiwc' 
LOCAL_FILE_PATH = 'ev_energy_consumption_model.pkl'


@st.cache_resource
def download_file_from_drive():
    if not os.path.exists(LOCAL_FILE_PATH):
        
        try:
         
            gdown.download(id=DRIVE_FILE_ID, output=LOCAL_FILE_PATH, quiet=False)
        except Exception as e:
            st.error(f"Download Error. Kripya confirm karein ki Drive link 'Anyone with the link' par set hai: {e}")
            st.stop()
    
    try:
        with open(LOCAL_FILE_PATH, 'rb') as f:
            model = pickle.load(f)
        st.sidebar.success("Model Loaded Successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f"Model Load Error: Check file access/corruption: {e}")
        st.stop()
        return None

# --- EXECUTE DOWNLOAD & LOAD MODEL ---
model = download_file_from_drive()

#  CORE PREDICTION SETUP 

st.set_page_config(layout="wide")

st.title("⚡ Smart EV Range Prediction Assistant")

# --- MODEL METRICS DISPLAY ---
st.sidebar.header("📊 Model Performance (RFR)")
st.sidebar.metric("R² Score (Accuracy)", "0.9997", "Excellent")
st.sidebar.metric("Mean Absolute Error (MAE)", "0.0076 kWh", "Very Low")

st.sidebar.subheader("Driving Mode Mapping")
st.sidebar.markdown("1: Eco | 2: Normal | 3: Sport")

# IMPORTANT: Feature Names for Model Input
FEATURE_NAMES = [
    'Speed_kmh', 'Acceleration_ms2', 'Battery_State_%', 'Battery_Voltage_V', 
    'Battery_Temperature_C', 'Slope_%', 'Temperature_C', 'Humidity_%', 
    'Wind_Speed_ms', 'Tire_Pressure_psi', 'Vehicle_Weight_kg', 'Distance_Travelled_km', 
    'Driving_Mode_2', 'Driving_Mode_3', 
    'Road_Type_2', 'Road_Type_3', 
    'Traffic_Condition_2', 'Traffic_Condition_3', 
    'Weather_Condition_2', 'Weather_Condition_3', 'Weather_Condition_4' 
]

def predict_energy_consumption_local(input_data_dict, loaded_model):
    """Handles data preparation and local prediction using the loaded model."""
    if loaded_model is None:
        return 0.0
        
    input_data = pd.DataFrame(0, index=[0], columns=FEATURE_NAMES)
    
    # Fill values from the received data and apply One-Hot Encoding
    for key, value in input_data_dict.items():
        if key in input_data.columns:
            input_data.loc[0, key] = value
        elif key in ['Driving_Mode', 'Road_Type', 'Traffic_Condition', 'Weather_Condition']:
            dummy_col = f"{key}_{value}"
            if dummy_col in input_data.columns:
                input_data.loc[0, dummy_col] = 1

    input_df = input_data[FEATURE_NAMES]
    prediction = loaded_model.predict(input_df)
    return float(prediction[0])

# INPUT MAPPING: Realistic placeholders
def prepare_input(speed, temp, mode, road, traffic, slope, battery_state):
    input_data = {
        "Speed_kmh": speed,
        "Temperature_C": temp,
        "Driving_Mode": mode,
        "Road_Type": road,
        "Traffic_Condition": traffic,
        "Slope_%": slope,
        "Battery_State_%": battery_state,
        "Distance_Travelled_km": 1.0, 
        "Acceleration_ms2": 0.5,           
        "Battery_Voltage_V": 380.0,        
        "Battery_Temperature_C": 30.0,     
        "Humidity_%": 65.0,                
        "Wind_Speed_ms": 3.0,              
        "Tire_Pressure_psi": 38.0,         
        "Vehicle_Weight_kg": 2100.0,       
        "Weather_Condition": 2 
    }
    return input_data

# --- MAIN INTERFACE (TABS) ---
tab1, tab2 = st.tabs(["🚀 Live Prediction Form", "💬 Smart Assistant (Chatbot)"])

# ====================================================================
# TAB 1: LIVE PREDICTION FORM
# ====================================================================

with tab1:
    st.header("Real-Time Energy Consumption Calculator")
    
    # ... (User Input Fields) ...
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
        
    # Prediction Button
    if st.button("Predict Range & Consumption", key='predict_btn'):
        
        if model is not None:
            with st.spinner('Calculating prediction from ML Model...'):
                
                input_data_dict = prepare_input(speed, temp, driving_mode, road_type, traffic_condition, slope, current_soc)
                
                # 2. Call the LOCAL prediction function directly
                consumption = predict_energy_consumption_local(input_data_dict, model)
                
                # --- Range Calculation ---
                TOTAL_USABLE_BATTERY_KWH = 60.0
                remaining_energy = TOTAL_USABLE_BATTERY_KWH * (current_soc / 100)
                
                predicted_range = remaining_energy / consumption if consumption > 0 else 0.0

                st.success("✅ Prediction Successful!")
                
                colA, colB, colC = st.columns(3)
                
                colA.metric("Predicted Consumption", f"{consumption:.4f} kWh/km")
                colB.metric("Remaining Battery Energy", f"{remaining_energy:.2f} kWh")
                colC.metric("Predicted Range", f"{predicted_range:.0f} km")
                
                st.subheader("💡 Analysis")
                st.info(f"On a **{road_type_name}** road in **{driving_mode_name} Mode**, your vehicle can travel approximately **{predicted_range:.0f} km**.")
                
        else:
            st.error("Model not loaded. Please ensure the model file is accessible.")
            
# ====================================================================
# TAB 2: SMART ASSISTANT (CHATBOT)
# ====================================================================

with tab2:
    st.header("Smart Assistant: Conversational Range Advice")
    st.info("💡 NOTE: This Chatbot tab is the future home for Smart Charging Recommendation logic.")
    
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{'role': 'assistant', 'content': 'Hello! I am your Smart EV Assistant. Ask me your range-related questions.'}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        response_text = f"Thank you for your question: '{prompt}'. Once the full Charging Recommendation logic is implemented, I will give you smart advice on range and nearby stations."
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.chat_message("assistant").write(response_text)
