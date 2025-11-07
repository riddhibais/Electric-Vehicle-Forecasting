import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import gdown 
import os 
import re 
from math import radians, sin, cos, sqrt, atan2 # <-- NEW: For Haversine distance


# 1. Google Drive Model ID
DRIVE_FILE_ID = '11DRnNwkkYM9OxZELxU93B0pvFjLQYiwc' 
LOCAL_FILE_PATH = 'ev_energy_consumption_model.pkl'

# --- DOWNLOAD & LOAD MODEL FUNCTION ---
@st.cache_resource
def download_file_from_drive():
    if not os.path.exists(LOCAL_FILE_PATH):
        try:
            gdown.download(id=DRIVE_FILE_ID, output=LOCAL_FILE_PATH, quiet=False)
        except Exception as e:
            st.error(f"Download Error. Check file access/Drive sharing settings: {e}")
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

# --- 3. CORE PREDICTION SETUP ---

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

# ====================================================================
# CHARGING STATION LOGIC (NEW SECTION)
# ====================================================================

# --- Dummy Charging Station Data (You can replace this with real data) ---
CHARGING_STATIONS_DATA = {
    'Station_Name': ['City Center Fast Charge', 'Highway Rest Stop', 'Mall Parking Station'],
    'Latitude': [21.2500, 21.2700, 21.2300], # Example Latitudes (Near Raipur)
    'Longitude': [81.6300, 81.6000, 81.6500] # Example Longitudes
}
STATIONS_DF = pd.DataFrame(CHARGING_STATIONS_DATA)
EARTH_RADIUS_KM = 6371

def haversine(lat1, lon1, lat2, lon2):
    """Calculates the distance between two points on the earth in kilometers."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return EARTH_RADIUS_KM * c

def calculate_nearest_station(user_lat, user_lon):
    """Finds the closest station to the user's location."""
    
    if STATIONS_DF.empty:
        return "No station data available."

    distances = STATIONS_DF.apply(
        lambda row: haversine(user_lat, user_lon, row['Latitude'], row['Longitude']),
        axis=1
    )
    
    closest_station_index = distances.idxmin()
    closest_station = STATIONS_DF.loc[closest_station_index]
    min_distance = distances.min()
    
    return (f"The nearest charging station is **{closest_station['Station_Name']}**.\n"
            f"It is approximately **{min_distance:.2f} km** away from your location.")

# ====================================================================
# NEW CHATBOT LOGIC FUNCTIONS (Unchanged from last step)
# ====================================================================

def handle_doubt_clearing(user_prompt):
    """Provides detailed explanations for model features."""
    prompt_lower = user_prompt.lower()
    
    if any(keyword in prompt_lower for keyword in ["slope", "road slope", "incline"]):
        return ("**Road Slope (%)**: This represents the steepness of the road. A positive slope (e.g., +5%) means driving uphill, which dramatically increases energy usage. A negative slope (e.g., -5%) means driving downhill, where regenerative braking can recover energy. In our model, **0% is a flat road**.")
    
    elif any(keyword in prompt_lower for keyword in ["soc", "battery state", "battery percentage"]):
        return ("**Battery State of Charge (SOC) %**: This is simply the remaining percentage of energy in your battery. This value helps calculate your remaining driving range: **Remaining Range = (Total Battery Energy * SOC) / Consumption Rate**.")

    elif any(keyword in prompt_lower for keyword in ["road type", "highway", "urban", "rural"]):
        return ("**Road Type**: This categorizes the driving environment, affecting speed limits and traffic. **1: Highway**, **2: Urban** (city driving), **3: Rural** (country roads). Each type has different typical consumption profiles.")

    elif any(keyword in prompt_lower for keyword in ["consumption", "kwh/km"]):
        return ("**Energy Consumption (kWh/km)**: This is the core output of the model—how many Kilowatt-hours (kWh) of energy your car uses to travel one kilometer. Lower is better. This rate depends heavily on speed, slope, and driving mode.")
    
    return None

def handle_prediction_chat(user_prompt, model):
    """Extracts parameters from text and returns a prediction."""
    prompt_lower = user_prompt.lower()
    
    if not any(keyword in prompt_lower for keyword in ["predict", "range", "consumption"]):
        return None # Not a prediction request
    
    # --- 1. Extract Numerical Values ---
    speed_match = re.search(r'(\d+)\s*km/?h|at\s*(\d+)', prompt_lower)
    battery_match = re.search(r'(\d+)\s*%', prompt_lower)
    
    speed = float(speed_match.group(1) or speed_match.group(2)) if speed_match else 60.0
    current_soc = float(battery_match.group(1)) if battery_match else 75.0
    
    # --- 2. Extract Categorical Values ---
    driving_mode = 2 # Default: Normal
    if "eco" in prompt_lower: driving_mode = 1
    elif "sport" in prompt_lower: driving_mode = 3

    road_type = 2 # Default: Urban
    if "highway" in prompt_lower: road_type = 1
    elif "rural" in prompt_lower: road_type = 3
    
    slope_match = re.search(r'slope\s*(\-?\+?\d+\.?\d*)', prompt_lower)
    slope = float(slope_match.group(1)) if slope_match else 0.0

    # --- 3. Run Prediction ---
    try:
        input_data_dict = prepare_input(speed, 25.0, driving_mode, road_type, 2, slope, current_soc)
        consumption = predict_energy_consumption_local(input_data_dict, model)
        
        TOTAL_USABLE_BATTERY_KWH = 60.0
        remaining_energy = TOTAL_USABLE_BATTERY_KWH * (current_soc / 100)
        predicted_range = remaining_energy / consumption if consumption > 0 else 0.0
        
        if consumption <= 0.0001:
            return "**Error**: Consumption too low. Please adjust inputs."
            
        return (f"Based on your query, the parameters used are: **Speed: {speed} km/h, Battery: {current_soc}%, Slope: {slope}%, Mode: {driving_mode}**.\n\n"
                f"**Predicted Consumption**: {consumption:.3f} kWh/km\n"
                f"**Predicted Range**: {predicted_range:.0f} km")

    except Exception:
        return "Sorry, I could not find enough parameters (Speed and Battery %) in your query to run the prediction model. Please provide them explicitly."


# ====================================================================
# MAIN INTERFACE (TABS)
# ====================================================================

tab1, tab2 = st.tabs(["🚀 Live Prediction Form", "💬 Smart Assistant (Chatbot)"])

# --- TAB 1: LIVE PREDICTION FORM (Unchanged) ---
# ... (Tab 1 content remains the same) ...

with tab1:
    st.header("Real-Time Energy Consumption Calculator")
    
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
        
    if st.button("Predict Range & Consumption", key='predict_btn'):
        
        if model is not None:
            with st.spinner('Calculating prediction from ML Model...'):
                input_data_dict = prepare_input(speed, temp, driving_mode, road_type, traffic_condition, slope, current_soc)
                consumption = predict_energy_consumption_local(input_data_dict, model)
                
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
            
# --- TAB 2: SMART ASSISTANT 

with tab2:
    st.header("Smart Assistant: Conversational Range Advice")
    st.info("💡 NOTE: Ask about model features, range prediction, or nearest charging stations (using default coordinates: 21.25°N, 81.63°E).")
    
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{'role': 'assistant', 'content': 'Hello! Ask me about the model, or try **"Find nearest charging station"** to use my default location (Raipur). 📍'}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        prompt_lower = prompt.lower()
        response_text = None
        
        with st.spinner('Thinking...'):
            
            # --- 1. Nearest Station Check ---
            if "nearest" in prompt_lower and ("station" in prompt_lower or "charger" in prompt_lower):
                # Using default coordinates for now (e.g., Raipur: 21.25, 81.63)
                user_lat = 21.2500 
                user_lon = 81.6300 
                
                response_text = calculate_nearest_station(user_lat, user_lon)
                response_text = f"Using default location (Lat: {user_lat}, Lon: {user_lon}):\n\n" + response_text
            
            # --- 2. Try Prediction ---
            if response_text is None:
                response_text = handle_prediction_chat(prompt, model)
            
            # --- 3. Try Doubt Clearing ---
            if response_text is None:
                response_text = handle_doubt_clearing(prompt)
            
            # --- 4. Generic Reply ---
            if response_text is None:
                response_text = "I'm sorry, I can only answer questions related to the **EV model features**, **predict range**, or **find the nearest charging station** (using a default location). Please rephrase your question."

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.chat_message("assistant").write(response_text)
