import streamlit as st
import common_functions as cf 
import re 
import pandas as pd 

# --- Load Model and Setup Sidebar ---
# The model loading logic is now resilient due to fixes in common_functions.py
model = cf.download_file_from_drive()

st.title("💬 Smart Assistant: Green Driving & Charging")
st.markdown("---") 

# ====================================================================
# CHATBOT LOGIC FUNCTIONS 
# ====================================================================

def handle_doubt_clearing(user_prompt):
    """Provides detailed explanations for model features when the user asks 'What is X?'"""
    prompt_lower = user_prompt.lower()
    
    if any(keyword in prompt_lower for keyword in ["slope", "road slope", "incline", "dhalan"]):
        return ("**🛣️ Road Slope (%)**: This describes the steepness of the road.\n"
                "* **Positive Slope (+5%)** means **driving uphill**, which increases energy usage.\n"
                "* **Negative Slope (-5%)** means **driving downhill**, where regenerative braking can **recover** energy.\n"
                "* **0%** means a **flat road**.")
    
    elif any(keyword in prompt_lower for keyword in ["soc", "battery state", "battery percentage", "range"]):
        return ("**🔋 Battery State of Charge (SOC) %**: This is the remaining percentage of energy in your battery, used to calculate your remaining driving range.")

    elif any(keyword in prompt_lower for keyword in ["consumption", "kwh/km", "average"]):
        return ("**⚡ Energy Consumption (kWh/km)**: This is the core output—the energy (kWh) the car uses per kilometer. **Lower is always better**.")
    
    elif any(keyword in prompt_lower for keyword in ["road type", "highway", "urban", "rural"]):
        return ("*Road Type: This categorizes the driving environment: **1: Highway, **2: Urban*, *3: Rural*. Each affects typical speed and traffic conditions.")
    
    return None

def handle_prediction_chat(user_prompt, model):
    """Extracts parameters from text (including slope) and returns a prediction."""
    prompt_lower = user_prompt.lower()
    
    # Check if this is a prediction request
    if not any(keyword in prompt_lower for keyword in ["predict", "range", "consumption"]):
        return None
    
    # --- 1. Extract Numerical Values ---
    speed_match = re.search(r'(\d+)\s*km/?h|at\s*(\d+)', prompt_lower)
    battery_match = re.search(r'(\d+)\s*%', prompt_lower)
    slope_match = re.search(r'slope\s*(\-?\+?\d+\.?\d*)', prompt_lower)

    # Assign extracted or default values
    # speed_match.group(1) is for '80 km/h', group(2) is for 'at 80'
    speed = float(speed_match.group(1) or speed_match.group(2)) if speed_match else 60.0
    current_soc = float(battery_match.group(1)) if battery_match else 75.0
    slope = float(slope_match.group(1)) if slope_match else 0.0
    
    # --- 2. Extract Categorical Values (Using Defaults: Normal, Urban, Moderate Traffic) ---
    driving_mode = 2 
    if "eco" in prompt_lower: driving_mode = 1
    elif "sport" in prompt_lower: driving_mode = 3

    road_type = 2 
    if "highway" in prompt_lower: road_type = 1
    elif "rural" in prompt_lower: road_type = 3
    
    # --- 3. Run Prediction ---
    try:
        if model is None:
             return "Model failed to load, prediction cannot be performed."

        input_data_dict = cf.prepare_input(
            speed=speed, 
            temp=25.0, # Default Temp
            mode=driving_mode, 
            road=road_type, 
            traffic=2, # Default Traffic
            slope=slope,
            battery_state=current_soc
        )
        consumption = cf.predict_energy_consumption_local(input_data_dict, model)
        
        predicted_range, co2_saved_kg = cf.calculate_range_metrics(consumption, current_soc)
        
        if consumption <= 0.0001:
            return "Consumption calculation failed (returned zero). Please adjust inputs or check model data."
        
        slope_analysis = f", Slope: **{slope}%**" if slope != 0.0 else ""
            
        return (f"**✅ Prediction Complete**\n\n"
                f"Based on: *Speed: {speed} km/h, Battery: {current_soc}%{slope_analysis}, Mode: {driving_mode}*.\n\n"
                f"*Predicted Consumption*: **{consumption:.3f} kWh/km**\n"
                f"*Predicted Range*: **{predicted_range:.0f} km**\n"
                f"*Green Skill: You are saving **{co2_saved_kg:.1f} kg CO2** on this range.")

    except Exception as e:
        # Generic error handling for failed regex or other issues
        return "Sorry, I could not find enough parameters (Speed and Battery %) in your query to run the prediction model. Please provide them explicitly."


# ====================================================================
# MAIN CHATBOT INTERFACE
# ====================================================================

st.info("💡 NOTE: Ask about model features (e.g., **'What is slope?'**), **predict range** (e.g., `predict range at 80km/h with 60% battery and 5% slope`), or **find the nearest charging station**.")

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': 'assistant', 'content': 'Hello! Ask me about the model, or try *"Find nearest charging station"* to use the map. 📍'}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    prompt_lower = prompt.lower()
    response_text = None
    
    with st.spinner('Thinking...'):
        
        # 1. Doubt Clearing Check (PRIORITY 1)
        response_text = handle_doubt_clearing(prompt)

        # 2. Nearest Station Check (Map Integration)
        if response_text is None and ("nearest" in prompt_lower and ("station" in prompt_lower or "charger" in prompt_lower or "map" in prompt_lower)):
            
            # This uses the coordinates defined in common_functions (currently Bengaluru) as the search center.
            user_lat, user_lon = cf.DEFAULT_LOCATION
            
            stations_df = cf.find_nearest_charging_stations(user_lat, user_lon)
            
            if not stations_df.empty:
                st.subheader("📍 Nearest Charging Stations (5km Radius - OpenStreetMap)")
                
                # Prepare data for map
                stations_df.rename(columns={'lat': 'latitude', 'lon': 'longitude'}, inplace=True)
                
                # st.map will attempt to center on the user's browser location if permission is granted, 
                # otherwise it centers on the station data.
                st.map(stations_df, zoom=12, use_container_width=True)
                
                # Get distance details
                nearest_details = cf.calculate_nearest_station_details(stations_df, user_lat, user_lon)
                
                st.info(f"The map attempts to use your current location, but the search was conducted around the default area (Lat: {user_lat:.2f}, Lon: {user_lon:.2f}).\n\n{nearest_details}")
                response_text = "Here are the free stations I found on the map!"
            
            else:
                response_text = f"I couldn't find any charging stations near the search area (within 5 km of the default location). The data might be missing for this area."


        # 3. Try Prediction 
        if response_text is None:
            response_text = handle_prediction_chat(prompt, model)
        
        # 4. Generic Reply 
        if response_text is None:
            response_text = "I'm sorry, I can only answer questions related to the **EV model features**, **predict range**, or **find the nearest charging station**."

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.chat_message("assistant").write(response_text)
