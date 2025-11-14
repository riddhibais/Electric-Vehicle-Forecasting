# pages/2_Smart_Assistant.py

import streamlit as st
import common_functions as cf 
import re # Needed for text extraction in chat

model = cf.download_file_from_drive()

st.title("💬 Smart Assistant")
st.markdown("---") 

# --- CHATBOT LOGIC FUNCTIONS (Need to be defined here) ---
# These functions use common_functions for calculations (cf.)

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
        return None
    
    # --- 1. Extract Numerical Values ---
    speed_match = re.search(r'(\d+)\s*km/?h|at\s*(\d+)', prompt_lower)
    battery_match = re.search(r'(\d+)\s*%', prompt_lower)
    
    speed = float(speed_match.group(1) or speed_match.group(2)) if speed_match else 60.0
    current_soc = float(battery_match.group(1)) if battery_match else 75.0
    
    # --- 2. Extract Categorical Values ---
    driving_mode = 2 
    if "eco" in prompt_lower: driving_mode = 1
    elif "sport" in prompt_lower: driving_mode = 3

    road_type = 2 
    if "highway" in prompt_lower: road_type = 1
    elif "rural" in prompt_lower: road_type = 3
    
    slope_match = re.search(r'slope\s*(\-?\+?\d+\.?\d*)', prompt_lower)
    slope = float(slope_match.group(1)) if slope_match else 0.0

    # --- 3. Run Prediction ---
    try:
        input_data_dict = cf.prepare_input(speed, 25.0, driving_mode, road_type, 2, slope, current_soc)
        consumption = cf.predict_energy_consumption_local(input_data_dict, model)
        
        predicted_range, co2_saved_kg = cf.calculate_range_metrics(consumption, current_soc)
        
        if consumption <= 0.0001:
            return "**Error**: Consumption too low. Please adjust inputs."
            
        return (f"Based on your query, the parameters used are: **Speed: {speed} km/h, Battery: {current_soc}%, Slope: {slope}%, Mode: {driving_mode}**.\n\n"
                f"**Predicted Consumption**: {consumption:.3f} kWh/km\n"
                f"**Predicted Range**: {predicted_range:.0f} km\n"
                f"**Green Skill**: You are saving **{co2_saved_kg:.1f} kg CO2** on this range.")

    except Exception:
        return "Sorry, I could not find enough parameters (Speed and Battery %) in your query to run the prediction model. Please provide them explicitly."


# --- MAIN CHATBOT INTERFACE ---

st.info("💡 NOTE: Ask about model features, range prediction, or nearest charging stations (using default coordinates: 21.25°N, 81.63°E).")

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': 'assistant', 'content': 'Hello! Ask me about the model, or try **"Find nearest charging station"** to use my default location. 📍'}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    prompt_lower = prompt.lower()
    response_text = None
    
    with st.spinner('Thinking...'):
        
        # 1. Nearest Station Check
        if "nearest" in prompt_lower and ("station" in prompt_lower or "charger" in prompt_lower):
            user_lat = 21.2500 
            user_lon = 81.6300 
            response_text = cf.calculate_nearest_station(user_lat, user_lon)
            response_text = f"Using default location (Lat: {user_lat}, Lon: {user_lon}):\n\n" + response_text
        
        # 2. Try Prediction 
        if response_text is None:
            response_text = handle_prediction_chat(prompt, model)
        
        # 3. Try Doubt Clearing 
        if response_text is None:
            response_text = handle_doubt_clearing(prompt)
        
        # 4. Generic Reply 
        if response_text is None:
            response_text = "I'm sorry, I can only answer questions related to the **EV model features**, **predict range**, or **find the nearest charging station** (using a default location). Please rephrase your question."

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.chat_message("assistant").write(response_text)
