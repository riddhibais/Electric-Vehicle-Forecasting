import streamlit as st
import common_functions as cf
import re 
import pandas as pd 

# --- Load Model and Setup Sidebar ---
model = cf.download_file_from_drive()

st.title("💬 Smart Assistant: Green Driving & Charging")
st.markdown("---") 

# ====================================================================
# CHATBOT LOGIC FUNCTIONS 
# ====================================================================

def handle_doubt_clearing(user_prompt):
    """Provides detailed explanations for model features."""
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
    
    if not any(keyword in prompt_lower for keyword in ["predict", "range", "consumption"]):
        return None
    
    # --- 1. Extract Numerical Values ---
    speed_match = re.search(r'(\d+)\s*km/?h|at\s*(\d+)', prompt_lower)
    battery_match = re.search(r'(\d+)\s*%', prompt_lower)
    slope_match = re.search(r'slope\s*(\-?\+?\d+\.?\d*)', prompt_lower)

    # Assign extracted or default values
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
        return "Sorry, I could not find enough parameters (Speed and Battery %) in your query to run the prediction model. Please provide them explicitly."


# ====================================================================
# MAIN CHATBOT INTERFACE (Bug Fixes and Final Logic)
# ====================================================================

st.info("💡 NOTE: Ask about model features, range prediction, or **find the nearest charging station near [City Name]**.")

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': 'assistant', 'content': 'Hello! Ask me about the model, or try *"Find nearest charging station near Pune"* to use the map. 📍'}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- SYNTAX ERROR FIX APPLIED HERE ---
prompt = st.chat_input("Type your question here...")

if prompt: # Standard check, avoids Walrus Operator issue
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    prompt_lower = prompt.lower()
    response_text = None
    
    with st.spinner('Thinking...'):
        
        # 1. Doubt Clearing
