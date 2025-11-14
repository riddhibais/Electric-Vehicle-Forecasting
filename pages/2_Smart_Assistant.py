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
    """Provides detailed explanations for model features like Slope."""
    prompt_lower = user_prompt.lower()
    
    if any(keyword in prompt_lower for keyword in ["slope", "road slope", "incline", "dhalan"]):
        return ("**🛣️ Road Slope (%)**: This represents the steepness of the road.\n"
                "* **Positive Slope (+5%)** means **driving uphill**, which dramatically increases energy usage.\n"
                "* **Negative Slope (-5%)** means **driving downhill**, allowing for energy **recovery** via regenerative braking.\n"
                "* **0%** means a **flat road**.")
    
    elif any(keyword in prompt_lower for keyword in ["soc", "battery state", "battery percentage", "range"]):
        return ("**🔋 Battery State of Charge (SOC) %**: This is the remaining percentage of energy in your battery, used to calculate your remaining driving range.")

    elif any(keyword in prompt_lower for keyword in ["consumption", "kwh/km", "average"]):
        return ("**⚡ Energy Consumption (kWh/km)**: This is the energy (kWh) your car uses to travel one kilometer. Lower consumption is better.")
    
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
    current_speed = float(speed_match.group(1) or speed_match.group(2)) if speed_match else 60.0
    current_soc = float(battery_match.group(1)) if battery_match else 75.0
    current_slope = float(slope_match.group(1)) if slope_match else 0.0

    # --- 2. Extract Categorical Values (Defaults) ---
    driving_mode = 2 # Default: Normal
    if "eco" in prompt_lower: driving_mode = 1
    elif "sport" in prompt_lower: driving_mode = 3

    road_type = 2 # Default: Urban
    traffic_condition = 2 # Default: Moderate
    temp = 25.0 # Default temp
    
    # --- 3. Run Prediction ---
    try:
        input_data_dict = cf.prepare_input(
            current_speed, 
            temp, 
            driving_mode, 
            road_type, 
            traffic_condition, 
            current_slope, # <-- Updated Slope value
            current_soc
        )
        consumption = cf.predict_energy_consumption_local(input_data_dict, model)
        
        predicted_range, co2_saved_kg = cf.calculate_range_metrics(consumption, current_soc)
        
        if consumption <= 0.0001 and model is not None:
