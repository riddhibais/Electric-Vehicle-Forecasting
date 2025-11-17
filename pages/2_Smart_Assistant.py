import streamlit as st
import common_functions as cf 
import re # Needed for text extraction in chat
import pandas as pd # Required for Streamlit map/data handling

# --- Load Model and Setup Sidebar ---
model = cf.download_file_from_drive()

st.title("💬 Smart Assistant: Green Driving & Charging")
st.markdown("---") 

# ====================================================================
# CHATBOT LOGIC FUNCTIONS (No changes needed here)
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
    
    # Check if this is a prediction request
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
# MAIN CHATBOT INTERFACE (Finalized Logic)
# ====================================================================

st.info("💡 NOTE: Ask about model features (e.g., **'What is slope?'**), **predict range** (e.g., `predict range at 80km/h with 60% battery`), or **find the nearest charging station near [City Name]**.")

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': 'assistant', 'content': 'Hello! Ask me about the model, or try *"Find nearest charging station near Pune"* to use the map. 📍'}]

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

        # 2. Nearest Station Check (Map Integration - FINAL LOGIC)
        if response_text is None and ("nearest" in prompt_lower and ("station" in prompt_lower or "charger" in prompt_lower or "map" in prompt_lower)):
            
            location_name = None
            search_center_lat, search_center_lon = None, None
            location_found = False # Flag to track if we should proceed to search

            # --- 2.1 Attempt to extract a city/location name from the prompt ---
            search_query_match = re.search(r'near\s+(.+)', prompt_lower)
            
            if search_query_match:
                # Case A: User provided a specific location (e.g., "near Mumbai")
                location_name = search_query_match.group(1).strip()
                st.info(f"Searching for stations near: **{location_name.title()}**")
                
                # Use Geocoding to get coordinates
                user_lat, user_lon, full_address = cf.get_coordinates_from_query(location_name)
                
                if user_lat is None:
                    # Geocoding failed
                    gmaps_url = cf.generate_gmaps_url(f"EV Charging Stations near {location_name.title()}", is_search=True)
                    response_text = (
                        f"❌ Sorry, I couldn't find the coordinates for **{location_name.title()}**. Please try a different name or a major city.\n\n"
                        f"**Tip:** You can search directly on [Google Maps]({gmaps_url})."
                    )
                else:
                    search_center_lat, search_center_lon = user_lat, user_lon
                    location_found = True # Location found, proceed to search
            
            else:
                # Case B: User asked for charging station without specifying location (PROMPT THE USER)
                response_text = (
                    "**🌎 Location Required!**\n\n"
                    "Please enter the **city or area name** for the search, like:\n"
                    "👉 **`Find nearest charging station near Mumbai`**"
                )
                
            # --- 2.2 Run Charging Station Search (Only if location was successfully found) ---
            if location_found: 
                
                # Assuming radius is 15km in common_functions.py
                stations_df = cf.find_nearest_charging_stations(search_center_lat, search_center_lon)
                
                if not stations_df.empty:
                    # Stations found via OSM
                    st.subheader(f"📍 Charging Stations Found (15km Radius)")
                    stations_df.rename(columns={'lat': 'latitude', 'lon': 'longitude'}, inplace=True)
                    st.map(stations_df, zoom=12, use_container_width=True)
                    
                    nearest_details = cf.calculate_nearest_station_details(stations_df, search_center_lat, search_center_lon)
                    gmaps_search_query = f"EV Charging Stations near {location_name.title()}"
                    gmaps_url = cf.generate_gmaps_url(gmaps_search_query, is_search=True)
                    
                    response_text = (
                        f"Here are the stations I found based on the 15 km search radius around **{location_name.title()}**.\n\n"
                        f"{nearest_details}\n\n"
                        f"**🗺️ External Map Link:**\n"
                        f"If you want to see more options and private chargers, click here: "
                        f"**➡️ [Search Charging Stations on Google Maps]({gmaps_url})**"
                    )
                
                else:
                    # Case C: No stations found via OSM, but location was valid (Direct Google Maps Link)
                    
                    gmaps_search_query = f"EV Charging Stations near {location_name.title()}"
                    gmaps_url = cf.generate_gmaps_url(gmaps_search_query, is_search=True)
                    
                    response_text = (
                        f"**⚠️ Search Result:** No free charging stations were found in the 15 km radius around **{location_name.title()}** based on OpenStreetMap (OSM) data.\n\n"
                        f"**View Now:** You can instantly see all available charging stations (public, private, etc.) in this area directly on Google Maps.\n"
                        f"**➡️ [Search Charging Stations on Google Maps]({gmaps_url})**"
                    )


        # 3. Try Prediction (Only if response_text is still None)
        if response_text is None:
            response_text = handle_prediction_chat(prompt, model)
        
        # 4. Generic Reply (Only if response_text is still None)
        if response_text is None:
            response_text = "I'm sorry, I can only answer questions related to the **EV model features**, **predict range**, or **find the nearest charging station near [City Name]**."

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.chat_message("assistant").write(response_text)
