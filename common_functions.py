# common_functions.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import gdown 
import os 
import re 
from math import radians, sin, cos, sqrt, atan2

# ==============================================================================
# ⚠️ CONFIGURATION: CONSTANTS & MODEL SETUP
# ==============================================================================

# Model ID and Path
DRIVE_FILE_ID = '11DRnNwkkYM9OxZELxU93B0pvFjLQYiwc' 
LOCAL_FILE_PATH = 'ev_energy_consumption_model.pkl'

# Green Skills and Vehicle Constants
TOTAL_USABLE_BATTERY_KWH = 60.0
# Avg CO2 emission of a petrol/diesel car per km (in kg)
EMISSION_FACTOR_KG_PER_KM = 0.18 

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

# GREEN SKILLS LOGIC
def calculate_range_metrics(consumption, current_soc):
    """Calculates remaining energy, predicted range, and CO2 savings."""
    remaining_energy = TOTAL_USABLE_BATTERY_KWH * (current_soc / 100)
    predicted_range = remaining_energy / consumption if consumption > 0 else 0.0
    
    # Emission Offset (Green Skill 1)
    co2_saved_kg = predicted_range * EMISSION_FACTOR_KG_PER_KM

    return predicted_range, co2_saved_kg

# ====================================================================
# CHARGING STATION LOGIC 
# ====================================================================

CHARGING_STATIONS_DATA = {
    'Station_Name': ['City Center Fast Charge', 'Highway Rest Stop', 'Mall Parking Station'],
    'Latitude': [21.2500, 21.2700, 21.2300], 
    'Longitude': [81.6300, 81.6000, 81.6500] 
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
