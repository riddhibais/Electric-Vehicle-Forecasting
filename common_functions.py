# common_functions.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import gdown 
import os 
import re 
from math import radians, sin, cos, sqrt, atan2
import requests # NEW: Standard library for making web requests (used for free Overpass API)

# ==============================================================================
# ⚠️ CONFIGURATION: CONSTANTS & MODEL SETUP
# ==============================================================================

# --- MAPS CONFIGURATION (NO KEY NEEDED) ---
gmaps_client = None # Google Maps client is now obsolete
DEFAULT_LOCATION = [21.2500, 81.6300] # Default: Raipur, CG (placeholder for map center)
EARTH_RADIUS_KM = 6371

# Model ID and Path
DRIVE_FILE_ID = '11DRnNwkkYM9OxZELxU93B0pvFjLQYiwc' 
LOCAL_FILE_PATH = 'ev_energy_consumption_model.pkl'

# Green Skills and Vehicle Constants
TOTAL_USABLE_BATTERY_KWH = 60.0
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
    
    # Fill values from the
