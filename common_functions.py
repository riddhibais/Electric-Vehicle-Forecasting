# common_functions.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import gdown 
import os 
import re 
from math import radians, sin, cos, sqrt, atan2
import googlemaps # New: Google Maps Library

# ==============================================================================
# ⚠️ CONFIGURATION: CONSTANTS & MODEL SETUP
# ==============================================================================

# --- API KEY & MAPS CLIENT ---
# Load API Key from .streamlit/secrets.toml
try:
    API_KEY = st.secrets["GOOGLE_MAPS_API_KEY"]
    gmaps_client = googlemaps.Client(key=API_KEY)
except (KeyError, AttributeError):
  #  st.error("Google Maps API Key not found in .streamlit/secrets.toml. Maps feature will be disabled.")
    gmaps_client = None

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
# CHARGING STATION LOGIC (USING Google Maps) 
# ====================================================================

# We will search for stations within a 5km radius of the user's default location.
DEFAULT_LOCATION = [21.2500, 81.6300] # Default: Raipur, CG (just a placeholder)
EARTH_RADIUS_KM = 6371

def haversine(lat1, lon1, lat2, lon2):
    """Calculates the distance between two points on the earth in kilometers."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return EARTH_RADIUS_KM * c


def find_nearest_charging_stations(user_lat, user_lon, radius_km=5):
    """
    Finds charging stations using Google Places API and returns a DataFrame
    for display on the Streamlit map.
    """
    if gmaps_client is None:
        return pd.DataFrame()
    
    try:
        # Google Places API Call: Search for charging stations within a 5km radius
        places_result = gmaps_client.places_nearby(
            location=(user_lat, user_lon),
            radius=radius_km * 1000, # Radius in meters
            type='electric_vehicle_charging_station'
        )
        
        stations = []
        for place in places_result.get('results', []):
            stations.append({
                'Station_Name': place.get('name', 'Unnamed Station'),
                'lat': place['geometry']['location']['lat'],
                'lon': place['geometry']['location']['lng']
            })
        
        if not stations:
            st.warning(f"No EV charging stations found within {radius_km} km of the location.")
            return pd.DataFrame()

        return pd.DataFrame(stations)
        
    except Exception as e:
        st.error(f"Error fetching stations from Google Maps API: {e}")
        return pd.DataFrame()

def calculate_nearest_station_details(stations_df, user_lat, user_lon):
    """
    Calculates distance to the nearest station from the fetched DataFrame.
    """
    if stations_df.empty:
        return "No stations data to calculate distance."

    distances = stations_df.apply(
        lambda row: haversine(user_lat, user_lon, row['lat'], row['lon']),
        axis=1
    )
    
    closest_station_index = distances.idxmin()
    closest_station = stations_df.loc[closest_station_index]
    min_distance = distances.min()
    
    return (f"The nearest charging station found is **{closest_station['Station_Name']}**.\n"
            f"It is approximately **{min_distance:.2f} km** away.")
