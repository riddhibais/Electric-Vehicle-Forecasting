# common_functions.py

import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim

import numpy as np
import pickle 
import gdown 
import os 
import re 
from math import radians, sin, cos, sqrt, atan2
import requests 

# ==============================================================================
# CONFIGURATION: CONSTANTS & MODEL SETUP
# ==============================================================================

gmaps_client = None 
DEFAULT_LOCATION = [21.2500, 81.6300] 
EARTH_RADIUS_KM = 6371

DRIVE_FILE_ID = '11DRnNwkkYM9OxZELxU93B0pvFjLQYiwc' 
LOCAL_FILE_PATH = 'ev_energy_consumption_model.pkl'

# Green Skills and Vehicle Constants
# 🟢 CHANGE: Increased to 60.0 kWh for higher range (approx 400 km @ 100%)
TOTAL_USABLE_BATTERY_KWH = 60.0 
EMISSION_FACTOR_KG_PER_KM = 0.18 
MODEL_SCALING_FACTOR = 70.0 

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
            st.error(f"Model Download Error: {e}")
    
    try:
        with open(LOCAL_FILE_PATH, 'rb') as f:
            model = pickle.load(f)
        st.sidebar.success("Model Loaded Successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f"Model Load Error: Check file/corruption. {e}")
        return None

# INPUT MAPPING
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
    
    global MODEL_SCALING_FACTOR 
    
    if loaded_model is None:
        return 0.11 

    try: 
        input_data = pd.DataFrame(0, index=[0], columns=FEATURE_NAMES)
        
        for key, value in input_data_dict.items():
            if key in input_data.columns:
                input_data.loc[0, key] = value
            elif key in ['Driving_Mode', 'Road_Type', 'Traffic_Condition', 'Weather_Condition']:
                dummy_col = f"{key}_{value}"
                if dummy_col in input_data.columns:
                    input_data.loc[0, dummy_col] = 1

        input_df = input_data[FEATURE_NAMES]
        prediction = loaded_model.predict(input_df)
        consumption = float(prediction[0])
        
        # FINAL SCALING FIX: Corrects consumption to a realistic kWh/km value
        consumption_scaled = max(consumption / MODEL_SCALING_FACTOR, 0.05)
        
        if consumption_scaled > 0.35:
             consumption_scaled = 0.35 

        return consumption_scaled
        
    except Exception as e:
        return 0.11 


# GREEN SKILLS LOGIC
def calculate_range_metrics(consumption, current_soc):
    
    global TOTAL_USABLE_BATTERY_KWH
    
    if not isinstance(consumption, (int, float)) or consumption <= 0.0001:
        return 0.0, 0.0 

    remaining_energy = TOTAL_USABLE_BATTERY_KWH * (current_soc / 100)
    predicted_range = remaining_energy / consumption if consumption > 0 else 0.0
    
    co2_saved_kg = predicted_range * EMISSION_FACTOR_KG_PER_KM

    return predicted_range, co2_saved_kg


# ====================================================================
# CHARGING STATION LOGIC (USING OPENSTREETMAP - OVERPASS API) 
# ====================================================================

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return EARTH_RADIUS_KM * c


def get_coordinates_from_query(query):
    geolocator = Nominatim(user_agent="EV_App_Assistant")
    try:
        location = geolocator.geocode(query, timeout=5)
        if location:
            return location.latitude, location.longitude, location.address
        return None, None, None
    except Exception:
        return None, None, None

def generate_gmaps_url(query, is_search=False):
    base_url = "https://www.google.com/maps/"
    if is_search:
        return f"https://www.google.com/maps/search/{query.replace(' ', '+')}"
    else:
        return f"https://www.google.com/maps/place/{query.replace(' ', '+')}"


def find_nearest_charging_stations(user_lat, user_lon, radius_km=5):
    
    overpass_url = "http://overpass-api.de/api/interpreter"
    radius_meters = radius_km * 1000
    
    overpass_query = f"""
    [out:json];
    node(around:{radius_meters},{user_lat},{user_lon})[amenity=charging_station];
    out center;
    """
    
    try:
        response = requests.get(overpass_url, params={'data': overpass_query}, timeout=15)
        response.raise_for_status() 
        data = response.json()
        
        stations = []
        for element in data.get('elements', []):
            if element.get('type') == 'node':
                stations.append({
                    'Station_Name': element.get('tags', {}).get('name', 'OSM Station'),
                    'lat': element['lat'],
                    'lon': element['lon']
                })
        
        if not stations:
            return pd.DataFrame()

        return pd.DataFrame(stations)
        
    except requests.exceptions.RequestException as e:
        return pd.DataFrame()

def calculate_nearest_station_details(stations_df, user_lat, user_lon):
    if stations_df.empty:
        return "No stations data to calculate distance."

    distances = stations_df.apply(
        lambda row: haversine(user_lat, user_lon, row['lat'], row['lon']),
        axis=1
    )
    
    closest_station_index = distances.idxmin()
    closest_station = stations_df.loc[closest_station_index]
    min_distance = distances.min()
    
    station_name = closest_station['Station_Name'] if closest_station['Station_Name'] != 'OSM Station' else f"Station at ({closest_station['lat']:.2f}, {closest_station['lon']:.2f})"
    
    return (f"The nearest charging station found (from OpenStreetMap) is **{station_name}**.\n"
            f"It is approximately **{min_distance:.2f} km** away.")
