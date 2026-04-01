import streamlit as st
import pandas as pd
import json
import os
import folium
import numpy as np

from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
from sklearn.neighbors import BallTree

# SPEED PROFILES
SPEED_OPTIONS = {
    "🚶 Walking": 5,
    "🚗 Driving": 30,
    "🚑 Ambulance": 50
}

# Load Data
@st.cache_data
def load_and_prepare_data():
    file_path = "health_facility_data_feb11.json"
    if not os.path.exists(file_path):
        st.error("JSON file not found.")
        return None

    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)

    df = pd.json_normalize(data["Licenced_HealthFacilities"])

    # Clean coordinates
    df['Geo_Location.Latitude'] = pd.to_numeric(df.get('Geo_Location.Latitude'), errors='coerce')
    df['Geo_Location.Longitude'] = pd.to_numeric(df.get('Geo_Location.Longitude'), errors='coerce')
    df = df.dropna(subset=['Geo_Location.Latitude', 'Geo_Location.Longitude'])

    # Level Category
    if 'Level' in df.columns:
        df['Level'] = df['Level'].astype(str).str.upper().str.strip()
        def categorize_level(level):
            if level in ['LEVEL 4', 'LEVEL 5', 'LEVEL 4B', 'LEVEL 6B', 'LEVEL 6A']:
                return 'Level 4+ (All services)'
            elif level in ['LEVEL 3A', 'LEVEL 3B']:
                return 'Level 3 (50% services)'
            elif level == 'LEVEL 2':
                return 'Below Level 3 (Basic services)'
            else:
                return 'Unknown'
        df['Level_Category'] = df['Level'].apply(categorize_level)

    return df

df_final = load_and_prepare_data()
if df_final is None:
    st.stop()

# Geocode & Nearest Functions
@st.cache_data
def geocode_location(place):
    geolocator = Nominatim(user_agent="kenya_health_locator")
    try:
        loc = geolocator.geocode(place + ", Kenya")
        if loc:
            return loc.latitude, loc.longitude
    except:
        pass
    return None, None

@st.cache_data
def find_nearest_hospitals(df, user_lat, user_lon, k=50):
    coords = df[['Geo_Location.Latitude', 'Geo_Location.Longitude']].values
    coords_rad = np.radians(coords)
    tree = BallTree(coords_rad, metric='haversine')
    user_point = np.radians([[user_lat, user_lon]])
    distances, indices = tree.query(user_point, k=k)
    distances_km = distances[0] * 6371
    nearest_df = df.iloc[indices[0]].copy()
    nearest_df['Distance_km'] = distances_km.round(1)
    return nearest_df

def add_travel_time(df, speed):
    df["Travel_time_minutes"] = (df["Distance_km"] / speed) * 60
    df["Travel_time_minutes"] = df["Travel_time_minutes"].round(1)
    return df

def filter_by_radius(df, radius_km):
    return df[df["Distance_km"] <= radius_km].sort_values("Distance_km")

# ================================================
# SIDEBAR
# ================================================
st.sidebar.title("🔍 Search")

user_location = st.sidebar.text_input("Enter your location", placeholder="Ruiru, Kitengela, Thika")

radius_km = st.sidebar.slider("Search radius (km)", 1, 100, 10)
num_results = st.sidebar.slider("Max hospitals to evaluate", 10, 200, 50)

speed_label = st.sidebar.selectbox("Travel mode", list(SPEED_OPTIONS.keys()))
speed = SPEED_OPTIONS[speed_label]

# Emergency Mode Toggle
emergency_mode = st.sidebar.toggle("🚨 Emergency Mode (Critical Care Only)")

# ← NEW: Clinical Input Text Box (as requested)
clinical_input = st.sidebar.text_area(
    "Clinical Input / Chief Complaint",
    placeholder="e.g. chest pain, labor pains, severe headache, burns, malaria, accident...",
    height=100
)

if emergency_mode:
    speed = SPEED_OPTIONS["🚑 Ambulance"]

# ================================================
# Main Processing
# ================================================
if user_location:
    lat, lon = geocode_location(user_location)
    if lat is None:
        st.error("Location not found. Try adding county (e.g. 'Kitengela, Kajiado')")
        st.stop()
    st.success(f"✅ Location found: {user_location}")
else:
    lat, lon = -1.29, 36.82

nearest_df = find_nearest_hospitals(df_final, lat, lon, num_results)
nearest_df = add_travel_time(nearest_df, speed)
df_filtered = filter_by_radius(nearest_df, radius_km)

if emergency_mode:
    df_filtered = df_filtered[df_filtered["Level_Category"] == "Level 4+ (All services)"].sort_values("Distance_km")

# Find Closest Reachable Hospital with valid phone
closest_reachable = None
if len(df_filtered) > 0:
    df_filtered = df_filtered.copy()
    df_filtered['PhoneNo'] = df_filtered['PhoneNo'].astype(str).str.strip()
    df_filtered['PhoneNo'] = df_filtered['PhoneNo'].replace(['nan', 'None', 'NULL', '', 'N/A', '(NULL)'], None)
    
    has_valid_phone = df_filtered['PhoneNo'].notna() & (df_filtered['PhoneNo'].str.len() > 5)
    
    if has_valid_phone.any():
        closest_reachable = df_filtered[has_valid_phone].iloc[0]
    else:
        closest_reachable = df_filtered.iloc[0]

# ================================================
# UI - Prominent Contact Card + Call Now
# ================================================
st.title("🏥 Kenya Hospital Locator")

if emergency_mode:
    st.error("🚨 EMERGENCY MODE ACTIVE: Showing only Level 4+ facilities")

if closest_reachable is not None:
    phone = closest_reachable.get('PhoneNo')
    name = closest_reachable.get('FacilityName', 'Unknown Facility')
    dist = closest_reachable.get('Distance_km', 0)
    time = closest_reachable.get('Travel_time_minutes', 0)

    phone_display = str(phone) if phone and str(phone).strip() not in ['None', 'nan'] else "No phone number available"

    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e40af, #3b82f6); 
                    padding: 28px; border-radius: 16px; text-align: center; 
                    margin: 20px 0; box-shadow: 0 10px 20px rgba(0,0,0,0.4);">
            <h2 style="color: #bae6fd; margin: 0 0 12px 0;">📞 Closest Reachable Hospital</h2>
            <h3 style="color: white; margin: 12px 0 18px 0;">{name}</h3>
            <p style="font-size: 38px; font-weight: bold; color: white; margin: 15px 0; letter-spacing: 2px;">
                {phone_display}
            </p>
            <p style="color: #bae6fd; font-size: 17px;">
                Distance: {dist:.1f} km • Travel Time: {time:.1f} min
            </p>
    """, unsafe_allow_html=True)

    # Call Now Button
    if phone and str(phone).strip() not in ['None', 'nan']:
        st.markdown(f"""
            <a href="tel:{phone}" target="_blank" style="text-decoration:none;">
                <button style="background-color: #22c55e; color: white; border: none; padding: 14px 32px; 
                font-size: 18px; font-weight: bold; border-radius: 12px; cursor: pointer; width: 100%;">
                    📲 Call Now: {phone}
                </button>
            </a>
        """, unsafe_allow_html=True)

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Hospitals Found", f"{len(df_filtered):,}")
if len(df_filtered) > 0:
    col2.metric("Closest Distance", f"{df_filtered['Distance_km'].min():.2f} km")
    col3.metric("Fastest Time", f"{df_filtered['Travel_time_minutes'].min():.1f} min")

# Table
st.subheader("Nearest Hospitals")
display_cols = ["FacilityName", "County", "Level", "Level_Category", "PhoneNo", "Distance_km", "Travel_time_minutes"]
existing_cols = [col for col in display_cols if col in df_filtered.columns]
st.dataframe(df_filtered[existing_cols].reset_index(drop=True), use_container_width=True)

# Map
st.subheader("Map")
m = folium.Map(location=[lat, lon], zoom_start=11)
folium.Marker([lat, lon], popup="Your Location", icon=folium.Icon(color="green")).add_to(m)
folium.Circle(radius=radius_km * 1000, location=[lat, lon], color="blue", fill=False).add_to(m)

marker_cluster = MarkerCluster().add_to(m)
color_map = {
    "Below Level 3 (Basic services)": "green",
    "Level 3 (50% services)": "orange",
    "Level 4+ (All services)": "red",
    "Unknown": "gray"
}

for _, row in df_filtered.iterrows():
    cat = row.get("Level_Category", "Unknown")
    icon_color = "darkpurple" if closest_reachable is not None and row.name == closest_reachable.name else color_map.get(cat, "gray")
    popup = f"<b>{row.get('FacilityName')}</b><br>Phone: {row.get('PhoneNo', 'N/A')}<br>Distance: {row.get('Distance_km'):.1f} km"
    folium.Marker([row['Geo_Location.Latitude'], row['Geo_Location.Longitude']], 
                  popup=popup, icon=folium.Icon(color=icon_color)).add_to(marker_cluster)

st_folium(m, width=1000, height=600)

st.caption("Data source: health_facility_data_feb11.json")