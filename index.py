import streamlit as st
import time
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('F:\\Downloads\\biofuel_data.csv')

data = load_data()

# Biofuel Predictor
def biofuel_predictor():
    st.title("Biofuel Yield Predictor")
    st.write("Predict biofuel yield using environmental and crop factors.")

    # Input Form
    crop = st.selectbox("Select Crop Type", data['crop_type'].unique())
    residue = st.slider("Residue Availability (tons)", 0, 100, step=1)
    temp = st.slider("Temperature (°C)", 0, 50, step=1)
    humidity = st.slider("Humidity (%)", 0, 100, step=1)
    water = st.slider("Water Availability (liters)", 0, 1000, step=10)

    # Model Preparation
    X = data[['crop_type', 'residue_availability', 'temperature', 'humidity', 'water_availability']]
    y = data['biofuel_yield']
    X = pd.get_dummies(X, columns=['crop_type'], drop_first=True)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Prepare Input for Prediction
    input_data = pd.DataFrame(0, index=[0], columns=X.columns)
    input_data['residue_availability'] = residue
    input_data['temperature'] = temp
    input_data['humidity'] = humidity
    input_data['water_availability'] = water

    # Set the crop type
    crop_column = f'crop_type_{crop}'
    if crop_column in input_data.columns:
        input_data[crop_column] = 1

    # Prediction
    predicted_yield = model.predict(input_data)
    st.write(f"Predicted Biofuel Yield: {predicted_yield[0]:.2f} liters")

# Real-Time Sensor Simulation
def sensor_simulation():
    st.title("Sensor Simulation")
    st.write("Simulated sensor readings in real time.")

    if st.button("Start Simulation"):
        with st.empty():
            while True:
                temp = random.uniform(20, 100)
                pressure = random.uniform(0, 10)
                flow_rate = random.uniform(0, 50)

                st.write(f"Temperature: {temp:.2f} °C, Pressure: {pressure:.2f} bar, Flow Rate: {flow_rate:.2f} L/min")
                time.sleep(1)

# Hydrogen Dispensing Simulation
def hydrogen_dispensing():
    st.title("Hydrogen Dispensing Simulation")
    st.write("Simulate hydrogen dispensing progress.")

    max_pressure = st.slider("Set Max Pressure (bar)", 100, 700, step=50)
    cooling_temp = st.slider("Set Cooling Temperature (°C)", -50, 0, step=5)
    target_hydrogen = st.slider("Target Hydrogen (kg)", 1, 10, step=1)

    if st.button("Start Dispensing"):
        current_pressure = 0
        dispensed_hydrogen = 0

        while dispensed_hydrogen < target_hydrogen:
            if current_pressure < max_pressure:
                current_pressure += random.uniform(5, 10)
                current_pressure = min(current_pressure, max_pressure)

            dispensed_hydrogen += 1
            st.write(f"Dispensed: {dispensed_hydrogen:.2f} Kg, Pressure: {current_pressure:.2f} bar")
            time.sleep(1)

# App Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Biofuel Predictor", "Sensor Simulation", "Hydrogen Dispensing"])

if page == "Biofuel Predictor":
    biofuel_predictor()
elif page == "Sensor Simulation":
    sensor_simulation()
elif page == "Hydrogen Dispensing":
    hydrogen_dispensing()
