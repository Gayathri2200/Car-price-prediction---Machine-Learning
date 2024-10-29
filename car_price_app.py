import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = pk.load(open('xgboost_car_price_model.pkl', 'rb'))

st.header('Car Price Prediction ML Model')

# Load and display data
cars_data = pd.read_csv("final_data.csv")

# UI inputs
name = st.selectbox('Select the Car Brand', cars_data['Name'].unique())
year = st.slider('Car Manufacture year', 1885, 2024)
km_driven = st.slider('No of Kilometers driven', 11, 200000)
fuel = st.selectbox('Select the Fuel type', cars_data['fuel_type'].unique())
transmission = st.selectbox('Select the Transmission type', cars_data['Transmission'].unique())
owner = st.selectbox('Select the Ownership', cars_data['Ownership'].unique())
mileage = st.slider('Car Mileage', 10, 40)
engine = st.slider('Car Engine capacity', 700, 5000)
power = st.slider('Car Max power', 0, 200)
seats = st.slider('No of seats', 5, 15)

# Encode input data the same way as the training data
if st.button('Predict'):
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, transmission, owner, mileage, engine, power, seats]],
        columns=['Name', 'year', 'km_driven', 'fuel_type', 'Transmission', 'Ownership', 'Mileage', 'Engine', 'Max Power', 'Seats']
    )

    # Rename columns to match model's expected feature names
    input_data_model = input_data_model.rename(columns={'km_driven': 'kilometers_driven', 'year': 'Year'})

    
    order = ['Name', 'fuel_type', 'Transmission', 'Ownership', 'Mileage', 'Engine', 'Max Power', 'Seats', 'kilometers_driven', 'Year']
    input_data_model = input_data_model[order]

    # Encode input data according to training data encoding
    for column in ['Name', 'fuel_type', 'Ownership', 'Transmission']:
        le = LabelEncoder()
        cars_data[column] = le.fit_transform(cars_data[column])
        input_data_model[column] = le.transform(input_data_model[column])

    # Show the transformed input data
    st.write(input_data_model)

    # Predict car price
    car_price = model.predict(input_data_model)
    st.markdown("Predicted Car Price: ₹" + str(round(car_price[0], 2)))





