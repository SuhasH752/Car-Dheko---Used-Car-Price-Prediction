import joblib
import pandas as pd
import streamlit as st
import os
import numpy as np

# Define the file paths
SAVE_DIR = r"C:\Users\Admin\guvi class"

MODEL_PATH = os.path.join(SAVE_DIR, "car_price_predictor.pkl")
ORDINAL_ENCODER_PATH = os.path.join(SAVE_DIR, "ordinal_encoder.pkl")
SCALER_PATH = os.path.join(SAVE_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(SAVE_DIR, "car_feature_names.pkl")
CATEGORICAL_FEATURES_PATH = os.path.join(SAVE_DIR, "categorical_features.pkl")
NUMERICAL_FEATURES_PATH = os.path.join(SAVE_DIR, "numerical_features.pkl")

# Load model and preprocessing objects
try:
    model = joblib.load(MODEL_PATH)
    ordinal_encoder = joblib.load(ORDINAL_ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    categorical_features = joblib.load(CATEGORICAL_FEATURES_PATH)
    numerical_features = joblib.load(NUMERICAL_FEATURES_PATH)
    st.success("✅ All model files loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

def preprocess_input(input_df):
    """Preprocess input using OrdinalEncoder and StandardScaler"""
    processed_df = input_df.copy()
    
    # Transform categorical features using OrdinalEncoder
    processed_df[categorical_features] = ordinal_encoder.transform(processed_df[categorical_features])
    
    # Scale numerical features
    processed_df[numerical_features] = scaler.transform(processed_df[numerical_features])
    
    return processed_df

# Extract categories from OrdinalEncoder
st.write("🔍 Extracting categories from encoder...")

# Get unique categories for each feature
manufacturer_categories = ordinal_encoder.categories_[list(categorical_features).index('manufacturer')]
fueltype_categories = ordinal_encoder.categories_[list(categorical_features).index('fueltype')]
bodytype_categories = ordinal_encoder.categories_[list(categorical_features).index('bodytype')]
transmission_categories = ordinal_encoder.categories_[list(categorical_features).index('transmission')]
city_categories = ordinal_encoder.categories_[list(categorical_features).index('City')]
model_variant_categories = ordinal_encoder.categories_[list(categorical_features).index('model_variant')]

# Create a mapping of manufacturer to available models
# This assumes your training data had consistent manufacturer-model relationships
manufacturer_model_mapping = {}
for manufacturer in manufacturer_categories:
    # Filter model variants that contain the manufacturer name (common pattern)
    manufacturer_models = [model for model in model_variant_categories if manufacturer.lower() in model.lower()]
    if not manufacturer_models:
        # If no direct matches, show all models
        manufacturer_models = list(model_variant_categories)
    manufacturer_model_mapping[manufacturer] = manufacturer_models

# Streamlit App
st.title("🚗 Car Price Predictor")

with st.form("car_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        manufacturer = st.selectbox("Brand*", manufacturer_categories, key="manufacturer")
        fueltype = st.selectbox("Fuel Type*", fueltype_categories)
        bodytype = st.selectbox("Body Type*", bodytype_categories)
        city = st.selectbox("City*", city_categories)
        
    with col2:
        transmission = st.selectbox("Transmission*", transmission_categories)
        
        # Dynamic model variant based on selected brand
        if manufacturer:
            available_models = manufacturer_model_mapping.get(manufacturer, model_variant_categories)
            model_variant = st.selectbox("Model*", available_models)
        else:
            model_variant = st.selectbox("Model*", model_variant_categories)
            
        modelyear = st.number_input("Model Year*", min_value=1990, max_value=2024, value=2020)
        seats = st.selectbox("Seats*", [4, 5, 6, 7, 8])
        ownernumber = st.selectbox("Previous Owners", [1, 2, 3, 4])
    
    # Simplified specifications section
    st.subheader("📊 Basic Information")
    col3, col4 = st.columns(2)
    
    with col3:
        kilometers = st.number_input("Kilometers Driven*", min_value=0, max_value=500000, value=50000, step=1000)
        
    with col4:
        registrationyear = st.number_input("Registration Year*", min_value=1990, max_value=2024, value=2020)
    
    # Set default values for removed technical specifications
    enginedisplacement = 1500  # Default value
    maxpower = 100  # Default value
    torque = 150  # Default value
    mileage = 15.0  # Default value
    variantid = 1  # Default value
    
    submitted = st.form_submit_button("🎯 Predict Car Price")

if submitted:
    # Create input data with user selections
    input_data = {
        'manufacturer': [manufacturer],
        'model_variant': [model_variant],
        'fueltype': [fueltype],
        'bodytype': [bodytype],
        'transmission': [transmission],
        'City': [city],
        'variantid': [variantid],
        'ownernumber': [ownernumber],
        'modelyear': [modelyear],
        'registrationyear': [registrationyear],
        'seats': [seats],
        'kilometers': [kilometers],
        'enginedisplacement': [enginedisplacement],
        'maxpower': [maxpower],
        'torque': [torque],
        'mileage': [mileage]
    }
    
    # Create DataFrame and ensure correct feature order
    input_df = pd.DataFrame(input_data)
    
    # Check if all required features are present
    missing_features = [f for f in feature_names if f not in input_df.columns]
    if missing_features:
        st.error(f"Missing features: {missing_features}")
        st.stop()
    
    input_df = input_df[feature_names]
    
    # Preprocess and predict
    try:
        processed_input = preprocess_input(input_df)
        price = model.predict(processed_input)[0]
        
        # Display results
        st.success(f"### 💰 Estimated Price: ₹ {int(price):,}")
        
        # Show car details
        st.info(f"""
        **Car Details:**
        - **Brand:** {manufacturer}
        - **Model:** {model_variant}
        - **Fuel Type:** {fueltype}
        - **Body Type:** {bodytype}
        - **Transmission:** {transmission}
        - **City:** {city}
        - **Model Year:** {modelyear}
        - **Registration Year:** {registrationyear}
        - **Seats:** {seats}
        - **Previous Owners:** {ownernumber}
        - **Kilometers Driven:** {kilometers:,} km
        """)
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

