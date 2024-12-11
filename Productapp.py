import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('final_xgb_model.pkl', 'rb'))

# Streamlit App
st.title("Product Sales Forecasting")

# Combined Store Type
store_type = st.radio("Store Type:", ["S2", "S3", "S4"])
store_type_mapping = {
    "S2": [1, 0, 0],  # S2 is active
    "S3": [0, 1, 0],  # S3 is active
    "S4": [0, 0, 1]   # S4 is active
}
store_type_inputs = store_type_mapping[store_type]

# Combined Location Type
location_type = st.radio("Location Type:", ["L2", "L3", "L4","L5"])
location_type_mapping = {
    "L2": [1, 0, 0,0],  # S2 is active
    "L3": [0, 1, 0,0],  # S3 is active
    "L4": [0, 0, 1,0],   # S4 is active
    "L5": [0, 0, 0,1]   # S5 is active
}
location_type_inputs = location_type_mapping[location_type]

# Combined Region Code
Region_code = st.radio("Region Code:", ["R2", "R3", "R4"])
Region_code_mapping = {
    "R2": [1, 0, 0],  # S2 is active
    "R3": [0, 1, 0],  # S3 is active
    "R4": [0, 0, 1],   # S4 is active
}
Region_code_inputs = Region_code_mapping[Region_code]

# Other binary features with "Yes/No" options
binary_features = {
    "Holiday": ["Yes", "No"],
    "Discount": ["Yes", "No"],
}

# Numeric features
numeric_features = [
    ("# Orders", "Number_of_orders"),
    ("Year", "Year"),
    ("Month", "Month"),
    ("Day of Week", "Day_of_Week"),
    ("Days Since Start", "Days_since_start")
]

# Collect binary inputs
binary_inputs = []
for feature, options in binary_features.items():
    # Use "Yes" and "No", then map to 1 and 0
    choice = st.radio(feature, options)
    binary_inputs.append(1 if choice == "Yes" else 0)

# Collect numeric inputs
numeric_inputs = []
for label, key in numeric_features:
    numeric_inputs.append(st.number_input(label, key=key))

# Button to make predictions
if st.button("Predict Sales"):
    # Combine all inputs
    features = np.array(store_type_inputs + location_type_inputs + Region_code_inputs + binary_inputs + numeric_inputs).reshape(1, -1)
    prediction = model.predict(features)[0]
    st.success(f"Predicted Sales: {prediction}")
