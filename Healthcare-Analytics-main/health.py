import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load dataset to get feature names
dataset_name = 'train.csv'  # Change this to your dataset name
df = pd.read_csv(dataset_name)

# Identify categorical and numerical features
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
numeric_features = df.select_dtypes(exclude=['object']).columns.tolist()

# Remove target variable from features
target_column = 'Stay'
if target_column in categorical_features:
    categorical_features.remove(target_column)
if target_column in numeric_features:
    numeric_features.remove(target_column)

# Create mappings for categorical features
category_mappings = {}
for feature in categorical_features:
    unique_values = df[feature].dropna().unique().tolist()
    category_mappings[feature] = {val: idx for idx, val in enumerate(unique_values)}

# Create mapping for target variable encoding
target_mapping = {val: idx for idx, val in enumerate(df[target_column].dropna().unique())}
inverse_target_mapping = {v: k for k, v in target_mapping.items()}

# Streamlit UI
st.title("Machine Learning Prediction App")
st.write("Enter values manually to get predictions.")

# Define input features
def user_input_features():
    user_data = {}

    for feature in numeric_features:
        user_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

    for feature in categorical_features:
        unique_values = list(category_mappings[feature].keys())
        selected_value = st.selectbox(f"Select {feature}", unique_values)
        user_data[feature] = category_mappings[feature][selected_value]  # Encode categorical values

    return pd.DataFrame([user_data])

# Manual input interface
input_df = user_input_features()

# Ensure feature order matches model training
input_df = input_df[numeric_features + categorical_features]

if st.button("Predict"):
    # Check if input DataFrame has the same columns as the model expects
    if set(input_df.columns) != set(df.drop(columns=[target_column]).columns):
        st.error("Feature mismatch! Ensure correct input.")
    else:
        prediction = model.predict(input_df)
        predicted_label = inverse_target_mapping.get(prediction[0], "Unknown")  # Decode output
        st.write("### Prediction:", predicted_label)
