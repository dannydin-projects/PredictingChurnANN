import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle

## Load model
model = load_model('ann_model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('le_gender.pkl', 'rb') as f:
    le_gender = pickle.load(f)

with open('ohe_geo.pkl', 'rb') as f:
    ohe_geo = pickle.load(f)

# Streamlit app
st.title("Customer Churn Prediction")  

st.header("User Input")

# User inputs with unique keys
geography = st.selectbox('Geography', ohe_geo.categories_[0], key='geography')
gender = st.selectbox('Gender', le_gender.classes_, key='gender')
age = st.slider('Age', 18, 92, key='age')
balance = st.number_input('Balance', key='balance')
credit_score = st.number_input('Credit Score', key='credit_score')
estimated_salary = st.number_input('Estimated Salary', key='estimated_salary')
tenure = st.slider('Tenure', 0, 10, key='tenure')
num_of_products = st.slider('Number of Products', 1, 4, key='num_products')
has_cr_card = st.selectbox('Has Credit Card', [0, 1], key='has_crcard')
is_active_member = st.selectbox('Is Active Member', [0, 1], key='is_active')

# Predict button
if st.button("Predict Churn"):
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [le_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography'
    geo_encoded = ohe_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict churn
    prediction = model.predict(input_data_scaled)
    churn_prob = prediction[0][0]

    st.divider()
    st.subheader("Prediction Result")
    
    if churn_prob > 0.5:
        st.error(f'⚠️ The customer is likely to churn with a probability of {churn_prob:.2f}')
    else:
        st.success(f'✅ The customer is unlikely to churn with a probability of {churn_prob:.2f}')