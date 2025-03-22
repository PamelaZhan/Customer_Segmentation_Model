import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st

# Set the page title and description
st.title("Customer Segmentation Predictor")
st.write("""
This app predicts a Customer Segmentation 
based on various customer characteristics.
""")

# Load the pre-trained model
with open("models/Kmodel.pkl", "rb") as pkl:
    k_model = pickle.load(pkl)


# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.subheader("Custermer Details")
    
    # Gender input
    Gender = st.selectbox("Gender", options=["Male", "Female"])

    # Age
    Age = st.slider("Age", min_value=18, max_value=95)

    # Customer Income
    Income = st.slider("Annual Income (in 1000 dollars)", min_value=0, max_value=200)
    
    # Spending Score
    Spending_Score = st.slider("Spending Score", min_value=0, max_value=100)
    
    # Submit button
    submitted = st.form_submit_button("Predict Customer Cluster")

# Handle the dummy variables to pass to the model
if submitted:
    # convert to integers
    age = int(Age)
    income = int(Income)
    spending_score = int(Spending_Score)

    # deal dummy feature
    Gender_Male = 1 if Gender == "Male" else 0
    Gender_Female = 1 if Gender == "Female" else 0
    


    # Prepare the input for prediction. This has to go in the same order as it was trained
    prediction_input = pd.DataFrame([[age, income, spending_score, Gender_Female, Gender_Male]])

    # Make prediction
    new_prediction = k_model.predict(prediction_input)

    # Display result
    st.subheader("Prediction Result:")
    st.write(f"The customer cluster number: {new_prediction[0]}")
    

st.write(
    """We used a machine learning (k-mean clustering) model to group customers (6 clusters). The elbow plot is shown as followed."""
)
st.image("clusters_on_two_features.png")
st.image("elbow.png")
