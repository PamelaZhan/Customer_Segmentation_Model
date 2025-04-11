import pandas as pd
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
    
    # create 2 columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender input
        Gender = st.selectbox("Gender", options=["Male", "Female"])
        # Age
        Age = st.slider("Age", min_value=18, max_value=95, value=38)

    with col2:
        # Customer Income
        Income = st.slider("Annual Income (in 1000 dollars)", min_value=0, max_value=200, value=60)    
        # Spending Score
        Spending_Score = st.slider("Spending Score", min_value=0, max_value=100, value=50)
    
    # Submit button
    submitted = st.form_submit_button("Predict Customer Cluster")

# meaningful names for clusters
group_names = ["Young High-Earner Big-Spender", "Middle-Aged High-Earner Frugal", "Young Low-Earner Big-Spender", 
                   "Senior Moderate-Earner Balanced-Spender","Young Moderate-Earner Balanced-Spender", "Middle-Aged Low-Earner Frugal"]

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
    st.write(f"The customer should be assigned to group: {group_names[new_prediction[0]]}")

# centroids of the Kmeans model    
centers = pd.DataFrame(k_model.cluster_centers_.astype(int), 
                       columns=['Age',	'Annual_Income',	'Spending_Score',	'Gender_Female',	'Gender_Male'])
centers.insert(loc=0, column='Group Name', value=group_names)
# display the centroids
st.write("A K-mean clustering model is used to group customers into 6 clusters.")
st.image("clusters.png")
st.write("Cluster centers (centroids):")
st.table(centers)
st.write("The elbow plot is shown as followed.")
st.image("elbow.png")
st.write("The Silhouette plot is shown as followed.")
st.image("silhouette.png")