# Customer_Segmentation_application
This app has been built using Streamlit and deployed with Streamlit community cloud

[Visit the app here](https://real-estate-price-predictor-app.streamlit.app/)

This application predicts the fair transaction price of a property before it's sold within a small county in New York state based on a dataset for transaction prices for previously sold properties on the market. The model aims to predict transaction prices with an average error of under $70,000.

## Features
- User-friendly interface powered by Streamlit.
- Input form to enter details such as property_tax, insurance, beds, baths, Bunglow/Condo, and other relevant factors.
- Real-time prediction of property price based on the trained model. Mean Absolute Error (MAE) < $70,000
- Accessible via Streamlit Community Cloud.

## Dataset
The application is trained on the **Mall Customer dataset**, a dataset of customers' characteristics and their consumption behavior. It includes features like:
- Gender: Gender of the customer
- Age: Age of the customer
- Income: Annual Income of the customers in 1000 dollars
- Spending_Score: Score assigned between 1-100 by the mall based on customer' spending behavior.


## Technologies Used
- **Streamlit**: For building the web application.
- **Scikit-learn**: For model training and evaluation.
- **Pandas** and **NumPy**: For data preprocessing and manipulation.
- **Matplotlib** and **Seaborn**: For exploratory data analysis and visualization (if applicable).

## Model
The predictive model is trained using the Mall Customer data. The Unsupervised Clustering K-means model is used.


#### Thank you for using the Customer Segmentation Application! Feel free to share your feedback.
