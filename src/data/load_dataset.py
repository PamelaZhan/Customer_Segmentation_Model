
import pandas as pd
from ..logging.logging import logging_decorator

@logging_decorator
def load_and_preprocess_data(data_path):
    
    # Import the data from 'mall_customers.csv'
    df = pd.read_csv(data_path)

    # drop Customer_ID
    df = df.drop('Customer_ID', axis = 1)          

    return df
