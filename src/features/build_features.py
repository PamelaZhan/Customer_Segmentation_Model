import pandas as pd
from ..logging.logging import logging_decorator

@logging_decorator
# create dummy features
def create_dummy(df):

    # Create dummy variables for 'Gender'
    df = pd.get_dummies(df, columns=['Gender'], dtype=int)

    # store the processed dataset in data/processed
    df.to_csv('data/processed/Processed_customers.csv', index=None)

    return df