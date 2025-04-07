
# import kmeans model
from sklearn.cluster import KMeans
import pickle
from ..logging.logging import logging_decorator


@logging_decorator
# Function to train the model
def train_kmodel(df):
    
    # train the model 
    kmodel = KMeans(n_clusters=6).fit(df)
       
    # Save the trained model
    with open('models/Kmodel.pkl', 'wb') as f:
        pickle.dump(kmodel, f)

    return kmodel

