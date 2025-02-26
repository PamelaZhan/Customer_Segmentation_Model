
# import kmeans model
from sklearn.cluster import KMeans
import pickle
from ..logging.logging import logging_decorator


@logging_decorator
# Function to train the model
def train_kmodel(df):
    
    # train our model on spending_score and annual_income
    kmodel = KMeans(n_clusters=6).fit(df)

    # Get the centroids of the clusters
    centroids = kmodel.cluster_centers_

    # Get cluster labels  
    y_pred = kmodel.labels_
       
    # Save the trained model
    with open('models/Kmodel.pkl', 'wb') as f:
        pickle.dump(kmodel, f)

    return kmodel, y_pred

