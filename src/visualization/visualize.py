
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
# import silhouette_score
from sklearn.metrics import silhouette_score
from ..logging.logging import logging_decorator

@logging_decorator
def plot_cluster_2features(df):
    # train our model on spending_score and annual_income
    kmodel = KMeans(n_clusters=5).fit(df[['Annual_Income','Spending_Score']])

    # Get the centroids of the clusters
    #centroids = kmodel.cluster_centers_

    # Get cluster labels  
    y_pred = kmodel.labels_
    df['Cluster'] = y_pred

    # plot these clusters
    sns.scatterplot(x='Annual_Income', y = 'Spending_Score', data=df, hue='Cluster', palette='colorblind')
    plt.savefig('clusters_on_two_features', dpi=300)
    plt.show()

@logging_decorator
def plot_correlation_heatmap(data):
    """
    Plot a correlation heatmap for the given data.
    
    Args:
        data (pandas.DataFrame): The input data.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr()*100, annot=True, fmt='.0f', cmap='RdBu_r')
    plt.title('Correlation Heatmap', fontsize=16)
    # Save the plot to a file
    plt.savefig('heatmap.png', dpi=300)
    # Show the plot
    plt.show()

@logging_decorator
def plot_elbow_silhouette(df):
    # Analyze clusters from 3 to 8     
    k = range(3,9) # the number of clusters
    K = [] # list to store the number of clusters
    ss = [] # list to store silhouette scores
    WCSS = []
    for i in k:
        kmodel = KMeans(n_clusters=i).fit(df )
        ypred = kmodel.labels_
        sil_score = silhouette_score(df, ypred) #silhouette_score, close to 1 indicates that the data points are well clustered 
        ss.append(sil_score)
        # calculate the WCSS scores, used to plot the Elbow Plot
        # WCSS-- Within Cluster Sum of Squared distances
        wcss_score = kmodel.inertia_
        WCSS.append(wcss_score)
        K.append(i)

    # Store the number of clusters and their respective WSS scores and Silhouette scores in a dataframe
    wss = pd.DataFrame({'cluster': K, 'WSS_Score':WCSS, 'Silhouette_Score':ss})

    # plot a Elbow plot. the Elbow joint point is optimal K value
    wss.plot(x='cluster', y = 'WSS_Score')
    plt.xlabel('No. of clusters')
    plt.ylabel('WSS Score')
    plt.title('Elbow Plot')
    # Save the plot to a file
    plt.savefig('elbow.png', dpi=300)
    plt.show()

    # plot the Silhouette plot
    wss.plot(x='cluster', y='Silhouette_Score')
    plt.xlabel('No. of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Plot')
    plt.savefig('silhouette.png', dpi=300)
    plt.show()

@logging_decorator
def plot_clusters(model, df):
    # centroids of clusters
    centroids = model.cluster_centers_
    # get first 3 clumns
    df3 = df[['Age', 'Annual_Income', 'Spending_Score']]
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot the data points, color-coded by cluster label
    ax.scatter(df3.iloc[:, 0], df3.iloc[:, 1], df3.iloc[:, 2], c=model.labels_, cmap='Accent', s=40, alpha=0.8)

    # Plot the centroids as red "X" markers
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', marker='X', s=200, label='Centroids')

    # Set plot labels and title
    ax.set_title('KMeans Clustering')
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual_Income')
    ax.set_zlabel('Spending_Score')
    ax.legend()
    plt.tight_layout()
    plt.savefig('clusters.png', dpi=300)
    # Show the plot
    plt.show()