
from src.data.load_dataset import load_and_preprocess_data
from src.features.build_features import create_dummy
from src.visualization.visualize import plot_clusters, plot_elbow_silhouette
from src.models.train_model import train_kmodel


if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/mall_customers.csv"
    df = load_and_preprocess_data(data_path)

    # Create dummy variables and separate features and target
    df = create_dummy(df)

    # Train the kmean using all features
    model = train_kmodel(df)

    # print cluster centers
    print(model.cluster_centers_.astype(int))
    # plot clusters trained by only two features: spending_score and annual_income
    plot_clusters(model, df)
    # plot elbow and silhouette
    plot_elbow_silhouette(df)

 
