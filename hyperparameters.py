datasets = [
    "Scaled_Dataset.csv", 
    "Scaled_PCA_n3.csv", 
    "Scaled_PCA_Variance_85.csv", 
    "Scaled_PCA_Variance_95.csv", 
    "Scaled_PCA_Variance_99.csv"
]

kmeans_hyp = {
    "clusters": [3, 4, 5, 6, 7, 8, 9, 10],
    "initialization_methods": ["kmeans++", "random"]
}

dbscan_hyp = {
    "min_samples": [2, 4, 8, 20, 24],
    "eps": [0.6, 0.726, 0.8, 0.9, 1.0]
}