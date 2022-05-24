from sklearn.metrics import silhouette_score, silhouette_samples

def calculate_silhouette_score(data, 
                               labels, 
                               metric="euclidean", 
                               sample_size=None, 
                               random_state=None):

    return silhouette_score(data, 
                            labels, 
                            metric=metric, 
                            sample_size=sample_size,
                            random_state=random_state)


def calculate_silhouette_samples(data, labels):
    
    return silhouette_samples(data, labels)