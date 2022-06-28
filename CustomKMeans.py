import numpy as np

class CustomKMeans:

    def __init__(
        self, n_centers=3, 
        init="random", 
        distance_metric="l2", 
        threshold=1e-4, 
        max_itr=1
    ):
        self.n_centers = n_centers
        self.threshold = threshold
        self.data = None
        self.centers = None
        self.prev_centers = None
        self.data_cluster = None
        self.distances = None
        self.distance_metric = distance_metric
        self.init = init
        self.n_dim = None
        self.max_itr = max_itr

    def fit(self, data):
        """ Initialize data for training

        Parameters:
            data: input features of shape (100, 2)
        """
        self.data = data
        self.n_dim = self.data.shape[-1]

        if self.init == "kmeans++":
            self.centers = self._generate_kmeans_plus_centers()
        elif self.init == "uniform":
            self.centers = self._generate_uniform_centers()
        else:
            self.centers = self._generate_random_sampled_centers()

    def _generate_kmeans_plus_centers(self):
        """KMeans++ initialization"""

        centers = []
        X = self.data

        initial_index = np.random.choice(range(X.shape[0]),)
        centers.append(np.asarray(X[initial_index, :].tolist()))

        for i in range(self.n_centers - 1):
            distance = np.sum((np.array(centers) - X[:, None, :])**2, axis=2)

            if i == 0:
                pdf = distance / np.sum(distance)
                centroid_new = X[
                    np.random.choice(range(X.shape[0]), replace=False, p=pdf.flatten())
                ]
            else:
                dist_min = np.min(distance, axis=1)

                pdf = dist_min / np.sum(dist_min)

                centroid_new = X[np.random.choice(range(X.shape[0]), replace=False, p=pdf)]

            centers.append(centroid_new.tolist())
        
        return np.array(centers)

    def _generate_uniform_centers(self):
        """Generate uniform centers"""

        centers = None
        return centers

    def _generate_random_sampled_centers(self):
        """Generate random sampled centers"""

        rand_index = np.random.randint(
            low=0, high=self.data.shape[0], size=self.n_centers
        )

        centers = self.data[rand_index]
        return centers

    def _l2(self, point1, point2):
        """Euclidean Distance"""

        dist = np.sqrt(np.sum(np.square(point1 - np.array(point2)), axis=1))
        return dist

    def _l1(self, point1, point2):
        """Absolute Distance"""

        dist = np.sum(np.absolute(point1 - point2))
        return dist

    def _calculate_distance(self):

        dist = []
        for center in self.centers:
            if self.distance_metric == "l1":
                abd = self._l1(self.data, center)
                dist.append(abd)
            else:
                sqrt = self._l2(self.data, center)
                dist.append(sqrt)

        self.distances = np.array(dist)
        return self.distances

    def _assign_clusters(self):
        
        self.data_cluster = np.argmin(self.distances, axis=0)
        return self.data_cluster

    def _update_centers(self):

        self.prev_centers = np.copy(self.centers)

        for i in range(self.n_centers):
            self.centers[i] = np.mean(
                self.data[self.data_cluster == i], axis=0
            )
        return self.centers

    def converge(self):

        current_itr = 1
        while current_itr < self.max_itr:
            self._calculate_distance()
            self._assign_clusters()
            self._update_centers()

            if self.is_optimal():
                break

            current_itr += 1

        if self.is_optimal():
            print("Optimial point reached after {} iterations".format(current_itr))
        else:
            print("Reached maximum iteration of {}".format(self.max_itr))

    def is_optimal(self):

        non_optimal = np.abs(self.prev_centers - self.centers) > self.threshold
        if non_optimal.astype(int).sum() == 0:
            return True
        return False