import numpy as np
import matplotlib.pyplot as plt


def dbscan(X, epsilon, min_samples):
    labels = np.zeros(X.shape[0], dtype=int)
    cluster_id = 0

    for i in range(X.shape[0]):
        if labels[i] != 0:
            continue

        neighbors = region_query(X, i, epsilon)
        if len(neighbors) < min_samples:
            labels[i] = -1  # (noise)
        else:
            cluster_id += 1
            expand_cluster(X, labels, i, neighbors, cluster_id, epsilon, min_samples)

    return labels


def region_query(X, point_idx, epsilon):
    neighbors = []
    for i in range(X.shape[0]):
        if point_idx != i and np.linalg.norm(X[point_idx] - X[i]) <= epsilon:  # Count euclidean distance
            neighbors.append(i)
    return neighbors


def expand_cluster(X, labels, point_idx, neighbors, cluster_id, epsilon, min_samples):
    labels[point_idx] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]
        if labels[neighbor_idx] == 0:
            labels[neighbor_idx] = cluster_id
            new_neighbors = region_query(X, neighbor_idx, epsilon)
            if len(new_neighbors) >= min_samples:
                neighbors.extend(new_neighbors)
        elif labels[neighbor_idx] == -1:
            labels[neighbor_idx] = cluster_id
        i += 1


np.random.seed(42)
X = np.random.rand(100, 2)
epsilon = 0.08
min_samples = 5

labels = dbscan(X, epsilon, min_samples)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c='gray', label='Dane przed DBSCAN')
plt.title('Przed DBSCAN')

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', label='Klastry po DBSCAN')
plt.title('Po DBSCAN')

from sklearn.cluster import DBSCAN

dbscan_sk = DBSCAN(eps=epsilon, min_samples=min_samples)
labels_sk = dbscan_sk.fit_predict(X)

plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels_sk, cmap='viridis', label='Klastry po DBSCAN scikit')
plt.title('Po DBSCAN scikit')

plt.show()
