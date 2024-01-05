import numpy as np
import matplotlib.pyplot as plt


def region_query(X, point_idx, epsilon):
    """
    Find all points within distance 'epsilon' of point X[point_idx].
    """
    neighbors = []
    for i, point in enumerate(X):
        if point_idx != i and np.linalg.norm(X[point_idx] - point) <= epsilon:  # Count euclidean distance
            neighbors.append(i)
    return set(neighbors)


def expand_cluster(X, labels, point_idx, neighbors, cluster_id, epsilon, min_samples):
    """
    Expand the cluster to include dense reachable points.
    """
    labels[point_idx] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor_idx = list(neighbors)[i]
        if labels[neighbor_idx] == -1:
            labels[neighbor_idx] = cluster_id  # change noise to border point
        elif labels[neighbor_idx] == 0:
            labels[neighbor_idx] = cluster_id  # label new point
            new_neighbors = region_query(X, neighbor_idx, epsilon)
            if len(new_neighbors) >= min_samples:
                neighbors = neighbors.union(new_neighbors)
        i += 1


def dbscan(X, epsilon, min_samples):
    """
    DBSCAN: Density-Based Spatial Clustering of Applications with Noise
    """
    labels = np.zeros(X.shape[0], dtype=int)
    cluster_id = 0

    for i in range(X.shape[0]):
        if labels[i] != 0:
            continue

        neighbors = region_query(X, i, epsilon)
        if len(neighbors) < min_samples:
            labels[i] = -1  # Label as noise
        else:
            cluster_id += 1
            expand_cluster(X, labels, i, neighbors, cluster_id, epsilon, min_samples)

    return labels


def test():
    np.random.seed(43)
    X = np.random.rand(150, 2)
    epsilon = 0.08
    min_samples = 5

    labels = dbscan(X, epsilon, min_samples)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c='gray', label='Dane przed DBSCAN')
    plt.title('Przed DBSCAN')
    plt.ylim(0, 1)

    plt.subplot(1, 3, 2)
    mask = labels != -1
    plt.scatter(X[mask, 0], X[mask, 1], c=labels[mask], cmap='viridis', label='Klastry po DBSCAN')
    plt.title('Po DBSCAN')
    plt.ylim(0, 1)

    from sklearn.cluster import DBSCAN

    dbscan_sk = DBSCAN(eps=epsilon, min_samples=min_samples)
    labels_sk = dbscan_sk.fit_predict(X)

    plt.subplot(1, 3, 3)
    mask_sk = labels_sk != -1
    plt.scatter(X[mask_sk, 0], X[mask_sk, 1], c=labels_sk[mask_sk], cmap='viridis', label='Klastry po DBSCAN scikit')
    plt.title('Po DBSCAN scikit')
    plt.ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    test()
