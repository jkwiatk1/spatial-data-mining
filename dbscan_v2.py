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
