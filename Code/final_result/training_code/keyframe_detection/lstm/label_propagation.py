import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def propagate_labels_by_similarity(features, labels, similarity_threshold=0.95):
    T = len(features)
    propagated = np.array(labels)
    key_indices = [i for i, v in enumerate(labels) if v >= 0.5]
    if not key_indices:
        return propagated

    sim_matrix = cosine_similarity(features)
    for i in range(T):
        if any(sim_matrix[i, j] > similarity_threshold for j in key_indices):
            propagated[i] = max(propagated[i], 0.5)
    return propagated