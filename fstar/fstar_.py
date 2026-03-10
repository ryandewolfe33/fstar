import scipy.sparse as sp
import numpy as np

import importlib.util


def _void_decorator(func):
    """A dummy decorator that does nothing."""
    return func


def conditional_njit(*args, **kwargs):
    """
    A decorator that applies @njit if Numba is installed, 
    otherwise acts as a pass-through decorator.
    """
    # Check if the 'numba' module is available
    if importlib.util.find_spec("numba") is not None:
        try:
            from numba import njit
            # Return the actual njit decorator, potentially with arguments
            if args and callable(args[0]):
                # Used as @conditional_njit on a function
                return njit(args[0])
            else:
                # Used as @conditional_njit() with arguments or without a function initially
                return njit(*args, **kwargs)
        except ImportError:
            # Fallback if numba is installed but njit is not importable for some reason
            pass
    
    # If numba is not installed or import failed, return the dummy decorator
    if args and callable(args[0]):
        # Used as @conditional_njit on a function
        return args[0]
    else:
        # Used as @conditional_njit() with arguments
        return _void_decorator


@conditional_njit
def _clustering_array_to_sparse(clustering):
    n_clusters = np.max(clustering) + 1
    n_nonoutliers = np.sum(clustering >= 0)
    indptr = np.empty(n_clusters + 1, dtype="int32")
    indices = np.empty(n_nonoutliers, dtype="int32")
    data = np.ones(n_nonoutliers, dtype="bool")
    clusters_argsort = np.argsort(clustering)
    next_index = 0
    for c in range(n_clusters):
        indptr[c] = next_index
        cluster_objects = np.arange(len(clustering))[clustering == c]
        n_cluster_objects = len(cluster_objects)
        indices[next_index : next_index + n_cluster_objects] = cluster_objects
        next_index += n_cluster_objects
    indptr[-1] = next_index  # update last indptr entry
    return indptr, indices, data


def clustering_array_to_sparse(clustering):
    """Convert a numpy clsutering array to a (clusters x objects) csr array
    Clusters are assumed to be contiguous 0-max(clustering), and -1 denotes objects with no clusters
    """
    indptr, indices, data = _clustering_array_to_sparse(clustering)
    clustering = sp.csr_matrix(
        (data, indices, indptr),
        shape=(len(indptr) - 1, len(clustering)),
        dtype="bool",
    )
    clustering.data[:] = True  # Sometime some entries are flipped to false, don't know why.
    return clustering


def node_clustering_to_edge_clustering(adjacency, node_clustering):
    """
    Project a node clustering onto the edges. An edge is in a cluster if both endpoints are in that cluster.
    Edge ids are computed by the data_index of a csr representation of the upper triangular adjacency matrix.
    """
    if sp.issparse(adjacency):
        adjacency = sp.triu(adjacency).tocsr()
    else:
        raise ValueError("adjacency must be a scipy.sparse array.")
    if sp.issparse(node_clustering):
        node_clustering = node_clustering.tocsr()
    elif isinstance(node_clustering, np.ndarray):
        node_clustering = clustering_array_to_sparse(node_clustering)
    else:
        raise ValueError("node_clustering must be a scipy.sparse array or a 1d numpy array.")
    
    # Need fast access to clusters per node
    node_clustering = node_clustering.transpose().tocsr()
    edge_clustering = sp.lil_array((len(adjacency.indices), node_clustering.shape[1]), dtype="bool")
    edge_id = 0
    for n1 in range(len(adjacency.indptr)-1):
        for n2 in adjacency.indices[adjacency.indptr[n1]:adjacency.indptr[n1+1]]:
            common_clusters = np.intersect1d(
                node_clustering.indices[node_clustering.indptr[n1]:node_clustering.indptr[n1+1]],
                node_clustering.indices[node_clustering.indptr[n2]:node_clustering.indptr[n2+1]]
                )
            edge_clustering[edge_id, common_clusters] = True
            edge_id += 1
    return edge_clustering.tocsc().transpose()


def edge_clustering_to_node_clustering(adjacency, edge_clustering):
    """
    Project an edge clustering onto the nodes. An node is in a cluster if it has at least one incident edge in that cluster.
    Edge ids are assumed to be the data_index of a csr representation of the upper triangular adjacency matrix.
    """
    if sp.issparse(adjacency):
        adjacency = sp.triu(adjacency).tocsr()
    else:
        raise ValueError("adjacency must be a scipy.sparse array.")
    if sp.issparse(edge_clustering):
        edge_clustering = edge_clustering.tocsc()
    elif isinstance(edge_clustering, np.ndarray):
        edge_clustering = clustering_array_to_sparse(edge_clustering)
    else:
        raise ValueError("node_clustering must be a scipy.sparse array or a 1d numpy array.")
    
    edge_clustering = edge_clustering.transpose().tocsr()
    node_clustering = sp.dok_array((edge_clustering.shape[1], adjacency.shape[0]), dtype="bool")
    edge_id = 0
    for n1 in range(len(adjacency.indptr)-1):
        for n2 in adjacency.indices[adjacency.indptr[n1]:adjacency.indptr[n1+1]]:
            edge_clusters = edge_clustering.indices[edge_clustering.indptr[edge_id]:edge_clustering.indptr[edge_id+1]]
            for cluster in edge_clusters:
                node_clustering[cluster, n1] = True
                node_clustering[cluster, n2] = True
            edge_id += 1
    return node_clustering.tocsr()


def fstar(c1, c2, outliers=True, drop_outliers=False, alpha=0.5):
    """
    Compute F*_wo between c1 and c2.

    Parameters
    ----------
    c1:scipy.sparse.array or np.ndarray - the first clustering
    c2:scipy.sparse_array or np.ndarray - the second clustering
    outliers:bool (defualt=True) - Flag to add outlier comparison term. If True compute F*_wo, if False compute F*_w
    drop_outliers:[False, 'c1', 'c2', 'either', 'both'] (default=False) - Flag to drop objects that consider outliers
        before computing score. Can be helpful for determining the quality of the clusters
        when extra outliers are not a concern.
    alpha:float (default=0.5) - A value between 0 and 1 to control the importance of matching in each direction.
        The default 0.5 is a symmetric measure, and 0/1 only looks at the best match for clustering in c1/c2.
    """
    # Massage type to a csr_array
    if sp.issparse(c1):
        c1 = c1.tocsr()
    elif isinstance(c1, np.ndarray):
        c1 = clustering_array_to_sparse(c1)
    else:
        raise ValueError("Fstar expects a scipy.sparse array or a 1d numpy array.")
    if sp.issparse(c2):
        c2 = c2.tocsr()
    elif isinstance(c2, np.ndarray):
        c2 = clustering_array_to_sparse(c2)
    else:
        raise ValueError("Fstar expects a scipy.sparse array or a 1d numpy array.")

    #Finding and dropping outliers is easier in csc
    c1 = c1.tocsc()
    c2 = c2.tocsc()

    if drop_outliers:
        c1_outliers = (c1.indptr[1:] - c1.indptr[:-1]) == 0
        c2_outliers = (c2.indptr[1:] - c2.indptr[:-1]) == 0
        if drop_outliers == "c1":
            drop = c1_outliers
        elif drop_outliers == "c2":
            drop = c2_outliers
        elif drop_outliers == "either":
            drop = np.bitwise_or(c1_outliers, c2_outliers)
        elif drop_outliers == "both":
            drop = np.bitwise_and(c1_outliers, c2_outliers)
        else:
            raise ValueError("drop_outliers must be one of ['c1', 'c2', 'either', 'both'].")

        c1 = c1[:, ~drop]
        c2 = c2[:, ~drop]

    # Neither have clusters
    if c1.shape[0] == 0 and c2.shape[0] == 0:
        return 1

    c1_outliers = (c1.indptr[1:] - c1.indptr[:-1]) == 0
    c2_outliers = (c2.indptr[1:] - c2.indptr[:-1]) == 0
    outlier_intersect = np.sum(c1_outliers * c2_outliers)
    outlier_fs = 0
    if outlier_intersect > 0:
        outlier_fs = outlier_intersect / (np.sum(c1_outliers) + np.sum(c2_outliers) - outlier_intersect)

    c1_outlier_prop = np.sum(c1_outliers) / len(c1_outliers)
    c2_outlier_prop = np.sum(c2_outliers) / len(c2_outliers)

    # Only one has clusters
    if c1.shape[0] == 0 or c2.shape[0] == 0:
        if not outliers:
            return 0
        return 0.5 * c1_outlier_prop * outlier_fs + 0.5 * c2_outlier_prop * outlier_fs

    # Back to csr for cluster matching
    c1 = c1.tocsr()
    c2 = c2.tocsr()

    c1_props = np.asarray(c1.sum(axis=1)).reshape(-1)
    c1_props = c1_props / np.sum(c1_props)
    c2_props = np.asarray(c2.sum(axis=1)).reshape(-1)
    c2_props = c2_props / np.sum(c2_props)

    intersect = c1.astype("float64") @ c2.transpose().astype("float64")
    rs = np.asarray(c1.sum(axis=1)).reshape(-1)  # row sums
    cs = np.asarray(c2.sum(axis=1)).reshape(-1)  # col sums
    rd = sp.diags(rs, dtype="float64")
    cd = sp.diags(cs, dtype="float64")
    nz = intersect.copy()  # Store non-zero data indices
    nz.data = np.ones_like(nz.data)
    union = rd @ nz + nz @ cd - intersect  # union is |p| + |l| - |p and l|
    union.data = 1/union.data
    fs = intersect.multiply(union)

    c1_fs = fs.max(axis=1).toarray().reshape(-1)
    c1_fs_average = np.sum(c1_fs * c1_props)
    c2_fs = fs.max(axis=0).toarray().reshape(-1)
    c2_fs_average = np.sum(c2_fs * c2_props)

    if not outliers:
        return alpha*c1_fs_average + (1-alpha)*c2_fs_average
    return alpha * (c1_outlier_prop * outlier_fs + (1-c1_outlier_prop)*c1_fs_average) \
        + (1-alpha) * (c2_outlier_prop * outlier_fs + (1-c2_outlier_prop)*c2_fs_average)
    