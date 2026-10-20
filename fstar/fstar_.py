import numpy as np
import scipy.sparse as sp


def clustering_array_to_sparse(clustering):
    """Convert a numpy clsutering array to a (clusters x objects) csr array
    Clusters are assumed to be contiguous 0-max(clustering), and -1 denotes objects with no clusters
    """
    object_ids = np.arange(len(clustering), dtype=clustering.dtype)
    inliers = clustering >= 0
    object_ids = object_ids[inliers]
    cluster_ids = clustering[inliers]
    data = np.ones_like(cluster_ids, dtype="bool")

    clustering = sp.coo_array(
        (data, (cluster_ids, object_ids))
    )
    clustering = clustering.tocsr()
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
    if isinstance(node_clustering, np.ndarray):
        node_clustering = clustering_array_to_sparse(node_clustering)
    elif not sp.issparse(node_clustering):
        raise ValueError("node_clustering must be a scipy.sparse array or a 1d numpy array.")

    # Need fast access to clusters per node
    node_clustering = node_clustering.tocsc()
    edge_clustering = sp.lil_array((len(adjacency.indices), node_clustering.shape[0]), dtype="bool")
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


def find_best_matches(c1, c2, return_match_ids=False):
    """ Find the best match for each cluster. Returns either
    two arrays of the F* scores for each cluster, or if
    `return_match_ids=True` then return four arrays, the first
    two the same as above and the second two are the cluster ids
    of the best match in the other clustering.

    Parameters
    ----------
    c1: scipy.sparse_array - the first clustering
    c2: scipy.sparse_array - the second clustering
    return_match_ids:bool (default=False) - Flag to include match ids in return
    """
    c1 = c1.tocsr()
    c2 = c2.tocsr()
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
    c2_fs = fs.max(axis=0).toarray().reshape(-1)

    if return_match_ids:
        c1_matches = fs.argmax(axis=1).toarray().reshape(-1)
        c2_matches = fs.argmax(axis=0).toarray().reshape(-1)
        return c1_fs, c2_fs, c1_matches, c2_matches

    return c1_fs, c2_fs


def fstar(
        c1: sp.sparray | np.ndarray,
        c2: sp.sparray | np.ndarray,
        outliers: bool = True,
        drop_outliers: bool | str =False,
        alpha: float = 0.5
    ) -> float:
    """ Compute F*_wo between c1 and c2.

    Parameters
    ----------
    c1:scipy.sparse.array or np.ndarray - the first clustering
    c2:scipy.sparse_array or np.ndarray - the second clustering
    outliers:bool (defualt=True) - Flag to add outlier comparison term. If True compute F*_wo, if
    False compute F*_w
    drop_outliers:[False, 'c1', 'c2', 'either', 'both'] (default=False) - Flag to drop objects that
        are consider outliers before computing score. Can be helpful for determining the quality of
        the clusters when extra outliers are not a concern.
    alpha:float (default=0.5) - A value between 0 and 1 to control the importance of matching in each
        direction. The default 0.5 is a symmetric measure, alpha=1 puts all the weight on the ability
        of each cluster in c1 to find a match and alpha=0 puts all the weight on the ability of each
        cluster in c2 to find a match.

    Returns
    -------
    float: the similarity scoree between c1 and c2.
    """
    if isinstance(c1, np.ndarray):
        c1 = clustering_array_to_sparse(c1)
    elif not sp.issparse(c1):
        raise ValueError("Fstar expects a scipy.sparse array or a 1d numpy array.")
    if isinstance(c2, np.ndarray):
        c2 = clustering_array_to_sparse(c2)
    elif not sp.issparse(c2):
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
            raise ValueError("drop_outliers must be Falsey or one of ['c1', 'c2', 'either', 'both'].")
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
        return outlier_fs * (alpha * c1_outlier_prop + (1-alpha) * c2_outlier_prop)

    c1_fs, c2_fs = find_best_matches(c1, c2)

    c1_props = np.asarray(c1.sum(axis=1)).reshape(-1)
    c1_props = c1_props / np.sum(c1_props)
    c1_fsw = np.sum(c1_fs * c1_props)
    c2_props = np.asarray(c2.sum(axis=1)).reshape(-1)
    c2_props = c2_props / np.sum(c2_props)
    c2_fsw = np.sum(c2_fs * c2_props)

    if outliers:
        c1_fsw = c1_outlier_prop * outlier_fs + (1-c1_outlier_prop) * c1_fsw
        c2_fsw = c2_outlier_prop * outlier_fs + (1-c2_outlier_prop) * c2_fsw

    return alpha * c1_fsw + (1-alpha) * c2_fsw
