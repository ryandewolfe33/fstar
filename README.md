# Fstar
A Pragmatic Method for Comparing Clusterings with Overlaps and Outliers.

## Getting Started
The package is currently pip installable but is not on pypi. For now, please clone this repository and install locally by following the instructions below.

```{shell}
git clone https://github.com/ryandewolfe33/fstar.git
cd fstar
pip install .
```

## Example
For examples of using fstar, please see the notebooks in the `experiments` folder.

## API
The main function is also named `fstar`, and computes a similarity score between two clusterings.
```
fstar.fstar(c1, c2, outliers=True,drop_outliers=False, alpha=0.5) -> float:
    Compute F*_wo between c1 and c2.

    Parameters
    ----------
    c1:scipy.sparse.array or np.ndarray - the first clustering
    c2:scipy.sparse_array or np.ndarray - the second clustering
    outliers:bool (defualt=True) - Flag to add outlier comparison term. If True compute F*_wo, if False compute F*_w
    drop_outliers:[False, 'c1', 'c2', 'either', 'both'] (default=False) - Flag to drop objects that consider outliers before computing score. Can be helpful for determining the quality of the clusters when extra outliers are not a concern.
    alpha:float (default=0.5) - A value between 0 and 1 to control the importance of matching in each direction. The default 0.5 is a symmetric measure, and 0/1 only looks at the best match for clustering in c1/c2.
    
    Returns
    -------
    float: the similarity scoree between c1 and c2.
```

There are also two helper function for working with graph aware comparisons. To compute the closed edge clustering of an edge clustering `e1`, run `node_clustering_to_edge_clustering(adjacency, edge_clustering_to_node_clustering(adjacency, e1))`.
```
fstar.node_clustering_to_edge_clustering(adjacency, node_clustering)
    Project a node clustering onto the edges. An edge is in a cluster if both endpoints are in that cluster.
    Edge ids are computed by the data_index of a csr representation of the upper triangular adjacency matrix.

    Parameters
    ----------
    adjacency:scipy.sparse.array - the adjacency matrix of the graph.
    node_clustering:scipy.sparse_array or np.ndarray - the node clustering. 
    
    Returns
    -------
    scipy.sparse.csr_array - the induced edge clustering. Edge ids are the data_index of a csr representation of the upper triangular adjacency matrix.
```
```
fstar.edge_clustering_to_node_clustering(adjacency, edge_clustering):
    """
    Project an edge clustering onto the nodes. An node is in a cluster if it has at least one incident edge in that cluster. Edge ids are assumed to be the data_index of a csr representation of the upper triangular adjacency matrix.
    """

    Parameters
    ----------
    adjacency:scipy.sparse.array - the adjacency matrix of the graph.
    node_clustering:scipy.sparse_array or np.ndarray - the edge clustering. Edge ids are assumed to be the data_index of a csr representation of the upper triangular adjacency matrix.
    
    Returns
    -------
    scipy.sparse.csr_array - the induced node clustering.
```


## Reference
Ryan DeWolfe, Paweł Prałat, and François Théberge. A Pragmatic Method for Comparing Clusterings with Overlaps and Outliers. Arxiv Preprint (2026). [https://doi.org/10.48550/arXiv.2602.14855]([https://doi.org/10.48550/arXiv.2602.14855)
