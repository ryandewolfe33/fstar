# This file implements a simple conversion of fstar to use the
# f1 or f1p scores instead for use in the comparions.

import numpy as np
import scipy.sparse as sp


def f1(c1: sp.sparray, c2: sp.sparray):
    # Inputs c1, c2: The clusterings to compare.
    # Both must be scipy csr arrays of size (n_clusters, n_objects)
    # with c[i,j] = 1 if object j is in cluster i.
    if c1.shape[0] == 0 and c2.shape[0] == 0: # No clusters
        return 1
    c1_sizes = c1.sum(axis=1)
    c1_props = c1_sizes / np.sum(c1_sizes)
    c2_sizes = c2.sum(axis=1)
    c2_props = c2_sizes / np.sum(c2_sizes)
    intersect = c1.astype("int64") @ c2.transpose().astype("int64")

    nz = intersect.copy() # Store non-zero data indices
    nz.data[:] = 1 # Set all non-zero entries to 1

    row_sizes = sp.diags(c1_sizes, dtype="int64")
    c1_size_matrix = row_sizes @ nz
    c1_size_matrix.data = 1 / c1_size_matrix.data
    precision = intersect.multiply(c1_size_matrix)

    col_sizes = sp.diags(c2_sizes, dtype="int64")
    c2_size_matrix = nz @ col_sizes
    c2_size_matrix.data = 1 / c2_size_matrix.data
    recall = intersect.multiply(c2_size_matrix)

    precision.data = 1 / precision.data
    recall.data = 1 / recall.data

    f1 = precision + recall
    f1.data = 2 / f1.data

    f1_l = f1.max(axis=1).toarray() # l for left, c1 compared to c2
    f1_lw = np.sum(f1_l * c1_props)
    f1_r = f1.max(axis=0).toarray() # r for right, c2 compared to c1
    f1_rw = np.sum(f1_r * c2_props)
    c1_o = c1.count_nonzero(axis=0) == 0
    c2_o = c2.count_nonzero(axis=0) == 0
    o_intersect = np.sum(c1_o * c2_o)
    o_f1 = 0 # outlier fstar term, only compute if intersect is non-zero
    if o_intersect > 0:
        precision = o_intersect / np.sum(c1_o)
        recall = o_intersect / np.sum(c2_o)
        o_f1 = 2 / (1/precision + 1/recall)
    alpha = 1 - np.sum(c1_o) / len(c1_o) # c1 outlier prop
    beta = 1 - np.sum(c2_o) / len(c2_o) # c2 outlier prop
    f1_l_wo = (1 - alpha) * o_f1 + alpha * f1_lw
    f1_r_wo = (1 - beta) * o_f1 + beta * f1_rw
    return 0.5 * (f1_l_wo + f1_r_wo)


def f1p(c1: sp.sparray, c2: sp.sparray):
    # Inputs c1, c2: The clusterings to compare.
    # Both must be scipy csr arrays of size (n_clusters, n_objects)
    # with c[i,j] = 1 if object j is in cluster i.
    if c1.shape[0] == 0 and c2.shape[0] == 0: # No clusters
        return 1
    c1_sizes = c1.sum(axis=1)
    c1_props = c1_sizes / np.sum(c1_sizes)
    c2_sizes = c2.sum(axis=1)
    c2_props = c2_sizes / np.sum(c2_sizes)
    intersect = c1.astype("int64") @ c2.transpose().astype("int64")

    nz = intersect.copy() # Store non-zero data indices
    nz.data[:] = 1 # Set all non-zero entries to 1
    row_sizes = sp.diags(c1_sizes, dtype="int64")
    c1_size_matrix = row_sizes @ nz
    col_sizes = sp.diags(c2_sizes, dtype="int64")
    c2_size_matrix = nz @ col_sizes

    size_product = c1_size_matrix.multiply(c2_size_matrix)
    size_product.data = 1/size_product.data
    fs = intersect.power(2).multiply(size_product)

    fs_l = fs.max(axis=1).toarray() # l for left, c1 compared to c2
    fs_lw = np.sum(fs_l * c1_props)
    fs_r = fs.max(axis=0).toarray() # r for right, c2 compared to c1
    fs_rw = np.sum(fs_r * c2_props)
    c1_o = c1.count_nonzero(axis=0) == 0
    c2_o = c2.count_nonzero(axis=0) == 0
    o_intersect = np.sum(c1_o * c2_o)
    o_fs = 0 # outlier fstar term, only compute if intersect is non-zero
    if o_intersect > 0:
        o_fs = o_intersect / (np.sum(c1_o) + np.sum(c2_o) - o_intersect)
    alpha = 1 - np.sum(c1_o) / len(c1_o) # c1 outlier prop
    beta = 1 - np.sum(c2_o) / len(c2_o) # c2 outlier prop
    fs_l_wo = (1 - alpha) * o_fs + alpha * fs_lw
    fs_r_wo = (1 - beta) * o_fs + beta * fs_rw
    return 2 * fs_l_wo * fs_r_wo / (fs_l_wo + fs_r_wo)
