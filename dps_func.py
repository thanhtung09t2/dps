"""
DPS Correntropy based, hierarchical density preserving data split
% 
%   R	    = DPS(A,levels,labs)
%   [R H]   = DPS(A,levels,labs)
%
% INPUT
%   A			Input data (rows = observations)
%   levels		Number of split levels, default: 3
%   labs        Labels for the data (optional, if no labels are given unsupervised split is performed)
%
% OUTPUT
%   R			Index array with rotation set with 2^LEVELS folds
%   H			Hierarchy of splits
%
% DESCRIPTION
% Density Preserving Sampling (DPS) divides the input dataset into a given
% number of folds (2^LEVELS) by maximizing the correntropy between the folds
% and can be used as an alternative for cross-validation. The procedure is
% deterministic, so unlike cross-validation it does not need to be repeated.
%
% REFERENCE
%   Budka, M. and Gabrys, B., 2012.
%   Density Preserving Sampling: Robust and Efficient Alternative to Cross-validation for Error Estimation.
%   IEEE Transactions on Neural Networks and Learning Systems, DOI: 10.1109/TNNLS.2012.2222925. 

""" 

import numpy as np

def dps(A, levels=3, labs=None):
    if labs is None or len(labs) == 0:
        labs = np.ones((A.shape[0],), dtype=int)
    if levels is None or levels == 0:
        levels = 3

    # Đánh số lại các nhãn để chỉ số bắt đầu từ 1
    if np.min(labs) < 1:
        labs = labs + 1 - np.min(labs)

    u = np.unique(labs).astype(int)
    t = np.full(int(np.max(u)) + 1, np.nan)
    t[u] = np.arange(1, len(u) + 1)  # Giữ chỉ số 1-based
    labs = t[labs.astype(int)]

    H = np.zeros((levels, A.shape[0]), dtype=int)

    idxs = [None] * (levels + 1)
    idxs[0] = [np.arange(A.shape[0])]

    for i in range(levels):
        idxs[i + 1] = []
        for j in range(len(idxs[i])):
            t = helper(A[idxs[i][j], :], labs[idxs[i][j]])
            idxs[i + 1].append(idxs[i][j][t[0]])
            idxs[i + 1].append(idxs[i][j][t[1]])

        for j, idx in enumerate(idxs[i + 1]):
            H[i, idx] = j + 1  # Giữ chỉ số 1-based

    R = H[-1, :]  # Note: R=0 is the remaining samples which cannot be allocated to the groups
    return R, H


def helper(A, labs):
    c = int(np.max(labs))
    cidx = [np.where(labs == i + 1)[0] for i in range(c)]  # Chỉ số 1-based
    csiz = [len(idx) for idx in cidx]

    siz = [0, 0]
    idx = []

    for i in range(c):
        BI = cidx[i]
        B = A[BI, :]

        m = len(BI)
        mask = np.ones(m, dtype=bool)

        BB = np.sum(B**2, axis=1).reshape(-1, 1)
        D = BB - 2 * np.dot(B, B.T) + BB.T
        np.fill_diagonal(D, np.inf)

        Dorg = D.copy()
        idx.append(np.full((2, (m + 1) // 2), np.nan, dtype=int))

        for j in range(m // 2):
            I, J = np.unravel_index(np.argmin(D), D.shape)
            mask[I] = mask[J] = False

            if (np.nanmean(Dorg[I, idx[i][0, :j]]) + np.nanmean(Dorg[J, idx[i][1, :j]])) < \
               (np.nanmean(Dorg[I, idx[i][1, :j]]) + np.nanmean(Dorg[J, idx[i][0, :j]])):
                idx[i][0, j] = J
                idx[i][1, j] = I
            else:
                idx[i][0, j] = I
                idx[i][1, j] = J

            D[I, :] = D[:, I] = np.inf
            D[J, :] = D[:, J] = np.inf

        if mask.any():
            I = np.where(mask)[0][0]
            if siz[0] < siz[1]:
                idx[i][0, -1] = I
            else:
                idx[i][1, -1] = I

        idx[i][0, ~np.isnan(idx[i][0, :])] = BI[idx[i][0, ~np.isnan(idx[i][0, :])].astype(int)]
        idx[i][1, ~np.isnan(idx[i][1, :])] = BI[idx[i][1, ~np.isnan(idx[i][1, :])].astype(int)]

        siz[0] += np.sum(~np.isnan(idx[i][0, :]))
        siz[1] += np.sum(~np.isnan(idx[i][1, :]))

    idx = np.concatenate(idx, axis=1)
    idxs = [idx[0, ~np.isnan(idx[0, :])].astype(int), idx[1, ~np.isnan(idx[1, :])].astype(int)]
    return idxs