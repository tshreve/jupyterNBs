import numpy as np
from numpy import random


# L2 norm of the results
def L2norm(A_EnKF, data_all, dsize=1.):
    # size total
    tsize = np.sum(dsize)
    # number of the ensemble
    nens = A_EnKF.shape[1]
    # number total dimension
    Ndim = A_EnKF.shape[0]
    # number of the measurements
    ndat = len(data_all)
    # allocate L2
    L2 = np.zeros(nens)
    # maximum value of data
    dmax = np.nanmax(np.abs(data_all))
    if dmax < 1e-10:
        dmax = 1e-10
    # do for all ensembles
    for i in np.arange(0, nens):
        dist = (A_EnKF[Ndim - ndat:Ndim, i] - data_all) / dmax
        L2[i] = (np.sum(dist ** 2 * dsize) / tsize) ** (1 / 2)
    return L2


# find the muta % of L2 needs to be mutated
def L2findMutation(x, muta):
    # sort data
    x_sort = np.sort(x)
    # find threshold value
    x_crit = x_sort[np.int(np.floor((1 - muta) * len(x)))]
    # return the index for mutation
    return x > x_crit, x_crit


# Mutation process
def Mutation(A_EnKF_p, Imut, new_std):
    # random generate
    random.seed()
    # nens
    nens = A_EnKF_p.shape[1]
    # mean value of the un mutation values
    new_mean = A_EnKF_p[:, ~Imut].mean(1)
    # re-distribute the parameters
    for p in np.arange(0, A_EnKF_p.shape[0]):
        A_EnKF_p[p, :] = new_mean[p] + new_std[p] * np.random.randn(1, nens)
    return A_EnKF_p