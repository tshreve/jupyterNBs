import numpy as np
#import dask.array as da

# EnKF is the ensemble Kalman Filter analysis processor with one core
# the algorithm is after Everson (2003)   
# A_EnKF = EnKF(A_EnKF = 2d array, data = 1d array, dataerr = 1d array)
# Input: A_EnKF, data, dataerr
#		"A_EnKF" = [[P1_1, P1_2, ..., P1_N],
#					[P2_1, P2_2, ..., P2_N],
#					...
#					[Pnp_1, Pnp_2, ..., Pnp_N],
#					[F1_1, F1_2, ..., F1_N],
#					[F2_1, F2_2, ..., F2_N],
#					...
#					[Fm_1, Fm_2, ..., Fm_N]]
#		in which: np = number of the parameters
#				  N  = number of ensembles
#				  m  = number of measurements
#		"data" = [d1, d2, ..., dm]
#		"dataerr" = [e1, e2, ..., em]
#
# Output: updated matrix: "A_EnKF"
def EnKF(A_EnKF, data, e_coef=0.1, dnorm=False):
    #print('EnKF start')
    # A matrix
    A = A_EnKF * 1.
    # Number of the measurement
    M = len(data)
    # Number of the ensemble
    N = A_EnKF.shape[1]
    # Number of the measurements + parameters
    Ndim = A_EnKF.shape[0]
    # Number of the parameters
    Np = Ndim - M

    # 1_N matrix
    OneN = np.ones([N, N]) / N
    #print('OneN finish')

    # A bar matrix: A_bar = A OneN
    A_bar = A @ OneN
    # A' = A - A_bar
    A_prm = A - A_bar
    #print('A_prm finish')

    # H matrix
    H1 = np.zeros([Np, M])
    H2 = np.identity(M)
    H = np.vstack((H1, H2)).transpose()
    #print('H finish')

    # Measurement Matrix
    d = np.kron(np.ones((N, 1)), data).transpose()
    # if the error is deformation dependent?
    if dnorm:
        E = d * np.random.rand(M, N) * e_coef
    else:
        E = np.random.normal(0,e_coef,size=[M, N])

    # measurement + pertubation
    D = d + E
    #print('D finish')

    # DHA = D - HA
    DHA = D - H @ A

    # HApE = HA' + E
    HApE = H @ A_prm + E
    #print('HApE finish')
    # Singular value decomposition
    U, S, V = np.linalg.svd(HApE)
    #U, S, V = da.linalg.svd(HApE)
    #print('svd finish')
    SIGinv = (S @ S.T) ** (-1)
    #print('SIGinv finish')
    
    # calculate the analysis ensemble
    X1 = SIGinv * U.transpose()
    X2 = X1 @ DHA
    X3 = U @ X2
    X4 = (H @ A_prm).transpose() @ X3
    Aa = A + A_prm @ X4

    return Aa