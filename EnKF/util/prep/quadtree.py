# Yan Zhan (C) 2019

import math
import numpy as np


# get the last row of a array
def getlastrow(matrix):
    if matrix.ndim == 1:
        return matrix
    else:
        return matrix[:, -1]


# check if the elements in the quad exceed the tolerance
def checktolerance(quad, tolerance):
    # if the matrix has 100% of NaN return NaN
    nNaN = np.count_nonzero(np.isnan(quad))
    nELM = quad.size
    if nNaN == nELM:
        return np.nan
    else:
        qdmin = np.nanmin(quad)
        qdmax = np.nanmax(quad)
        if (qdmax - qdmin) > tolerance:
            return 'inf'
        else:
            return np.nanmean(quad)


# check if the elements in the quad exceed the tolerance
def checktolerance_std(quad, tolerance):
    # if the matrix has 100% of NaN return NaN
    nNaN = np.count_nonzero(np.isnan(quad))
    nELM = quad.size
    if nNaN == nELM:
        return np.nan
    else:
        qdstd = np.nanstd(quad)
        if qdstd > tolerance:
            return 'inf'
        else:
            return np.nanmean(quad)


# checking the size of the matrix can be solved by quadtree
def checkdepth(a):
    rows = a.shape[0]
    return int(math.log(rows, 2))


# seperate the parent quad into 4 children quad
def seperate(quad):
    ndim = int(checkdepth(quad))
    # ~ print(ndim)
    NW = quad[0:2 ** (ndim - 1), 0:2 ** (ndim - 1)]
    NE = quad[0:2 ** (ndim - 1), 2 ** (ndim - 1):2 ** ndim]
    SE = quad[2 ** (ndim - 1):2 ** ndim, 2 ** (ndim - 1):2 ** ndim]
    SW = quad[2 ** (ndim - 1):2 ** ndim, 0:2 ** (ndim - 1)]
    return NW, SW, SE, NE


# fill a matrix with zeros if the matrix is not 2^N * 2^N
def fillMat(a):
    if a.size == 1:
        return a
    else:
        nrows = a.shape[0]
        ncols = a.shape[1]
        longside = np.max([nrows, ncols])
        targetlength = int(2 ** np.ceil(math.log(longside, 2)))
        # create a 2^N * 2^N matrix full of NaNs
        anew = np.zeros((targetlength, targetlength))
        anew.fill(np.nan)
        # put the original matix in the middle of it
        crow = int(np.ceil((targetlength - nrows) / 2))
        ccol = int(np.ceil((targetlength - ncols) / 2))
        anew[crow:crow + nrows, ccol:ccol + ncols] = a
        return anew, crow, ccol, nrows, ncols


##############################################################
########################## qtBranch ##########################
##############################################################
# create the quad tree branch, and find the leaf by tolerance
# Input: node (current index info of original matrix)
#		 node = [row col depth]
#		 quad (the current quad matrix)
#		 quad = N * N matrix (N = 2^n)
#		 tolerance (the difference between max and min value in quad)
#		 qtstruct (storing all node, node history)
# 		 qtstruct = [node1
#					 node2
#					 ...
#					 nodeN]
#		 qvalue (storting the mean value of the quad)
#		 qvalue = [value1, value2, ..., valueN]
#
# Output: qtstruct, qvalue
#
def qtBranch(node, quad, tolerance, qtstruct, qvalue, nansize=9):
    # get depth
    depth = node[2]

    # check if the tree find the leaf
    if depth != -1:
        # check quad if meet the tolerance
        # qdtol = checktolerance(quad, tolerance)
        qdtol, RMSE, validity = qttol1(quad, tolerance, nansize=nansize)
        # if exceed the tolerace, seperate it
        if qdtol == np.inf:
            # seperate the branch
            NW, SW, SE, NE = seperate(quad)
            # get new nodes
            # node = [row col depth]
            # row & col are the top-left corner index
            nNW = node + np.array([0, 0, -1])
            nNE = node + np.array([0, 2 ** (depth - 1), -1])
            nSE = node + np.array([2 ** (depth - 1), 2 ** (depth - 1), -1])
            nSW = node + np.array([2 ** (depth - 1), 0, -1])

            # create branch, order: NW -> SW -> SE -> NE
            qtstruct, qvalue = qtBranch(nNW, NW, tolerance, qtstruct, qvalue, nansize=nansize)
            qtstruct, qvalue = qtBranch(nSW, SW, tolerance, qtstruct, qvalue, nansize=nansize)
            qtstruct, qvalue = qtBranch(nSE, SE, tolerance, qtstruct, qvalue, nansize=nansize)
            qtstruct, qvalue = qtBranch(nNE, NE, tolerance, qtstruct, qvalue, nansize=nansize)
        else:
            # attach the branch structure at the end of the tree structure
            qtstruct = np.vstack((qtstruct, node))
            qtattach = [qdtol, RMSE, validity]
            qvalue = np.vstack((qvalue, qtattach))

    return qtstruct, qvalue

# USing RMSE to determine the growth of the quadtree
def qttol1(quad, tolerance, nansize=9):
    # if the matrix has 100% of NaN return NaN
    nNaN = np.count_nonzero(~np.isnan(quad))
    nELM = quad.size
    # initiate RMSE
    RMSE = np.inf
    if nNaN == 0:
        qtval = np.nan
    elif nNaN == nELM:
        RMSE = np.nanstd(quad)
        if RMSE > tolerance:
            qtval = np.inf
        else:
            qtval = np.nanmean(quad)
    else:
        if nELM >= 2**nansize:
            qtval = np.inf
        else:
            RMSE = np.nanstd(quad)
            if RMSE > tolerance:
                RMSE = np.nan
                qtval = np.inf
            else:
                qtval = np.nanmean(quad)
            
    # validity is the propotion of the area with data
    validity = nNaN / nELM
    return qtval, RMSE, validity


def qtXYmesh(x, y, qtstruct):
    # mesh X, Y
    [X, Y] = np.meshgrid(x, y)

    # document the original shape
    nrows_og = X.shape[0]
    ncols_og = X.shape[1]

    for i in range(1, qtstruct.shape[0]):
        # get the left-bottom corner index of every quad
        xIndex_min = int(qtstruct[i, 1])
        yIndex_min = int(qtstruct[i, 0])
        # get the size of each quad
        dxy = int(2 ** qtstruct[i, 2])
        # calculate the right-top corner index of quad
        xIndex_max = xIndex_min + dxy - 1
        yIndex_max = yIndex_min + dxy - 1
        # storing the data, if the data is in the original range
        if xIndex_max <= ncols_og and yIndex_max <= nrows_og:
            # get X, Y matrix
            X[yIndex_min:yIndex_max, xIndex_min:xIndex_max] = (x[xIndex_min] + x[xIndex_max]) / 2.0
            Y[yIndex_min:yIndex_max, xIndex_min:xIndex_max] = (y[yIndex_min] + y[yIndex_max]) / 2.0
    return X, Y


##############################################################
########################## qtMatrix ##########################
##############################################################
# apply quadtree partition to a Matrix, return a new matrix 
# with same size as the original Matrix just for plot
#
# Input: origMat (original 2D matrix storing the data)
#		 tolerance (the difference between the max and min values in a quad)
#
# Output: newMat (the new matrix in the original shape, with quadtree
#					partitional data)
#
def qtMatrix(origMat, tolerance):
    # document the original shape
    nrows_og = origMat.shape[0]
    ncols_og = origMat.shape[1]
    # fill the matrix to be 2^N * 2^N
    origMat, crow, ccol, nrows, ncols = fillMat(origMat)
    # get matrix information
    depth = checkdepth(origMat)
    # initialization
    root = np.array([0, 0, depth])
    nodeini = [-999, -999, -999]
    qvalueini = [-999, -999, -999]
    # new matrix
    newMat = np.zeros(origMat.shape)

    # build quadtree
    qtstruct, qvalue = qtBranch(root, origMat, tolerance, nodeini, qvalueini)

    # re-value the Matrix with the new value of the quad
    for i in range(1, qtstruct.shape[0]):
        irow = int(qtstruct[i, 0])
        icol = int(qtstruct[i, 1])
        dxy = int(2 ** qtstruct[i, 2])
        newMat[irow:(irow + dxy), icol:(icol + dxy)] = qvalue[i, 0]
        newMat[irow, icol:(icol + dxy)] = np.nan
        newMat[irow:(irow + dxy), icol] = np.nan
    # return the new matrix in the original shape
    return newMat[crow:crow + nrows, ccol:ccol + ncols]


############################################################
########################## qtData ##########################
############################################################
# using quadtree partition create the data for EnKF (x, y, z)
#
# Inputs: x (array of data's x location)
#		  x = [x1, x2, ..., xn]
#		  y (array of data's y location)
#		  y = [y1, y2, ..., ym]
#		  Z (2D matrix of the data in an order of x, y mesh)
#		  Z = [[Z11, Z12, ..., Z1n],
#			   [Z21, Z22, ..., Z2n],
#			   ...
#			   [Zm1, Zm2, ..., Zmn]]
#		  tolerance (the difference between the max and min values in a quad)
#
# Outputs: 'x': xqt[1:], 'y': yqt[1:], 'dx': xsize[1:], 'dy': ysize[1:],
#           'data': darray[1:], 'RMSE': RMSE[1:], 'validity': validity[1:],
#           'qt_data': qt_data
def qtData(x: object, y: object, Z: object, num=800, snum=50, ini_nansize=8) -> object:
    # initialization
    xqt = [-999]
    yqt = [-999]
    xsize = [-999]
    ysize = [-999]
    darray = [-999]
    RMSE = [-999]
    validity = [-999]

    # add one more element at the end of the x, y array
    x = np.hstack((x, 2 * x[x.shape[0] - 1] - x[x.shape[0] - 2]))
    y = np.hstack((y, 2 * y[y.shape[0] - 1] - y[y.shape[0] - 2]))

    # fill the matrix to be 2^N * 2^N
    Z, crow, ccol, nrows, ncols = fillMat(Z)
    # get matrix depth
    depth = checkdepth(Z)
    # initialization
    root = np.array([0, 0, depth])
    nodeini = [-999, -999, -999]
    qvalueini = [-999, -999, -999]

    # make the number of the quad go into the range of (num+-snum)
    # initialize the parameter
    # initial tolerance = std of the whole matrix
    tol = np.nanstd(Z)
    # initial tolerance increment/decrement
    dtol = np.nanstd(Z)
    # initial number of the quad
    cnum = 1
    # initial status
    if cnum < num - snum:
        cstat = -1
    elif cnum > num + snum:
        cstat = 1
    else:
        cstat = 0
    
    nansize = ini_nansize * 1
    # do while the number of the quad is outside the range
    n_nansize = 0
    while cstat != 0:
        # quadtree partition
        qtstruct, qvalue = qtBranch(root, Z, tol, nodeini, qvalueini,nansize=nansize)
        # number of the data
        cnum = qtstruct.shape[0]
        pstat = cstat * 1.
        # current status
        if cnum < num - snum:
            cstat = -1
        elif cnum > num + snum:
            cstat = 1
        else:
            cstat = 0      
     
        if cstat == 1:
            # adjust the tolerance
            if pstat * cstat == 1.:
                n_nansize += 1
                dtol = dtol * 2
            else:
                dtol = dtol / 2
            tol = tol + dtol 
        elif cstat == -1:
            n_nansize = 0
            dtol = dtol / 2
            tol = tol - dtol
            
        if n_nansize > 6:
            nansize += 1
            n_nansize = 0
            
        print('num={},tol={},nansize={}'.format(cnum,tol,nansize))

    # Create quadtree partitioned Matrix for plot
    # new matrix
    Iplotqt = np.nan * np.ones(Z.shape)
    # plotMat
    plotMat = np.nan * np.ones(Z.shape)
    # index of the quad in plotting matrix
    n = 0

    # re-value find the center (x, y) of the quad
    for i in range(1, qtstruct.shape[0]):
        # get the left-bottom corner index and the quad size
        irow = int(qtstruct[i, 0])
        icol = int(qtstruct[i, 1])
        dxy = int(2 ** qtstruct[i, 2])

        # storing the data, if the data is in the original range
        if irow >= crow and (irow + dxy) <= (crow + nrows) \
                and icol >= ccol and (icol + dxy) <= (ccol + ncols) \
                and np.isnan(qvalue[i, 0]) == False:
            # get the position of the quad center
            xval = (x[icol - ccol] + x[icol + dxy - ccol]) / 2.0
            yval = (y[irow - crow] + y[irow + dxy - crow]) / 2.0
            dx = +(x[icol + dxy - ccol] - x[icol - ccol])
            dy = -(y[irow + dxy - crow] - y[irow - crow])

            # create downsampled array for EnKF
            # append x, y locations
            xqt = np.hstack((xqt, xval))
            yqt = np.hstack((yqt, yval))
            # append x, y size
            xsize = np.hstack((xsize, dx))
            ysize = np.hstack((ysize, dy))
            # append data
            darray = np.hstack((darray, qvalue[i, 0]))
            # append RMSE
            RMSE = np.hstack((RMSE, qvalue[i, 1]))
            # append validity
            validity = np.hstack((validity, qvalue[i, 2]))

            # matrix for plot
            plotMat[irow:(irow + dxy), icol:(icol + dxy)] = qvalue[i, 0]
            # Index matrix for converting array to plotting matrix
            Iplotqt[irow:(irow + dxy), icol:(icol + dxy)] = n
            # update the index
            n = n + 1

    # update matrix for plot and its index
    qt_plot = plotMat[crow:crow + nrows, ccol:ccol + ncols]
    qt_data = Iplotqt[crow:crow + nrows, ccol:ccol + ncols]
    # X & Y for qt_plot
    X, Y = np.meshgrid(x[1:], y[1:])
    
    print('quadtree partiation finished, number of datanum={}'.format(cnum))
    return {'x': xqt[1:], 'y': yqt[1:], 'dx': xsize[1:], 'dy': ysize[1:],
            'data': darray[1:], 'RMSE': RMSE[1:], 'validity': validity[1:],
            'qt_data': qt_data, 'qt_plot': qt_plot, 'qtX': X, 'qtY': Y}


# mask the data
def mask(x, y, z, rMask):
    R2 = np.array(x) ** 2 + np.array(y) ** 2
    xnew = x[R2 < rMask ** 2]
    ynew = y[R2 < rMask ** 2]
    znew = z[R2 < rMask ** 2]
    return xnew, ynew, znew
