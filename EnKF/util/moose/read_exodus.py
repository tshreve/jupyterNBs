import netCDF4
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.tri as mtri
from scipy.interpolate import interp1d
import vtk

##############################################################################
# read the surface displacement fromt the exodus file
# method = 'node': the date extracted on the nodes
#        = 'coord': the date interpolated on the input coordinates
def read_surf_uw(filename, method='node', coord=[]):
    
    # read the file
    nc = netCDF4.Dataset(filename)
    # read coordinates
    X_all = nc.variables['coordx'][:]
    Y_all = nc.variables['coordy'][:]
    # read displacement
    U_all = nc.variables['vals_nod_var1'][:]
    W_all = nc.variables['vals_nod_var2'][:]
    # close the file
    nc.close()
    
    # Get the surface data
    surface = np.where(Y_all[:]==0.)
    # r
    x_node = X_all[surface[0]]
    # data on the surface
    u_raw = U_all[1,surface[0]]
    w_raw = W_all[1,surface[0]]

    # storing the data in an pandas dataframe
    disp2d = pd.DataFrame({'r':x_node,'u':u_raw,'w':w_raw})
    # sort the data according to the distance
    disp2d = disp2d.sort_values('r')
    
    # if method is coordinate, then interpolate the data
    if method=='coord':
        fu = interp1d(disp2d.r, disp2d.u)
        fw = interp1d(disp2d.r, disp2d.w)
        u_int = fu(coord)
        w_int = fw(coord)
        # storing the data in an pandas dataframe
        disp2d = pd.DataFrame({'r':coord,'u':u_int,'w':w_int})
    
    return disp2d

##############################################################################
# read surface displacement from MOOSE output file
# And interpolate into the desired coordinates
# by Tara Shreve & Yan Zhan (2021)
# Input: fname = string (file name)
#        xdata = numpy array (1d or 2d), x coordinates
#        ydata = numpy array (1d or 2d), y coordinates
# Output: umodel, vmodel, wmodel = numpy array (same shape as xdata)
#         three components of deformation
def read_moose_uvw(fname, xdata, ydata):

    nc = netCDF4.Dataset(fname)

    coordx = nc.variables['coordx']
    coordy = nc.variables['coordy']
    coordz = nc.variables['coordz']

    #!Find better way to sort values and interpolate
    surface = np.where(coordz[:]==0.)
    distancex = coordx[surface[0]]
    distancey = coordy[surface[0]]

    Ux_all = nc.variables['vals_nod_var1']
    Uy_all = nc.variables['vals_nod_var2']
    Uz_all = nc.variables['vals_nod_var3']

    Ux_moo_3d = Ux_all[1,surface[0]]
    Uy_moo_3d = Uy_all[1,surface[0]]
    Uz_moo_3d = Uz_all[1,surface[0]]
    
    umodel = griddata((distancex.data,distancey.data), Ux_moo_3d.data,
                      (xdata, ydata), method='cubic')
    vmodel = griddata((distancex.data,distancey.data), Uy_moo_3d.data,
                      (xdata, ydata), method='cubic')
    wmodel = griddata((distancex.data,distancey.data), Uz_moo_3d.data,
                      (xdata, ydata), method='cubic')
    
    return umodel, vmodel, wmodel

##############################################################################
# read n'th step data and write into PyVista UnstructuredGrid
# By Yan Zhan (2022)
# Input: fname = string (file name)
#        nstep (int):  the time step
# output: PyVista UnstructuredGrid Data
def exodus2PyVista(filename, nstep=1):
    # read exodus by netCFD4
    model = netCDF4.Dataset(filename)
    
    # dimension of the model
    dim = model.dimensions['num_dim'].size
    # number nodes of the element
    nne = model.dimensions['num_nod_per_el1'].size
    
    # read coordinates
    X_all = np.ma.getdata(model.variables['coordx'][:])
    Y_all = np.ma.getdata(model.variables['coordy'][:])
    
    if dim==3:
        print('read 3D model')
        Z_all = np.ma.getdata(model.variables['coordz'][:])
    elif dim==2:
        print('read 2D model')
        Z_all = np.zeros(X_all.shape)
    else:
        raise ValueError('dim = {} is not supported, (need to be 2 or 3)'.format(dim))
        
    # ensemble the points
    points = np.vstack([X_all,Y_all,Z_all]).T
    # how node is mapped
    elem_node = np.ma.getdata(model.variables['connect1'][:])-1
    
    # create PyVista UnstructuredGrid
    if dim==3:
        if nne==4:
            grid = pv.UnstructuredGrid({vtk.VTK_TETRA: elem_node}, points)
        elif nne==10:
            grid = pv.UnstructuredGrid({vtk.VTK_QUADRATIC_TETRA: elem_node}, points)
        else:
            raise ValueError('The element type is not supported')
    elif dim==2:
        if nne==3:
            grid = pv.UnstructuredGrid({vtk.VTK_TRIANGLE: elem_node}, points)
        elif nne==6:
            grid = pv.UnstructuredGrid({vtk.VTK_QUADRATIC_TRIANGLE: elem_node}, points)
        else:
            raise ValueError('The element type is not supported')
    else:
        raise ValueError('dim = {} is not supported, (need to be 2 or 3)'.format(dim))
    
    try:
        # get the name of the variables for nodal data 
        name_nod_var = getNames(model,'name_nod_var')
            # write the data in the PyVista mesh (nodal)
        for i, nnv in enumerate(name_nod_var):
            grid[nnv] = model.variables['vals_nod_var{}'.format(i+1)][:][nstep]
    except:
        print('No nodal data found')
    
    try:
        # get the name of the variables element data 
        name_elem_var = getNames(model,'name_elem_var')
        # write the data in the PyVista mesh (element)
        for i, nev in enumerate(name_elem_var):
            grid[nev] = model.variables['vals_elem_var{}eb1'.format(i+1)][:][nstep]
    except:
        print('No element data found')
            
    # close the model
    model.close()
    
    return grid

##############################################################################
# read n'th step data along a boundary and write into PyVista UnstructuredGrid
# By Yan Zhan (2022)
# Input: fname = string (file name)
#        bd_name = string (boundary name)
#        nstep (int):  the time step
# output: PyVista UnstructuredGrid Data
def exodus_getBD(filename, bd_name = 'chamber', nstep = 1):
    # read model
    model = netCDF4.Dataset(filename)

    # dimension of the model
    dim = model.dimensions['num_dim'].size
    # number nodes of the element
    nne = model.dimensions['num_nod_per_el1'].size

    # get name of the boundarys
    ss_names = getNames(model, key='ss_names')
    # if the name is valid
    if bd_name in ss_names:
        # find the index of the boundary name
        bd_sel = ss_names.index(bd_name) + 1
    else:
        raise ValueError('Unknown bd_name: {}, Valid: {}'.format(bd_name,ss_names))

    # Side set side list (P14 3.11.3)
    side_ss = model.variables['side_ss{}'.format(bd_sel)][:]
    # Side set element list (P14 3.11.2)
    elem_ss = model.variables['elem_ss{}'.format(bd_sel)][:]
    # Element connectivity
    connect = model.variables['connect1'][:]
    # read coordinates
    X_all = np.ma.getdata(model.variables['coordx'][:])
    Y_all = np.ma.getdata(model.variables['coordy'][:])
    if dim==3:
        Z_all = np.ma.getdata(model.variables['coordz'][:])
    elif dim==2:
        Z_all = Y_all * 0.
    else:
        raise ValueError('dim = {} is not supported, (need to be 2 or 3)'.format(dim))

    # side set node ordering
    QUAD = np.array([[1,2,5],
                     [2,3,6],
                     [3,4,7],
                     [4,1,8]])
    TRIANGLE = np.array([[1,2,4],
                         [2,3,5],
                         [3,1,6]])
    TETRA = np.array([[1,2,4,5,9,8],
                      [2,3,4,6,10,9],
                      [1,4,3,8,10,7],
                      [1,3,2,7,6,5]])

    # determine element type (P24 Table 2)
    # side set node ordering
    if dim==2:
        # 2 dimension
        if nne==3:
            # 3 node Triangle
            norder_ss = TRIANGLE[:,0:2]
            vtkCellType = vtk.VTK_LINE
        elif nne==6:
            # 6 node Triangle
            norder_ss = TRIANGLE
            vtkCellType = vtk.VTK_QUADRATIC_EDGE
        else:
            raise ValueError('The element type is not supported')
    elif dim==3:
        # 3 dimension
        if nne==4:
            # 4 node tetra
            norder_ss = TETRA[:,0:3]
            vtkCellType = vtk.VTK_TRIANGLE
        elif nne==10:
            # 10 node tetra
            norder_ss = TETRA
            vtkCellType = vtk.VTK_QUADRATIC_TRIANGLE
        else:
            raise ValueError('The element type is not supported')
    else:
        raise ValueError('dim = {} is not supported, (need to be 2 or 3)'.format(dim))

    # find the node number in each side element
    elem_node_surf = []
    for elem_cur,side_cur in zip(connect[elem_ss-1], side_ss):
        elem_node_surf.append(elem_cur[norder_ss[side_cur-1]-1])
    # generate node of elements along the boundary
    elem_node_surf = np.array(elem_node_surf) - 1

    # generate VTK object
    # nodes locations
    points = np.vstack([X_all,Y_all,Z_all]).T
    # element connectivity
    elem_node = np.ma.getdata(model.variables['connect1'][:])-1
    # generate pyVista UnstructuredGrid
    grid = pv.UnstructuredGrid({vtkCellType: elem_node_surf}, points)

    # make the other nodes nan
    id_all = np.arange(0,grid.n_points)
    id_surf = np.unique(elem_node_surf.reshape(-1))
    id_keep = np.isin(id_all, id_surf)
    grid.points[~id_keep,:] = np.nan

    try:
        # get the name of the variables for nodal data 
        name_nod_var = getNames(model,'name_nod_var')
            # write the data in the PyVista mesh (nodal)
        for i, nnv in enumerate(name_nod_var):
            grid[nnv] = model.variables['vals_nod_var{}'.format(i+1)][:][nstep]
            grid[nnv][~id_keep] = np.nan
    except:
        print('No nodal data found')

    try:
        # get the name of the variables element data 
        name_elem_var = getNames(model,'name_elem_var')
        # write the data in the PyVista mesh (element)
        for i, nev in enumerate(name_elem_var):
            grid[nev] = model.variables['vals_elem_var{}eb1'.format(i+1)][:][nstep][elem_ss-1]
    except:
        print('No element data found')

    return grid

##############################################################################
# Get Names in a catalog (key)
# By Yan Zhan (2022)
def getNames(model, key='name_nod_var'):
    # name of the element variables
    name_var = []
    for vname in np.ma.getdata(model.variables[key][:]):
        Iend = np.where(vname==b'')[0][0]
        name_var.append(''.join(vname[:Iend].astype('U8')))
    return name_var

##############################################################################
# Get the mesh of the 2D model
# currently only support triangluar mesh
def get2Dmesh(model, elemType='tri'):
    # read coordinates
    X_all = model.variables['coordx'][:]
    Y_all = model.variables['coordy'][:]
    # how node is mapped
    elem_node = model.variables['connect1'][:]
    # Element ID
    elem_id = model.variables['elem_num_map'][:]
    # Node ID
    node_id = model.variables['node_num_map'][:]
    if elemType=='tri':
        # Create triangulation.
        triang = mtri.Triangulation(np.ma.getdata(X_all),
                                    np.ma.getdata(Y_all),
                                    np.ma.getdata(elem_node-1))
    else:
        raise ValueError('Not support elemType={}'.format(elemType))
    return triang

##############################################################################
# Get the values of the variable
# input: model (netCFD4.Dataset),
#        var_name (string): the name of the variable
#        level (string): the variable stored in which level (node or element)
#        nstep (int):  the time step
# output: the values of the variable
def getVar(model,var_name='disp_r',level='node',nstep=1):
    # name of the element variables
    if level=='node':
        name_all_var = getNames(model,key='name_nod_var')
    elif level=='elem':
        name_all_var = getNames(model,key='name_elem_var')
    else:
        raise ValueError('Unknown level: {}'.format(level))
    # check if the variable name exists
    if var_name in name_all_var:
        # find the index
        var_sel = name_all_var.index(var_name) + 1
        # extract the data
        if level=='node':
            vals = model.variables['vals_nod_var{}'.format(var_sel)][:][nstep]
        else:
            vals = model.variables['vals_elem_var{}eb1'.format(var_sel)][:][nstep]
        # unmask
        vals = np.ma.getdata(vals)
    else:
        raise ValueError('Unknown var_name: {}, Valid var: {}'.format(var_name, name_all_var))
    
    return vals

##############################################################################
# Find the node ID, if the node is along the boundary (bd_name)
def find_BDnodes(model, bd_name='chamber'):
    # how node is mapped
    elem_node = model.variables['connect1'][:]
    # name of the boundarys
    ss_names = getNames(model, key='ss_names')
    # if the name is valid
    if bd_name in ss_names:
        # find the index of the boundary name
        bd_sel = ss_names.index(bd_name) + 1
        # the element id of the boundary
        elem_ss = model.variables['elem_ss{}'.format(bd_sel)][:]
    else:
        raise ValueError('Unknown bd_name: {}, Valid: {}'.format(bd_name,ss_names))

    return find_node_chain(elem_node[elem_ss-1][:])

##############################################################################
# Get the variable's value along the boundary
# input: model (netCFD4.Dataset),
#        bd_name (string): the name of the boundary
#        var_name (string): the name of the variable
#        level (string): the variable stored in which level (node or element)
#        nstep (int):  the time step
#        powerIWD: power number of the inversed distance weighted average
#                  if level is element, the nodal variable are intetpolated
#                  by the element values using the IWD
# output: the boundary node's x, y, and values of the variable
def getBDvar(model,bd_name='chamber',var_name='disp_r',level='node',
             nstep=1,powerIDW=2.):
    # find the boundary nodes
    BD_node = find_BDnodes(model, bd_name=bd_name)
    # mesh of the model
    mesh = get2Dmesh(model, elemType='tri')
    # get the value of the variable
    vals = getVar(model,var_name=var_name,level=level,nstep=nstep)
    # If the variable is evaluated on the element
    if level=='elem':
        # allocate the nodal values
        vals_node = np.zeros(len(BD_node))
        # get the nodal values
        for i, node_sel in enumerate(BD_node):
            vals_node[i] = IDW_NodeVals(node_sel,vals,
                            model.variables['coordx'][:],
                            model.variables['coordy'][:],
                            model.variables['connect1'][:],
                            powerIDW=powerIDW)
    elif level=='node':
        # no need to interpolate
        vals_node = vals[BD_node-1]
    else:
        raise ValueError('Unknown level: {}'.format(level))
            
    return mesh.x[BD_node-1], mesh.y[BD_node-1], vals_node
            
##############################################################################
# Get the chain of boundary nodes       
def find_node_chain(bd_elem):
    # define head and tail with the first row
    head = bd_elem[0][0]
    tail = bd_elem[0][1]
    # delete the first row
    bd_elem = np.delete(bd_elem, 0, axis=0)
    # allocate the node_chain
    node_chain = [head,tail]
    # do while bd_elem is not empty
    while len(bd_elem)>0:
        for i,bd_e in enumerate(bd_elem):
            if head in bd_e:
                if bd_e[0]==head:
                    head = bd_e[1]
                elif bd_e[1]==head:
                    head = bd_e[0]
                node_chain.insert(0,head)
                bd_elem = np.delete(bd_elem, i, axis=0)
                break
            elif tail in bd_e:
                if bd_e[0]==tail:
                    tail = bd_e[1]
                elif bd_e[1]==tail:
                    tail = bd_e[0]
                node_chain.append(tail)    
                bd_elem = np.delete(bd_elem, i, axis=0)
                break
    return np.array(node_chain)
    
    