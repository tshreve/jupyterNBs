import numpy as np
import gmsh
from scipy.interpolate import interp2d
from .basic import spheroid
import xarray as xr

# generate mesh for the 3D Yang model
# by Yan Zhan (2021)
# Input: meshname = string (mesh file name without '.msh')
#        param = 1d array for all parameter needed for the Yang 3d model
#              = [x0,y0,z0,r_vertical,r_horizontal,dipping,azimuth]
def gmsh_Yang3d(meshname, param):
    box3d_ellipsoid_void(name=meshname,
                        x0=param[0]*1e3,y0=param[1]*1e3,z0=param[2]*1e3,
                        r1=param[4]*1e3,r2=param[4]*1e3,r3=param[3]*1e3,
                        theta=param[5],phi=param[6],
                        xmin=-20e3,ymin=-20e3,zmin=-20e3,
                        xmax=20e3,ymax=20e3,zmax=0
                       )
    return 1


# Create 3d with an ellipsoidal void (2020/12/3)
# adding the feature that the ellipsoid can rotate (2021/9/7)
# adding the refinement around the pressure source (2021/9/13)
# adding the topography option (2021/9/25)
# name = file name
# center of the ellipsoid (x0, y0, z0) (m)
# note: z0 is negative if BSL
# ellipsoid's three axes in x, y, z directions 
# if no roatation: (r1, r2, r3) (m)
# angle between ellipsoid's z axis and global z: theta (rad)
# azimuth of the trace of ellipsoid's z axis: phi (rad)
# The range of the Box (x/y/z,min/max) (m)
# "refine" controls the refinement of the mesh
# If using the topography: 
# topo_x, topo_y: 1D Arrays defining the data point coordinates (regular).
# topo_Z: 2D array of elevation rank = len(topo_x)*len(topo_y) 
def box3d_ellipsoid_void(name='box3d.msh', x0=0, y0=0, z0=-3e3, 
                         r1=1e3, r2=500, r3=100,
                         theta=np.pi/3, phi=0,
                         xmin = -1e4, xmax = 1e4,
                         ymin = -1e4, ymax = 1e4,
                         zmin = -1e4, zmax = 0, refine=1., 
                         topo_utm = 0, 
                         order=2, refine_ball=True, chamber_refine=1.
                        ):

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("{}".format(name))

    # allocate for all surfaces
    shells = []

    # size of the mesh at corners
    lc = ((xmax-xmin)+(ymax-ymin)+(zmax-zmin))/30.

    # points
    gmsh.model.geo.addPoint(xmin, ymin, zmin, lc, 1)
    gmsh.model.geo.addPoint(xmax, ymin, zmin, lc, 2)
    gmsh.model.geo.addPoint(xmax, ymax, zmin, lc, 3)
    gmsh.model.geo.addPoint(xmin, ymax, zmin, lc, 4)
    gmsh.model.geo.addPoint(xmin, ymin, zmax, lc, 5)
    gmsh.model.geo.addPoint(xmax, ymin, zmax, lc, 6)
    gmsh.model.geo.addPoint(xmax, ymax, zmax, lc, 7)
    gmsh.model.geo.addPoint(xmin, ymax, zmax, lc, 8)

    # lines
    # bottom 4 lines
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    # top 4 lines
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 7, 6)
    gmsh.model.geo.addLine(7, 8, 7)
    gmsh.model.geo.addLine(8, 5, 8)
    # lateral 4 lines
    gmsh.model.geo.addLine(1, 5, 9)
    gmsh.model.geo.addLine(2, 6, 10)
    gmsh.model.geo.addLine(3, 7, 11)
    gmsh.model.geo.addLine(4, 8, 12)

    # faces
    # bottom
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 21)
    gmsh.model.geo.addPlaneSurface([21], 1)
    # left
    gmsh.model.geo.addCurveLoop([-4, 12, 8, -9], 22)
    gmsh.model.geo.addPlaneSurface([22], 2)
    # front
    gmsh.model.geo.addCurveLoop([-1, 9, 5, -10], 23)
    gmsh.model.geo.addPlaneSurface([23], 3)
    # right
    gmsh.model.geo.addCurveLoop([-2, 10, 6, -11], 24)
    gmsh.model.geo.addPlaneSurface([24], 4)
    # back
    gmsh.model.geo.addCurveLoop([-3, 11, 7, -12], 25)
    gmsh.model.geo.addPlaneSurface([25], 5)
    # top
    gmsh.model.geo.addCurveLoop([-5, -8, -7, -6], 26)
    gmsh.model.geo.addPlaneSurface([26], 6)

    # assemble 6 surface to one loop
    s1 = gmsh.model.geo.addSurfaceLoop([1, 2, 3, 4, 5, 6])
    shells.append(s1)

    # create the sphere
    #s2 = cheeseHole(x, y, z, r, lcar3)
    s2, chamber_surface = spheroid(x0, y0, z0, r1, r2, r3, theta, -phi, chamber_refine=chamber_refine)
    shells.append(s2)

    # create the box + sphere
    v1 = gmsh.model.geo.addVolume(shells, 1)

    # add to physical boundarys
    gmsh.model.addPhysicalGroup(2, [1], 101)
    gmsh.model.setPhysicalName(2, 101, "bottom")
    gmsh.model.addPhysicalGroup(2, [2, 4], 102)
    gmsh.model.setPhysicalName(2, 102, "leftright")
    gmsh.model.addPhysicalGroup(2, [3, 5], 103)
    gmsh.model.setPhysicalName(2, 103, "frontback")
    gmsh.model.addPhysicalGroup(2, [6], 104)
    gmsh.model.setPhysicalName(2, 104, "top")
    gmsh.model.addPhysicalGroup(2, chamber_surface, 201)
    gmsh.model.setPhysicalName(2, 201, "chamber")

    # add domain
    ps = gmsh.model.addPhysicalGroup(3, [v1])
    gmsh.model.setPhysicalName(3, ps, "domain")

    gmsh.model.geo.synchronize()

    
    if refine_ball:
        # refine the mesh *
        # The value of this field is VIn inside a spherical ball, 
        # VOut outside. The ball is defined by
        # ||dX||^2 < R^2 &&
        # dX = (X - XC)^2 + (Y-YC)^2 + (Z-ZC)^2
        # If Thickness is > 0, the mesh size is interpolated between 
        # VIn and VOut in a layer around the ball of the prescribed thickness.
        gmsh.model.mesh.field.add("Ball", 1)
        gmsh.model.mesh.field.setNumber(1, "VIn", min(r1,r2,r3)/refine*10)
        gmsh.model.mesh.field.setNumber(1, "VOut", lc)
        gmsh.model.mesh.field.setNumber(1, "Radius", -z0)
        gmsh.model.mesh.field.setNumber(1, "Thickness", -z0)
        gmsh.model.mesh.field.setNumber(1, "XCenter", x0)
        gmsh.model.mesh.field.setNumber(1, "YCenter", y0)
        gmsh.model.mesh.field.setNumber(1, "ZCenter", 0)
        gmsh.model.mesh.field.setAsBackgroundMesh(1)

    
    '''
    gmsh.model.mesh.field.add("Box", 2)
    gmsh.model.mesh.field.setNumber(2, "VIn", min(r1,r2,r3)/refine*2)
    gmsh.model.mesh.field.setNumber(2, "VOut", lc)
    gmsh.model.mesh.field.setNumber(2, "Thickness", -z0)
    gmsh.model.mesh.field.setNumber(2, "XMin", x0+z0)
    gmsh.model.mesh.field.setNumber(2, "XMax", x0-z0)
    gmsh.model.mesh.field.setNumber(2, "YMin", y0+z0)
    gmsh.model.mesh.field.setNumber(2, "YMax", y0-z0)
    gmsh.model.mesh.field.setNumber(2, "ZMin", 0)
    gmsh.model.mesh.field.setNumber(2, "ZMax", z0-max(r1,r2,r3))
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    '''
    
    '''
    # Say we would like to obtain mesh elements with size lc/30 near curve 2 and
    # point 5, and size lc elsewhere. To achieve this, we can use two fields:
    # "Distance", and "Threshold". We first define a Distance field (`Field[1]') on
    # points 5 and on curve 2. This field returns the distance to point 5 and to
    # (100 equidistant points on) curve 2.
    gmsh.model.mesh.field.add("Distance", 2)
    gmsh.model.mesh.field.setNumbers(2, "PointsList", [9])
    # We then define a `Threshold' field, which uses the return value of the
    # `Distance' field 1 in order to define a simple change in element size
    # depending on the computed distances
    #
    # SizeMax -                     /------------------
    #                              /
    #                             /
    #                            /
    # SizeMin -o----------------/
    #          |                |    |
    #        Point         DistMin  DistMax
    gmsh.model.mesh.field.add("Threshold", 3)
    gmsh.model.mesh.field.setNumber(3, "InField", 2)
    gmsh.model.mesh.field.setNumber(3, "SizeMin", min(r1,r2,r3)/refine)
    gmsh.model.mesh.field.setNumber(3, "SizeMax", lc)
    gmsh.model.mesh.field.setNumber(3, "DistMin", max(r1,r2,r3))
    gmsh.model.mesh.field.setNumber(3, "DistMax", -z0)
    gmsh.model.mesh.field.setAsBackgroundMesh(3)
    '''
    
    '''
    # Say we want to modulate the mesh element sizes using a mathematical function
    # of the spatial coordinates. We can do this with the MathEval field:
    gmsh.model.mesh.field.add("MathEval", 3)
    Ffd3a = "(((x{:+.2f})^2+(y{:+.2f})^2+(z{:+.2f})^2)^0.5".format(-x0,-y0,-z0)
    Ffd3b = "+((x{:+.2f})^2+(y{:+.2f})^2+z^2)^0.5{:+.2f})/10{:+.2f}".format(-x0,-y0,z0,min(r1,r2,r3))
    gmsh.model.mesh.field.setString(3, "F", Ffd3a+Ffd3b)
    gmsh.model.mesh.field.setAsBackgroundMesh(3)
    '''
    
    '''
    # Say we want to modulate the mesh element sizes using a mathematical function
    # of the spatial coordinates. We can do this with the MathEval field:
    gmsh.model.mesh.field.add("MathEval", 3)
    Ffd3 = "(((x{:+.2f})^2+(y{:+.2f})^2+(z{:+.2f})^2)^0.5/1e3+1)^1.2*{:.2f}".format(-x0,-y0,-z0/2,min(r1,r2,r3)/refine)
    gmsh.model.mesh.field.setString(3, "F", Ffd3)
    gmsh.model.mesh.field.setAsBackgroundMesh(3)
    '''
    
    
    # first generate a 2D mesh
    gmsh.model.mesh.generate(2)
    
    
    # if there is topography
    if type(topo_utm) == xr.core.dataarray.DataArray:
        # interpolate the topography
        f = interp2d(topo_utm.x, topo_utm.y, topo_utm.sel(band=1))
        # get the node information
        nodeTags, nodeCoords = gmsh.model.mesh.getNodesForPhysicalGroup(2,104)
        # get nodal x,y,z
        x = nodeCoords[0::3] * 1.
        y = nodeCoords[1::3] * 1.
        z = nodeCoords[2::3] * 1.
        # allocate the surface z coodinate
        znew = z * 0.
        # interpolate surdace z from the topography data
        for i in range(0, len(x)):
            znew[i] = f(x[i], y[i])

        # set the 2D mesh node coordinates
        for i in range(0, len(nodeTags)):
            gmsh.model.mesh.setNode(nodeTags[i], (x[i],y[i],znew[i]), parametricCoord=[])
    
    # now generate 3D mesh
    gmsh.model.mesh.generate(3)

    # save the mesh
    gmsh.write(name)
    gmsh.finalize()
    
    # reset order
    if order > 1:
        reset_order_3d(name, order)
    
    return 0

def reset_order_3d(name, order):
    gmsh.initialize()
    gmsh.open(name)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(order)
    gmsh.write(name)
    gmsh.finalize()
    
    return 0


'''
def box3d_ellipsoid_void_surf(name='box3d.msh', x0=0, y0=0, z0=-3e3, 
                         r1=1e3, r2=500, r3=100,
                         theta=np.pi/3, phi=0,
                         xmin = -1e4, xmax = 1e4,
                         ymin = -1e4, ymax = 1e4,
                         zmin = -1e4, zmax = 0, refine=1, 
                         topo_x = [], topo_y = [], topo_Z = [], 
                         order=2
                        ):

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("{}".format(name))

    # allocate for all surfaces
    shells = []

    # size of the mesh at corners
    lc = ((xmax-xmin)+(ymax-ymin)+(zmax-zmin))/30

    # points
    gmsh.model.geo.addPoint(xmin, ymin, zmin, lc/refine, 1)
    gmsh.model.geo.addPoint(xmax, ymin, zmin, lc/refine, 2)
    gmsh.model.geo.addPoint(xmax, ymax, zmin, lc/refine, 3)
    gmsh.model.geo.addPoint(xmin, ymax, zmin, lc/refine, 4)
    gmsh.model.geo.addPoint(xmin, ymin, zmax, lc/refine, 5)
    gmsh.model.geo.addPoint(xmax, ymin, zmax, lc/refine, 6)
    gmsh.model.geo.addPoint(xmax, ymax, zmax, lc/refine, 7)
    gmsh.model.geo.addPoint(xmin, ymax, zmax, lc/refine, 8)
    gmsh.model.geo.addPoint(x0, y0, 0, -z0/10./refine, 9)

    # lines
    # bottom 4 lines
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    # top 4 lines
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 7, 6)
    gmsh.model.geo.addLine(7, 8, 7)
    gmsh.model.geo.addLine(8, 5, 8)
    # lateral 4 lines
    gmsh.model.geo.addLine(1, 5, 9)
    gmsh.model.geo.addLine(2, 6, 10)
    gmsh.model.geo.addLine(3, 7, 11)
    gmsh.model.geo.addLine(4, 8, 12)
    # top diag line
    gmsh.model.geo.addLine(5, 9, 13)
    gmsh.model.geo.addLine(6, 9, 14)
    gmsh.model.geo.addLine(7, 9, 15)
    gmsh.model.geo.addLine(8, 9, 16)

    # faces
    # bottom
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 21)
    gmsh.model.geo.addPlaneSurface([21], 1)
    # left
    gmsh.model.geo.addCurveLoop([-4, 12, 8, -9], 22)
    gmsh.model.geo.addPlaneSurface([22], 2)
    # front
    gmsh.model.geo.addCurveLoop([-1, 9, 5, -10], 23)
    gmsh.model.geo.addPlaneSurface([23], 3)
    # right
    gmsh.model.geo.addCurveLoop([-2, 10, 6, -11], 24)
    gmsh.model.geo.addPlaneSurface([24], 4)
    # back
    gmsh.model.geo.addCurveLoop([-3, 11, 7, -12], 25)
    gmsh.model.geo.addPlaneSurface([25], 5)
    # top-1
    gmsh.model.geo.addCurveLoop([-5, 13, -14], 26)
    gmsh.model.geo.addPlaneSurface([26], 6)
    # top-2
    gmsh.model.geo.addCurveLoop([-6, 14, -15], 27)
    gmsh.model.geo.addPlaneSurface([27], 7)
    # top-3
    gmsh.model.geo.addCurveLoop([-7, 15, -16], 28)
    gmsh.model.geo.addPlaneSurface([28], 8)
    # top-4
    gmsh.model.geo.addCurveLoop([-8, 16, -13], 29)
    gmsh.model.geo.addPlaneSurface([29], 9)

    # assemble 6 surface to one loop
    s1 = gmsh.model.geo.addSurfaceLoop([1, 2, 3, 4, 5, 6, 7, 8, 9])
    shells.append(s1)

    # create the sphere
    #s2 = cheeseHole(x, y, z, r, lcar3)
    s2, chamber_surface = spheroid(x0, y0, z0, r1, r2, r3, theta, -phi, refine=1.)
    shells.append(s2)

    # create the box + sphere
    v1 = gmsh.model.geo.addVolume(shells, 1)

    # add to physical boundarys
    gmsh.model.addPhysicalGroup(2, [1], 101)
    gmsh.model.setPhysicalName(2, 101, "bottom")
    gmsh.model.addPhysicalGroup(2, [2, 4], 102)
    gmsh.model.setPhysicalName(2, 102, "leftright")
    gmsh.model.addPhysicalGroup(2, [3, 5], 103)
    gmsh.model.setPhysicalName(2, 103, "frontback")
    gmsh.model.addPhysicalGroup(2, [6,7,8,9], 104)
    gmsh.model.setPhysicalName(2, 104, "top")
    gmsh.model.addPhysicalGroup(2, chamber_surface, 201)
    gmsh.model.setPhysicalName(2, 201, "chamber")

    # add domain
    ps = gmsh.model.addPhysicalGroup(3, [v1])
    gmsh.model.setPhysicalName(3, ps, "domain")

    gmsh.model.geo.synchronize()
    
    # first generate a 2D mesh
    gmsh.model.mesh.generate(2)
    
    # if there is topography
    if len(topo_x) != 0:
        # interpolate the topography
        f = interp2d(topo_x, topo_y, topo_Z)
        # get the node information
        nodeTags, nodeCoords, parametricCoord = gmsh.model.mesh.getNodes()
        # get nodal x,y,z
        x = nodeCoords[0::3] * 1.
        y = nodeCoords[1::3] * 1.
        z = nodeCoords[2::3] * 1.
        # find surface nodes
        surf_x = x[z==0] * 1.
        surf_y = y[z==0] * 1.
        # allocate the surface z coodinate
        surf_z = surf_x * 0
        # interpolate surdace z from the topography data
        for i in range(0, len(surf_x)):
            surf_z[i] = f(surf_x[i], surf_y[i])

        # allocate the z coordinate
        znew = z * 1.
        # assign the new values into the orginal z
        znew[z==0] = surf_z * 1.
    
        # set the 2D mesh node coordinates
        for i in range(0, len(nodeTags)):
            gmsh.model.mesh.setNode(nodeTags[i], (x[i],y[i],znew[i]), parametricCoord)
    
    # now generate 3D mesh
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(order)    
    
    # save the mesh
    gmsh.write(name)

    gmsh.finalize()

    # print
    print('{} created'.format(name))
    return gmsh
'''