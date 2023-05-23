# The Python API is entirely defined in the `gmsh.py' module (which contains the
# full documentation of all the functions in the API):
import gmsh
import numpy as np

# Create axisymmetric 2d with an elliptical void
# parameters (2020/12/3)
# name = file name
# zc = depth of the center of the ellipse (m) (negative)
# r1 = half height of the ellipse (m) (positive)
# r2 = half width of the ellipse (m) (positive)
# xmax = maximum distance of the box (m) (positive)
# zmax = maximum height of the box (m) (positive)
def axisymmetric2d_ellipse_void(name='Ellipse_A2d', 
                                zc=-3e3, r1=0.5e3, r2=1e3,
                                xmax=10e3, zmax=10e3, refine=1.,order=2):

    ## size of the corners
    # top left corner
    lc_tl = (-r1-zc)/10/refine
    # top right corner
    lc_tr = xmax/10/refine
    # bottom right corner
    lc_br = xmax/10/refine
    # bottom left corner
    lc_bl = (zmax + zc - r1)/10/refine


    # Before using any functions in the Python API, Gmsh must be initialized:
    gmsh.initialize()

    # By default Gmsh will not print out any messages: in order to output messages
    # on the terminal, just set the "General.Terminal" option to 1:
    gmsh.option.setNumber("General.Terminal", 1)

    # Next we add a new model named "t1" (if gmsh.model.add() is not called a new
    # unnamed model will be created on the fly, if necessary):
    gmsh.model.add("{}".format(name))

    # The Python API provides direct access to each supported geometry kernel. The
    # built-in kernel is used in this first tutorial: the corresponding API
    # functions have the `gmsh.model.geo' prefix.

    # The first type of `elementary entity' in Gmsh is a `Point'. To create a point
    # with the built-in geometry kernel, the Python API function is
    # gmsh.model.geo.addPoint():
    # - the first 3 arguments are the point coordinates (x, y, z)
    # - the next (optional) argument is the target mesh size (the "characteristic
    #   length") close to the point
    # - the last (optional) argument is the point tag (a stricly positive integer
    #   that uniquely identifies the point)
    # top left
    gmsh.model.geo.addPoint(0, 0, 0, lc_tl, 1)
    # top right
    gmsh.model.geo.addPoint(xmax, 0, 0, lc_tr, 2)
    # bottom right
    gmsh.model.geo.addPoint(xmax, -zmax, 0, lc_br, 3)
    # bottom left
    gmsh.model.geo.addPoint(0, -zmax, 0, lc_bl, 4)
    ## for ellipse
    # start
    gmsh.model.geo.addPoint(0, zc+r1, 0, r2/10/refine, 5)
    # center
    gmsh.model.geo.addPoint(0, zc, 0, (r1+r2)/20/refine, 6)
    # major axis
    gmsh.model.geo.addPoint(r2, zc, 0, r1/10/refine, 7)
    # end
    gmsh.model.geo.addPoint(0, zc-r1, 0, r2/10/refine, 8)

    # add line
    gmsh.model.geo.addLine(1, 5, 1)
    gmsh.model.geo.addEllipseArc(5,6,7,7,tag=2)
    gmsh.model.geo.addEllipseArc(7,6,7,8,tag=3)
    gmsh.model.geo.addLine(8, 4, 4)
    gmsh.model.geo.addLine(4, 3, 5)
    gmsh.model.geo.addLine(3, 2, 6)
    gmsh.model.geo.addLine(2, 1, 7)

    # boundarys
    gmsh.model.addPhysicalGroup(1, [5], 101)
    gmsh.model.setPhysicalName(1, 101, "bottom")
    gmsh.model.addPhysicalGroup(1, [6], 102)
    gmsh.model.setPhysicalName(1, 102, "right")
    gmsh.model.addPhysicalGroup(1, [2,3], 103)
    gmsh.model.setPhysicalName(1, 103, "chamber")
    gmsh.model.addPhysicalGroup(1, [7], 104)
    gmsh.model.setPhysicalName(1, 104, "top")

    # create surface
    gmsh.model.geo.addCurveLoop([1,2,3,4,5,6,7], 99)
    face = gmsh.model.geo.addPlaneSurface([99])
    ps = gmsh.model.addPhysicalGroup(2, [face])
    gmsh.model.setPhysicalName(2, ps, "domain")

    gmsh.model.geo.synchronize()

    # We can then generate a 2D mesh...
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)

    # ... and save it to disk
    gmsh.write("{}.msh".format(name))

    gmsh.finalize()
    
    # print
    print('{} created'.format(name))
    
    

# Create axisymmetric 2d with an elliptical void
# parameters (2020/12/3)
# name = file name
# zc = depth of the center of the ellipse (m) (negative)
# r1 = half height of the ellipse (m) (positive)
# r2 = half width of the ellipse (m) (positive)
# rc = radius of the conduit
# xmax = maximum distance of the box (m) (positive)
# zmax = maximum height of the box (m) (positive)
#
# Figure shows the id of points "1." and lines "(1)"
# 1.--------(8)--------2.
# |                    | 
#(1)                   |
# |                    |
# 5.------(2)------\   |
# |                |   |
# 6.               7.  |
# |                |   |
# |       /8.-(3)-/   (7)
# |    (4)             |
# |     |              |
# |     10. 9.         |
# |     |              |
#(9)    |              |
# |    (5)             |
# |     |              |
# |     |              |
# |     |              |
# 4.(10)11.----(6)-----3.

def axisymmetric2d_ellipse_conduit(name='Ellipse_A2d', 
                                zc=-3e3, r1=0.5e3, r2=1e3,
                                xmax=10e3, zmax=10e3, rc=100,refine=1.):

    
    ## size of the corners
    # top left corner
    lc_tl = (-r1-zc)/10
    # top right corner
    lc_tr = xmax/10
    # bottom right corner
    lc_br = xmax/10
    # bottom left corner
    lc_bl = (zmax + zc - r1)/10
    
    ## Find the fillet corner
    # slope
    kte = np.linspace(0,r1/r2*2,1000)
    # slope equation
    eq0 = rc*kte/np.sqrt(1+kte**2)+r2*kte/np.sqrt(kte**2+(r1/r2)**2)-2*rc
    # solve the equation
    imin = np.argmin(np.abs(eq0))
    # tangant point
    xte0 = r2*kte[imin]/np.sqrt(kte[imin]**2+(r1/r2)**2)
    yte0 = -xte0*(r1/r2)**2/kte[imin]
    # fillet circle center
    xfc0 = 2 * rc
    yfc0 = yte0 - rc/np.sqrt(1+kte[imin]**2)

    # Before using any functions in the Python API, Gmsh must be initialized:
    gmsh.initialize()

    # By default Gmsh will not print out any messages: in order to output messages
    # on the terminal, just set the "General.Terminal" option to 1:
    gmsh.option.setNumber("General.Terminal", 1)

    # Next we add a new model named "t1" (if gmsh.model.add() is not called a new
    # unnamed model will be created on the fly, if necessary):
    gmsh.model.add("{}".format(name))

    # The Python API provides direct access to each supported geometry kernel. The
    # built-in kernel is used in this first tutorial: the corresponding API
    # functions have the `gmsh.model.geo' prefix.

    # The first type of `elementary entity' in Gmsh is a `Point'. To create a point
    # with the built-in geometry kernel, the Python API function is
    # gmsh.model.geo.addPoint():
    # - the first 3 arguments are the point coordinates (x, y, z)
    # - the next (optional) argument is the target mesh size (the "characteristic
    #   length") close to the point
    # - the last (optional) argument is the point tag (a stricly positive integer
    #   that uniquely identifies the point)
    # top left
    gmsh.model.geo.addPoint(0, 0, 0, lc_tl*refine, 1)
    # top right
    gmsh.model.geo.addPoint(xmax, 0, 0, lc_tr*refine, 2)
    # bottom right
    gmsh.model.geo.addPoint(xmax, -zmax, 0, lc_br*refine, 3)
    # bottom left
    gmsh.model.geo.addPoint(0, -zmax, 0, rc/4*refine, 4)
    ## for ellipse
    # start
    gmsh.model.geo.addPoint(0, zc+r1, 0, rc/2*refine, 5)
    # center
    gmsh.model.geo.addPoint(0, zc, 0, (r1+r2)/20*refine, 6)
    # major axis
    gmsh.model.geo.addPoint(r2, zc, 0, r1/10*refine, 7)
    # end
    #gmsh.model.geo.addPoint(0, zc-r1, 0, r2/10, 8)
    
    ## Conduit
    # tang ellip
    gmsh.model.geo.addPoint(xte0, yte0+zc, 0, rc/2*refine, 8)
    # tang center
    gmsh.model.geo.addPoint(xfc0, yfc0+zc, 0, rc/2*refine, 9)
    # tang conduit
    gmsh.model.geo.addPoint(rc, yfc0+zc, 0, rc/4*refine, 10)
    # bottom
    gmsh.model.geo.addPoint(rc, -zmax, 0, rc/4*refine, 11)
    
    # add line
    gmsh.model.geo.addLine(1, 5, 1)
    
    gmsh.model.geo.addEllipseArc(5,6,7,7,tag=2)
    gmsh.model.geo.addEllipseArc(7,6,7,8,tag=3)
    gmsh.model.geo.addEllipseArc(8,9,10,10,tag=4)
    
    gmsh.model.geo.addLine(10, 11, 5)
    gmsh.model.geo.addLine(11, 3, 6)
    gmsh.model.geo.addLine(3, 2, 7)
    gmsh.model.geo.addLine(2, 1, 8)
    
    gmsh.model.geo.addLine(4, 5, 9)
    gmsh.model.geo.addLine(11, 4, 10)

    # boundarys
    gmsh.model.addPhysicalGroup(1, [6], 101)
    gmsh.model.setPhysicalName(1, 101, "solid_bottom")
    gmsh.model.addPhysicalGroup(1, [7], 102)
    gmsh.model.setPhysicalName(1, 102, "solid_right")
    gmsh.model.addPhysicalGroup(1, [8], 103)
    gmsh.model.setPhysicalName(1, 103, "solid_top")
    # liquid inlet
    gmsh.model.addPhysicalGroup(1, [10], 104)
    gmsh.model.setPhysicalName(1, 104, "liquid_inlet")
    # solid-liquid interface
    gmsh.model.addPhysicalGroup(1, [2,3,4,5], 105)
    gmsh.model.setPhysicalName(1, 105, "sl_interface")
    
    # left (no need)
    gmsh.model.addPhysicalGroup(1, [9], 106)
    gmsh.model.setPhysicalName(1, 106, "liquid_left")
    gmsh.model.addPhysicalGroup(1, [1], 107)
    gmsh.model.setPhysicalName(1, 107, "solid_left")
    
    # create surface
    gmsh.model.geo.addCurveLoop([1,2,3,4,5,6,7,8], 99)
    face1 = gmsh.model.geo.addPlaneSurface([99])
    ps1 = gmsh.model.addPhysicalGroup(2, [face1])
    gmsh.model.setPhysicalName(2, ps1, "domain_solid")
    
    gmsh.model.geo.addCurveLoop([10,9,2,3,4,5], 100)
    face2 = gmsh.model.geo.addPlaneSurface([100])
    ps2 = gmsh.model.addPhysicalGroup(2, [face2])
    gmsh.model.setPhysicalName(2, ps2, "domain_liquid")

    gmsh.model.geo.synchronize()

    # We can then generate a 2D mesh...
    gmsh.model.mesh.generate(2)

    # ... and save it to disk
    gmsh.write("{}.msh".format(name))

    gmsh.finalize()
    
    # print
    print('{} created'.format(name))