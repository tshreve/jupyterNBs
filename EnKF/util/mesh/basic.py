import gmsh
import numpy as np


# This function will create a spheroidal surface. We don't specify
# tags manually, and let the functions return them automatically:
# parameter:
# (x,y,z) location of the center
# r1, r2, r3, half-length of axises in x, y, z directions
# refine = refinement of the mesh dafault 1, element size ~ (r1+r2+r3)/15
# return:
# sph_volume = spheroid volume 3D
# sph_surface = spheroid surface for boundary condition
def spheroid(x0=0, y0=0, z0=-3e3, r1=1e3, r2=500, r3=100,
             theta=np.pi/3, phi=0, chamber_refine=1.):
    
    center = np.array([x0, y0, z0])
    loc1 = center + rotation(rotation(np.array([ 0,  0,  0]),theta,axis='x'),phi,axis='z')
    loc2 = center + rotation(rotation(np.array([ r1, 0,  0]),theta,axis='x'),phi,axis='z')
    loc3 = center + rotation(rotation(np.array([ 0, r2,  0]),theta,axis='x'),phi,axis='z')
    loc4 = center + rotation(rotation(np.array([ 0,  0, r3]),theta,axis='x'),phi,axis='z')
    loc5 = center + rotation(rotation(np.array([-r1, 0,  0]),theta,axis='x'),phi,axis='z')
    loc6 = center + rotation(rotation(np.array([ 0,-r2,  0]),theta,axis='x'),phi,axis='z')
    loc7 = center + rotation(rotation(np.array([ 0,  0,-r3]),theta,axis='x'),phi,axis='z')
    
    esize = min(r1,r2,r3)/chamber_refine
    
    p1 = gmsh.model.geo.addPoint(loc1[0], loc1[1], loc1[2], esize)
    p2 = gmsh.model.geo.addPoint(loc2[0], loc2[1], loc2[2], esize)
    p3 = gmsh.model.geo.addPoint(loc3[0], loc3[1], loc3[2], esize)
    p4 = gmsh.model.geo.addPoint(loc4[0], loc4[1], loc4[2], esize)
    p5 = gmsh.model.geo.addPoint(loc5[0], loc5[1], loc5[2], esize)
    p6 = gmsh.model.geo.addPoint(loc6[0], loc6[1], loc6[2], esize)
    p7 = gmsh.model.geo.addPoint(loc7[0], loc7[1], loc7[2], esize)

    c1 = gmsh.model.geo.addEllipseArc(p2, p1, p7, p7)
    c2 = gmsh.model.geo.addEllipseArc(p7, p1, p7, p5)
    c3 = gmsh.model.geo.addEllipseArc(p5, p1, p4, p4)
    c4 = gmsh.model.geo.addEllipseArc(p4, p1, p4, p2)
    c5 = gmsh.model.geo.addEllipseArc(p2, p1, p3, p3)
    c6 = gmsh.model.geo.addEllipseArc(p3, p1, p3, p5)
    c7 = gmsh.model.geo.addEllipseArc(p5, p1, p6, p6)
    c8 = gmsh.model.geo.addEllipseArc(p6, p1, p6, p2)
    c9 = gmsh.model.geo.addEllipseArc(p7, p1, p3, p3)
    c10 = gmsh.model.geo.addEllipseArc(p3, p1, p3, p4)
    c11 = gmsh.model.geo.addEllipseArc(p4, p1, p6, p6)
    c12 = gmsh.model.geo.addEllipseArc(p6, p1, p6, p7)

    l1 = gmsh.model.geo.addCurveLoop([c5, c10, c4])
    l2 = gmsh.model.geo.addCurveLoop([c9, -c5, c1])
    l3 = gmsh.model.geo.addCurveLoop([c12, -c8, -c1])
    l4 = gmsh.model.geo.addCurveLoop([c8, -c4, c11])
    l5 = gmsh.model.geo.addCurveLoop([-c10, c6, c3])
    l6 = gmsh.model.geo.addCurveLoop([-c11, -c3, c7])
    l7 = gmsh.model.geo.addCurveLoop([-c2, -c7, -c12])
    l8 = gmsh.model.geo.addCurveLoop([-c6, -c9, c2])

    # We need non-plane surfaces to define the spherical holes. Here we use the
    # `gmsh.model.geo.addSurfaceFilling()' function, which can be used for
    # surfaces with 3 or 4 curves on their boundary. With the he built-in
    # kernel, if the curves are circle arcs, ruled surfaces are created;
    # otherwise transfinite interpolation is used.
    #
    # With the OpenCASCADE kernel, `gmsh.model.occ.addSurfaceFilling()' uses a
    # much more general generic surface filling algorithm, creating a BSpline
    # surface passing through an arbitrary number of boundary curves. The
    # `gmsh.model.geo.addThruSections()' allows to create ruled surfaces (see
    # `t19.py').

    s1 = gmsh.model.geo.addSurfaceFilling([l1])
    s2 = gmsh.model.geo.addSurfaceFilling([l2])
    s3 = gmsh.model.geo.addSurfaceFilling([l3])
    s4 = gmsh.model.geo.addSurfaceFilling([l4])
    s5 = gmsh.model.geo.addSurfaceFilling([l5])
    s6 = gmsh.model.geo.addSurfaceFilling([l6])
    s7 = gmsh.model.geo.addSurfaceFilling([l7])
    s8 = gmsh.model.geo.addSurfaceFilling([l8])
    
    # create surface group
    sph_surface = [s1, s2, s3, s4, s5, s6, s7, s8]
    # create volume
    sph_volume = gmsh.model.geo.addSurfaceLoop(sph_surface)

    return sph_volume, sph_surface

# rotation by x,y, or z axis (Sep 2021)
# original vectors X = [[x1,y1,z1],[x2,y2,z2],...]
# output vectors X* = [[x1*,y1*,z1*],[x2*,y2*,z2*],...]
def rotation(X, theta, axis='x'):
    if axis=='x':
        RM = np.array([[ 1, 0            , 0            ],
                       [ 0, np.cos(theta),-np.sin(theta)],
                       [ 0, np.sin(theta), np.cos(theta)]])
    elif axis=='y':
        RM = np.array([[ np.cos(theta), 0, np.sin(theta)],
                       [ 0            , 1, 0            ],
                       [-np.sin(theta), 0, np.cos(theta)]])
    elif axis=='z':
        RM = np.array([[ np.cos(theta),-np.sin(theta), 0],
                       [ np.sin(theta), np.cos(theta), 0],
                       [ 0            , 0            , 1]])
    else:
        print('error: wrong axis setting')
    
    return np.matmul(RM, X.T)




