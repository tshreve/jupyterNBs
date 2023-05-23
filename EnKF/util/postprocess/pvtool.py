import numpy as np
import pyvista as pv

def pv_yang_ellipsoid(RP):
    # create ellipsoid
    ep = pv.ParametricEllipsoid(xradius=RP[4]*RP[3]*1e3,yradius=RP[4]*RP[3]*1e3,zradius=RP[3]*1e3)
    # rotate
    ep = ep.rotate_x((np.pi/2-RP[5])/np.pi*180, inplace=False)
    ep = ep.rotate_z((-RP[6])/np.pi*180, inplace=False)
    # move
    ep.points[:,0] = ep.points[:,0] + RP[0]*1e3
    ep.points[:,1] = ep.points[:,1] + RP[1]*1e3
    ep.points[:,2] = ep.points[:,2] + RP[2]*1e3
    
    return ep