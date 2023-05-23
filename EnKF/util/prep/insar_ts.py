import numpy as np
import pandas as pd
from pyproj import Transformer
import h5py
import pyvista as pv

# projection transform (Ambrym)
LL2UTM = Transformer.from_crs("epsg:4326", "epsg:32759")

def read_hd5_tara(data_file_name,geom_file_name,lon_lim,lat_lim,
                  dsize=200,volcx=0,volcy=0,create_topo=True,create_plane=True):

    # open the h5 files
    h5dat = h5py.File(data_file_name, mode='r')
    h5geo = h5py.File(geom_file_name, mode='r')

    # get the dates of the data
    datestr = h5dat['date'][:].astype('U8')
    # convert it to time
    time = pd.to_datetime(datestr, format='%Y%m%d')
    # number of the time step
    nstep = len(datestr)
    # read timesereis 
    timeseries = h5dat['timeseries'][:]

    # read geometry
    latitude = h5geo['latitude'][:]
    longitude = h5geo['longitude'][:]
    height = h5geo['height'][:]
    # looking angle
    azimuthAngle = h5geo['azimuthAngle'][:]
    incidenceAngle = h5geo['incidenceAngle'][:]

    # get 1d array for InSAR data
    lat_1d = np.nanmean(latitude, axis=1)
    lon_1d = np.nanmean(longitude, axis=0)

    # find the index of the data boundary
    i_ymin = np.nanargmin(np.abs(lat_1d - lat_lim[0]))
    i_ymax = np.nanargmin(np.abs(lat_1d - lat_lim[1]))
    i_xmin = np.nanargmin(np.abs(lon_1d - lon_lim[0]))
    i_xmax = np.nanargmin(np.abs(lon_1d - lon_lim[1]))

    # crop lat & lon
    latitude = latitude[i_ymax:i_ymin, i_xmin:i_xmax] * 1
    longitude = longitude[i_ymax:i_ymin, i_xmin:i_xmax] * 1
    height  = height[i_ymax:i_ymin, i_xmin:i_xmax] * 1
    # crop time series data
    timeseries = timeseries[:,i_ymax:i_ymin, i_xmin:i_xmax] * 1
    # crop looking angle
    azimuthAngle = azimuthAngle[i_ymax:i_ymin, i_xmin:i_xmax] * 1
    incidenceAngle = incidenceAngle[i_ymax:i_ymin, i_xmin:i_xmax] * 1

    ## transform lat/lon data to UTM
    pdat0 = timeseries[0].reshape(-1)
    plon = longitude.reshape(-1)
    plat = latitude.reshape(-1)
    pAzm = azimuthAngle.reshape(-1)
    pInc = incidenceAngle.reshape(-1)
    modz = height.reshape(-1)
    # convert to UTM
    utmx,utmy = LL2UTM.transform(plat,plon)
    # model x and y (volcano centered)
    modx = utmx - volcx
    mody = utmy - volcy

    # if create 3D
    if create_topo:
        points_3d = np.vstack([modx,mody,modz*2]).T
        Inan = np.any(np.isnan([modx,mody,modz*2]),axis=0)
        point_cloud3d = pv.PolyData(points_3d[~Inan,:])
        for n in range(0, nstep):
            point_cloud3d['t{}'.format(n)] = timeseries[n].reshape(-1)[~Inan]

    # find nan value
    Inan = np.any(np.isnan([plon,plat,pdat0]),axis=0)
    # remove all nan values
    modx = modx[~Inan]
    mody = mody[~Inan]
    pAzm = pAzm[~Inan]
    pInc = pInc[~Inan]

    # generate a PyVista point cloud for the data in UTM
    points = np.vstack([modx,mody,np.zeros(modx.shape)]).T
    point_cloud = pv.PolyData(points)
    point_cloud.clear_data()
    # storing the time series into the point cloud
    for n in range(0, nstep):
        pdat = timeseries[n].reshape(-1)
        pdat = pdat[~Inan]
        point_cloud['t{}'.format(n)] = pdat
        if n>0:
            point_cloud['dt{}'.format(n)] = pdat - point_cloud['t{}'.format(n-1)]

    # storing the looking angle into the point cloud
    point_cloud['azm'] = pAzm
    point_cloud['inc'] = pInc
    # calculate looking angle vector
    point_cloud['lkag_x'] = np.cos(-pAzm/180*np.pi - np.pi/2) * np.sin(pInc/180*np.pi)
    point_cloud['lkag_y'] = -np.sin(-pAzm/180*np.pi - np.pi/2) * np.sin(pInc/180*np.pi)
    point_cloud['lkag_z'] = np.cos(pInc/180*np.pi)

    ## Create Uniform grid
    # create a new uniform grid for further processing
    Xnew, Ynew = np.meshgrid(np.arange(modx.min(),modx.max()+dsize,dsize),
                             np.arange(mody.min(),mody.max()+dsize,dsize))
    # flatten the grid
    px_new = Xnew.reshape(-1)
    py_new = Ynew.reshape(-1)
    # create a new point cloud
    points_new = np.vstack([px_new,py_new,np.zeros(px_new.shape)]).T
    point_cloud_new = pv.PolyData(points_new)
    point_cloud_new.clear_data()
    # Interpolate the data
    point_cloud_new = point_cloud_new.interpolate(point_cloud,radius=dsize*1.2,
                                                  strategy='null_value',null_value=np.nan)
    ### create 2D for the data
    if create_plane:
        # create a plane type data
        plane = pv.Plane(center=[(modx.min()+modx.max())/2,(mody.min()+mody.max())/2,0],
                 i_size=(modx.max()-modx.min()), i_resolution=int((modx.max()-modx.min())/dsize),
                 j_size=(mody.max()-mody.min()), j_resolution=int((mody.max()-mody.min())/dsize),
                 direction=[0,0,1],
                )
        plane.clear_data()
        plane = plane.interpolate(point_cloud,radius=dsize*1.2,strategy='null_value',null_value=np.nan)
    
    return_item = {'nstep':nstep, 'time':time, 't_str':datestr,
                   'orig_point': point_cloud, 'unif_point': point_cloud_new}
    if create_plane:
        return_item['plane'] = plane
    if create_topo:
        return_item['topo_point'] = point_cloud3d
    
    return return_item
    