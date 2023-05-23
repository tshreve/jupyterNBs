import rioxarray as rx
import pyvista as pv
import numpy as np

# read DEM model
def read_geotiff(filename,LL2UTM,center=[0,0],dinv=1):
    # Read in the data
    data = rx.open_rasterio(filename)
    values = np.asarray(data)[0,::dinv,::dinv]
    # Make a mesh
    lon, lat = np.meshgrid(data['x'][::dinv], data['y'][::dinv])
    # transform to the UTM
    utmx, utmy = LL2UTM.transform(lat,lon)
    # create pyvista structure
    mesh = pv.StructuredGrid(utmx-center[0], utmy-center[1], np.zeros(lon.shape))
    mesh['data'] = values.ravel(order='F')

    return mesh

# read NASA STRM hgt file
def read_hgt(hgt_name):

    fn = 'data/{}.hgt'.format(hgt_name)

    if hgt_name[0] == 'N':
        lat_min = float(hgt_name[1:3])
    else:
        lat_min = -float(hgt_name[1:3])
    lat_max = lat_min + 1.

    if hgt_name[3] == 'E':
        lon_min = float(hgt_name[4:7])
    else:
        lon_min = -float(hgt_name[4:7])
    lon_max = lon_min + 1.

    siz = os.path.getsize(fn)
    dim = int(math.sqrt(siz/2))
    assert dim*dim*2 == siz, 'Invalid file size'

    Lon = np.linspace(lon_min,lon_max,dim)
    Lat = np.linspace(lat_max,lat_min,dim)
    data = numpy.fromfile(fn, numpy.dtype('>i2'), dim*dim).reshape((dim, dim))

    ds = xr.DataArray(data=data,
                      dims=['y','x'],
                      coords={'y':Lat,'x': Lon}
                     )
    
    return ds