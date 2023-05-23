import matplotlib.colors as colors
import numpy as np
from matplotlib import cm

# Function for creating a shaded data with elevation data
def shade_data_elev(DATA_all, TOPO_all, cb, colormap='Spectral',
                    fraction=1, blend_mode='soft', vert_exag=50,dx=1,dy=1
                   ):
    # define colormap
    cmap = cm.get_cmap(colormap)
    # normalize data
    data_norm = (DATA_all + cb) / cb / 2
    # transfer 2D array to 1D array
    data_norm_1d = data_norm.flatten()
    # create rgb 1D array
    rgb_1d = np.zeros((data_norm_1d.shape[0], 3))
    Inonnan = np.where(~np.isnan(data_norm_1d))
    rgb_1d[Inonnan, :] = cmap(data_norm_1d[Inonnan])[:, :3]
    # reshape the array to 2D*rgb
    RGB_2d = np.reshape(rgb_1d, (data_norm.shape[0], data_norm.shape[1], 3))

    # Adding elevation as the intensity of the RGB map
    LS = colors.LightSource(azdeg=315, altdeg=45)
    RGB_2d_shade = LS.shade_rgb(RGB_2d, TOPO_all, fraction=fraction, 
                                blend_mode=blend_mode,vert_exag=vert_exag,
                                dx=dx,dy=dy
                               )
    # mask the nan value as grey shaded topography
    norm_elev = colors.Normalize(np.nanmin(TOPO_all), np.nanmax(TOPO_all))
    elev = norm_elev(TOPO_all)
    rgb_mask = LS.hillshade(elev, vert_exag=50)
    # mask NaN
    RGB_2d_shade[np.isnan(data_norm), 0] = rgb_mask[np.isnan(data_norm)]
    RGB_2d_shade[np.isnan(data_norm), 1] = rgb_mask[np.isnan(data_norm)]
    RGB_2d_shade[np.isnan(data_norm), 2] = rgb_mask[np.isnan(data_norm)]
    
    return RGB_2d_shade