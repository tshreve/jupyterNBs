import sys
import os
import numpy as np

# run moose in python
# by Yan Zhan (2021)
# Input: software = path to the software
#        inputfile = name of the inputfile without ".i"
#        overwrite = string to overwrite the command of MOOSE
def run_moose(software, inputfile, overwrite, mpi=False, ncpu=4):
    # run MOOSE
    if mpi:
        # run moose on a multiple cores (ncpu), MOOSE environment is needed
        os.system('mpiexec -n {} {} -i {}.i {}'.format(ncpu,software,inputfile,overwrite))
    else:
        # run moose on a single core (DOF < 20,0000)
        os.system('{} -i {}.i {}'.format(software,inputfile,overwrite))
    return 1


# generate the overwrite for the input file
def overwrite(meshfile, outputfile, E=30e9, nu=0.25, density=0, dP=10e6):
    
    if '3d' in meshfile:
        zcomp = 'z'
    elif '2d' in meshfile:
        zcomp = 'y'
    else:
        raise ValueError('Add 2d/3d in the name of meshfile')
    
    g = 9.8
    # allocate overwritting string
    res = ''
    # add mesh file name
    res = ''.join((res, 'Mesh/file={} '.format(meshfile) ))
    # add outputfile name
    res = ''.join((res, 'Outputs/file_base={} '.format(outputfile)))
    # add Young's Modules
    res = ''.join((res, 'Materials/elasticity_tensor/youngs_modulus=\'{:.3e}\' '.format(E)))
    # add Poisson's Ratio
    res = ''.join((res, 'Materials/elasticity_tensor/poissons_ratio=\'{:.3f}\' '.format(nu)))
    if gravity:
        # add body force
        res = ''.join((res, 'Kernels/weight/value=\'{:.3e}\' '.format(-density*g)))
        # add initial stress
        res = ''.join((res, 'Functions/rhogh/value=\'{:.3e}*{}\' '.format(density*g,zcomp)))
        # add boundary load
        res = ''.join((res, 'Functions/boundary_pressure/value=\'{:.3e}*{}+{:.3e}\' '.format(-density*g,zcomp,dP)))
    else:
        # add boundary load
        res = ''.join((res, 'BCs/pressure_x/factor={:.3e} '.format(dP)))
        res = ''.join((res, 'BCs/pressure_y/factor={:.3e} '.format(dP)))
        res = ''.join((res, 'BCs/pressure_z/factor={:.3e} '.format(dP)))
        

    return res