import numpy as np
import cmath

# pi
pi = np.pi


def mogi_disp(x, y, x0=0, y0=0, d=3e3, a=5e2, dP=100e6, E=75e9, nu=0.25):
    # Input
    # x: 1d array
    # y: 1d array (y = np.zeros(x.shape) if in axial-symmertic 2D model)
    # magma chamber location (x0, y0) [m]
    # magma chamber depth: d (downward posititve) [m]
    # magma chamber radius: a (always positive) [m]
    # overpressure: dP (expanding as positive) [Pa]
    # Young's Modulus: E [Pa]
    # Poisson's ration: nu [1]
    # Output
    # displacement in x, y, z direction = (du, dv, dw)

    # shear modulus
    mu = E / 2 / (1 + nu)
    # distance to the center (x0, y0)
    R = ((x - x0) ** 2 + (y - y0) ** 2) ** (1 / 2)
    # load term
    A = 3 * (a ** 3) * dP / (4 * mu)
    # displacement
    du = A * (x - x0) / ((d ** 2 + R ** 2) ** (3 / 2))
    dv = A * (y - y0) / ((d ** 2 + R ** 2) ** (3 / 2))
    dw = A * d / ((d ** 2 + R ** 2) ** (3 / 2))
    return du, dv, dw

def mogi_vol(x, y, x0=0, y0=0, d=3e3, dV=1e5, E=30e6, nu=0.25):
    # Input
    # x: 1d array
    # y: 1d array (y = np.zeros(x.shape) if in axial-symmertic 2D model)
    # magma chamber location (x0, y0) [m]
    # magma chamber depth: d (downward posititve) [m]
    # magma chamber radius: a (always positive) [m]
    # overpressure: dP (expanding as positive) [Pa]
    # Young's Modulus: E [Pa]
    # Poisson's ration: nu [1]
    # Output
    # displacement in x, y, z direction = (du, dv, dw)

    # distance to the center (x0, y0)
    R = ((x - x0) ** 2 + (y - y0) ** 2) ** (1 / 2)
    # load term
    A = dV * (1-nu) / pi
    # displacement
    du = A * (x - x0) / ((d ** 2 + R ** 2) ** (3 / 2))
    dv = A * (y - y0) / ((d ** 2 + R ** 2) ** (3 / 2))
    dw = A * d / ((d ** 2 + R ** 2) ** (3 / 2))
    return du, dv, dw

def mogi_vol_pool(enkf_x,enkf_y,RP):
    # calculate the model
    return mogi_vol(enkf_x, enkf_y,
                    x0=RP[0]*1e3, y0=RP[1]*1e3, d=-RP[2]*1e3,
                    dV=RP[3]*1e6,E=30e9, nu=0.25)

def Okada_cen2cor(param_cen):
    # get parameters
    xc = param_cen[:, 0, :]
    yc = param_cen[:, 1, :]
    zc = param_cen[:, 2, :]
    # r1: distance between Xi-Xf
    r1 = param_cen[:, 3, :]
    # r2: distance between Xt-Xb
    r2 = param_cen[:, 4, :]
    U = param_cen[:, 5, :]
    theta = param_cen[:, 6, :]  # degree
    # phi
    phi = param_cen[:, 7, :]  # degree
    # calculate Okada paramter:
    delta = theta
    alfa = np.pi / 2 - phi / 180 * np.pi
    beta = alfa + np.pi / 2
    xi = xc + r1 * np.cos(alfa) + r2 * np.cos(beta)
    xf = xc - r1 * np.cos(alfa) + r2 * np.cos(beta)
    yi = yc + r1 * np.sin(alfa) + r2 * np.sin(beta)
    yf = yc - r1 * np.sin(alfa) + r2 * np.sin(beta)
    zt = zc + r2 * np.sin(theta / 180 * np.pi)
    zb = zc - r2 * np.sin(theta / 180 * np.pi)

    PE_p = np.zeros(param_cen.shape)
    PE_p[:, 0, :] = xi
    PE_p[:, 1, :] = yi
    PE_p[:, 2, :] = xf
    PE_p[:, 3, :] = yf
    PE_p[:, 4, :] = zt
    PE_p[:, 5, :] = zb
    PE_p[:, 6, :] = U
    PE_p[:, 7, :] = delta

    return PE_p


# Okada_ddv
def Okada_ddv(xi, yi, xf, yf, zt, zb,
              U, delta, mu, nu,
              x, y, ddv):
    Ux, Uy, Uz, \
    dwdx, dwdy, eea, \
    gamma1, gamma2, \
    duxx, duxy, \
    duyx, duyy = Okada85("tensile",
                         xi, yi, xf, yf, zt, zb,
                         U, delta, mu, nu,
                         x, y
                         )

    disp = Ux * ddv[0, :] + Uy * ddv[1, :] + Uz * ddv[2, :]

    return disp


# rectangular dislocation Green's function (forward model)
# compute the displacement, strain and tilt due to a rectangular
# dislocation based on Okada (1985). All parameters are in SI units

# OUTPUT
# u         horizontal (East component) deformation
# v         horizontal (North component) deformation
# w         vertical (Up component) deformation
# dwdx      ground tilt (East component)
# dwdy      ground tilt (North component)
# eea       areal strain
# gamma1    shear strain
# gamma2    shear strain

# FAULT PARAMETERS
# fault     a string that define the kind of fault: strike, dip or tensile
# xi        x start
# yi        y start
# xf        x finish
# yf        y finish
# zt        top
#           (positive downward and defined as depth below the reference surface)
# zb        bottom; zb > zt
#           (positive downward and defined as depth below the reference surface)
# U         fault slip
#           strike slip fault: U > 0 right lateral strike slip
#           dip slip fault   : U > 0 reverse slip
#           tensile fault    : U > 0 tensile opening fault
# delta     dip angle from horizontal reference surface (pi/2 = vertical fault)
#           delta can be between 0 and pi/2 but must be different from zero!
# phi       strike angle from North (pi/2 = fault trace parallel to the East)

# CRUST PARAMETERS
# mu        shear modulus
# nu        Poisson's ratio

# GEODETIC BENCHMARKS
# x,y       benchmark location (must be COLUMN vectors)
# *************************************************************************
# Y. Okada (1985). Surface deformation due to shear and tensile faults in a
# half-space. Bull Seism Soc Am 75(4), 1135-1154.
# *************************************************************************
# Note:
# the displacement derivatives duxx duxy duyx duyy are presented only
# for testing purposes (check against Table 2 of Okada, 1985). The
# dimension of duxx duxy duyx duyy are m/km
def Okada85(fault,
            xi, yi, xf, yf, zt, zb,
            U, delta, mu, nu,
            x, y
            ):
    # [1] Set the parameters for the Okada (1985) dislocation model ***********
    # Lame's first parameter
    lamda = 2 * mu * nu / (1 - 2 * nu)
    # change fault geometry units from m to km
    xi = 0.001 * xi
    yi = 0.001 * yi
    xf = 0.001 * xf
    yf = 0.001 * yf
    zt = 0.001 * zt
    zb = 0.001 * zb

    # change bechmarks location units from m to km
    x = 0.001 * x
    y = 0.001 * y

    # check that dip angle delta is different from zero
    phi = atan2((xf - xi), (yf - yi))

    if abs(delta) < 1E-3:
        delta = 1E-3
        print('[Okada85] dip angle delta corrected')

    # L, W: length (L) in KM and width (W) in KM of the rectangular fault
    L = sqrt((xf - xi) ** 2 + (yi - yf) ** 2)
    if zb > zt:
        W = abs((zb - zt) / sin(delta))
    else:
        W = abs((zb - zt) / sin(delta))
        print('[Okada85] fault top zt deeper than fault bottom zb')

    # x0,y0:coordinates in KM of left lower corner of the rectangular fault
    # (see Figure 1; Okada, 1985)
    x0 = xi + (zb - zt) * cot(delta) * cos(phi)
    y0 = yi - (zb - zt) * cot(delta) * sin(phi)

    # z0: depth in KM of of left lower corner of the rectangular fault
    z0 = zb

    # translate the coordinates of the points where the displacement is computed
    # in the coordinates systen centered in (x0,y0)
    xxn = x - x0
    yyn = y - y0
    d = z0

    # rotate the coordinate system to be coherent with the model coordinate
    # system of Figure 1 (Okada, 1985)
    xxp = sin(phi) * xxn + cos(phi) * yyn
    yyp = -cos(phi) * xxn + sin(phi) * yyn

    # [2] Compute the displacement and displacement gradient matrix ***********
    # Okada (1985), equation (30)
    p = yyp * cos(delta) + d * sin(delta)
    q = yyp * sin(delta) - d * cos(delta)

    if fault == 'strike':
        # strike slip fault displacement and displacement gradient matrix. Okada (1985), equation (25)
        [ux1, uy1, uz1, duxx1, duxy1, duyx1, duyy1, duzx1, duzy1] = ok8525(xxp, p, q, mu, lamda, -U, delta)
        [ux2, uy2, uz2, duxx2, duxy2, duyx2, duyy2, duzx2, duzy2] = ok8525(xxp, p - W, q, mu, lamda, -U, delta)
        [ux3, uy3, uz3, duxx3, duxy3, duyx3, duyy3, duzx3, duzy3] = ok8525(xxp - L, p, q, mu, lamda, -U, delta)
        [ux4, uy4, uz4, duxx4, duxy4, duyx4, duyy4, duzx4, duzy4] = ok8525(xxp - L, p - W, q, mu, lamda, -U, delta)
    elif fault == 'dip':
        # dip slip fault displacement and displacement gradient matrix. Okada (1985), equation (26)
        [ux1, uy1, uz1, duxx1, duxy1, duyx1, duyy1, duzx1, duzy1] = ok8526(xxp, p, q, mu, lamda, U, delta)
        [ux2, uy2, uz2, duxx2, duxy2, duyx2, duyy2, duzx2, duzy2] = ok8526(xxp, p - W, q, mu, lamda, U, delta)
        [ux3, uy3, uz3, duxx3, duxy3, duyx3, duyy3, duzx3, duzy3] = ok8526(xxp - L, p, q, mu, lamda, U, delta)
        [ux4, uy4, uz4, duxx4, duxy4, duyx4, duyy4, duzx4, duzy4] = ok8526(xxp - L, p - W, q, mu, lamda, U, delta)
    else:
        # tensile fault displacement and displacement gradient matrix. Okada (1985), equation (27)
        [ux1, uy1, uz1, duxx1, duxy1, duyx1, duyy1, duzx1, duzy1] = ok8527(xxp, p, q, mu, lamda, U, delta)
        [ux2, uy2, uz2, duxx2, duxy2, duyx2, duyy2, duzx2, duzy2] = ok8527(xxp, p - W, q, mu, lamda, U, delta)
        [ux3, uy3, uz3, duxx3, duxy3, duyx3, duyy3, duzx3, duzy3] = ok8527(xxp - L, p, q, mu, lamda, U, delta)
        [ux4, uy4, uz4, duxx4, duxy4, duyx4, duyy4, duzx4, duzy4] = ok8527(xxp - L, p - W, q, mu, lamda, U, delta)

    # displacement, Chinnery's notation, Okada (1985), equation (24)
    Upx = (ux1 - ux2 - ux3 + ux4)
    Upy = (uy1 - uy2 - uy3 + uy4)

    # displacement gradient matrix, Chinnery's notation, Okada (1985), equation (24)
    Dpuxx = (duxx1 - duxx2 - duxx3 + duxx4)
    Dpuxy = (duxy1 - duxy2 - duxy3 + duxy4)
    Dpuyx = (duyx1 - duyx2 - duyx3 + duyx4)
    Dpuyy = (duyy1 - duyy2 - duyy3 + duyy4)
    Dpuzx = (duzx1 - duzx2 - duzx3 + duzx4)
    Dpuzy = (duzy1 - duzy2 - duzy3 + duzy4)

    # Rotate the horizontal displacement components Upx and Upy back
    u = sin(phi) * Upx - cos(phi) * Upy
    v = sin(phi) * Upy + cos(phi) * Upx
    w = (uz1 - uz2 - uz3 + uz4)

    # Rotate the displacement gradient matrix back
    duxx = Dpuxx * sin(phi) ** 2 - (Dpuxy + Dpuyx) * sin(phi) * cos(phi) + Dpuyy * cos(phi) ** 2
    duxy = Dpuxy * sin(phi) ** 2 + (Dpuxx - Dpuyy) * sin(phi) * cos(phi) - Dpuyx * cos(phi) ** 2
    duyx = Dpuyx * sin(phi) ** 2 + (Dpuxx - Dpuyy) * sin(phi) * cos(phi) - Dpuxy * cos(phi) ** 2
    duyy = Dpuyy * sin(phi) ** 2 + (Dpuxy + Dpuyx) * sin(phi) * cos(phi) + Dpuxx * cos(phi) ** 2
    dwdx = Dpuzx * sin(phi) - Dpuzy * cos(phi)  # ground tilt (East)
    dwdy = Dpuzy * sin(phi) + Dpuzx * cos(phi)  # ground tilt (North)

    # [3] Scale ground tilt ***************************************************
    dwdx = 0.001 * dwdx
    dwdy = 0.001 * dwdy

    # [4] Compute and scale shear and areal strain ****************************
    # Strains
    eea = 0.001 * (duxx + duyy)  # areal strain, equation (5)
    gamma1 = 0.001 * (duxx - duyy)  # shear strain
    gamma2 = 0.001 * (duxy + duyx)  # shear strain

    return u, v, w, dwdx, dwdy, eea, gamma1, gamma2, duxx, duxy, duyx, duyy


# function [ux uy uz duxx duxy duyx duyy duzx duzy ] = ok8525(csi,eta,q,mu,lambda,U1,delta)
# strike slip fault, Green's function (forward model) based on eq. (25) of
# Okada (1985). All paremeters are in SI units
# csi       dummy coordinate, defined in (22) and used in the integral (23)
# eta       dummy coordinate, defined in (22) and used in the integral (23)
# q         projected coordinate, see (12)
# mu        shear modulus
# lambda    Lame's first elastic parameter
# delta     dip angle from horizontal reference surface (90 degrees = vertical fault)
# U1        strike slip (U1 > 0 : left lateral strike slip)
# *************************************************************************
# Y. Okada (1985). Surface deformation due to shear and tensile faults in a
# half-space. Bull Seism Soc Am 75(4), 1135-1154.
# *************************************************************************
# ==========================================================================
# USGS Software Disclaimer
# The software and related documentation were developed by the U.S.
# Geological Survey (USGS) for use by the USGS in fulfilling its mission.
# The software can be used, copied, modified, and distributed without any
# fee or cost. Use of appropriate credit is requested.
#
# The USGS provides no warranty, expressed or implied, as to the correctness
# of the furnished software or the suitability for any purpose. The software
# has been tested, but as with any complex software, there could be undetected
# errors. Users who find errors are requested to report them to the USGS.
# The USGS has limited resources to assist non-USGS users; however, we make
# an attempt to fix reported problems and help whenever possible.
# ==========================================================================
def ok8525(csi, eta, q, mu, lamda, U1, delta):
    # GENERAL PARAMETERS
    alpha = mu / (lamda + mu)

    # Okada (1985), equation (30)
    ytilde = eta * cos(delta) + q * sin(delta)
    dtilde = eta * sin(delta) - q * cos(delta)
    R2 = csi ** 2 + ytilde ** 2 + dtilde ** 2
    R = sqrt(R2)
    X2 = csi ** 2 + q ** 2
    X = sqrt(X2)

    # check singularity condition (iii), pg 1148, Okada (1985)
    if abs(R + eta) < 1E-16:
        Reta = 0
        lnReta = -log(R - eta)
    else:
        Reta = 1 / (R + eta)
        lnReta = log(R + eta)

    # Okada (1985), equation (36)
    Ac = (2 * R + csi) / (R ** 3 * (R + csi) ** 2)
    An = (2 * R + eta) / (R ** 3 * (R + eta) ** 2)

    # Okada (1985), equation (40) and (41)
    # check singularity for cos(delta)=0
    if cos(delta) < 1E-16:  # Okada (1985), equation (41)
        K1 = alpha * csi * q / (R * (R + dtilde) ** 2)
        K3 = alpha * (sin(delta) / (R + dtilde)) * (csi ** 2 / (R * (R + dtilde)) - 1)
    else:
        K3 = alpha * (1 / cos(delta)) * (q * Reta / R - ytilde / (R * (R + dtilde)))
        K1 = alpha * (csi / cos(delta)) * (1 / (R * (R + dtilde)) - sin(delta) * Reta / R)

    K2 = alpha * (-sin(delta) / R + q * cos(delta) * Reta / R) - K3

    # *** DISPLACEMENT ********************************************************
    # check singularity for cos(delta)=0
    if cos(delta) < 1E-16:  # Okada (1985), equation (29)
        I1 = -0.5 * alpha * csi * q / (R + dtilde) ** 2
        I3 = 0.5 * alpha * (eta / (R + dtilde) + ytilde * q / (R + dtilde) ** 2 - lnReta)
        I4 = -alpha * q / (R + dtilde)
        I5 = -alpha * csi * sin(delta) / (R + dtilde)
    else:  # Okada (1985), equation (28)
        I4 = alpha * (1 / cos(delta)) * (log(R + dtilde) - sin(delta) * lnReta)
        if abs(csi) < 1E-16:  # check singularity condition (ii), pp 1148, Okada (1985)
            I5 = 0
        else:
            I5 = alpha * (2 / cos(delta)) * atan(
                (eta * (X + q * cos(delta)) + X * (R + X) * sin(delta)) / (csi * (R + X) * cos(delta)))

        I3 = alpha * ((1 / cos(delta)) * ytilde / (R + dtilde) - lnReta) + tan(delta) * I4
        I1 = alpha * ((-1 / cos(delta)) * (csi / (R + dtilde))) - tan(delta) * I5

    I2 = alpha * (-lnReta) - I3

    # strike slip displacement: Okada (1985), equation (25)
    if abs(q) < 1E-16:  # check singularity condition (i), pp 1148, Okada (1985)
        ux = -(U1 / (2 * pi)) * (csi * q * Reta / R + I1 * sin(delta))
    else:
        ux = -(U1 / (2 * pi)) * (csi * q * Reta / R + atan(csi * eta / (q * R)) + I1 * sin(delta))

    uy = -(U1 / (2 * pi)) * (ytilde * q * Reta / R + q * cos(delta) * Reta + I2 * sin(delta))
    uz = -(U1 / (2 * pi)) * (dtilde * q * Reta / R + q * sin(delta) * Reta + I4 * sin(delta))

    # *** STRAINS **************************************************************
    # check singularity for cos(delta)=0; # Okada (1985), equation (35) and (34)
    if cos(delta) < 1E-16:  # Okada (1985), equation (35)
        J1 = 0.5 * alpha * (q / (R + dtilde) ** 2) * (2 * csi ** 2 / (R * (R + dtilde)) - 1)
        J2 = 0.5 * alpha * (csi * sin(delta) / (R + dtilde) ** 2) * (2 * q ** 2 / (R * (R + dtilde)) - 1)
    else:  # # Okada (1985), equation (34)
        J1 = alpha * (1 / cos(delta)) * (csi ** 2 / (R * (R + dtilde) ** 2) - 1 / (R + dtilde)) - tan(delta) * K3
        J2 = alpha * (1 / cos(delta)) * (csi * ytilde / (R * (R + dtilde) ** 2)) - tan(delta) * K1

    J3 = alpha * (-csi * Reta / R) - J2
    J4 = alpha * (-cos(delta) / R - q * sin(delta) * Reta / R) - J1

    # strike slip strains: Okada (1985), equation (31)
    duxx = (U1 / (2 * pi)) * (csi ** 2 * q * An - J1 * sin(delta))
    duxy = (U1 / (2 * pi)) * (csi ** 3 * dtilde / (R ** 3 * (eta ** 2 + q ** 2)) - (csi ** 3 * An + J2) * sin(delta))
    duyx = (U1 / (2 * pi)) * (csi * q * cos(delta) / R ** 3 + (csi * q ** 2 * An - J2) * sin(delta))
    duyy = (U1 / (2 * pi)) * (ytilde * q * cos(delta) / R ** 3 + (
        q ** 3 * An * sin(delta) - 2 * q * sin(delta) * Reta / R - (csi ** 2 + eta ** 2) * cos(
            delta) / R ** 3 - J4) * sin(
        delta))
    # *************************************************************************


    # *** TILTS ***************************************************************
    # strike slip tilts: Okada (1985), equation (37)
    duzx = (U1 / (2 * pi)) * (-csi * q ** 2 * An * cos(delta) + (csi * q / R ** 3 - K1) * sin(delta))
    duzy = (U1 / (2 * pi)) * (dtilde * q * cos(delta) / R ** 3 + (
        csi ** 2 * q * An * cos(delta) - sin(delta) / R + ytilde * q / R ** 3 - K2) * sin(delta))
    # *************************************************************************

    return ux, uy, uz, duxx, duxy, duyx, duyy, duzx, duzy


# function [ux uy uz duxx duxy duyx duyy duzx duzy ] = ok8526(csi,eta,q,mu,lambda,U2,delta)
# dip slip fault, Green's function (forward model) based on eq. (26) of
# Okada (1985). All paremeters are in SI units
# csi       dummy coordinate, defined in (22) and used in the integral (23)
# eta       dummy coordinate, defined in (22) and used in the integral (23)
# q         projected coordinate, see (12)
# mu        shear modulus
# lambda    Lame's first elastic parameter
# delta     dip angle from horizontal reference surface (90 degrees = vertical fault)
# U2        dip slip (U2 > 0 : the southern section of the fault goes up)
# *************************************************************************
# Y. Okada (1985). Surface deformation due to shear and tensile faults in a
# half-space. Bull Seism Soc Am 75(4), 1135-1154.
# *************************************************************************
# ==========================================================================
# USGS Software Disclaimer
# The software and related documentation were developed by the U.S.
# Geological Survey (USGS) for use by the USGS in fulfilling its mission.
# The software can be used, copied, modified, and distributed without any
# fee or cost. Use of appropriate credit is requested.
#
# The USGS provides no warranty, expressed or implied, as to the correctness
# of the furnished software or the suitability for any purpose. The software
# has been tested, but as with any complex software, there could be undetected
# errors. Users who find errors are requested to report them to the USGS.
# The USGS has limited resources to assist non-USGS users however, we make
# an attempt to fix reported problems and help whenever possible.
# ==========================================================================
def ok8526(csi, eta, q, mu, lamda, U2, delta):
    # *** GENERAL PARAMETERS **************************************************
    # Okada (1985), equation (30)
    ytilde = eta * cos(delta) + q * sin(delta)
    dtilde = eta * sin(delta) - q * cos(delta)
    R2 = csi ** 2 + ytilde ** 2 + dtilde ** 2
    R = sqrt(R2)
    X2 = csi ** 2 + q ** 2
    X = sqrt(X2)

    alpha = mu / (lamda + mu)
    Rcsi = 1 / (R + csi)

    # check singularity condition (iii), pg 1148, Okada (1985)
    if abs(R + eta) < 1E-16:
        Reta = 0
        lnReta = -log(R - eta)
    else:
        Reta = 1 / (R + eta)
        lnReta = log(R + eta)

    # Okada (1985), equation (36)
    Ac = (2 * R + csi) / (R ** 3 * (R + csi) ** 2)
    An = (2 * R + eta) / (R ** 3 * (R + eta) ** 2)

    # Okada (1985), equation (40) and (41)
    # check singularity for cos(delta)=0
    if cos(delta) < 1E-16:  # Okada (1985), equation (41)
        K1 = alpha * csi * q / (R * (R + dtilde) ** 2)
        K3 = alpha * (sin(delta) / (R + dtilde)) * (csi ** 2 / (R * (R + dtilde)) - 1)
    else:
        K3 = alpha * (1 / cos(delta)) * (q * Reta / R - ytilde / (R * (R + dtilde)))
        K1 = alpha * (csi / cos(delta)) * (1 / (R * (R + dtilde)) - sin(delta) * Reta / R)

    K2 = alpha * (-sin(delta) / R + q * cos(delta) * Reta / R) - K3

    # *** DISPLACEMENT ********************************************************
    # check singularity for cos(delta)=0
    if cos(delta) < 1E-16:  # Okada (1985), equation (29)
        I1 = -0.5 * alpha * csi * q / (R + dtilde) ** 2
        I3 = 0.5 * alpha * (eta / (R + dtilde) + ytilde * q / (R + dtilde) ** 2 - lnReta)
        I4 = -alpha * q / (R + dtilde)
        I5 = -alpha * csi * sin(delta) / (R + dtilde)
    else:  # Okada (1985), equation (28)
        I4 = alpha * (1 / cos(delta)) * (log(R + dtilde) - sin(delta) * lnReta)
        if abs(csi) < 1E-16:  # check singularity condition (ii), pp 1148, Okada (1985)
            I5 = 0
        else:
            I5 = alpha * (2 / cos(delta)) * atan(
                (eta * (X + q * cos(delta)) + X * (R + X) * sin(delta)) / (csi * (R + X) * cos(delta)))

        I3 = alpha * ((1 / cos(delta)) * ytilde / (R + dtilde) - lnReta) + tan(delta) * I4
        I1 = alpha * ((-1 / cos(delta)) * (csi / (R + dtilde))) - tan(delta) * I5

    I2 = alpha * (-lnReta) - I3

    # dip slip displacement: Okada (1985), equation (26)
    ux = -(U2 / (2 * pi)) * (q / R - I3 * sin(delta) * cos(delta))
    if abs(q) < 1E-16:  # check singularity condition (i), pp 1148, Okada (1985)
        uy = -(U2 / (2 * pi)) * (ytilde * q * Rcsi / R + - I1 * sin(delta) * cos(delta))
        uz = -(U2 / (2 * pi)) * (dtilde * q * Rcsi / R + - I5 * sin(delta) * cos(delta))
    else:
        uy = -(U2 / (2 * pi)) * (
            ytilde * q * Rcsi / R + cos(delta) * atan(csi * eta / (q * R)) - I1 * sin(delta) * cos(delta))
        uz = -(U2 / (2 * pi)) * (
            dtilde * q * Rcsi / R + sin(delta) * atan(csi * eta / (q * R)) - I5 * sin(delta) * cos(delta))

    # *** STRAINS **************************************************************
    # check singularity for cos(delta)=0 # Okada (1985), equation (35) and (34)
    if cos(delta) < 1E-16:  # Okada (1985), equation (35)
        J1 = 0.5 * alpha * (q / (R + dtilde) ** 2) * (2 * csi ** 2 / (R * (R + dtilde)) - 1)
        J2 = 0.5 * alpha * (csi * sin(delta) / (R + dtilde) ** 2) * (2 * q ** 2 / (R * (R + dtilde)) - 1)
    else:  # # Okada (1985), equation (34)
        J1 = alpha * (1 / cos(delta)) * (csi ** 2 / (R * (R + dtilde) ** 2) - 1 / (R + dtilde)) - tan(delta) * K3
        J2 = alpha * (1 / cos(delta)) * (csi * ytilde / (R * (R + dtilde) ** 2)) - tan(delta) * K1

    J3 = alpha * (-csi * Reta / R) - J2
    J4 = alpha * (-cos(delta) / R - q * sin(delta) * Reta / R) - J1

    # dip slip strains: Okada (1985), equation (32)
    duxx = (U2 / (2 * pi)) * (csi * q / R ** 3 + J3 * sin(delta) * cos(delta))
    duxy = (U2 / (2 * pi)) * (ytilde * q / R ** 3 - sin(delta) / R + J1 * sin(delta) * cos(delta))
    duyx = (U2 / (2 * pi)) * (ytilde * q / R ** 3 + q * cos(delta) * Reta / R + J1 * sin(delta) * cos(delta))
    duyy = (U2 / (2 * pi)) * (
        ytilde ** 2 * q * Ac - (2 * ytilde * Rcsi / R + csi * cos(delta) * Reta / R) * sin(delta) + J2 * sin(
            delta) * cos(
            delta))

    # *** TILTS ***************************************************************
    # dip slip tilts: Okada (1985), equation (38)
    duzx = (U2 / (2 * pi)) * (dtilde * q / R ** 3 + q * sin(delta) * Reta / R + K3 * sin(delta) * cos(delta))
    duzy = (U2 / (2 * pi)) * (ytilde * dtilde * q * Ac -
                              (2 * dtilde * Rcsi / R + csi * sin(delta) * Reta / R) * sin(delta) + K1 * sin(
        delta) * cos(delta))

    return ux, uy, uz, duxx, duxy, duyx, duyy, duzx, duzy


# function [ux uy uz duxx duxy duyx duyy duzx duzy ] = ok8527(csi,eta,q,mu,lamda,U3,delta)
# tensile fault, Green's function (forward model) based on eq. (26) of
# Okada (1985). All paremeters are in SI units
# csi       dummy coordinate, defined in (22) and used in the integral (23)
# eta       dummy coordinate, defined in (22) and used in the integral (23)
# q         projected coordinate, see (12)
# mu        shear modulus
# lamda    Lame's first elastic parameter
# delta     dip angle from horizontal reference surface (90 degrees = vertical fault)
# U3        tensile opening (U3 > 0 : dike opening)
# *************************************************************************
# Y. Okada (1985). Surface deformation due to shear and tensile faults in a
# half-space. Bull Seism Soc Am 75(4), 1135-1154.
# *************************************************************************
# ==========================================================================
# USGS Software Disclaimer
# The software and related documentation were developed by the U.S.
# Geological Survey (USGS) for use by the USGS in fulfilling its mission.
# The software can be used, copied, modified, and distributed without any
# fee or cost. Use of appropriate credit is requested.
#
# The USGS provides no warranty, expressed or implied, as to the correctness
# of the furnished software or the suitability for any purpose. The software
# has been tested, but as with any complex software, there could be undetected
# errors. Users who find errors are requested to report them to the USGS.
# The USGS has limited resources to assist non-USGS users however, we make
# an attempt to fix reported problems and help whenever possible.
# ==========================================================================
def ok8527(csi, eta, q, mu, lamda, U3, delta):
    # *** GENERAL PARAMETERS **************************************************
    # Okada (1985), equation (30)
    ytilde = eta * cos(delta) + q * sin(delta)
    dtilde = eta * sin(delta) - q * cos(delta)
    R2 = csi ** 2 + ytilde ** 2 + dtilde ** 2
    R = sqrt(R2)
    X2 = csi ** 2 + q ** 2
    X = sqrt(X2)

    alpha = mu / (lamda + mu)
    Rcsi = 1 / (R + csi)

    # check singularity condition (iii), pg 1148, Okada (1985)
    # if abs(R + eta) < 1E-16:
    #     Reta = 0
    #     lnReta = -log(R - eta)
    # else:
    #
    Reta = 1 / (R + eta)
    lnReta = log(R + eta)
    Ic1148 = abs(R + eta) < 1E-16
    Reta[Ic1148] = 0
    lnReta[Ic1148] = -log(R[Ic1148] - eta[Ic1148])

    # Okada (1985), equation (36)
    Ac = (2 * R + csi) / (R ** 3 * (R + csi) ** 2)
    An = (2 * R + eta) / (R ** 3 * (R + eta) ** 2)

    # Okada (1985), equation (40) and (41)
    # check singularity for cos(delta)=0
    if cos(delta) < 1E-16:  # Okada (1985), equation (41)
        K1 = alpha * csi * q / (R * (R + dtilde) ** 2)
        K3 = alpha * (sin(delta) / (R + dtilde)) * (csi ** 2 / (R * (R + dtilde)) - 1)
    else:
        K3 = alpha * (1 / cos(delta)) * (q * Reta / R - ytilde / (R * (R + dtilde)))
        K1 = alpha * (csi / cos(delta)) * (1 / (R * (R + dtilde)) - sin(delta) * Reta / R)

    K2 = alpha * (-sin(delta) / R + q * cos(delta) * Reta / R) - K3

    # *** DISPLACEMENT ********************************************************
    # check singularity for cos(delta)=0
    if cos(delta) < 1E-16:  # Okada (1985), equation (29)
        I1 = -0.5 * alpha * csi * q / (R + dtilde) ** 2
        I3 = 0.5 * alpha * (eta / (R + dtilde) + ytilde * q / (R + dtilde) ** 2 - lnReta)
        I4 = -alpha * q / (R + dtilde)
        I5 = -alpha * csi * sin(delta) / (R + dtilde)
    else:  # Okada (1985), equation (28)
        I4 = alpha * (1 / cos(delta)) * (log(R + dtilde) - sin(delta) * lnReta)

        I5 = alpha * (2 / cos(delta)) * atan(
            (eta * (X + q * cos(delta)) + X * (R + X) * sin(delta)) / (csi * (R + X) * cos(delta)))
        # check singularity condition (ii), pp 1148, Okada (1985)
        I5[abs(csi) < 1E-16] = 0

        I3 = alpha * ((1 / cos(delta)) * ytilde / (R + dtilde) - lnReta) + tan(delta) * I4
        I1 = alpha * ((-1 / cos(delta)) * (csi / (R + dtilde))) - tan(delta) * I5

    I2 = alpha * (-lnReta) - I3

    # tensile displacement: Okada (1985), equation (27)
    ux = (U3 / (2 * pi)) * (q ** 2 * Reta / R - I3 * sin(delta) ** 2)
    # if abs(q) < 1E-16:  # check singularity condition (i), pp 1148, Okada (1985)
    uyI = (U3 / (2 * pi)) * (-dtilde * q * Rcsi / R - sin(delta) * (csi * q * Reta / R) - I1 * sin(delta) ** 2)
    uzI = (U3 / (2 * pi)) * (ytilde * q * Rcsi / R + cos(delta) * (csi * q * Reta / R) - I5 * sin(delta) ** 2)
    # else:
    uy = (U3 / (2 * pi)) * (
        -dtilde * q * Rcsi / R - sin(delta) * (csi * q * Reta / R - atan(csi * eta / (q * R))) - I1 * sin(
            delta) ** 2)
    uz = (U3 / (2 * pi)) * (
        ytilde * q * Rcsi / R + cos(delta) * (csi * q * Reta / R - atan(csi * eta / (q * R))) - I5 * sin(delta) ** 2)
    uy[abs(q) < 1E-16] = uyI[abs(q) < 1E-16]
    uz[abs(q) < 1E-16] = uzI[abs(q) < 1E-16]

    # *** STRAINS **************************************************************
    # check singularity for cos(delta)=0 # Okada (1985), equation (35) and (34)
    if cos(delta) < 1E-16:  # Okada (1985), equation (35)
        J1 = 0.5 * alpha * (q / (R + dtilde) ** 2) * (2 * csi ** 2 / (R * (R + dtilde)) - 1)
        J2 = 0.5 * alpha * (csi * sin(delta) / (R + dtilde) ** 2) * (2 * q ** 2 / (R * (R + dtilde)) - 1)
    else:  # # Okada (1985), equation (34)
        J1 = alpha * (1 / cos(delta)) * (csi ** 2 / (R * (R + dtilde) ** 2) - 1 / (R + dtilde)) - tan(delta) * K3
        J2 = alpha * (1 / cos(delta)) * (csi * ytilde / (R * (R + dtilde) ** 2)) - tan(delta) * K1

    J3 = alpha * (-csi * Reta / R) - J2
    J4 = alpha * (-cos(delta) / R - q * sin(delta) * Reta / R) - J1

    # tensile strains: Okada (1985), equation (33)
    duxx = -(U3 / (2 * pi)) * (csi * q ** 2 * An + J3 * sin(delta) ** 2)
    duxy = -(U3 / (2 * pi)) * (-dtilde * q / R ** 3 - csi ** 2 * q * An * sin(delta) + J1 * sin(delta) ** 2)
    duyx = -(U3 / (2 * pi)) * (q ** 2 * cos(delta) / R ** 3 + q ** 3 * An * sin(delta) + J1 * sin(delta) ** 2)
    duyy = -(U3 / (2 * pi)) * (
        (ytilde * cos(delta) - dtilde * sin(delta)) * q ** 2 * Ac - q * sin(2 * delta) * Rcsi / R - (
            csi * q ** 2 * An - J2) * sin(delta) ** 2)

    # *** TILTS ***************************************************************
    # tensile tilts: Okada (1985), equation (39)
    duzx = -(U3 / (2 * pi)) * (q ** 2 * sin(delta) / R ** 3 - q ** 3 * An * cos(delta) + K3 * sin(delta) ** 2)
    duzy = -(U3 / (2 * pi)) * (
        (ytilde * sin(delta) + dtilde * cos(delta)) * q ** 2 * Ac + csi * q ** 2 * An * sin(delta) * cos(delta) - (
            2 * q * Rcsi / R - K1) * sin(delta) ** 2)

    return ux, uy, uz, duxx, duxy, duyx, duyy, duzx, duzy


# frequently used functions
def sin(x):
    return np.sin(x)


def cos(x):
    return np.cos(x)


def tan(x):
    return sin(x) / cos(x)


def cot(x):
    return cos(x) / sin(x)


def log(x):
    return np.log(x)


def atan(x):
    return np.arctan(x)


def atan2(x, y):
    return np.arctan2(x, y)


def sqrt(x):
    return np.sqrt(x)
