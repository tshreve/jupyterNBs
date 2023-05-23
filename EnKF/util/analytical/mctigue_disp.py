# McTigue (1988) model
def mctigue_disp(x, y, x0=0, y0=0, d=5e3, a=5e2, dP=100e6, E=30e9, nu=0.25):
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
    # dimensionless excess pressure (pressure/shear modulus)
    P_G = dP / mu
    # coordinates system centered in (x0,y0)
    xxn = x - x0
    yyn = y - y0
    # radial distance from source center
    r = (xxn ** 2 + yyn ** 2) ** (1/2)
    # dimensionless coordinates
    csi = xxn / d
    psi = yyn / d
    rho = r / d
    e = a / d
    # constant and expression used in the formulas 
    f1 = 1 / ((rho ** 2 + 1) ** 1.5)
    f2 = 1 / ((rho ** 2 + 1) ** 2.5)
    c1 = e ** 3 / (7 - 5 * nu)
    # displacement (dimensionless) [McTigue (1988), eq. (52) and (53)] 
    uzbar = e ** 3 * (1 - nu) * f1 * (1 - c1 * (0.5 * (1 + nu) - 3.75 * (2 - nu)/(rho ** 2 + 1)))
    urbar = rho * uzbar

    # displacement (dimensional) 
    u = urbar * P_G * d * xxn / (r + 1e-16)
    v = urbar * P_G * d * yyn / (r + 1e-16)
    w = uzbar * P_G * d
    return u, v, w