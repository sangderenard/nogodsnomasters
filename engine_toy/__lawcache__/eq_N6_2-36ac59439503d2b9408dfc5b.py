def _lambdifygenerated(D_e, a, r, r_e):
    return D_e*(1 - exp(-a*(r - r_e)))**2
