def _lambdifygenerated(H, K_u, M_s, mu_0, psi, theta):
    return -H*M_s*mu_0*cos(psi - theta) + K_u*sin(theta)**2
