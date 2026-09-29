def _lambdifygenerated(E_L, E_R, E_T, nu_LT, nu_RT, sigma_L, sigma_R, sigma_T):
    return sigma_T/E_T - nu_RT*sigma_R/E_R - nu_LT*sigma_L/E_L
