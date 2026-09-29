def _lambdifygenerated(E_L, E_R, E_T, nu_LR, nu_TR, sigma_L, sigma_R, sigma_T):
    return -nu_TR*sigma_T/E_T + sigma_R/E_R - nu_LR*sigma_L/E_L
