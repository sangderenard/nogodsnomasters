def _lambdifygenerated(E_L, E_R, E_T, nu_RL, nu_TL, sigma_L, sigma_R, sigma_T):
    return -nu_TL*sigma_T/E_T - nu_RL*sigma_R/E_R + sigma_L/E_L
