def _lambdifygenerated(I_B, N, m_i, omega_B, p_i):
    return 0.5*I_B*omega_B*transpose + (builtins.sum((1/2)*p_i**2/m_i for i in range(1, N+1)))
