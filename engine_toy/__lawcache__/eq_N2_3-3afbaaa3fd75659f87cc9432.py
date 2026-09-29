def _lambdifygenerated(F_i, N, omega_B, tau_B, v_i):
    return omega_B*tau_B + (builtins.sum(F_i*v_i for i in range(1, N+1)))
