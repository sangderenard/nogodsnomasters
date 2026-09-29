def _lambdifygenerated(J_k, N, omega_k, Dummy_342):
    return (builtins.sum(J_k*omega_k**2/Dummy_342**2 for k in range(1, N+1)))
