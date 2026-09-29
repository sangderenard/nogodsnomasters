def _lambdifygenerated(N, i, j, m_p, Dummy_349, Dummy_348, Dummy_347):
    return (builtins.sum(m_p*(Dummy_347**2*(1 if i == j else 0) - Dummy_348*Dummy_349) for p in range(1, N+1)))
