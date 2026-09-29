def _lambdifygenerated(c_ij, k_ij, Dummy_317, n, r, v_parallel):
    return n*(c_ij*v_parallel + k_ij*(-Dummy_317 + r))
