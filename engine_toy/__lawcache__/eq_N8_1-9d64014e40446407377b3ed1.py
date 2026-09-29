def _lambdifygenerated(delta_S, mu_k, mu_s, v_S, Dummy_324):
    return mu_k + (-mu_k + mu_s)*exp(-abs(Dummy_324/v_S)**delta_S)
