def _lambdifygenerated(K_tc, K_te, b_cut, h_chip):
    return b_cut*(K_tc*h_chip + K_te)
