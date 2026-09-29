def _lambdifygenerated(F_N, Dummy_330, epsilon_L):
    return -epsilon_L*amax(numpy.asarray([0,-1 + F_N/Dummy_330]), axis=0) + 1
