def _lambdifygenerated(Delta_theta, Dummy_339, Dummy_338):
    return select([greater(Delta_theta, (1/2)*Dummy_339),less(Delta_theta, -1/2*Dummy_339),True], [Dummy_338*(Delta_theta - 1/2*Dummy_339),Dummy_338*(Delta_theta + (1/2)*Dummy_339),0], default=nan)
