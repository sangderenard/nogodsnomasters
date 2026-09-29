def _lambdifygenerated(Dummy_359, Dummy_358, Dummy_357, l, l_0):
    return select([greater(l, l_0),True], [Dummy_357*(l - l_0) + Dummy_358*Dummy_359,0], default=nan)
