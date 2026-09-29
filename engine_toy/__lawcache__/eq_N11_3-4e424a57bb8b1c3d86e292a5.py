def _lambdifygenerated(Dummy_362, Dummy_361, Dummy_360):
    return select([less(Dummy_360, 0),True], [-Dummy_360*Dummy_362,-Dummy_360*Dummy_361], default=nan)
