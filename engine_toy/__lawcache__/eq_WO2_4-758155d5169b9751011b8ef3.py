def _lambdifygenerated(MC, MC_fsp, MC_ref, beta_T):
    return beta_T*(amin(numpy.asarray([MC,MC_fsp]), axis=0) - amin(numpy.asarray([MC_fsp,MC_ref]), axis=0))
