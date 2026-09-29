def _lambdifygenerated(A_p, C_D, rho, u, v_i):
    return -0.5*A_p*C_D*rho*(-u + v_i)*abs(u - v_i)
