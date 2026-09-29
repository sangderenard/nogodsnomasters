def _lambdifygenerated(G, m_i, m_j, x_i, x_j):
    return G*m_i*m_j*(-x_i + x_j)/abs(x_i - x_j)**3
