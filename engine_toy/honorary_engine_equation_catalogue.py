import sympy as sp

t, x, y, z = sp.symbols('t x y z')
nabla = sp.Function('nabla')  # Symbolic placeholder for del operator where explicit coords are omitted
nabla_op = sp.Symbol('nabla')  # same operator used bare (e.g. -i*hbar*nabla, nabla**2), never applied to a field
# The catalogue keeps rotation matrices, deformation gradients and density
# matrices as plain scalar Symbol/Function objects rather than MatrixSymbol,
# so genuine matrix operators (.T, sp.Determinant, sp.Trace) raise on them.
# These three placeholders are the same "operator whose exact type is
# omitted" idiom nabla already uses above, applied to transpose/det/trace.
transpose = sp.Function('transpose')
det = sp.Function('det')
tr = sp.Function('tr')
cross = sp.Function('cross')  # a x b, since plain Symbol/Function has no vector algebra

# =====================================================================
# 1. NEWTON
# =====================================================================
m_i, v_i, p_i, F_i, x_i = sp.symbols('m_i v_i p_i F_i x_i', cls=sp.Function)
R, omega_B, I_B, tau_B = sp.symbols('R omega_B I_B tau_B', cls=sp.Function)
# N1
eq_N1_1 = sp.Eq(sp.Derivative(x_i(t), t), p_i(t)/m_i(t))
eq_N1_2 = sp.Eq(sp.Derivative(p_i(t), t), F_i(t))
eq_N1_3 = sp.Eq(sp.Derivative(R(t), t), R(t) * sp.MatrixSymbol('omega_B_cross', 3, 3))
eq_N1_4 = sp.Eq(transpose(R(t)) * R(t), sp.Identity(3), evaluate=False)
eq_N1_5 = sp.Eq(det(R(t)), 1)
eq_N1_6 = sp.Eq(I_B(t)*sp.Derivative(omega_B(t), t) + cross(omega_B(t), I_B(t)*omega_B(t)), tau_B(t))

# N2
K, U, L, H_ham, q_a, p_a, Q_nc = sp.symbols('K U L H q_a p_a Q^nc', cls=sp.Function)
eq_N2_1 = sp.Eq(K(t), sp.Sum(p_i(t)**2/(2*m_i(t)), ('i', 1, sp.Symbol('N'))) + 0.5*(transpose(omega_B(t)) * I_B(t) * omega_B(t)))
eq_N2_2 = sp.Eq(sp.Symbol('F_cons_i'), -sp.Derivative(U(t), x_i(t)))
eq_N2_3 = sp.Eq(sp.Derivative(K(t), t), sp.Sum(F_i(t)*v_i(t), ('i', 1, sp.Symbol('N'))) + tau_B(t)*omega_B(t))
eq_N2_4 = sp.Eq(sp.Derivative(sp.Derivative(L(t), sp.Derivative(q_a(t), t)), t) - sp.Derivative(L(t), q_a(t)), Q_nc(t))
eq_N2_5 = sp.Eq(L(t), K(t) - U(t))
eq_N2_6 = sp.Eq(sp.Derivative(q_a(t), t), sp.Derivative(H_ham(t), p_a(t)))
eq_N2_7 = sp.Eq(sp.Derivative(p_a(t), t), -sp.Derivative(H_ham(t), q_a(t)) + Q_nc(t))

# N3
k_ij, r_ij, l0_ij, c_ij, v_para = sp.symbols('k_ij r l_{0;ij} c_ij v_parallel')
n_vec = sp.Symbol('n')
eq_N3_1 = sp.Eq(sp.Symbol('U_ij'), 0.5 * k_ij * (r_ij - l0_ij)**2)
eq_N3_2 = sp.Eq(sp.Symbol('F_i'), (k_ij*(r_ij - l0_ij) + c_ij*v_para)*n_vec)
eq_N3_3 = sp.Eq(sp.Symbol('F_j'), -sp.Symbol('F_i'))
eq_N3_4 = sp.Eq(sp.Symbol('P_diss'), c_ij * v_para**2)

# N4
G, m_j, x_j, Phi, rho, V, g_vec, C_D, A_p, u = sp.symbols('G m_j x_j Phi rho V g C_D A_p u')
eq_N4_1 = sp.Eq(sp.Symbol('F_{i<-j}'), G*m_i(t)*m_j*(x_j - x_i(t))/sp.Abs(x_j - x_i(t))**3)
eq_N4_2 = sp.Eq(sp.Derivative(Phi, x, 2) + sp.Derivative(Phi, y, 2) + sp.Derivative(Phi, z, 2), 4*sp.pi*G*rho)
eq_N4_3 = sp.Eq(sp.Symbol('F'), -m_i(t)*nabla(Phi))
eq_N4_4 = sp.Eq(sp.Symbol('F_b'), -rho*V*g_vec)
eq_N4_5 = sp.Eq(sp.Symbol('F_d'), -0.5*rho*C_D*A_p*sp.Abs(v_i(t) - u)*(v_i(t) - u))

# N5
g_n, lambda_n, lambda_t, mu_f, v_t = sp.symbols('g_n lambda_n lambda_t mu_f v_t')
e_r, vn_m, vn_p, J_n, ma, mb, Ia, Ib, ra, rb = sp.symbols('e_r v_n^- v_n^+ J_n m_a m_b I_a I_b r_a r_b')
E_s, R_s, delta, nu_a, nu_b, Ea, Eb, Ra, Rb = sp.symbols('E^* R^* delta nu_a nu_b E_a E_b R_a R_b')
eq_N5_1 = g_n >= 0
eq_N5_2 = lambda_n >= 0
eq_N5_3 = sp.Eq(g_n * lambda_n, 0)
eq_N5_4 = sp.Abs(lambda_t) <= mu_f * lambda_n
eq_N5_5 = sp.Eq(lambda_t, -mu_f * lambda_n * v_t / sp.Abs(v_t))
eq_N5_6 = sp.Eq(vn_p, -e_r * vn_m)
eq_N5_7 = sp.Eq(J_n, -(1+e_r)*vn_m / (1/ma + 1/mb + (ra*n_vec)**2/Ia + (rb*n_vec)**2/Ib)) # Simplified cross product term as scalar rep
eq_N5_8 = sp.Eq(sp.Symbol('F_n'), (4/3)*E_s*sp.sqrt(R_s)*delta**(3/2))
eq_N5_9 = sp.Eq(1/E_s, (1-nu_a**2)/Ea + (1-nu_b**2)/Eb)
eq_N5_10 = sp.Eq(1/R_s, 1/Ra + 1/Rb)

# N6
kb, rb_v, rb0, ka, ta, ta0, kd, nd, phid, deld, eps_ij, sig_ij, q_i, q_j, eps_c = sp.symbols('k_b r_b r_b^0 k_a theta_a theta_a^0 k_d n_d phi_d delta_d epsilon_ij sigma_ij q_i q_j epsilon')
De, a_m, r_e = sp.symbols('D_e a r_e')
eq_N6_1 = sp.Eq(sp.Symbol('U(R)'), sp.Sum((kb/2)*(rb_v-rb0)**2, ('b', 1, sp.Symbol('N'))) + sp.Sum((ka/2)*(ta-ta0)**2, ('a', 1, sp.Symbol('N'))) + sp.Sum(kd*(1+sp.cos(nd*phid-deld)), ('d', 1, sp.Symbol('N'))) + sp.Sum(4*eps_ij*((sig_ij/r_ij)**12 - (sig_ij/r_ij)**6) + q_i*q_j/(4*sp.pi*eps_c*r_ij), ('ij', 1, sp.Symbol('N'))))
eq_N6_2 = sp.Eq(sp.Symbol('U_M(r)'), De*(1 - sp.exp(-a_m*(r_ij - r_e)))**2)

# N7
g_q = sp.Function('g')(q_a(t), t)
eq_N7_1 = sp.Eq(g_q, 0)
m_var, m_in, m_out, v_in, v_out, F_ext = sp.symbols('m m_in m_out v_in v_out F_ext')
eq_N7_2 = sp.Eq(sp.Derivative(m_var*v_i(t), t), F_ext + sp.Sum(m_in*v_in, ('idx_in', 1, sp.Symbol('N'))) - sp.Sum(m_out*v_out, ('out', 1, sp.Symbol('N'))))

# =====================================================================
# 2. TIMOSHENKO
# =====================================================================
gamma, w, theta_t, kappa_b, Q_t, M_t, kappa_s, G_t, A_t, E_t, I_t = sp.symbols('gamma w theta kappa_b Q M kappa_s G A E I', cls=sp.Function)
# T1
eq_T1_1 = sp.Eq(gamma(x,t), sp.Derivative(w(x,t), x) - theta_t(x,t))
eq_T1_2 = sp.Eq(kappa_b(x,t), sp.Derivative(theta_t(x,t), x))
eq_T1_3 = sp.Eq(Q_t(x,t), kappa_s(x)*G_t(x)*A_t(x)*gamma(x,t))
eq_T1_4 = sp.Eq(M_t(x,t), E_t(x)*I_t(x)*kappa_b(x,t))

# T2
rho_t, q_t, c_t = sp.symbols('rho q c', cls=sp.Function)
eq_T2_1 = sp.Eq(rho_t(x)*A_t(x)*sp.Derivative(w(x,t), t, 2), sp.Derivative(Q_t(x,t), x) + q_t(x,t))
eq_T2_2 = sp.Eq(rho_t(x)*I_t(x)*sp.Derivative(theta_t(x,t), t, 2), sp.Derivative(M_t(x,t), x) + Q_t(x,t) + c_t(x,t))

# T3
u_t, Ip_t, varphi_t, J_t, f_x, m_x = sp.symbols('u I_p varphi J f_x m_x', cls=sp.Function)
eq_T3_1 = sp.Eq(rho_t(x)*A_t(x)*sp.Derivative(u_t(x,t), t, 2), sp.Derivative(E_t(x)*A_t(x)*sp.Derivative(u_t(x,t), x), x) + f_x(x,t))
eq_T3_2 = sp.Eq(rho_t(x)*Ip_t(x)*sp.Derivative(varphi_t(x,t), t, 2), sp.Derivative(G_t(x)*J_t(x)*sp.Derivative(varphi_t(x,t), x), x) + m_x(x,t))

# T4
U_b, N_t, B_s, C_s, K_e, M_e, q_vec, C_mat, f_int, f_ext = sp.symbols('U_b N B_s C_s K_e M_e q_vec C f_int f_ext')
eq_T4_1 = sp.Eq(U_b, 0.5 * sp.Integral(E_t(x)*A_t(x)*sp.Derivative(u_t(x,t),x)**2 + E_t(x)*I_t(x)*sp.Derivative(theta_t(x,t),x)**2 + kappa_s(x)*G_t(x)*A_t(x)*(sp.Derivative(w(x,t),x)-theta_t(x,t))**2 + G_t(x)*J_t(x)*sp.Derivative(varphi_t(x,t),x)**2, x))
eq_T4_2 = sp.Eq(K_e, sp.Integral(B_s * C_s * B_s, x)) # Transpose implied
eq_T4_3 = sp.Eq(M_e, sp.Integral(N_t * rho_t(x) * N_t, x))
eq_T4_4 = sp.Eq(sp.Symbol('M') * sp.Derivative(q_vec, t, 2) + C_mat * sp.Derivative(q_vec, t) + f_int, f_ext)

# T5
K_mat, phi_k, omega_k, a_k, zeta_k, alpha_M, beta_K = sp.symbols('K phi_k omega_k a_k zeta_k alpha_M beta_K')
eq_T5_1 = sp.Eq(K_mat * phi_k, omega_k**2 * sp.Symbol('M') * phi_k)
eq_T5_2 = sp.Eq(phi_k * sp.Symbol('M') * phi_k, sp.KroneckerDelta(1, 1)) # Delta ij abstraction
eq_T5_3 = sp.Eq(sp.Derivative(a_k, t, 2) + 2*zeta_k*omega_k*sp.Derivative(a_k, t) + omega_k**2 * a_k, phi_k * f_ext)
eq_T5_4 = sp.Eq(C_mat, alpha_M * sp.Symbol('M') + beta_K * K_mat)

# T6
K_tan, K_mat_part, K_geo, P_cr, K_L, L_len, eps_th, alpha_T, T_temp, T_0, sig_t, eps_t, eps_p = sp.symbols('K_tangent K_material K_geometric P_cr K_L L epsilon^th alpha_T T T_0 sigma epsilon epsilon^p')
eq_T6_1 = sp.Eq(K_tan, K_mat_part + K_geo)
eq_T6_2 = sp.Eq(P_cr, sp.pi**2 * E_t(x) * I_t(x) / (K_L * L_len)**2)
eq_T6_3 = sp.Eq(eps_th, sp.Integral(alpha_T, (T_temp, T_0, T_temp)))
eq_T6_4 = sp.Eq(sig_t, E_t(x) * (eps_t - eps_th - eps_p))

# T7
q_i_vec, K_ii, f_i, K_ib, q_b, K_red, K_bb, K_bi, h_t, D_b, nu_t = sp.symbols('q_i K_ii f_i K_ib q_b K_red K_bb K_bi h D_b nu')
eq_T7_1 = sp.Eq(q_i_vec, K_ii**-1 * (f_i - K_ib * q_b))
eq_T7_2 = sp.Eq(K_red, K_bb - K_bi * K_ii**-1 * K_ib)
eq_T7_3 = sp.Eq(rho_t(x) * h_t * sp.Derivative(w(x,y,t), t, 2) + D_b * nabla(nabla(w(x,y,t))), q_t(x,t)) # nabla^4 abstraction
eq_T7_4 = sp.Eq(D_b, E_t(x) * h_t**3 / (12 * (1 - nu_t**2)))


# =====================================================================
# 3. FARADAY
# =====================================================================
B_f, E_f, D_f, H_f, J_f_vec, rho_f_f, P_f, M_f, eps0_f, mu0_f = sp.symbols('B E D H J_f rho_f P M epsilon_0 mu_0', cls=sp.Function)
# F1
eq_F1_1 = sp.Eq(sp.Derivative(B_f(t), t), -nabla(E_f(t))) # curl abstraction
eq_F1_2 = sp.Eq(sp.Derivative(D_f(t), t), nabla(H_f(t)) - J_f_vec(t))
eq_F1_3 = sp.Eq(nabla(D_f(t)), rho_f_f(t)) # div
eq_F1_4 = sp.Eq(nabla(B_f(t)), 0)
eq_F1_5 = sp.Eq(sp.Derivative(rho_f_f(t), t) + nabla(J_f_vec(t)), 0)
eq_F1_6 = sp.Eq(D_f(t), eps0_f(t)*E_f(t) + P_f(t))
eq_F1_7 = sp.Eq(B_f(t), mu0_f(t)*(H_f(t) + M_f(t)))
eq_F1_8 = sp.Eq(D_f(t), sp.Symbol('epsilon')*E_f(t))
eq_F1_9 = sp.Eq(B_f(t), sp.Symbol('mu')*H_f(t))
eq_F1_10 = sp.Eq(sp.Symbol('J_c'), sp.Symbol('sigma_e')*E_f(t))

# F2
d0, d1, d2, b_f, e_f, M_eps, M_mu, M_sig, j_f = sp.symbols('d_0 d_1 d_2 b e M_epsilon M_mu^-1 M_sigma j')
eq_F2_1 = sp.Eq(d1*d0, 0)
eq_F2_2 = sp.Eq(d2*d1, 0)
eq_F2_3 = sp.Eq(sp.Derivative(b_f, t), -d1*e_f)
eq_F2_4 = sp.Eq(M_eps*sp.Derivative(e_f, t), d1*M_mu*b_f - M_sig*e_f - j_f)

# F3
u_EM, S_vec, F_L, f_L, T_tens = sp.symbols('u_EM S F f T')
eq_F3_1 = sp.Eq(u_EM, 0.5*(E_f(t)*D_f(t) + B_f(t)*H_f(t)))
eq_F3_2 = sp.Eq(S_vec, E_f(t) * H_f(t)) # cross
eq_F3_3 = sp.Eq(sp.Derivative(u_EM, t) + nabla(S_vec), -J_f_vec(t)*E_f(t))
eq_F3_4 = sp.Eq(F_L, sp.Symbol('q')*(E_f(t) + v_i(t)*B_f(t))) # cross
eq_F3_5 = sp.Eq(f_L, rho_f_f(t)*E_f(t) + J_f_vec(t)*B_f(t))
eq_F3_6 = sp.Eq(T_tens, eps0_f(t)*(E_f(t)**2 - 0.5*E_f(t)**2) + 1/mu0_f(t)*(B_f(t)**2 - 0.5*B_f(t)**2))
eq_F3_7 = sp.Eq(sp.Symbol('F_matter'), sp.Integral(T_tens, sp.Symbol('A')) - sp.Derivative(sp.Integral(eps0_f(t)*E_f(t)*B_f(t), sp.Symbol('V')), t))

# F4
phi_f, A_f, sig_surf, K_f = sp.symbols('phi A sigma_f^surface K_f')
eq_F4_1 = sp.Eq(nabla(sp.Symbol('epsilon')*nabla(phi_f)), -rho_f_f(t))
eq_F4_2 = sp.Eq(E_f(t), -nabla(phi_f))
eq_F4_3 = sp.Eq(B_f(t), nabla(A_f))
eq_F4_4 = sp.Eq(E_f(t), -nabla(phi_f) - sp.Derivative(A_f, t))
eq_F4_5 = sp.Eq(n_vec*sp.Symbol('Delta E'), 0)
eq_F4_6 = sp.Eq(n_vec*sp.Symbol('Delta B'), 0)
eq_F4_7 = sp.Eq(n_vec*sp.Symbol('Delta D'), sig_surf)
eq_F4_8 = sp.Eq(n_vec*sp.Symbol('Delta H'), K_f)
eq_F4_9 = sp.Eq(E_f(t), 0) # tangential PEC

# F5
gamma_f, w0, wp, tau_D, chi, q_J, q_d, eps_pp, delta_s, R_s_f, w_f = sp.symbols('gamma omega_0 omega_p tau_D chi q_J q_d epsilon\'\' delta R_s omega')
eq_F5_1 = sp.Eq(sp.Derivative(P_f(t), t, 2) + gamma_f*sp.Derivative(P_f(t), t) + w0**2 * P_f(t), eps0_f(t)*wp**2*E_f(t))
eq_F5_2 = sp.Eq(tau_D*sp.Derivative(P_f(t), t) + P_f(t), eps0_f(t)*chi*E_f(t))
eq_F5_3 = sp.Eq(q_J, 0.5*sp.Symbol('sigma_e')*sp.Abs(E_f(t))**2)
eq_F5_4 = sp.Eq(q_d, 0.5*w_f*eps_pp*sp.Abs(E_f(t))**2)
eq_F5_5 = sp.Eq(delta_s, sp.sqrt(2/(w_f*sp.Symbol('mu')*sp.Symbol('sigma_e'))))
eq_F5_6 = sp.Eq(R_s_f, 1/(sp.Symbol('sigma_e')*delta_s))

# F6
A_c, i_c, s_c, v_c, phi_c, R_c, C_c, L_c, U_c, U_L, lam_c, V_line, I_line, L_p, R_p, C_p, G_p, Gamma_c, Z_L, Z_0, Y_port, Y_pp, Y_pi, Y_ii, Y_ip = sp.symbols('A_c i s v phi R C L U_C U_L lambda V I L\' R\' C\' G\' Gamma Z_L Z_0 Y_port Y_pp Y_pi Y_ii Y_ip')
eq_F6_1 = sp.Eq(A_c*i_c, s_c)
eq_F6_2 = sp.Eq(v_c, A_c*phi_c)
eq_F6_3 = sp.Eq(sp.Symbol('v_R'), R_c*sp.Symbol('i_R'))
eq_F6_4 = sp.Eq(sp.Symbol('i_C'), C_c*sp.Derivative(sp.Symbol('v_C'), t))
eq_F6_5 = sp.Eq(sp.Symbol('v_L'), L_c*sp.Derivative(sp.Symbol('i_L'), t))
eq_F6_6 = sp.Eq(U_c, 0.5*C_c*sp.Symbol('v_C')**2)
eq_F6_7 = sp.Eq(U_L, 0.5*L_c*sp.Symbol('i_L')**2)
eq_F6_8 = sp.Eq(i_c, sp.Derivative(sp.Symbol('Q'), t))
eq_F6_9 = sp.Eq(v_c, sp.Derivative(lam_c, t))
eq_F6_10 = sp.Eq(sp.Derivative(V_line, z), -L_p*sp.Derivative(I_line, t) - R_p*I_line)
eq_F6_11 = sp.Eq(sp.Derivative(I_line, z), -C_p*sp.Derivative(V_line, t) - G_p*V_line)
eq_F6_12 = sp.Eq(Gamma_c, (Z_L - Z_0)/(Z_L + Z_0))
eq_F6_13 = sp.Eq(Y_port, Y_pp - Y_pi * Y_ii**-1 * Y_ip)

# F7
E_m, w_m, beta_g, kc, f_mnp, c_med, m_i_idx, n_i_idx, p_i_idx, a_g, b_g, d_g = sp.symbols('E_m omega_m beta k_c f_mnp c_medium m n p a_g b_g d_g')
eq_F7_1 = sp.Eq(nabla(1/sp.Symbol('mu')*nabla(E_m)), w_m**2 * sp.Symbol('epsilon') * E_m) # curl(curl) abstracted
eq_F7_2 = sp.Eq(beta_g**2, w_m**2*sp.Symbol('mu')*sp.Symbol('epsilon') - kc**2)
eq_F7_3 = sp.Eq(f_mnp, c_med/2 * sp.sqrt((m_i_idx/a_g)**2 + (n_i_idx/b_g)**2 + (p_i_idx/d_g)**2))


# =====================================================================
# 4. NAVIER-STOKES
# =====================================================================
rho_n, u_n, S_m, p_n, tau_n, g_n_vec, f_other, S_p, E_n, q_n, S_E, e_n, Y_s, j_s, omega_s, S_s = sp.symbols('rho u S_m p tau g f_other S_p E q S_E e Y_s j_s omega_s S_s', cls=sp.Function)
# NS1
eq_NS1_1 = sp.Eq(sp.Derivative(rho_n(t), t) + nabla(rho_n(t)*u_n(t)), S_m(t))
eq_NS1_2 = sp.Eq(sp.Derivative(rho_n(t)*u_n(t), t) + nabla(rho_n(t)*u_n(t)*u_n(t) + p_n(t)), nabla(tau_n(t)) + rho_n(t)*g_n_vec(t) + f_other(t) + S_p(t))
eq_NS1_3 = sp.Eq(sp.Derivative(rho_n(t)*E_n(t), t) + nabla((rho_n(t)*E_n(t) + p_n(t))*u_n(t)), nabla(tau_n(t)*u_n(t) - q_n(t)) + rho_n(t)*g_n_vec(t)*u_n(t) + f_other(t)*u_n(t) + S_E(t))
eq_NS1_4 = sp.Eq(E_n(t), e_n(t) + 0.5*sp.Abs(u_n(t))**2)
eq_NS1_5 = sp.Eq(sp.Derivative(rho_n(t)*Y_s(t), t) + nabla(rho_n(t)*Y_s(t)*u_n(t) + j_s(t)), omega_s(t) + S_s(t))
eq_NS1_6 = sp.Eq(sp.Sum(omega_s(t), ('s', 1, sp.Symbol('N'))), 0)
eq_NS1_7 = sp.Eq(sp.Sum(S_s(t), ('s', 1, sp.Symbol('N'))), S_m(t))

# NS2
D_tens, mu_n, zeta_n, k_n, T_n, h_s, D_s, j_s0, x_s, c_tot, D_sr, J_s_mol = sp.symbols('D mu zeta k T h_s D_s j_s^0 x_s c_tot mathcal{D}_{sr} J_s')
eq_NS2_1 = sp.Eq(D_tens, 0.5*(nabla(u_n(t)) + nabla(u_n(t)))) # grad u + grad u^T
eq_NS2_2 = sp.Eq(tau_n(t), 2*mu_n*(D_tens - (1/3)*nabla(u_n(t))) + zeta_n*nabla(u_n(t))) # Identity implied
eq_NS2_3 = sp.Eq(q_n(t), -k_n*nabla(T_n) + sp.Sum(h_s*j_s(t), ('s', 1, sp.Symbol('N'))))
eq_NS2_4 = sp.Eq(sp.Sum(j_s(t), ('s', 1, sp.Symbol('N'))), 0)
eq_NS2_5 = sp.Eq(sp.Sum(Y_s(t), ('s', 1, sp.Symbol('N'))), 1)
eq_NS2_6 = sp.Eq(j_s0, -rho_n(t)*D_s*nabla(Y_s(t)))
eq_NS2_7 = sp.Eq(j_s(t), j_s0 - Y_s(t)*sp.Sum(sp.Symbol('j_r^0'), ('r', 1, sp.Symbol('N'))))
eq_NS2_8 = sp.Eq(-nabla(x_s), sp.Sum((sp.Symbol('x_r')*J_s_mol - x_s*sp.Symbol('J_r'))/(c_tot*D_sr), ('r', 1, sp.Symbol('N'))))

# NS3
M_bar, R_u, z_alg = sp.symbols('bar{M} R_u z')
eq_NS3_1 = sp.Eq(p_n(t), sp.Function('p')(rho_n(t), e_n(t), Y_s(t), z_alg))
eq_NS3_2 = sp.Eq(T_n, sp.Function('T')(rho_n(t), e_n(t), Y_s(t), z_alg))
eq_NS3_3 = sp.Eq(p_n(t), rho_n(t)*R_u*T_n/M_bar)
eq_NS3_4 = sp.Eq(1/M_bar, sp.Sum(Y_s(t)/sp.Symbol('M_s'), ('s', 1, sp.Symbol('N'))))

# NS4
eq_NS4_1 = sp.Eq(nabla(u_n(t)), 0)
eq_NS4_2 = sp.Eq(rho_n(t)*(sp.Derivative(u_n(t), t) + u_n(t)*nabla(u_n(t))), -nabla(p_n(t)) + mu_n*nabla(nabla(u_n(t))) + sp.Symbol('f'))

# NS5
u_w, v_I, u_I_n, f_surf, gamma_s, R1, R2 = sp.symbols('u_w v_I u_n f_surface gamma_s R_1 R_2')
eq_NS5_1 = sp.Eq(u_n(t), u_w)
eq_NS5_2 = sp.Eq(u_I_n, (u_n(t) - v_I)*n_vec)
eq_NS5_3 = sp.Eq(sp.Symbol('Delta(rho u_n)'), 0)
eq_NS5_4 = sp.Eq(sp.Symbol('Delta(rho u_n u - sigma n)'), f_surf)
eq_NS5_5 = sp.Eq(sp.Symbol('sigma'), -p_n(t) + tau_n(t))
eq_NS5_6 = sp.Eq(sp.Symbol('Delta p'), gamma_s*(1/R1 + 1/R2))

# NS6
Delta_p, L_p, r_p, Q_p, f_D, D_p, U_vel, K_minor, C_d, A_area, m_dot_ns, p_0, T_0, gamma_gas, R_s_gas = sp.symbols('Delta_p L r Q f_D D U K_minor C_d A dot{m} p_0 T_0 gamma R_s')
eq_NS6_1 = sp.Eq(Delta_p, (8*mu_n*L_p/(sp.pi*r_p**4))*Q_p)
eq_NS6_2 = sp.Eq(Delta_p, f_D*(L_p/D_p)*rho_n(t)*U_vel*sp.Abs(U_vel)/2 + K_minor*rho_n(t)*U_vel*sp.Abs(U_vel)/2)
eq_NS6_3 = sp.Eq(Q_p, C_d*A_area*sp.sign(Delta_p)*sp.sqrt(2*sp.Abs(Delta_p)/rho_n(t)))
eq_NS6_4 = sp.Eq(m_dot_ns, C_d*A_area*p_0*sp.sqrt(gamma_gas/(R_s_gas*T_0))*(2/(gamma_gas+1))**((gamma_gas+1)/(2*(gamma_gas-1))))
eq_NS6_5 = (sp.Symbol('p_b')/p_0) <= (2/(gamma_gas+1))**(gamma_gas/(gamma_gas-1))

# NS7
u_prime, mu_t, k_t, nu_t, C_s_les, Delta_les, D_tilde = sp.symbols('u\' mu_t k_t nu_t C_s Delta tilde{D}')
eq_NS7_1 = sp.Eq(-rho_n(t)*sp.Symbol('overline{u\'u\'}'), 2*mu_t*D_tens - (2/3)*(rho_n(t)*k_t + mu_t*nabla(u_n(t))))
eq_NS7_2 = sp.Eq(nu_t, (C_s_les*Delta_les)**2 * sp.sqrt(2 * D_tilde * D_tilde)) # inner product abstracted
eq_NS7_3 = sp.Eq(u_n(t)*sp.Derivative(u_n(t), x) + sp.Symbol('v')*sp.Derivative(u_n(t), y), -1/rho_n(t)*sp.Derivative(p_n(t), x) + sp.Symbol('nu')*sp.Derivative(u_n(t), y, 2))
eq_NS7_4 = sp.Eq(sp.Derivative(p_n(t), y), 0)
eq_NS7_5 = sp.Eq(sp.Derivative(u_n(t), x) + sp.Derivative(sp.Symbol('v'), y), 0)

# NS8
Re, Ma, Pe, Kn, c_s_ns, alpha_ns, lambda_mfp = sp.symbols('Re Ma Pe Kn c_s alpha lambda_mfp')
eq_NS8_1 = sp.Eq(Re, rho_n(t)*U_vel*L_p/mu_n)
eq_NS8_2 = sp.Eq(Ma, U_vel/c_s_ns)
eq_NS8_3 = sp.Eq(Pe, U_vel*L_p/alpha_ns)
eq_NS8_4 = sp.Eq(Kn, lambda_mfp/L_p)


# =====================================================================
# 5. BJERKNES
# =====================================================================
p_b, z_b, rho_b, g_b, R_mix, T_b, theta_b, p_0_b, R_d, c_pd, Omega_b, u_b, f_b, u_g_b, Gamma_b, l_b, chi_b, K_b = sp.symbols('p z rho g R_mix T theta p_0 R_d c_pd Omega u f u_g Gamma l chi K')
# B1
eq_B1_1 = sp.Eq(sp.Derivative(p_b, z_b), -rho_b*g_b)
eq_B1_2 = sp.Eq(p_b, rho_b*R_mix*T_b)
eq_B1_3 = sp.Eq(theta_b, T_b*(p_0_b/p_b)**(R_d/c_pd))
eq_B1_4 = sp.Eq(f_b*sp.Symbol('hat{z}')*u_g_b, -1/rho_b*nabla(p_b)) # cross prod abs; horizontal nabla reuses the shared placeholder
eq_B1_5 = sp.Eq(sp.Derivative(Gamma_b, t), -sp.Integral(1/rho_b, p_b)) # closed loop abs
eq_B1_6 = sp.Eq(Gamma_b, sp.Integral(u_b, l_b))
eq_B1_7 = sp.Eq(sp.Derivative(chi_b, t), nabla(K_b*nabla(chi_b)))

# B2
p_s = sp.Function('p_s')  # saturation vapor pressure, called as p_s(T)
T_0_b, L_b, R_s_b, A_A, B_A, C_A, p_unit, RH, p_v, s_b, T_d = sp.symbols('T_0 L R_s A_A B_A C_A p_unit RH p_v s T_d')
eq_B2_1 = sp.Eq(sp.ln(p_s(T_b)/p_s(T_0_b)), sp.Integral(sp.Function('L')(T_b)/(R_s_b*T_b**2), (T_b, T_0_b, T_b)))
eq_B2_2 = sp.Eq(p_s(T_b), p_s(T_0_b)*sp.exp((L_b/R_s_b)*(1/T_0_b - 1/T_b)))
eq_B2_3 = sp.Eq(sp.log(p_s(T_b)/p_unit, 10), A_A - B_A/(T_b + C_A))
eq_B2_4 = sp.Eq(RH, p_v/p_s(T_b))
eq_B2_5 = sp.Eq(s_b, RH - 1)
eq_B2_6 = sp.Eq(p_s(T_d), p_v)

# B3
S_eq, r_b, a_w, gamma_s_b, M_w, rho_l, R_u_b, A_K, B_K, i_b, n_solute, r_c, S_c = sp.symbols('S_eq r a_w gamma_s M_w rho_l R_u A_K B_K i n_solute r_c S_c')
eq_B3_1 = sp.Eq(sp.Function('S_eq')(r_b), a_w*sp.exp(2*gamma_s_b*M_w/(rho_l*R_u_b*T_b*r_b)))
eq_B3_2 = sp.Eq(S_eq, sp.exp(A_K/r_b - B_K/r_b**3))
eq_B3_3 = sp.Eq(A_K, 2*gamma_s_b*M_w/(rho_l*R_u_b*T_b))
eq_B3_4 = sp.Eq(B_K, 3*i_b*n_solute*M_w/(4*sp.pi*rho_l))
eq_B3_5 = sp.Eq(r_c, sp.sqrt(3*B_K/A_K))
eq_B3_6 = sp.Eq(sp.ln(S_c), sp.sqrt(4*A_K**3/(27*B_K)))

# B4
j_m, alpha_e, T_s_b, M_b, alpha_c, T_v, alpha_b, j_mol = sp.symbols('j_m alpha_e T_s M alpha_c T_v alpha j_mol')
eq_B4_1 = sp.Eq(j_m, alpha_e*p_s(T_s_b)*sp.sqrt(M_b/(2*sp.pi*R_u_b*T_s_b)) - alpha_c*p_v*sp.sqrt(M_b/(2*sp.pi*R_u_b*T_v)))
eq_B4_2 = sp.Eq(j_m, alpha_b*(p_s(T_b) - p_v)*sp.sqrt(M_b/(2*sp.pi*R_u_b*T_b)))
eq_B4_3 = sp.Eq(j_mol, j_m/M_b)
eq_B4_4 = sp.Eq(j_mol, alpha_b*(p_s(T_b) - p_v)/sp.sqrt(2*sp.pi*M_b*R_u_b*T_b))

# B5
m_cond, D_v, rho_v_inf, rho_v_s, T_p, m_b, c_p_b, k_g, T_g, L_v, P_rad = sp.symbols('dot{m}_cond D_v rho_{v;infty} rho_{v;s} T_p m c_p k_g T_g L_v P_rad')
eq_B5_1 = sp.Eq(m_cond, 4*sp.pi*r_b*D_v*(rho_v_inf - sp.Function('rho_v_s')(T_p, r_b, a_w)))
eq_B5_2 = sp.Eq(sp.Derivative(r_b, t), m_cond/(4*sp.pi*rho_l*r_b**2))
eq_B5_3 = sp.Eq(m_b*c_p_b*sp.Derivative(T_p, t), 4*sp.pi*r_b*k_g*(T_g - T_p) + L_v*m_cond + P_rad)

# B6
m_v, m_evap, S_v, m_l, m_freeze, m_melt, S_l, m_i_b, S_i, Q_latent, L_f = sp.symbols('dot{m}_v dot{m}_evap S_v dot{m}_l dot{m}_freeze dot{m}_melt S_l dot{m}_i S_i dot{Q}_latent L_f')
eq_B6_1 = sp.Eq(m_v, -m_cond + m_evap + S_v)
eq_B6_2 = sp.Eq(m_l, m_cond - m_evap - m_freeze + m_melt + S_l)
eq_B6_3 = sp.Eq(m_i_b, m_freeze - m_melt + S_i)
eq_B6_4 = sp.Eq(Q_latent, L_v*(m_cond - m_evap) + L_f*(m_freeze - m_melt))

# B7
m_p, v_p, rho_g, V_p, F_d, F_other_b, v_t_b, mu_g = sp.symbols('m_p v_p rho_g V_p F_d F_other v_t mu_g')
eq_B7_1 = sp.Eq(m_p*sp.Derivative(v_p, t), m_p*g_b - rho_g*V_p*g_b + F_d + F_other_b)
eq_B7_2 = sp.Eq(v_t_b, 2*r_b**2*(rho_b - rho_g)*g_b/(9*mu_g))

# B8
n_dist, C_op, B_op, S_n, K_coll, a_break, b_daught, r_star, Delta_g_v, Delta_G_star, J_nuc, J_0, k_B_b = sp.symbols('n C B S_n K a b r_* Delta_g_v Delta_G_* J J_0 k_B', cls=sp.Function)
m_mass, x_b, M_mass = sp.symbols('m x M')  # mass/position coordinates, not operators -- kept out of the cls=Function batch above
m_prime = sp.Symbol("m'")
# C_op/B_op are written as functions of the coagulation coordinate m alone;
# their dependence on the whole distribution n is implicit, the same way
# nabla/transpose/det/tr/cross above gesture at an operator without a full
# sympy operator-on-a-function representation (n itself can't be passed
# unapplied as an argument, so "C[n](m)" collapses to "C_op(m)" here).
eq_B8_1 = sp.Eq(sp.Derivative(n_dist(m_mass, x_b, t), t) + nabla(v_p*n_dist(m_mass, x_b, t)) + sp.Derivative(m_b*n_dist(m_mass, x_b, t), m_mass), C_op(m_mass) + B_op(m_mass) + S_n(t))
eq_B8_2 = sp.Eq(C_op(m_mass), 0.5*sp.Integral(K_coll(m_prime, m_mass-m_prime)*n_dist(m_prime)*n_dist(m_mass-m_prime), m_prime) - n_dist(m_mass)*sp.Integral(K_coll(m_mass, m_prime)*n_dist(m_prime), m_prime))
eq_B8_3 = sp.Eq(B_op(m_mass), sp.Integral(a_break(M_mass)*b_daught(m_mass, M_mass)*n_dist(M_mass), M_mass) - a_break(m_mass)*n_dist(m_mass))
eq_B8_4 = sp.Eq(sp.Integral(m_mass*b_daught(m_mass, M_mass), m_mass), M_mass)
eq_B8_5 = sp.Eq(r_star(t), 2*gamma_s_b/Delta_g_v(t))
eq_B8_6 = sp.Eq(Delta_G_star(t), 16*sp.pi*gamma_s_b**3/(3*Delta_g_v(t)**2))
eq_B8_7 = sp.Eq(J_nuc(t), J_0(t)*sp.exp(-Delta_G_star(t)/(k_B_b(t)*T_b)))

# B9
Gamma_s, nabla_s, u_f, j_s_surf, j_s_dep, j_s_evap, S_s_surf, h_f, gamma_sv, gamma_sl, gamma_lv, theta_e = sp.symbols('Gamma_s nabla_s u_f j_s^surf j_{s;deposit} j_{s;evaporate} S_s^surf h_f gamma_{sv} gamma_{sl} gamma_{lv} theta_e')
eq_B9_1 = sp.Eq(sp.Derivative(Gamma_s, t) + nabla_s*(Gamma_s*u_f + j_s_surf), j_s_dep - j_s_evap + S_s_surf)
eq_B9_2 = sp.Eq(h_f, Gamma_s/rho_l)
eq_B9_3 = sp.Eq(gamma_sv - gamma_sl, gamma_lv*sp.cos(theta_e))

# B10
beta_ext, Q_ext, T_trans, x_mie, k_h, m_r, C_sca, Q_sca, a_n, b_n, psi_n, xi_n = sp.symbols('beta_ext Q_ext mathcal{T} x k_h m_r C_sca Q_sca a_n b_n psi_n xi_n', cls=sp.Function)
lambda_w, s_path = sp.symbols('lambda s')  # wavelength and path length are plain arguments, not operators
z_mie = sp.Symbol('z_mie')
def _riccati_deriv(fn, arg):
    """d(fn)/d(arg) via a dummy variable and Subs, since sympy's Derivative
    needs an atomic diff-wrt target and some Riccati-Bessel arguments below
    are a product m_r(t)*x_mie(t), not a single Symbol/Function."""
    return sp.Subs(sp.Derivative(fn(z_mie), z_mie), z_mie, arg)
eq_B10_1 = sp.Eq(beta_ext(lambda_w), sp.Integral(Q_ext(r_b, lambda_w)*sp.pi*r_b**2*n_dist(r_b), r_b))
eq_B10_2 = sp.Eq(T_trans(t), sp.exp(-sp.Integral(beta_ext(t), s_path)))
eq_B10_3 = sp.Eq(sp.Symbol('x'), k_h(t)*r_b)
eq_B10_4 = sp.Eq(C_sca(t), (8*sp.pi/3)*k_h(t)**4*r_b**6 * sp.Abs((m_r(t)**2 - 1)/(m_r(t)**2 + 2))**2)
eq_B10_5 = sp.Eq(Q_ext(t), (2/sp.Symbol('x')**2)*sp.Sum((2*sp.Symbol('n')+1)*sp.re(a_n(sp.Symbol('n')) + b_n(sp.Symbol('n'))), ('n', 1, sp.oo)))
eq_B10_6 = sp.Eq(Q_sca(t), (2/sp.Symbol('x')**2)*sp.Sum((2*sp.Symbol('n')+1)*(sp.Abs(a_n(sp.Symbol('n')))**2 + sp.Abs(b_n(sp.Symbol('n')))**2), ('n', 1, sp.oo)))
eq_B10_7 = sp.Eq(a_n(t), (m_r(t)*psi_n(m_r(t)*x_mie(t))*_riccati_deriv(psi_n, x_mie(t)) - psi_n(x_mie(t))*_riccati_deriv(psi_n, m_r(t)*x_mie(t))) / (m_r(t)*psi_n(m_r(t)*x_mie(t))*_riccati_deriv(xi_n, x_mie(t)) - xi_n(x_mie(t))*_riccati_deriv(psi_n, m_r(t)*x_mie(t))))
eq_B10_8 = sp.Eq(b_n(t), (psi_n(m_r(t)*x_mie(t))*_riccati_deriv(psi_n, x_mie(t)) - m_r(t)*psi_n(x_mie(t))*_riccati_deriv(psi_n, m_r(t)*x_mie(t))) / (psi_n(m_r(t)*x_mie(t))*_riccati_deriv(xi_n, x_mie(t)) - m_r(t)*xi_n(x_mie(t))*_riccati_deriv(psi_n, m_r(t)*x_mie(t))))


# =====================================================================
# 6. GIBBS
# =====================================================================
dU, T_g, dS, p_g, dV, mu_s_g, dn_s, H_g, U_g, A_g, S_g, G_g, V_g = sp.symbols('dU T dS p dV mu_s dn_s H U A S G V')
# G1
eq_G1_1 = sp.Eq(dU, T_g*dS - p_g*dV + sp.Sum(mu_s_g*dn_s, ('s', 1, sp.Symbol('N'))))
eq_G1_2 = sp.Eq(H_g, U_g + p_g*V_g)
eq_G1_3 = sp.Eq(A_g, U_g - T_g*S_g)
eq_G1_4 = sp.Eq(G_g, U_g + p_g*V_g - T_g*S_g)
eq_G1_5 = sp.Eq(sp.Symbol('dA'), -S_g*sp.Symbol('dT') - p_g*dV + sp.Sum(mu_s_g*dn_s, ('s', 1, sp.Symbol('N'))))
eq_G1_6 = sp.Eq(sp.Symbol('dG'), -S_g*sp.Symbol('dT') + V_g*sp.Symbol('dp') + sp.Sum(mu_s_g*dn_s, ('s', 1, sp.Symbol('N'))))

# G2
C_p_g, C_V_g, alpha_V, kappa_T, n_g = sp.symbols('C_p C_V alpha_V kappa_T n')
eq_G2_1 = sp.Eq(S_g, -sp.Derivative(G_g, T_g))
eq_G2_2 = sp.Eq(V_g, sp.Derivative(G_g, p_g))
eq_G2_3 = sp.Eq(mu_s_g, sp.Derivative(G_g, n_g))
eq_G2_4 = sp.Eq(C_p_g, sp.Derivative(H_g, T_g))
eq_G2_5 = sp.Eq(C_V_g, sp.Derivative(U_g, T_g))
eq_G2_6 = sp.Eq(alpha_V, 1/V_g * sp.Derivative(V_g, T_g))
eq_G2_7 = sp.Eq(kappa_T, -1/V_g * sp.Derivative(V_g, p_g))
eq_G2_8 = sp.Eq(C_p_g - C_V_g, T_g*V_g*alpha_V**2 / kappa_T)
eq_G2_9 = sp.Eq(sp.Derivative(S_g, p_g), -sp.Derivative(V_g, T_g)) # Maxwell
eq_G2_10 = sp.Eq(S_g*sp.Symbol('dT') - V_g*sp.Symbol('dp') + sp.Sum(n_g*sp.Symbol('dmu_s'), ('s', 1, sp.Symbol('N'))), 0) # Gibbs-Duhem

# G3
h_s_circ, h_s_circ_T0, c_p_s_circ, s_s_circ, s_s_circ_T0, g_s_circ = sp.symbols('h_s^circ h_s^circ(T_0) c_{p;s}^circ s_s^circ s_s^circ(T_0) g_s^circ')
eq_G3_1 = sp.Eq(h_s_circ, h_s_circ_T0 + sp.Integral(c_p_s_circ, (T_g, sp.Symbol('T_0'), T_g)))
eq_G3_2 = sp.Eq(s_s_circ, s_s_circ_T0 + sp.Integral(c_p_s_circ/T_g, (T_g, sp.Symbol('T_0'), T_g)))
eq_G3_3 = sp.Eq(g_s_circ, h_s_circ - T_g*s_s_circ)

# G4
mu_s_circ, R_u_g, a_s_g, mu_s_tilde, z_s_g, F_c_g, phi_g, gamma_s_g, x_s_g, f_s_g, p_circ, phi_s_g, y_s_g = sp.symbols('mu_s^circ R_u a_s tilde{mu}_s z_s F_c phi gamma_s x_s f_s p^circ phi_s y_s')
eq_G4_1 = sp.Eq(mu_s_g, mu_s_circ + R_u_g*T_g*sp.ln(a_s_g))
eq_G4_2 = sp.Eq(mu_s_tilde, mu_s_g + z_s_g*F_c_g*phi_g)
eq_G4_3 = sp.Eq(a_s_g, gamma_s_g*x_s_g)
eq_G4_4 = sp.Eq(a_s_g, f_s_g/p_circ)
eq_G4_5 = sp.Eq(f_s_g, phi_s_g*y_s_g*p_g)

# G5
a_vdw, b_vdw, a_helm, rho_g, Y_g, s_spec, e_spec = sp.symbols('a b a_helm rho Y s e')
eq_G5_1 = sp.Eq(p_g*V_g, n_g*R_u_g*T_g)
eq_G5_2 = sp.Eq((p_g + a_vdw*n_g**2/V_g**2)*(V_g - n_g*b_vdw), n_g*R_u_g*T_g)
eq_G5_3 = sp.Eq(p_g, rho_g**2 * sp.Derivative(a_helm, rho_g))
eq_G5_4 = sp.Eq(s_spec, -sp.Derivative(a_helm, T_g))
eq_G5_5 = sp.Eq(e_spec, a_helm + T_g*s_spec)

# G6
G_min, A_elem, b_elem, T_alpha, T_beta, p_alpha, p_beta, mu_s_alpha, mu_s_beta, p_coex, L_coex, Delta_v, Delta_s_coex, F_rule, C_rule, P_rule = sp.symbols('G A_elem b T^alpha T^beta p^alpha p^beta mu_s^alpha mu_s^beta p_{coex} L Delta_v Delta_s F C P')
eq_G6_1 = sp.Eq(A_elem*n_g, b_elem)
eq_G6_2 = sp.Eq(T_alpha, T_beta)
eq_G6_3 = sp.Eq(p_alpha, p_beta)
eq_G6_4 = sp.Eq(mu_s_alpha, mu_s_beta)
eq_G6_5 = sp.Eq(sp.Derivative(p_coex, T_g), L_coex/(T_g*Delta_v))
eq_G6_6 = sp.Eq(L_coex, T_g*Delta_s_coex)
eq_G6_7 = sp.Eq(F_rule, C_rule - P_rule + 2)

# G7
xi_dot, L_xi = sp.symbols('dot{xi} L_xi')
eq_G7_1 = sp.Eq(xi_dot, -L_xi*sp.Derivative(G_g, sp.Symbol('xi')))
eq_G7_2 = L_xi >= 0

# G8
U_dot, Q_dot_g, W_dot_g, m_dot_in_g, h_in, m_dot_out_g, h_out, S_dot_g, Q_dot_k, T_k, s_in, s_out, S_gen_dot, mu_JT, v_spec, c_p_spec = sp.symbols('dot{U} dot{Q} dot{W} dot{m}_{in} h dot{m}_{out} h dot{S} dot{Q}_k T_k s s dot{S}_{gen} mu_{JT} v c_p')
eq_G8_1 = sp.Eq(U_dot, Q_dot_g - W_dot_g + sp.Sum(m_dot_in_g*h_in, ('idx_in', 1, sp.Symbol('N'))) - sp.Sum(m_dot_out_g*h_out, ('out', 1, sp.Symbol('N'))))
eq_G8_2 = sp.Eq(S_dot_g, sp.Sum(Q_dot_k/T_k, ('k', 1, sp.Symbol('N'))) + sp.Sum(m_dot_in_g*s_in, ('idx_in', 1, sp.Symbol('N'))) - sp.Sum(m_dot_out_g*s_out, ('out', 1, sp.Symbol('N'))) + S_gen_dot)
eq_G8_3 = S_gen_dot >= 0
eq_G8_4 = sp.Eq(mu_JT, (T_g*sp.Derivative(v_spec, T_g) - v_spec)/c_p_spec)
eq_G8_5 = sp.Eq(sp.Symbol('Delta T'), mu_JT*sp.Symbol('Delta p'))


# =====================================================================
# 7. BRAGG
# =====================================================================
R_n_alpha, n_i, a_i, r_alpha, a_i_vec, b_j_vec, G_hkl, h_idx, k_idx, l_idx, b1, b2, b3, d_hkl, theta_br, n_br, lambda_br, F_G, f_alpha = sp.symbols('R_{n;alpha} n_i a_i r_alpha a_i b_j G_{hkl} h k l b_1 b_2 b_3 d_{hkl} theta n lambda F(G) f_alpha')
# BR1
eq_BR1_1 = sp.Eq(R_n_alpha, sp.Sum(n_i*a_i, ('i', 1, 3)) + r_alpha)
eq_BR1_2 = sp.Eq(a_i_vec * b_j_vec, 2*sp.pi*sp.KroneckerDelta(1, 1)) # dot prod abstraction
eq_BR1_3 = sp.Eq(G_hkl, h_idx*b1 + k_idx*b2 + l_idx*b3)
eq_BR1_4 = sp.Eq(2*d_hkl*sp.sin(theta_br), n_br*lambda_br)
eq_BR1_5 = sp.Eq(F_G, sp.Sum(f_alpha * sp.exp(sp.I * G_hkl * r_alpha), ('alpha', 1, sp.Symbol('N'))))

# BR2
eps_br, u_br, sig_br, C_br, eps_p_br, eps_th_br, alpha_br, T_br, T0_br, z_br, psi_br, F_br, x_br, X_br, Fe_br, Fp_br, P_br, Psi_br, J_br = sp.symbols('epsilon u sigma C epsilon^p epsilon^{th} alpha T T_0 z psi F x X F_e F_p P Psi J')
eq_BR2_1 = sp.Eq(eps_br, 0.5*(nabla(u_br) + nabla(u_br))) # transpose abstracted
eq_BR2_2 = sp.Eq(sig_br, C_br*(eps_br - eps_p_br - eps_th_br))
eq_BR2_3 = sp.Eq(eps_th_br, sp.Integral(alpha_br, (T_br, T0_br, T_br)))
eq_BR2_4 = sp.Eq(sig_br, sp.Derivative(psi_br, eps_br))
eq_BR2_5 = sp.Eq(C_br, sp.Derivative(sig_br, eps_br))
eq_BR2_6 = sp.Eq(F_br, sp.Derivative(x_br, X_br))
eq_BR2_7 = sp.Eq(F_br, Fe_br*Fp_br)
eq_BR2_8 = sp.Eq(P_br, sp.Derivative(Psi_br, F_br))
eq_BR2_9 = sp.Eq(sig_br, J_br**-1 * P_br * F_br) # transpose abstracted
eq_BR2_10 = sp.Eq(J_br, det(F_br))

# BR3
s_br, sig_eq, f_y, sig_y, eps_bar_p, eps_p_dot, lam_dot = sp.symbols('s sigma_eq f_y sigma_y bar{epsilon}^p dot{epsilon}^p dot{lambda}')
eq_BR3_1 = sp.Eq(s_br, sp.Add(sig_br, -sp.Rational(1, 3)*tr(sig_br)*sp.Identity(3), evaluate=False), evaluate=False)
eq_BR3_2 = sp.Eq(sig_eq, sp.sqrt((3/2)*s_br*s_br)) # inner prod abstracted
eq_BR3_3 = f_y <= 0
eq_BR3_4 = sp.Eq(f_y, sig_eq - sig_y)
eq_BR3_5 = sp.Eq(eps_p_dot, lam_dot * (3*s_br)/(2*sig_eq))
eq_BR3_6 = sp.Eq(sp.Derivative(eps_bar_p, t), lam_dot)
eq_BR3_7 = lam_dot >= 0
eq_BR3_8 = sp.Eq(lam_dot*f_y, 0)

# BR4
tau_a, s_a, n_a, gamma_dot_a, gamma_dot_0, g_a, m_r_br, rho_mobile, b_burg, v_disl, tau_c, tau_0, alpha_G, G_shear, rho_disl = sp.symbols('tau_a s_a n_a dot{gamma}_a dot{gamma}_0 g_a m_r rho_mobile b v_disl tau_c tau_0 alpha_G G rho_disl')
eq_BR4_1 = sp.Eq(tau_a, s_a * sig_br * n_a)
eq_BR4_2 = sp.Eq(eps_p_dot, sp.Sum(gamma_dot_a * (s_a * n_a), ('a', 1, sp.Symbol('N')))) # sym abstracted
eq_BR4_3 = sp.Eq(gamma_dot_a, gamma_dot_0 * sp.Abs(tau_a/g_a)**(1/m_r_br) * sp.sign(tau_a))
eq_BR4_4 = sp.Eq(sp.Symbol('dot{gamma}'), rho_mobile * b_burg * v_disl)
eq_BR4_5 = sp.Eq(tau_c, tau_0 + alpha_G * G_shear * b_burg * sp.sqrt(rho_disl))

# BR5
U_eam, F_alpha_i, rho_alpha_j, phi_alpha_ij, D_defect, D_0, E_a_defect, R_u_br, c_defect, S_c_defect, j_c, M_c, mu_c = sp.symbols('U F_{alpha_i} rho_{alpha_j} phi_{alpha_i;alpha_j} D D_0 E_a R_u c S_c j_c M_c mu_c')
eq_BR5_1 = sp.Eq(U_eam, sp.Sum(F_alpha_i + 0.5*sp.Sum(phi_alpha_ij, ('j', 1, sp.Symbol('N'))), ('i', 1, sp.Symbol('N')))) # inner func abstracted
eq_BR5_2 = sp.Eq(D_defect, D_0 * sp.exp(-E_a_defect/(R_u_br*T_br)))
eq_BR5_3 = sp.Eq(sp.Derivative(c_defect, t), nabla(D_defect*nabla(c_defect)) + S_c_defect)
eq_BR5_4 = sp.Eq(j_c, -M_c*nabla(mu_c))

# BR6
F_func, f_Gibbs, kappa_c, kappa_eta, eta_br, f_elastic, L_eta = sp.symbols('mathcal{F} f_Gibbs kappa_c kappa_eta eta f_elastic L_eta')
eq_BR6_1 = sp.Eq(F_func, sp.Integral(f_Gibbs + (kappa_c/2)*sp.Abs(nabla(c_defect))**2 + (kappa_eta/2)*sp.Abs(nabla(eta_br))**2 + f_elastic, sp.Symbol('V')))
eq_BR6_2 = sp.Eq(sp.Derivative(c_defect, t), nabla(M_c * nabla(sp.Derivative(F_func, c_defect))))
eq_BR6_3 = sp.Eq(sp.Derivative(eta_br, t), -L_eta * sp.Derivative(F_func, eta_br))

# BR7
G_frac, G_c, K_I, Y_frac, a_frac, E_prime, E_frac, nu_frac, E_frac_diff, g_d, d_frac, l_frac = sp.symbols('G G_c K_I Y a E\' E nu mathcal{E} g(d) d ell')
eq_BR7_1 = G_frac >= G_c
eq_BR7_2 = sp.Eq(K_I, Y_frac*sig_br*sp.sqrt(sp.pi*a_frac))
eq_BR7_3 = sp.Eq(G_frac, K_I**2 / E_prime)
eq_BR7_4 = sp.Eq(E_prime, E_frac)
eq_BR7_5 = sp.Eq(E_prime, E_frac/(1 - nu_frac**2))
eq_BR7_6 = sp.Eq(E_frac_diff, sp.Integral(g_d*psi_br + (G_c/2)*(d_frac**2/l_frac + l_frac*sp.Abs(nabla(d_frac))**2), sp.Symbol('V')))
eq_BR7_7 = (0 <= d_frac) & (d_frac <= 1)
eq_BR7_8 = sp.Derivative(d_frac, t) >= 0
eq_BR7_9 = sp.Eq(sp.Derivative(a_frac, sp.Symbol('N_fatigue')), sp.Symbol('C') * sp.Symbol('Delta K')**sp.Symbol('m'))

# BR8
psi_nk, u_nk, v_nk, hbar, E_n_band, m_star_inv, Phi_force, u_ia, D_dyn, e_knu, w_knu, n_B, k_B_br, C_knu, x_ph, k_ij_br, tau_knu = sp.symbols('psi_{nk} u_{nk} v_{nk} hbar E_n (m^*)^{-1} Phi u_{i;alpha} D e_{k;nu} omega_{k;nu} n_B k_B C_{k;nu} x k_{ij} tau_{k;nu}')
eq_BR8_1 = sp.Eq(psi_nk, sp.exp(sp.I * sp.Symbol('k') * r_alpha) * u_nk)
eq_BR8_2 = sp.Eq(v_nk, hbar**-1 * nabla(E_n_band))
eq_BR8_3 = sp.Eq(m_star_inv, hbar**-2 * sp.Derivative(E_n_band, sp.Symbol('k_i'), sp.Symbol('k_j')))
eq_BR8_4 = sp.Eq(Phi_force, sp.Derivative(U_eam, u_ia, sp.Symbol('u_{j,beta}')))
eq_BR8_5 = sp.Eq(D_dyn * e_knu, w_knu**2 * e_knu)
eq_BR8_6 = sp.Eq(n_B, 1/(sp.exp(hbar*w_knu/(k_B_br*T_br)) - 1))
eq_BR8_7 = sp.Eq(C_knu, k_B_br*x_ph**2 * sp.exp(x_ph)/(sp.exp(x_ph)-1)**2)
eq_BR8_8 = sp.Eq(x_ph, hbar*w_knu/(k_B_br*T_br))
eq_BR8_9 = sp.Eq(k_ij_br, 1/sp.Symbol('V') * sp.Sum(C_knu * v_nk * v_nk * tau_knu, ('knu', 1, sp.Symbol('N'))))

# BR9
eps_semi, phi_semi, q_e, p_semi, n_semi, ND, NA, rho_trap, E_semi, J_n, mu_n, D_n, J_p, mu_p, D_p, G_semi, R_semi = sp.symbols('epsilon phi q_e p n N_D^+ N_A^- rho_{trap} E J_n mu_n D_n J_p mu_p D_p G R')
eq_BR9_1 = sp.Eq(-nabla(eps_semi*nabla(phi_semi)), q_e*(p_semi - n_semi + ND - NA) + rho_trap)
eq_BR9_2 = sp.Eq(E_semi, -nabla(phi_semi))
eq_BR9_3 = sp.Eq(J_n, q_e*mu_n*n_semi*E_semi + q_e*D_n*nabla(n_semi))
eq_BR9_4 = sp.Eq(J_p, q_e*mu_p*p_semi*E_semi - q_e*D_p*nabla(p_semi))
eq_BR9_5 = sp.Eq(sp.Derivative(n_semi, t), q_e**-1 * nabla(J_n) + G_semi - R_semi)
eq_BR9_6 = sp.Eq(sp.Derivative(p_semi, t), -q_e**-1 * nabla(J_p) + G_semi - R_semi)
eq_BR9_7 = sp.Eq(D_n/mu_n, k_B_br*T_br/q_e)
eq_BR9_8 = sp.Eq(D_p/mu_p, k_B_br*T_br/q_e)

# BR10
f_FD, E_semi_lvl, mu_semi, E_c, D_c, E_v, D_v_semi, R_SRH, n_i_semi, tau_p, n_1, tau_n, p_1, R_rad, B_rad, R_Auger, C_n, C_p_semi, f_t, c_n, e_n, c_p, e_p_trap, I_semi, I_s, V_semi, n_d_semi, E_g, alpha_V_semi, beta_V = sp.symbols('f_{FD} E mu E_c D_c E_v D_v R_{SRH} n_i tau_p n_1 tau_n p_1 R_{rad} B R_{Auger} C_n C_p f_t c_n e_n c_p e_p I I_s V n_d E_g alpha_V beta_V')
eq_BR10_1 = sp.Eq(f_FD, 1/(sp.exp((E_semi_lvl - mu_semi)/(k_B_br*T_br)) + 1))
eq_BR10_2 = sp.Eq(n_semi, sp.Integral(D_c*f_FD, (E_semi_lvl, E_c, sp.oo)))
eq_BR10_3 = sp.Eq(p_semi, sp.Integral(D_v_semi*(1 - f_FD), (E_semi_lvl, -sp.oo, E_v)))
eq_BR10_4 = sp.Eq(R_SRH, (n_semi*p_semi - n_i_semi**2)/(tau_p*(n_semi+n_1) + tau_n*(p_semi+p_1)))
eq_BR10_5 = sp.Eq(R_rad, B_rad*(n_semi*p_semi - n_i_semi**2))
eq_BR10_6 = sp.Eq(R_Auger, (C_n*n_semi + C_p_semi*p_semi)*(n_semi*p_semi - n_i_semi**2))
eq_BR10_7 = sp.Eq(sp.Derivative(f_t, t), c_n*n_semi*(1-f_t) - e_n*f_t - c_p*p_semi*f_t + e_p_trap*(1-f_t))
eq_BR10_8 = sp.Eq(I_semi, I_s*(sp.exp(q_e*V_semi/(n_d_semi*k_B_br*T_br)) - 1))
eq_BR10_9 = sp.Eq(sp.Function('E_g')(T_br), sp.Function('E_g')(0) - alpha_V_semi*T_br**2/(T_br + beta_V))

# BR11
M_mag, gamma_mag, mu_0_mag, H_eff, alpha_G_mag, M_s_mag, F_mag, P_pol, L_P, chi_mag, C_mag, T_C = sp.symbols('M gamma mu_0 H_{eff} alpha_G M_s mathcal{F} P L_P chi C T_C')
eq_BR11_1 = sp.Eq(sp.Derivative(M_mag, t), -gamma_mag*mu_0_mag*M_mag*H_eff + (alpha_G_mag/M_s_mag)*M_mag*sp.Derivative(M_mag, t)) # cross prod abs
eq_BR11_2 = sp.Eq(H_eff, -mu_0_mag**-1 * sp.Derivative(F_mag, M_mag))
eq_BR11_3 = sp.Eq(sp.Derivative(P_pol, t), -L_P * sp.Derivative(F_mag, P_pol))
eq_BR11_4 = sp.Eq(chi_mag, C_mag/(T_br - T_C))

# BR12
sig_ve, E_ve, eps_ve, eta_ve, tau_ve, G_ve, G_inf, G_a, tau_a_ve, Psi_nh, mu_nh, I_1, J_nh, lam_nh = sp.symbols('sigma E epsilon eta tau G G_infty G_a tau_a Psi mu I_1 J lambda')
eq_BR12_1 = sp.Eq(sig_ve, E_ve*eps_ve + eta_ve*sp.Derivative(eps_ve, t))
eq_BR12_2 = sp.Eq(sp.Derivative(sig_ve, t) + sig_ve/tau_ve, E_ve*sp.Derivative(eps_ve, t))
eq_BR12_3 = sp.Eq(tau_ve, eta_ve/E_ve)
eq_BR12_4 = sp.Eq(sp.Function('G_ve')(t), G_inf + sp.Sum(G_a*sp.exp(-t/tau_a_ve), ('a', 1, sp.Symbol('N'))))
eq_BR12_5 = sp.Eq(sp.Function('sig_ve')(t), sp.Integral(sp.Function('G_ve')(t - sp.Symbol('s'))*sp.Derivative(sp.Function('eps_ve')(sp.Symbol('s')), sp.Symbol('s')), (sp.Symbol('s'), -sp.oo, t)))
eq_BR12_6 = sp.Eq(Psi_nh, (mu_nh/2)*(I_1 - 3) - mu_nh*sp.ln(J_nh) + (lam_nh/2)*(sp.ln(J_nh))**2)
eq_BR12_7 = sp.Eq(I_1, tr(transpose(F_br) * F_br))
eq_BR12_8 = J_nh > 0
eq_BR12_9 = sp.Eq(P_br, sp.Derivative(Psi_nh, F_br))


# =====================================================================
# 8. HAMILTON
# =====================================================================
psi_h, H_op, E_n_h, n_ket, O_op, rho_op = sp.symbols('psi H E_n n O varrho')
# H1
eq_H1_1 = sp.Eq(sp.I*hbar*sp.Derivative(psi_h, t), H_op*psi_h)
eq_H1_2 = sp.Eq(H_op*n_ket, E_n_h*n_ket)
eq_H1_3 = sp.Eq(H_op, sp.conjugate(H_op)) # dagger abs
eq_H1_4 = sp.Eq(psi_h * psi_h, 1) # inner prod abs
eq_H1_5 = sp.Eq(sp.Symbol('langle O rangle'), psi_h * O_op * psi_h)
eq_H1_6 = sp.Eq(sp.Symbol('langle O rangle'), tr(rho_op * O_op))
eq_H1_7 = sp.Eq(sp.I*hbar*sp.Symbol('S')*sp.Derivative(sp.Symbol('c'), t), H_op*sp.Symbol('c'))
eq_H1_8 = sp.Eq(H_op*sp.Symbol('c'), E_n_h*sp.Symbol('S')*sp.Symbol('c'))

# H2
m_e, q_h, A_h, phi_h, V_h, rho_P, j_P = sp.symbols('m q A phi V rho_P j_P')
eq_H2_1 = sp.Eq(H_op, (-sp.I*hbar*nabla_op - q_h*A_h)**2/(2*m_e) + q_h*phi_h + V_h)
eq_H2_2 = sp.Eq(rho_P, sp.Abs(psi_h)**2)
eq_H2_3 = sp.Eq(j_P, (hbar/m_e)*sp.im(sp.conjugate(psi_h)*nabla(psi_h)) - (q_h/m_e)*A_h*sp.Abs(psi_h)**2)
eq_H2_4 = sp.Eq(sp.Derivative(rho_P, t) + nabla(j_P), 0)

# H3
gamma_a, L_a = sp.symbols('gamma_a L_a')
eq_H3_1 = sp.Eq(sp.Derivative(rho_op, t), -(sp.I/hbar)*(H_op*rho_op - rho_op*H_op)) # commutator abs
eq_H3_2 = sp.Eq(sp.Derivative(rho_op, t), -(sp.I/hbar)*(H_op*rho_op - rho_op*H_op) + sp.Sum(gamma_a*(L_a*rho_op*sp.conjugate(L_a) - 0.5*(sp.conjugate(L_a)*L_a*rho_op + rho_op*sp.conjugate(L_a)*L_a)), ('a', 1, sp.Symbol('N'))))

# H4
H_e, Z_I, e_q, eps_0, r_i, R_I, r_j, V_NN, R_J, E_BO, E_e = sp.symbols('H_e Z_I e epsilon_0 r_i R_I r_j V_{NN} R_J E_{BO} E_e')
eq_H4_1 = sp.Eq(H_e, -sp.Sum((hbar**2/(2*m_e))*nabla_op**2, ('i', 1, sp.Symbol('N'))) - sp.Sum(Z_I*e_q**2/(4*sp.pi*eps_0*sp.Abs(r_i - R_I)), ('iI_pair', 1, sp.Symbol('N'))) + sp.Sum(e_q**2/(4*sp.pi*eps_0*sp.Abs(r_i - r_j)), ('i_lt_j', 1, sp.Symbol('N'))))
eq_H4_2 = sp.Eq(V_NN, sp.Sum(Z_I*sp.Symbol('Z_J')*e_q**2/(4*sp.pi*eps_0*sp.Abs(R_I - R_J)), ('I_lt_J', 1, sp.Symbol('N'))))
eq_H4_3 = sp.Eq(E_BO, E_e + V_NN)

# H5
F_I, M_I, H_mw = sp.symbols('F_I M_I H^{mw}')
eq_H5_1 = sp.Eq(F_I, -nabla(E_BO)) # grad w.r.t R_I
eq_H5_2 = sp.Eq(M_I*sp.Derivative(R_I, t, 2), F_I)
eq_H5_3 = sp.Eq(nabla(E_BO), 0) # at R_*
eq_H5_4 = sp.Eq(H_mw, sp.Derivative(E_BO, R_I, R_J)/sp.sqrt(M_I*sp.Symbol('M_J')))
eq_H5_5 = sp.Eq(H_mw*sp.Symbol('e_k'), sp.Symbol('omega_k')**2 * sp.Symbol('e_k'))

# H6
F_op, h_op, J_op, K_op, F_mat, C_mat, S_mat, eps_mat, E_HF = sp.symbols('F h J K F_mat C S varepsilon E_{HF}')
eq_H6_1 = sp.Eq(F_op, h_op + sp.Sum(J_op - K_op, ('j', 1, 'occ')))
eq_H6_2 = sp.Eq(F_mat*C_mat, S_mat*C_mat*eps_mat)
eq_H6_3 = sp.Eq(E_HF, sp.Sum(sp.Symbol('h_{ii}'), ('i', 1, sp.Symbol('N'))) + 0.5*sp.Sum(sp.Symbol('J_{ij}') - sp.Symbol('K_{ij}'), ('ij_pair', 1, sp.Symbol('N'))) + V_NN)

# H7
v_ext, v_H, v_xc, phi_i, eps_i, n_r, f_i_occ, E_n_func, T_s_func, E_xc_func = sp.symbols('v_{ext} v_H v_{xc} phi_i varepsilon_i n f_i E T_s E_{xc}')
eq_H7_1 = sp.Eq((-0.5*nabla_op**2 + v_ext + v_H + v_xc)*phi_i, eps_i*phi_i)
eq_H7_2 = sp.Eq(n_r, sp.Sum(f_i_occ*sp.Abs(phi_i)**2, ('i', 1, sp.Symbol('N'))))
eq_H7_3 = sp.Eq(v_H, sp.Integral(n_r/sp.Abs(r_i - r_j), r_j))
eq_H7_4 = sp.Eq(v_xc, sp.Derivative(E_xc_func, n_r))
eq_H7_5 = sp.Eq(E_n_func, T_s_func + sp.Integral(v_ext*n_r, r_i) + 0.5*sp.Integral(n_r*sp.Symbol('n(r\')')/sp.Abs(r_i - sp.Symbol('r\'')), (r_i, sp.Symbol('r\''))) + E_xc_func + V_NN)

# H8
t_ij, w_fi, E_f_h, E_i_h, H_prime, d_dipole, E_field, Gamma_if, rho_f_h, T_prob = sp.symbols('t_{ij} omega_{fi} E_f E_i H\' d E Gamma_{i->f} rho_f mathcal{T}')
eq_H8_1 = sp.Eq(H_op, sp.Sum(eps_i*sp.Symbol('|i><i|'), ('i', 1, sp.Symbol('N'))) + sp.Sum(t_ij*sp.Symbol('|i><j|'), ('i_ne_j', 1, sp.Symbol('N'))))
eq_H8_2 = sp.Eq(sp.Symbol('t_{ji}'), sp.conjugate(t_ij))
eq_H8_3 = sp.Eq(hbar*w_fi, E_f_h - E_i_h)
eq_H8_4 = sp.Eq(sp.Function('H_prime')(t), -d_dipole * sp.Function('E_field')(t))
eq_H8_5 = sp.Eq(Gamma_if, (2*sp.pi/hbar)*sp.Abs(sp.Symbol('<f|H\'|i>'))**2 * sp.Function('rho_f_h')(E_f_h))
eq_H8_6 = sp.Eq(T_prob, sp.exp(-(2/hbar)*sp.Integral(sp.sqrt(2*m_e*(V_h - E_i_h)), x)))
eq_H8_7 = sp.Eq(sp.Symbol('E_0'), 0.5*hbar*sp.Symbol('omega'))

# H9
c_a, E_a_h, R_dot, d_ab, O_exp, x_exp, p_exp, E_0_var = sp.symbols('c_a E_a dot{R} d_{ab} O x p E_0')
eq_H9_1 = sp.Eq(sp.I*hbar*sp.Derivative(c_a, t), E_a_h*c_a - sp.I*hbar*sp.Sum(R_dot * d_ab * sp.Symbol('c_b'), ('b', 1, sp.Symbol('N'))))
eq_H9_2 = sp.Eq(d_ab, sp.Symbol('<phi_a|nabla_R|phi_b>'))
eq_H9_3 = sp.Eq(sp.Derivative(O_exp, t), (sp.I/hbar)*sp.Symbol('<[H,O]>') + sp.Symbol('<partial_t O>'))
eq_H9_4 = sp.Eq(sp.Derivative(x_exp, t), p_exp/m_e)
eq_H9_5 = sp.Eq(sp.Derivative(p_exp, t), -sp.Symbol('<nabla V>'))
eq_H9_6 = E_0_var <= sp.Symbol('<psi|H|psi>/<psi|psi>')


# =====================================================================
# 9. CURIE
# =====================================================================
lam_c, t_half, N_c, tau_c, A_c, b_ji, S_c_nu, N_vec, A_mat, S_vec, N_dec = sp.symbols('lambda t_{1/2} N tau A b_{j->i} S N_vec A_mat S_vec N_{decayed}')
# C1
eq_C1_1 = sp.Eq(lam_c, sp.ln(2)/t_half)
eq_C1_2 = sp.Eq(sp.Derivative(N_c, tau_c), -lam_c*N_c)
eq_C1_3 = sp.Eq(N_c, sp.Symbol('N_i(0)')*sp.exp(-lam_c*tau_c))
eq_C1_4 = sp.Eq(A_c, lam_c*N_c)

# C2
eq_C2_1 = sp.Eq(sp.Derivative(N_c, t), sp.Sum(b_ji*lam_c*sp.Symbol('N_j'), ('j', 1, sp.Symbol('N'))) - lam_c*N_c + S_c_nu)
eq_C2_2 = sp.Eq(sp.Sum(sp.Symbol('b_{i->k}'), ('k', 1, sp.Symbol('N'))), 1)
eq_C2_3 = sp.Eq(sp.Derivative(N_vec, t), A_mat*N_vec + S_vec)
eq_C2_4 = sp.Eq(sp.Symbol('N(t+Delta t)'), sp.exp(A_mat*sp.Symbol('Delta t'))*sp.Symbol('N(t)'))

# C3
eq_C3_1 = sp.Eq(N_dec, sp.Function('Binomial')(N_c, 1 - sp.exp(-lam_c*sp.Symbol('Delta tau'))))
eq_C3_2 = sp.Eq(sp.Symbol('P_{survive}'), sp.exp(-sp.Integral(lam_c, tau_c)))

# C4
Q_c, c_light, m_ini, m_fin, S_r, y_ikr, E_rad = sp.symbols('Q c m_{initial} m_{final} S_r y_{ik;r} E')
eq_C4_1 = sp.Eq(Q_c, (sp.Sum(m_ini, ('i', 1, sp.Symbol('N'))) - sp.Sum(m_fin, ('f', 1, sp.Symbol('N'))))*c_light**2)
eq_C4_2 = sp.Eq(sp.Function('S_r')(E_rad), sp.Sum(A_c*sp.Sum(sp.Symbol('b_{ik}')*sp.Function('y_ikr')(E_rad), ('k', 1, sp.Symbol('N'))), ('i', 1, sp.Symbol('N'))))

# C5
I_s, I_0, mu_rad, S_stop, P_dep, E_dep_i, D_abs, m_dose = sp.symbols('I I_0 mu S P_{deposited} E_{dep;i} D_{absorbed} m')
eq_C5_1 = sp.Eq(I_s, I_0*sp.exp(-sp.Integral(mu_rad, x)))
eq_C5_2 = sp.Eq(sp.Derivative(E_rad, sp.Symbol('s')), -S_stop)
eq_C5_3 = sp.Eq(sp.Symbol('range'), sp.Integral(1/S_stop, (E_rad, 0, sp.Symbol('E_0'))))
eq_C5_4 = sp.Eq(P_dep, sp.Sum(A_c*E_dep_i, ('i', 1, sp.Symbol('N'))))
eq_C5_5 = sp.Eq(D_abs, sp.Symbol('E_{deposited}')/m_dose)

# C6
r_rxn, sigma_rxn, phi_flux, B_bind, a_v, a_s_c, a_c_c, a_a_c, delta_c, A_nuc, Z_nuc = sp.symbols('r sigma phi B a_v a_s a_c a_a delta A Z')
eq_C6_1 = sp.Eq(r_rxn, N_c*sp.Integral(sigma_rxn*phi_flux, E_rad))
eq_C6_2 = sp.Eq(B_bind, a_v*A_nuc - a_s_c*A_nuc**(2/3) - a_c_c*Z_nuc*(Z_nuc-1)/A_nuc**(1/3) - a_a_c*(A_nuc - 2*Z_nuc)**2/A_nuc + delta_c)


# =====================================================================
# 10. EINSTEIN
# =====================================================================
gamma_e, v_e, c_e, p_e, m_e_ein, E_e_ein, tau_e, F_e, a_e = sp.symbols('gamma v c p m E tau F a')
# E1
eq_E1_1 = sp.Eq(gamma_e, (1 - v_e**2/c_e**2)**-0.5)
eq_E1_2 = sp.Eq(p_e, gamma_e*m_e_ein*v_e)
eq_E1_3 = sp.Eq(E_e_ein, gamma_e*m_e_ein*c_e**2)
eq_E1_4 = sp.Eq(E_e_ein**2, c_e**2 * p_e**2 + m_e_ein**2 * c_e**4)
eq_E1_5 = sp.Eq(v_e, c_e**2 * p_e / E_e_ein)
eq_E1_6 = sp.Eq(sp.Symbol('dtau'), sp.Symbol('dt')/gamma_e)
eq_E1_7 = sp.Eq(E_e_ein, c_e*sp.Abs(p_e)) # photons

# E2
eq_E2_1 = sp.Eq(sp.Derivative(p_e, t), F_e)
eq_E2_2 = sp.Eq(sp.Derivative(E_e_ein, t), F_e*v_e) # dot prod
eq_E2_3 = sp.Eq(a_e, (F_e - (F_e*v_e)*v_e/c_e**2)/(gamma_e*m_e_ein))

# E3
V_boost, t_prime, x_prime, y_prime, z_prime, p_mu_prime, Lambda_mu_nu, p_nu, eta_mu_nu, f_obs, f_src, beta_e = sp.symbols('V t\' x\' y\' z\' p\'^mu Lambda^mu_nu p^nu eta f_{obs} f_{src} beta')
eq_E3_1 = sp.Eq(t_prime, gamma_e*(t - V_boost*x/c_e**2))
eq_E3_2 = sp.Eq(x_prime, gamma_e*(x - V_boost*t))
eq_E3_3 = sp.Eq(y_prime, y)
eq_E3_4 = sp.Eq(z_prime, z)
eq_E3_5 = sp.Eq(p_mu_prime, Lambda_mu_nu * p_nu)
eq_E3_6 = sp.Eq(sp.Symbol('Lambda^T eta Lambda'), eta_mu_nu)
eq_E3_7 = sp.Eq(eta_mu_nu, sp.Matrix([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
eq_E3_8 = sp.Eq(f_obs/f_src, sp.sqrt((1 - beta_e)/(1 + beta_e)))

# E4
p_mu, F_mu_nu, u_nu, u_mu, q_e_ein, E_i, B_k = sp.symbols('p^mu F^mu_nu u^nu u_mu q E_i B_k')
eq_E4_1 = sp.Eq(sp.Derivative(p_mu, tau_e), q_e_ein*F_mu_nu*u_nu)
eq_E4_2 = sp.Eq(u_mu*sp.Symbol('u^mu'), -c_e**2)
eq_E4_3 = sp.Eq(sp.Symbol('F^{0i}'), E_i/c_e)
eq_E4_4 = sp.Eq(sp.Symbol('F^{ij}'), sp.Symbol('epsilon^{ijk}')*B_k)
eq_E4_5 = sp.Eq(sp.Derivative(p_e, t), q_e_ein*(sp.Symbol('E') + v_e*sp.Symbol('B')))

# E5
ds_sq, g_mu_nu, dx_mu, dx_nu, Gamma_alpha_mu_nu, g_alpha_beta, R_rho_sigma_mu_nu, R_mu_nu, R_scalar, G_mu_nu, Lambda_cc, T_mu_nu = sp.symbols('ds^2 g_{mu;nu} dx^mu dx^nu Gamma^alpha_{mu;nu} g^{alpha;beta} R^rho_{sigma;mu;nu} R_{mu;nu} R G_{mu;nu} Lambda T_{mu;nu}')
eq_E5_1 = sp.Eq(ds_sq, g_mu_nu*dx_mu*dx_nu)
eq_E5_2 = sp.Eq(ds_sq, -c_e**2 * sp.Symbol('dtau')**2)
eq_E5_3 = sp.Eq(Gamma_alpha_mu_nu, 0.5*g_alpha_beta*(sp.Derivative(g_mu_nu, dx_mu) + sp.Derivative(g_mu_nu, dx_nu) - sp.Derivative(g_mu_nu, sp.Symbol('x^beta')))) # abstract deriv
eq_E5_4 = sp.Eq(sp.Derivative(sp.Symbol('x^alpha'), tau_e, 2) + Gamma_alpha_mu_nu*sp.Derivative(dx_mu, tau_e)*sp.Derivative(dx_nu, tau_e), 0)
eq_E5_5 = sp.Eq(R_rho_sigma_mu_nu, sp.Derivative(Gamma_alpha_mu_nu, dx_mu) - sp.Derivative(Gamma_alpha_mu_nu, dx_nu) + Gamma_alpha_mu_nu*Gamma_alpha_mu_nu - Gamma_alpha_mu_nu*Gamma_alpha_mu_nu) # indices abstracted
eq_E5_6 = sp.Eq(R_mu_nu, sp.Symbol('R^alpha_{mu,alpha,nu}'))
eq_E5_7 = sp.Eq(R_scalar, sp.Symbol('g^{mu,nu}')*R_mu_nu)
eq_E5_8 = sp.Eq(G_mu_nu, R_mu_nu - 0.5*R_scalar*g_mu_nu)
eq_E5_9 = sp.Eq(G_mu_nu + Lambda_cc*g_mu_nu, (8*sp.pi*G/c_e**4)*T_mu_nu)
eq_E5_10 = sp.Eq(nabla(sp.Symbol('T^{mu,nu}')), 0)

# E6
Phi_e, rho_e, xi_nu = sp.symbols('Phi rho xi_nu')
eq_E6_1 = sp.Eq(sp.Symbol('g_{00}'), -(1 + 2*Phi_e/c_e**2))
eq_E6_2 = sp.Eq(nabla(nabla(Phi_e)), 4*sp.pi*G*rho_e)
eq_E6_3 = sp.Eq(nabla(xi_nu), 0) # sym abstract
eq_E6_4 = sp.Symbol('p_mu*xi^mu')


# =====================================================================
# 11. LAVOISIER
# =====================================================================
A_es, z_s_l, nu_sr, nu_p, nu_m, M_s_l, M_e_l, b_l, n_s_l, Q_l, F_c_l = sp.symbols('A_{es} z_s nu_{sr} nu^+ nu^- M_s M_e b n_s Q F_c')
# L1
eq_L1_1 = sp.Eq(nu_sr, nu_p - nu_m)
eq_L1_2 = sp.Eq(sp.Symbol('A')*sp.Symbol('nu'), 0)
eq_L1_3 = sp.Eq(sp.Symbol('z^T')*sp.Symbol('nu'), 0)
eq_L1_4 = sp.Eq(M_s_l, sp.Sum(A_es*M_e_l, ('e', 1, sp.Symbol('N'))))
eq_L1_5 = sp.Eq(b_l, sp.Symbol('A')*n_s_l)
eq_L1_6 = sp.Eq(Q_l, F_c_l*sp.Sum(z_s_l*n_s_l, ('s', 1, sp.Symbol('N'))))

# L2
V_l, r_r_l, n_s_bound, c_s_l, omega_s_l = sp.symbols('V r_r n_s^{boundary} c_s omega_s')
eq_L2_1 = sp.Eq(sp.Derivative(n_s_l, t), V_l*sp.Sum(nu_sr*r_r_l, ('r', 1, sp.Symbol('N'))) + sp.Derivative(n_s_bound, t))
eq_L2_2 = sp.Eq(c_s_l, n_s_l/V_l)
eq_L2_3 = sp.Eq(sp.Derivative(c_s_l, t), sp.Sum(nu_sr*r_r_l, ('r', 1, sp.Symbol('N'))) + sp.Derivative(n_s_bound, t)/V_l - c_s_l*sp.Derivative(V_l, t)/V_l)
eq_L2_4 = sp.Eq(omega_s_l, M_s_l*sp.Sum(nu_sr*r_r_l, ('r', 1, sp.Symbol('N'))))

# L3
k_p, k_m, A_k, b_k_l, E_a_l, R_u_l, T_l, M_eff, alpha_s, P_r_l, k_0_l, k_inf, k_eff, F_fall = sp.symbols('k^+ k^- A_k b_k E_a R_u T [M]_{eff} alpha_s P_r k_0 k_infty k_{eff} F_{falloff}')
eq_L3_1 = sp.Eq(r_r_l, k_p*sp.Product(c_s_l**nu_m, ('s', 1, sp.Symbol('N'))) - k_m*sp.Product(c_s_l**nu_p, ('s', 1, sp.Symbol('N'))))
eq_L3_2 = sp.Eq(sp.Symbol('k(T)'), A_k*T_l**b_k_l * sp.exp(-E_a_l/(R_u_l*T_l)))
eq_L3_3 = sp.Eq(M_eff, sp.Sum(alpha_s*c_s_l, ('s', 1, sp.Symbol('N'))))
eq_L3_4 = sp.Eq(P_r_l, k_0_l*M_eff/k_inf)
eq_L3_5 = sp.Eq(k_eff, k_inf * (P_r_l/(1 + P_r_l)) * F_fall)

# L4
Delta_r_G, mu_s_l, K_a_l, Delta_r_G_circ, Q_a_l, a_s_l, c_circ, Delta_nu, q_chem, Delta_r_H, s_gen_rxn = sp.symbols('Delta_rG mu_s K_a Delta_rG^circ Q_a a_s c^circ Delta_{nu} dot{q}_{chem} Delta_rH dot{s}_{gen;rxn}')
eq_L4_1 = sp.Eq(Delta_r_G, sp.Sum(nu_sr*mu_s_l, ('s', 1, sp.Symbol('N'))))
eq_L4_2 = sp.Eq(K_a_l, sp.exp(-Delta_r_G_circ/(R_u_l*T_l)))
eq_L4_3 = sp.Eq(Q_a_l, sp.Product(a_s_l**nu_sr, ('s', 1, sp.Symbol('N'))))
eq_L4_4 = sp.Eq(Q_a_l, K_a_l) # at eq
eq_L4_5 = sp.Eq(a_s_l, c_s_l/c_circ)
eq_L4_6 = sp.Eq(k_p/k_m, (c_circ)**Delta_nu * K_a_l)
eq_L4_7 = sp.Eq(Delta_nu, sp.Sum(sp.Symbol('nu_s'), ('s', 1, sp.Symbol('N'))))
eq_L4_8 = sp.Eq(q_chem, -sp.Sum(Delta_r_H*r_r_l, ('r', 1, sp.Symbol('N'))))
eq_L4_9 = sp.Eq(s_gen_rxn, -(1/T_l)*sp.Sum(r_r_l*Delta_r_G, ('r', 1, sp.Symbol('N'))))
eq_L4_10 = s_gen_rxn >= 0

# L5
J_s_l, D_s_l, phi_l, u_l, E_eq, E_circ, n_e_l, j_l, j_0, alpha_a, alpha_c_l, eta_l, m_dep, Q_pass = sp.symbols('J_s D_s phi u E_{eq} E^circ n_e j j_0 alpha_a alpha_c eta m_{deposited} Q_{passed}')
eq_L5_1 = sp.Eq(J_s_l, -D_s_l*nabla(c_s_l) - (z_s_l*F_c_l*D_s_l/(R_u_l*T_l))*c_s_l*nabla(phi_l) + c_s_l*u_l)
eq_L5_2 = sp.Eq(E_eq, E_circ - (R_u_l*T_l/(n_e_l*F_c_l))*sp.ln(Q_a_l))
eq_L5_3 = sp.Eq(j_l, j_0*(sp.exp(alpha_a*n_e_l*F_c_l*eta_l/(R_u_l*T_l)) - sp.exp(-alpha_c_l*n_e_l*F_c_l*eta_l/(R_u_l*T_l))))
eq_L5_4 = sp.Eq(eta_l, sp.Symbol('E') - E_eq)
eq_L5_5 = sp.Eq(m_dep, Q_pass*M_s_l/(n_e_l*F_c_l))

# L6
pH, a_H, K_a_acid, a_A, a_HA, k_H_s, p_s_l, theta_l, k_a_l, k_d_l, Gamma_ads, Gamma_max = sp.symbols('pH a_{H^+} K_a^{acid} a_{A^-} a_{HA} k_{H;s} p_s theta k_a k_d Gamma_{ads} Gamma_{max}')
eq_L6_1 = sp.Eq(pH, -sp.log(a_H, 10))
eq_L6_2 = sp.Eq(K_a_acid, a_H*a_A/a_HA)
eq_L6_3 = sp.Eq(c_s_l, k_H_s*p_s_l)
eq_L6_4 = sp.Eq(sp.Derivative(theta_l, t), k_a_l*c_s_l*(1-theta_l) - k_d_l*theta_l)
eq_L6_5 = sp.Eq(Gamma_ads, Gamma_max*theta_l)

# L7
P_l, a_r_l = sp.symbols('P a_r', cls=sp.Function)
N_l, nu_r_l = sp.symbols('N nu_r')
eq_L7_1 = sp.Eq(sp.Derivative(P_l(N_l, t), t), sp.Sum(a_r_l(N_l-nu_r_l)*P_l(N_l-nu_r_l, t) - a_r_l(N_l)*P_l(N_l, t), ('r', 1, sp.Symbol('N'))))


# =====================================================================
# 12. FOURIER
# =====================================================================
q_c_f, k_f, T_f = sp.symbols('q_c k T', cls=sp.Function)
# FO1
eq_FO1_1 = sp.Eq(q_c_f(t), -k_f(t)*nabla(T_f(t)))
rho_f_t, c_p_f, q_dot_f = sp.symbols('rho c_p dot{q}')
eq_FO1_2 = sp.Eq(rho_f_t*c_p_f*sp.Derivative(T_f(t), t), nabla(k_f(t)*nabla(T_f(t))) + q_dot_f)
eq_FO1_3 = (nabla(T_f(t)) * k_f(t) * nabla(T_f(t))) / T_f(t)**2 >= 0

# FO2
h_f_t, z_f, L_alpha, Delta_f_alpha, U_i_f, H_ij_f, P_ext_f = sp.symbols('h z L_alpha Delta_{f_alpha} U_i H_{ij} P_i^{external}')
eq_FO2_1 = sp.Eq(sp.Function('h_f_t')(T_f(t), z_f) - sp.Function('h_f_t')(sp.Symbol('T_0'), sp.Symbol('z_0')), sp.Integral(c_p_f, T_f(t)) + sp.Sum(L_alpha*Delta_f_alpha, ('alpha', 1, sp.Symbol('N'))))
eq_FO2_2 = sp.Eq(sp.Derivative(U_i_f, t), sp.Sum(H_ij_f*(sp.Symbol('T_j') - sp.Symbol('T_i')), ('j', 1, sp.Symbol('N'))) + P_ext_f)
eq_FO2_3 = sp.Eq(sp.Symbol('T_i'), sp.Function('T')(U_i_f, sp.Symbol('composition'), sp.Symbol('z_i')))
eq_FO2_4 = sp.Eq(H_ij_f, k_f(t)*sp.Symbol('A')/sp.Symbol('L'))
eq_FO2_5 = sp.Eq(H_ij_f, sp.Symbol('H_{ji}'))

# FO3
q_pp, h_c_f, T_s_f, h_tc, T_1, T_2, j_m_f, n_vec_f = sp.symbols('q\'\' h_c T_s h_{tc} T_1 T_2 j_m n')
eq_FO3_1 = sp.Eq(q_pp, h_c_f*(T_s_f - T_f(t)))
eq_FO3_2 = sp.Eq(q_pp, h_tc*(T_1 - T_2))
eq_FO3_3 = sp.Eq(sp.Symbol('Delta(q_n + j_m h)'), 0)

# FO4
B_nu, h_p, nu_f, c_f, k_B_f, E_b_f, sig_SB, P_net_f, eps_f, A_f, T_env_f = sp.symbols('B_nu h_p nu c k_B E_b sigma_{SB} P_{net} epsilon A T_{env}')
eq_FO4_1 = sp.Eq(sp.Function('B_nu')(T_f(t)), (2*h_p*nu_f**3/c_f**2) * 1/(sp.exp(h_p*nu_f/(k_B_f*T_f(t))) - 1))
eq_FO4_2 = sp.Eq(E_b_f, sig_SB*T_f(t)**4)
eq_FO4_3 = sp.Eq(P_net_f, eps_f*sig_SB*A_f*(T_s_f**4 - T_env_f**4))

# FO5
G_i_f, F_ij_f, J_j_f, J_i_f, eps_i_f, T_i_f, P_i_f, A_i_f, A_j_f, F_ji_f = sp.symbols('G_i F_{ij} J_j J_i epsilon_i T_i P_i A_i A_j F_{ji}')
eq_FO5_1 = sp.Eq(G_i_f, sp.Sum(F_ij_f*J_j_f, ('j', 1, sp.Symbol('N'))))
eq_FO5_2 = sp.Eq(J_i_f, eps_i_f*sig_SB*T_i_f**4 + (1 - eps_i_f)*G_i_f)
eq_FO5_3 = sp.Eq(P_i_f, A_i_f*(J_i_f - G_i_f))
eq_FO5_4 = sp.Eq(sp.Sum(F_ij_f, ('j', 1, sp.Symbol('N'))), 1)
eq_FO5_5 = sp.Eq(A_i_f*F_ij_f, A_j_f*F_ji_f)

# FO6
I_nu, Omega_f, kappa_a, kappa_s, p_nu_f, S_nu = sp.symbols('I_nu Omega kappa_a kappa_s p_nu S_nu', cls=sp.Function)
eq_FO6_1 = sp.Eq((1/c_f)*sp.Derivative(I_nu(t), t) + Omega_f(t)*nabla(I_nu(t)), -(kappa_a(t) + kappa_s(t))*I_nu(t) + kappa_a(t)*sp.Function('B_nu')(T_f(t)) + kappa_s(t)*sp.Integral(p_nu_f(t)*I_nu(t), sp.Symbol('Omega\'')))

# FO7
C_min, m_dot_h, c_ph, m_dot_c, c_pc, C_r, NTU, U_f, A_f_ht, eps_ht, Q_dot_ht, T_hin, T_cin, J_te, sig_e, E_te, S_T, q_te, Pi_te = sp.symbols('C_{min} dot{m}_h c_{p;h} dot{m}_c c_{p;c} C_r NTU U A epsilon dot{Q} T_{h;in} T_{c;in} J sigma_e E S_T q Pi')
eq_FO7_1 = sp.Eq(C_min, sp.Min(m_dot_h*c_ph, m_dot_c*c_pc))
eq_FO7_2 = sp.Eq(C_r, C_min/sp.Symbol('C_{max}'))
eq_FO7_3 = sp.Eq(NTU, U_f*A_f_ht/C_min)
eq_FO7_4 = sp.Eq(eps_ht, (1 - sp.exp(-NTU*(1 - C_r)))/(1 - C_r*sp.exp(-NTU*(1 - C_r))))
eq_FO7_5 = sp.Eq(eps_ht, NTU/(1 + NTU)) # Cr=1
eq_FO7_6 = sp.Eq(Q_dot_ht, eps_ht*C_min*(T_hin - T_cin))
eq_FO7_7 = sp.Eq(J_te, sig_e*(E_te - S_T*nabla(T_f(t))))
eq_FO7_8 = sp.Eq(q_te, Pi_te*J_te - k_f(t)*nabla(T_f(t)))
eq_FO7_9 = sp.Eq(Pi_te, S_T*T_f(t))

# FO8
Bi, h_c_fo, L_c_fo, k_fo, Fo, alpha_fo, Delta_t_fo, rho_fo, c_p_fo = sp.symbols('Bi h_c L_c k Fo alpha Delta_t rho c_p')
eq_FO8_1 = sp.Eq(Bi, h_c_fo*L_c_fo/k_fo)
eq_FO8_2 = sp.Eq(Fo, alpha_fo*Delta_t_fo/L_c_fo**2)
eq_FO8_3 = sp.Eq(alpha_fo, k_fo/(rho_fo*c_p_fo))


# =====================================================================
# 13. BOLTZMANN
# =====================================================================
f_s, v_bo, F_s, m_s_bo, C_s_bo, S_s_bo, B_bo, Omega_bo = sp.symbols('f_s v F_s m_s C_s S_s B Omega', cls=sp.Function)
# BO1
eq_BO1_1 = sp.Eq(sp.Derivative(f_s(t), t) + v_bo(t)*nabla(f_s(t)) + (F_s(t)/m_s_bo(t))*nabla(f_s(t)), C_s_bo(f_s(t)) + S_s_bo(t))
eq_BO1_2 = sp.Eq(C_s_bo(f_s(t)), sp.Integral(B_bo(t)*(sp.Symbol('f\'f_*\'') - f_s(t)*sp.Symbol('f_*')), (sp.Symbol('v_*'), sp.Symbol('Omega'))))

# BO2
n_bo, rho_bo, u_bo, P_bo, e_tr, q_bo = sp.symbols('n rho u P e_{transl} q')
eq_BO2_1 = sp.Eq(n_bo, sp.Integral(f_s(t), v_bo(t)))
eq_BO2_2 = sp.Eq(rho_bo, m_s_bo(t)*n_bo)
eq_BO2_3 = sp.Eq(rho_bo*u_bo, m_s_bo(t)*sp.Integral(v_bo(t)*f_s(t), v_bo(t)))
eq_BO2_4 = sp.Eq(P_bo, m_s_bo(t)*sp.Integral((v_bo(t)-u_bo)*(v_bo(t)-u_bo)*f_s(t), v_bo(t)))
eq_BO2_5 = sp.Eq(rho_bo*e_tr, (m_s_bo(t)/2)*sp.Integral(sp.Abs(v_bo(t)-u_bo)**2*f_s(t), v_bo(t)))
eq_BO2_6 = sp.Eq(q_bo, (m_s_bo(t)/2)*sp.Integral(sp.Abs(v_bo(t)-u_bo)**2*(v_bo(t)-u_bo)*f_s(t), v_bo(t)))

# BO3
f_M, k_B_bo, T_bo, C_BGK, tau_bo, mu_bo, Pr, Kn_bo = sp.symbols('f_M k_B T C_{BGK} tau mu Pr Kn')
eq_BO3_1 = sp.Eq(f_M, n_bo*(m_s_bo(t)/(2*sp.pi*k_B_bo*T_bo))**(3/2) * sp.exp(-m_s_bo(t)*sp.Abs(v_bo(t)-u_bo)**2/(2*k_B_bo*T_bo)))
eq_BO3_2 = sp.Eq(C_BGK, -(f_s(t) - f_M)/tau_bo)
eq_BO3_3 = sp.Eq(mu_bo, sp.Symbol('p')*tau_bo)
eq_BO3_4 = sp.Eq(Pr, 1)
eq_BO3_5 = sp.Eq(f_s(t), sp.Symbol('f^{(0)}') + Kn_bo*sp.Symbol('f^{(1)}')) # plus dots

# BO4
H_bo = sp.Symbol('H')
eq_BO4_1 = sp.Eq(sp.Integral(C_s_bo(f_s(t))*sp.Symbol('{1, mv, 1/2mv^2}'), v_bo(t)), 0)
eq_BO4_2 = sp.Eq(H_bo, sp.Integral(f_s(t)*sp.ln(f_s(t)/sp.Symbol('f_*')), (x, v_bo(t))))
eq_BO4_3 = sp.Derivative(H_bo, t) <= 0 # collis

# BO5
lam_mfp, sig_coll, L_bo = sp.symbols('lambda_{mfp} sigma_{coll} L')
eq_BO5_1 = sp.Eq(lam_mfp, 1/(sp.sqrt(2)*n_bo*sig_coll))
eq_BO5_2 = sp.Eq(Kn_bo, lam_mfp/L_bo)

# BO6
Z_bo, beta_bo, A_bo, U_bo, S_bo, n_FD, n_BE, eps_bo, mu_bo_chem = sp.symbols('Z beta A U S bar{n}_{FD} bar{n}_{BE} epsilon mu')
eq_BO6_1 = sp.Eq(Z_bo, tr(sp.exp(-beta_bo*sp.Symbol('H'))))
eq_BO6_2 = sp.Eq(beta_bo, (k_B_bo*T_bo)**-1)
eq_BO6_3 = sp.Eq(A_bo, -k_B_bo*T_bo*sp.ln(Z_bo))
eq_BO6_4 = sp.Eq(U_bo, -sp.Derivative(sp.ln(Z_bo), beta_bo))
eq_BO6_5 = sp.Eq(S_bo, k_B_bo*(sp.ln(Z_bo) + beta_bo*U_bo))
eq_BO6_6 = sp.Eq(n_FD, 1/(sp.exp(beta_bo*(eps_bo - mu_bo_chem)) + 1))
eq_BO6_7 = sp.Eq(n_BE, 1/(sp.exp(beta_bo*(eps_bo - mu_bo_chem)) - 1))

# BO7
zeta_bo, F_cons_bo, dW, D_bo = sp.symbols('zeta F_{cons} dW D')
eq_BO7_1 = sp.Eq(sp.Symbol('dx'), v_bo(t)*sp.Symbol('dt'))
eq_BO7_2 = sp.Eq(m_s_bo(t)*sp.Symbol('dv'), F_cons_bo*sp.Symbol('dt') - zeta_bo*v_bo(t)*sp.Symbol('dt') + sp.sqrt(2*zeta_bo*k_B_bo*T_bo)*dW)
eq_BO7_3 = sp.Eq(D_bo, k_B_bo*T_bo/zeta_bo)

# BO8
psi_bo, v_E, Sig_t, Sig_s, S_bo_rad = sp.symbols('psi v Sigma_t Sigma_s S', cls=sp.Function)
eq_BO8_1 = sp.Eq((1/v_E(t))*sp.Derivative(psi_bo(t), t) + Omega_bo(t)*nabla(psi_bo(t)) + Sig_t(t)*psi_bo(t), sp.Integral(Sig_s(t)*psi_bo(t), (sp.Symbol('E\''), sp.Symbol('Omega\''))) + S_bo_rad(t))


# =====================================================================
# 14. NOETHER
# =====================================================================
# NO1
Q_N, L_n, q_dot_a, eta_a, H_n, tau_n, B_n, j_mu_n, L_cal, phi_a, K_mu = sp.symbols('Q_N L dot{q}_a eta_a H tau B j^mu mathcal{L} phi_a K^mu')
eq_NO1_1 = sp.Eq(Q_N, sp.Sum(sp.Derivative(L_n, q_dot_a)*eta_a, ('a', 1, sp.Symbol('N'))) - H_n*tau_n - B_n)
eq_NO1_2 = sp.Eq(sp.Derivative(Q_N, t), 0)
eq_NO1_3 = sp.Eq(j_mu_n, sp.Sum(sp.Derivative(L_cal, sp.Symbol('(partial_mu phi_a)'))*sp.Symbol('delta phi_a'), ('a', 1, sp.Symbol('N'))) - K_mu)
eq_NO1_4 = sp.Eq(nabla(j_mu_n), 0) # partial_mu j^mu

# NO2
R_Q, Q_t1, Q_t0, F_Q, S_Q = sp.symbols('R_Q Q(t_1) Q(t_0) mathcal{F}_Q S_Q')
eq_NO2_1 = sp.Eq(R_Q, Q_t1 - Q_t0 + sp.Integral(F_Q*sp.Symbol('n'), (t, sp.Symbol('A'))) - sp.Integral(S_Q, (t, sp.Symbol('V'))))

# NO3
DQ_a, DQ_b, DQ_link = sp.symbols('DeltaQ_{a->b}^{(a)} DeltaQ_{a->b}^{(b)} DeltaQ_{link}')
eq_NO3_1 = sp.Eq(DQ_a + DQ_b, 0)
eq_NO3_2 = sp.Eq(sp.Symbol('Delta Q_a') + DQ_link + sp.Symbol('Delta Q_b'), 0)

# NO4
A_n, Dn, Dn_b, F_c_n, z_n, DQ_bnd, d2_b, rho_f_n, J_f_n = sp.symbols('A Delta_n Delta_n_{boundary} F_c z Delta_Q_{boundary} d_2b rho_f J_f')
eq_NO4_1 = sp.Eq(A_n*Dn, A_n*Dn_b)
eq_NO4_2 = sp.Eq(F_c_n*z_n*Dn, DQ_bnd) # transpose abstract
eq_NO4_3 = sp.Eq(d2_b, 0)
eq_NO4_4 = sp.Eq(sp.Symbol('Gauss residual'), nabla(sp.Symbol('D')) - rho_f_n)
eq_NO4_5 = sp.Eq(sp.Derivative(rho_f_n, t) + nabla(J_f_n), 0)

# NO5
rho_pos, T_pos, n_s_pos, Y_s_sum, S_gen_pos, rho_hat_dag, rho_hat_tr, rho_hat_pos, P_diss, R_ortho, R_det = sp.symbols('rho T n_s Y_s dot{S}_{gen} hat{varrho} hat{varrho}^dagger hat{varrho}_pos P_{diss} R R_det')
eq_NO5_1 = rho_pos > 0
eq_NO5_2 = T_pos > 0
eq_NO5_3 = n_s_pos >= 0
eq_NO5_4 = sp.Eq(sp.Sum(Y_s_sum, ('s', 1, sp.Symbol('N'))), 1)
eq_NO5_5 = S_gen_pos >= 0
eq_NO5_6 = sp.Eq(rho_hat_dag, sp.conjugate(rho_hat_dag))
eq_NO5_7 = sp.Eq(tr(rho_hat_dag), 1)
eq_NO5_8 = rho_hat_pos >= 0 # positive semi-def abstract
eq_NO5_9 = P_diss >= 0
eq_NO5_10 = sp.Eq(transpose(R_ortho) * R_ortho, sp.Identity(3), evaluate=False)
eq_NO5_11 = sp.Eq(det(R_ortho), 1)

# NO6
r_Q_norm, a_Q, r_Q_tol, Q_scale = sp.symbols('r_Q a_Q r_Q^{tol} Q_{scale}')
eq_NO6_1 = sp.Eq(r_Q_norm, sp.Abs(R_Q) / (a_Q + r_Q_tol*Q_scale))
eq_NO6_2 = sp.Eq(sp.Symbol('||y_h - y_{h/2}||'), 2**sp.Symbol('p') * sp.Symbol('||y_{h/2} - y_{h/4}||'))

# =====================================================================
# 15. COUPLING (X1-X4)
# =====================================================================
P_mech, F_x, v_x, tau_x, omega_x, P_elec, V_x, I_x, P_heat, Q_dot_x, P_chem, mu_s_x, n_dot_s_x = sp.symbols('P_{mech} F v tau omega P_{electric} V I P_{heat} dot{Q} P_{chemical} mu_s dot{n}_s')
eq_X1_1 = sp.Eq(P_mech, F_x*v_x + tau_x*omega_x)
eq_X1_2 = sp.Eq(P_elec, V_x*I_x)
eq_X1_3 = sp.Eq(P_heat, Q_dot_x)
eq_X1_4 = sp.Eq(P_chem, sp.Sum(mu_s_x*n_dot_s_x, ('s', 1, sp.Symbol('N'))))

m_dot_x, rho_x, u_x, v_b_x, n_x, m_dot_s_x, Y_s_x, j_s_x = sp.symbols('dot{m} rho u v_b n dot{m}_s Y_s j_s')
eq_X2_1 = sp.Eq(m_dot_x, sp.Integral(rho_x*(u_x - v_b_x)*n_x, sp.Symbol('A')))
eq_X2_2 = sp.Eq(m_dot_s_x, sp.Integral((rho_x*Y_s_x*(u_x - v_b_x) + j_s_x)*n_x, sp.Symbol('A')))

Q_tot, w_i, Q_i, Q_dot_tot, Q_dot_i = sp.symbols('Q_{total} w_i Q_i dot{Q}_{total} dot{Q}_i')
eq_X3_1 = sp.Eq(Q_tot, sp.Sum(w_i*Q_i, ('i', 1, sp.Symbol('N'))))
eq_X3_2 = sp.Eq(Q_dot_tot, sp.Sum(w_i*Q_dot_i, ('i', 1, sp.Symbol('N'))))

M_x, m_i_x, P_x, p_i_x, E_x, E_i_x, N_s_x, N_si_x = sp.symbols('M m_i P p_i E E_i N_s N_{s;i}')
eq_X4_1 = sp.Eq(M_x, sp.Sum(w_i*m_i_x, ('i', 1, sp.Symbol('N'))))
eq_X4_2 = sp.Eq(P_x, sp.Sum(w_i*p_i_x, ('i', 1, sp.Symbol('N'))))
eq_X4_3 = sp.Eq(E_x, sp.Sum(w_i*E_i_x, ('i', 1, sp.Symbol('N'))))
eq_X4_4 = sp.Eq(N_s_x, sp.Sum(w_i*N_si_x, ('i', 1, sp.Symbol('N'))))

y_c, L_map, R_map, xi = sp.symbols('y_{coarse} L R xi')
eq_X5_1 = sp.Eq(sp.Function('R_map')(sp.Function('L_map')(y_c, xi)), y_c)


# =====================================================================
# 18. TARTAGLIA (External Ballistics and Flight Mechanics)
# =====================================================================
v_vec, u_wind, v_rel, M_mach, c_s, z = sp.symbols('v u_{wind} v_{rel} M c_s z')
# TA1
eq_TA1_1 = sp.Eq(v_rel, v_vec - u_wind)
eq_TA1_2 = sp.Eq(M_mach, sp.Abs(v_rel) / c_s)

# TA2
F_aero, q_dyn, S_ref, C_D, C_L, n_vec, F_Magnus = sp.symbols('F_{aero} q S C_D C_L n F_{Magnus}')
p_spin, d_ref, C_Ypalpha = sp.symbols('p d C_{Yp\\alpha}')
# Cross products abstracted as functions/multiplication for sympy mapping
cross_n_vrel = sp.Function('cross')(n_vec, v_rel)
cross_vrel_n_vrel = sp.Function('cross')(v_rel, cross_n_vrel)
eq_TA2_1 = sp.Eq(F_aero, -q_dyn * S_ref * (C_D * (v_rel / sp.Abs(v_rel)) + C_L * (cross_vrel_n_vrel / sp.Abs(cross_vrel_n_vrel))) + F_Magnus)
eq_TA2_2 = sp.Eq(F_Magnus, q_dyn * S_ref * (p_spin * d_ref / sp.Abs(v_rel)) * C_Ypalpha * cross_n_vrel)

# TA3
M_aero, C_Malpha, C_Mp, C_Mq, omega_t = sp.symbols('M_{aero} C_{M\\alpha} C_{Mp} C_{Mq} \\omega_t')
eq_TA3_1 = sp.Eq(M_aero, q_dyn * S_ref * d_ref * (C_Malpha * cross_n_vrel + C_Mp * (p_spin * d_ref / (2 * sp.Abs(v_rel))) * n_vec + C_Mq * (omega_t * d_ref / (2 * sp.Abs(v_rel)))))

# TA4
S_g, I_x, I_y = sp.symbols('S_g I_x I_y')
eq_TA4_1 = sp.Eq(S_g, (I_x**2 * p_spin**2) / (4 * I_y * q_dyn * S_ref * d_ref * C_Malpha))
eq_TA4_2 = S_g > 1


# =====================================================================
# 19. PIOBERT (Internal Ballistics and Confined Propellant)
# =====================================================================
r_burn, w, beta_burn, P_chamber, alpha_burn = sp.symbols('r w beta P \\alpha')
m_dot_g, rho_p, V_p0, phi_burn, f_unburned, theta_form = sp.symbols('\\dot{m}_g \\rho_p V_{p;0} \\phi f \\theta')
# P1
eq_P1_1 = sp.Eq(r_burn, sp.Derivative(w, t))
eq_P1_2 = sp.Eq(r_burn, beta_burn * P_chamber**alpha_burn)
eq_P1_3 = sp.Eq(m_dot_g, rho_p * V_p0 * sp.Derivative(phi_burn, f_unburned) * sp.Derivative(f_unburned, t))
eq_P1_4 = sp.Eq(phi_burn, (1 - f_unburned) * (1 + theta_form * f_unburned))

# P2
V_c, eta_covol, m_g, R_s_gas, T_g = sp.symbols('V_c \\eta m_g R_s T_g')
V_c_t = sp.Function('V_c')(t)
eq_P2_1 = sp.Eq(P_chamber * (V_c_t - eta_covol * m_g - V_p0 * (1 - phi_burn)), m_g * R_s_gas * T_g)

# P3
c_v, e_ex, Q_dot_loss = sp.symbols('c_v e_{ex} \\dot{Q}_{loss}')
eq_P3_1 = sp.Eq(sp.Derivative(m_g * c_v * T_g, t), m_dot_g * e_ex - P_chamber * sp.Derivative(V_c_t, t) - Q_dot_loss)

# P4
V_chamber, A_bore, x_proj, m_proj, P_base, F_resist = sp.symbols('V_{chamber} A_{bore} x m_{proj} P_{base} F_{resist}')
P_bulk, P_breech = sp.symbols('P_{bulk} P_{breech}')
x_proj_t = sp.Function('x')(t)
eq_P4_1 = sp.Eq(V_c_t, V_chamber + A_bore * x_proj_t)
eq_P4_2 = sp.Eq(m_proj * sp.Derivative(x_proj_t, t, 2), P_base * A_bore - sp.Function('F_{resist}')(x_proj_t))
eq_P4_3 = sp.Eq(P_base, P_bulk / (1 + m_g / (3 * m_proj)))
eq_P4_4 = sp.Eq(P_breech, P_base * (1 + m_g / (2 * m_proj)))


# =====================================================================
# 20. ZELDOVICH (Unconstrained Detonation and Blast Waves)
# =====================================================================
P_CJ, rho_0, D_CJ, rho_CJ, v_CJ, c_s_CJ = sp.symbols('P_{CJ} \\rho_0 D_{CJ} \\rho_{CJ} v_{CJ} c_{s;CJ}')
# Z1
eq_Z1_1 = sp.Eq(P_CJ, rho_0 * D_CJ**2 * (1 - rho_0 / rho_CJ))
eq_Z1_2 = sp.Eq(D_CJ, v_CJ + c_s_CJ)

# Z2
G_ls, u_vec, S_T = sp.symbols('G u S_T')
eq_Z2_1 = sp.Eq(sp.Derivative(G_ls, t) + u_vec * nabla(G_ls), S_T * sp.Abs(nabla(G_ls))) # Inner product abstracted

# Z3
R_shock, xi_0, E_blast, D_shock = sp.symbols('R \\xi_0 E D')
R_shock_t = sp.Function('R')(t)
eq_Z3_1 = sp.Eq(R_shock_t, xi_0 * (E_blast / rho_0)**(1/5) * t**(2/5))
eq_Z3_2 = sp.Eq(sp.Function('D_shock')(t), sp.Derivative(R_shock_t, t))
eq_Z3_3 = sp.Eq(sp.Function('D_shock')(t), (2/5) * (R_shock_t / t))

# Z4
P_t, P_0, P_s, t_a, t_star = sp.symbols('P(t) P_0 P_s t_a t_*')
eq_Z4_1 = sp.Eq(P_t, P_0 + P_s * sp.exp(-(t - t_a) / t_star) * (1 - (t - t_a) / t_star))


# =====================================================================
# 21. OTTO (Constrained Chamber Combustion and Cyclic Work)
# =====================================================================
theta, a_crank, l_rod, A_c = sp.symbols('\\theta a l A_c')
V_theta, V_c_otto = sp.symbols('V(\\theta) V_c')
# O1
eq_O1_1 = sp.Eq(V_theta, V_c_otto + A_c * (l_rod + a_crank - a_crank * sp.cos(theta) - sp.sqrt(l_rod**2 - a_crank**2 * sp.sin(theta)**2)))

# O2
U_dot_u, P_cyl, V_dot_u, Q_dot_u, m_dot_b, h_u = sp.symbols('\\dot{U}_u P \\dot{V}_u \\dot{Q}_u \\dot{m}_b h_u')
U_dot_b, V_dot_b, Q_dot_b = sp.symbols('\\dot{U}_b \\dot{V}_b \\dot{Q}_b')
eq_O2_1 = sp.Eq(U_dot_u, -P_cyl * V_dot_u - Q_dot_u - m_dot_b * h_u)
eq_O2_2 = sp.Eq(U_dot_b, -P_cyl * V_dot_b - Q_dot_b + m_dot_b * h_u)

# O3
x_b, a_wiebe, theta_0, Delta_theta, m_wiebe, m_total = sp.symbols('x_b a \\theta_0 \\Delta\\theta m m_{total}')
eq_O3_1 = sp.Eq(x_b, 1 - sp.exp(-a_wiebe * ((theta - theta_0) / Delta_theta)**(m_wiebe + 1)))
eq_O3_2 = sp.Eq(m_dot_b, m_total * sp.Derivative(x_b, theta) * sp.Derivative(theta, t))

# O4
Q_dot_wall, h_c_woschni, A_wall, T_gas, T_wall = sp.symbols('\\dot{Q} h_c A_{wall} T_{gas} T_{wall}')
C_woschni, d_bore, v_char = sp.symbols('C d v_{char}')
eq_O4_1 = sp.Eq(Q_dot_wall, h_c_woschni * A_wall * (T_gas - T_wall))
eq_O4_2 = sp.Eq(h_c_woschni, C_woschni * d_bore**-0.2 * P_cyl**0.8 * sp.Symbol('T')**-0.53 * v_char**0.8)

# O5
m_dot_valve, C_D_valve, A_valve, R_gas, Psi_flow, P_T = sp.symbols('\\dot{m} C_D A_{valve} R \\Psi P_T')
eq_O5_1 = sp.Eq(m_dot_valve, C_D_valve * sp.Function('A_{valve}')(theta) * (sp.Symbol('P_0') / sp.sqrt(R_gas * sp.Symbol('T_0'))) * sp.Function('\\Psi')(P_T / sp.Symbol('P_0')))


# =====================================================================
# 23. DE LAVAL (Nozzles and Compressible Expansion)
# =====================================================================
A, A_star, M, gamma = sp.symbols('A A^* M gamma')
# DL1
eq_DL1_1 = sp.Eq(A/A_star, (1/M) * ((2/(gamma + 1)) * (1 + (gamma - 1)/2 * M**2))**((gamma + 1)/(2*(gamma - 1))))

# DL2
m_dot, P_0, T_0, R_s, P_static, T_static = sp.symbols('dot{m} P_0 T_0 R_s P T')
eq_DL2_1 = sp.Eq(m_dot, (P_0 * A_star / sp.sqrt(T_0)) * sp.sqrt(gamma / R_s) * (2/(gamma + 1))**((gamma + 1)/(2*(gamma - 1))))
eq_DL2_2 = sp.Eq(P_0/P_static, (1 + (gamma - 1)/2 * M**2)**(gamma/(gamma - 1)))
eq_DL2_3 = sp.Eq(T_0/T_static, 1 + (gamma - 1)/2 * M**2)

# DL3
C_F, P_e, P_a, A_e = sp.symbols('C_F P_e P_a A_e')
term1 = sp.sqrt((2*gamma**2)/(gamma - 1) * (2/(gamma + 1))**((gamma + 1)/(gamma - 1)) * (1 - (P_e/P_0)**((gamma - 1)/gamma)))
term2 = ((P_e - P_a)/P_0) * (A_e/A_star)
eq_DL3_1 = sp.Eq(C_F, term1 + term2)


# =====================================================================
# 24. TSIOLKOVSKY (Reactive Thrust and Kinematics)
# =====================================================================
F_thrust, v_e, c_eff, I_sp, g_0 = sp.symbols('F_{thrust} v_e c I_{sp} g_0')
# TS1
eq_TS1_1 = sp.Eq(F_thrust, m_dot * v_e + (P_e - P_a) * A_e)
eq_TS1_2 = sp.Eq(F_thrust, m_dot * c_eff)
eq_TS1_3 = sp.Eq(c_eff, v_e + (P_e - P_a)*A_e/m_dot)
eq_TS1_4 = sp.Eq(c_eff, I_sp * g_0)

# TS2
Delta_V, m_0, m_f = sp.symbols('DeltaV m_0 m_f')
eq_TS2_1 = sp.Eq(Delta_V, c_eff * sp.ln(m_0 / m_f))

# TS3
E_mech, v_vec = sp.symbols('mathcal{E} v')
eq_TS3_1 = sp.Eq(sp.Derivative(E_mech, t), F_thrust * v_vec)
eq_TS3_2 = sp.Eq(F_thrust * v_vec, m_dot * c_eff * v_vec)


# =====================================================================
# 25. ARCHIMEDES (Hydrostatics and Buoyancy)
# =====================================================================
F_b_vec, P_field, n_vec, dA, rho_f, V_sub, g_vec, x_vec, dV = sp.symbols('F_b P n dA rho_f V_{sub} g x dV')
rho_f_func = sp.Function('rho_f')(x_vec)
# AR1
eq_AR1_1 = sp.Eq(F_b_vec, -sp.Integral(P_field * n_vec, sp.Symbol('partial V')))
eq_AR1_2 = sp.Eq(F_b_vec, rho_f * V_sub * g_vec)
eq_AR1_3 = sp.Eq(F_b_vec, sp.Integral(rho_f_func * g_vec, V_sub))

# AR2
GM_bar, I_w, BG_bar = sp.symbols('overline{GM} I_w overline{BG}')
eq_AR2_1 = sp.Eq(GM_bar, I_w / V_sub - BG_bar)
eq_AR2_2 = GM_bar > 0


# =====================================================================
# ENGINE / EQUATION REGISTRY
# =====================================================================
# Everything above is a flat script: every eq_* name is a module global.
# This section discovers them rather than hand-duplicating them into a
# second data structure, so appending a new numbered engine section
# above (as this file's own history has already done twice) needs no
# update here.

import re as _re

# prefix -> (engine display name, markdown section number or None if this
# engine has equations here but isn't in HONORARY_ENGINE_EQUATION_CATALOGUE.md)
ENGINE_PREFIXES = {
    'N':  ('Newton', 1),
    'T':  ('Timoshenko', 2),
    'F':  ('Faraday', 3),
    'NS': ('Navier-Stokes', 4),
    'B':  ('Bjerknes', 5),
    'G':  ('Gibbs', 6),
    'BR': ('Bragg', 7),
    'H':  ('Hamilton', 8),
    'C':  ('Curie', 9),
    'E':  ('Einstein', 10),
    'L':  ('Lavoisier', 11),
    'FO': ('Fourier', 12),
    'BO': ('Boltzmann', 13),
    'NO': ('Noether', 14),
    'X':  ('Coupling', 15),
    'TA': ('Tartaglia', 18),
    'P':  ('Piobert', 19),
    'Z':  ('Zeldovich', 20),
    'O':  ('Otto', 21),
    'DL': ('De Laval', 23),
    'TS': ('Tsiolkovsky', 24),
    'AR': ('Archimedes', 25),
}

_EQ_NAME_RE = _re.compile(r'^eq_([A-Z]+)(\d+)_(\d+[a-z]?)$')


def _discover_equations():
    """Group every eq_* module global by its honorary-engine prefix."""
    by_engine = {}
    for name, value in globals().items():
        m = _EQ_NAME_RE.match(name)
        if m is None:
            continue
        prefix = m.group(1)
        engine, _section = ENGINE_PREFIXES.get(prefix, (prefix, None))
        by_engine.setdefault(engine, {})[name] = value
    return by_engine


def available_engines():
    """Every honorary engine that actually has equations in this module."""
    return sorted(_discover_equations())


def equations_by_engine(engine):
    """All eq_* objects for one engine, keyed by their eq_ name."""
    by_engine = _discover_equations()
    if engine not in by_engine:
        raise KeyError(f"no equations for engine {engine!r}; available: {available_engines()}")
    return by_engine[engine]


def equations_for(*engines):
    """A merged {name: equation} dict for a chosen subset of engines.

    This is the entry point for making a modular decision about which
    equation sets a sim build actually needs: pick the engines, get
    exactly their laws, nothing implicitly pulled in from the rest of
    the catalogue.
    """
    by_engine = _discover_equations()
    out = {}
    for engine in engines:
        if engine not in by_engine:
            raise KeyError(f"no equations for engine {engine!r}; available: {available_engines()}")
        out.update(by_engine[engine])
    return out


# A few illustrative starting bundles, not an authoritative taxonomy.
# See HONORARY_ENGINE_EQUATION_CATALOGUE.md section 16.4 ("a practical
# chamber convergence witness") and the shared lineage of one gun's
# ballistics engines (Tartaglia/Piobert/Zeldovich/Otto/De Laval/Tsiolkovsky).
ENGINE_SETS = {
    'chamber_witness': ('Lavoisier', 'Bjerknes', 'Fourier', 'Faraday'),
    'ballistics_stack': ('Newton', 'Timoshenko', 'Tartaglia', 'Piobert',
                         'Zeldovich', 'Otto', 'De Laval', 'Tsiolkovsky'),
}


# =====================================================================
# SYMBOL IDENTITY REGISTRY
# =====================================================================
# Section 0.1 of the source catalogue: "Symbols have namespaces and
# units; identical spelling is not sufficient to identify physical
# state." This file's own code is not exempt: it reuses bare display
# strings like "M" and "A" for genuinely different physical quantities
# across engines -- entirely legitimately, because each engine's local
# Python name (M_t, M_mag, M_b, M_x, ...) already disambiguates them for
# anyone reading this module. The moment two equations from different
# engines are combined into one expression, that protection disappears:
# sympy sees only the display string, and Symbol('M') built by
# Timoshenko IS, to sympy, Symbol('M') built by Bjerknes.
#
# This registry distinguishes two things that look identical to sympy:
#   - a name that COLLIDES (same string, different physical quantity)
#   - a name that is genuinely SHARED (same string, same quantity, used
#     by more than one engine, sometimes with one engine owning its
#     evolution and another only reading it)
# and gives one dynamic-resolution policy for both, `describe_symbol`.
# It is a hand-curated sample verified against this file's own text, not
# an exhaustive audit of every repeated token here -- see
# `raw_token_report` for the (much longer, uncurated) rest.

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolUsage:
    engine: str
    python_name: str
    note: str = ""


@dataclass(frozen=True)
class SymbolIdentity:
    key: str
    description: str
    usages: tuple


# token -> tuple of DISTINCT identities that all print as that token.
# More than one identity under a token means: same spelling, NOT the
# same physical quantity -- do not combine expressions using it across
# engines without renaming one of them first.
KNOWN_COLLISIONS = {
    'M': (
        SymbolIdentity(
            'mach_number', 'Local Mach number M = |v|/c_s.',
            (SymbolUsage('De Laval', 'M', 'DL1 area-Mach relation'),
             SymbolUsage('Tartaglia', 'M_mach', 'TA1 flight Mach number'))),
        SymbolIdentity(
            'bending_moment', "Timoshenko's beam bending-moment resultant.",
            (SymbolUsage('Timoshenko', 'M_t', 'T1/T2 beam kinematics and dynamics'),)),
        SymbolIdentity(
            'magnetization_field',
            "Magnetization M in B = mu0(H+M). Bragg (BR11) owns its LLG "
            "evolution, Faraday (F1) only reads it as a constitutive closure "
            "-- see the Faraday section preamble: \"Shared polarization/"
            "magnetization states have one owner.\"",
            (SymbolUsage('Faraday', 'M_f', 'F1 constitutive closure, reader'),
             SymbolUsage('Bragg', 'M_mag', 'BR11 Landau-Lifshitz-Gilbert dynamics, owner'))),
        SymbolIdentity(
            'molar_mass', 'Species molar mass in the Hertz-Knudsen phase-flux formula.',
            (SymbolUsage('Bjerknes', 'M_b', 'B2/B4 saturation and interfacial flux'),)),
        SymbolIdentity(
            'total_mass', 'Aggregate restricted mass of a population of instances.',
            (SymbolUsage('Coupling', 'M_x', 'X4 fine-to-coarse restriction'),)),
    ),
    'A': (
        SymbolIdentity(
            'area', 'A cross-section, surface, or heat-exchanger area.',
            (SymbolUsage('Timoshenko', 'A_t', 'T1 beam cross-section'),
             SymbolUsage('Navier-Stokes', 'A_area', 'NS6 passage cross-section'),
             SymbolUsage('Fourier', 'A_f', 'FO4 radiating surface area'),
             SymbolUsage('Fourier', 'A_f_ht', 'FO7 heat-exchanger area'))),
        SymbolIdentity(
            'electromagnetic_vector_potential',
            'The EM vector potential A, with B = curl(A).',
            (SymbolUsage('Faraday', 'A_f', 'F4 electrostatics/potential closure'),
             SymbolUsage('Hamilton', 'A_h', 'H2 minimal-coupling Hamiltonian'))),
        SymbolIdentity(
            'helmholtz_free_energy', 'A = U - T*S.',
            (SymbolUsage('Gibbs', 'A_g', 'G1 fundamental potentials'),
             SymbolUsage('Boltzmann', 'A_bo', 'BO6 statistical ensembles'))),
        SymbolIdentity(
            'radioactive_activity', 'Decay activity A = lambda*N.',
            (SymbolUsage('Curie', 'A_c', 'C1 decay and activity'),)),
        SymbolIdentity(
            'nucleon_number', 'Mass number A in the semi-empirical binding-energy formula.',
            (SymbolUsage('Curie', 'A_nuc', 'C6 binding-energy estimate'),)),
        SymbolIdentity(
            'element_incidence_matrix',
            'Element-by-species incidence matrix, A*n = element totals.',
            (SymbolUsage('Lavoisier', "sp.Symbol('A') inline", 'L1 stoichiometry/composition'),
             SymbolUsage('Noether', 'A_n', 'NO4 chemical conservation check'))),
    ),
}

# The source markdown (section 0.1, and again inside Bragg's BR11) flags
# two more traps by NAME rather than by sympy token: a physical constant
# and an honorary engine that happen to share a word. Neither is a sympy
# symbol collision in THIS file (the Faraday constant is always written
# F_c here, never bare F), but "just share a name" was explicitly part
# of the brief, and these are the two traps the source text itself calls
# out by name.
NAME_VS_DOMAIN_TRAPS = (
    ("Faraday constant (F_c, electrochemistry)",
     "Faraday engine (electromagnetic fields, networks and waves)",
     "Unrelated by physics, related only by Michael Faraday's name."),
    ("Curie-Weiss law (chi = C/(T-T_C), BR11 magnetism)",
     "Curie engine (nuclides, decay and radiation)",
     "Source text: \"not the Curie nuclear engine and not a universal magnetic law.\""),
)


def describe_symbol(token, engine=None):
    """Resolve what a bare display token like 'M' or 'A' actually means.

    Policy:
      - If `token` is a known collision and `engine` is given, return the
        one SymbolIdentity whose usages list that engine.
      - If `token` is a known collision and `engine` is omitted, return
        every candidate identity -- the caller must disambiguate; this
        function never guesses which one was meant.
      - If `token` is not in the curated table, fall back to a live scan
        of this module's own source (`raw_token_report`), so the caller
        still sees every section that uses it, tagged as "not curated"
        rather than a false assurance either way.
    """
    if token in KNOWN_COLLISIONS:
        identities = KNOWN_COLLISIONS[token]
        if engine is None:
            return identities
        matches = tuple(i for i in identities if any(u.engine == engine for u in i.usages))
        if not matches:
            other = sorted({u.engine for i in identities for u in i.usages})
            raise KeyError(f"{token!r} is curated for {other}, not {engine!r}")
        return matches
    occurrences = raw_token_report().get(token)
    if occurrences is None:
        raise KeyError(f"{token!r} does not appear as a bare sp.symbols()/sp.Symbol() token")
    return {'curated': False, 'occurrences': occurrences}


def namespaced_symbol(token, identity_key):
    """A fresh, collision-free sympy Symbol for programmatic disambiguated use.

    Two calls with the same (token, identity_key) are equal (same name,
    same assumptions), so this is safe to call repeatedly instead of
    caching one instance by hand.
    """
    return sp.Symbol(f"{token}^{{{identity_key}}}")


_TOKEN_RE = _re.compile(r"sp\.symbols\('([^']*)'\)|sp\.Symbol\('([^']*)'\)")
_SECTION_HEADER_RE = _re.compile(r'^#\s*\d+\.\s*(.+)$')


def raw_token_report():
    """Live, best-effort scan of this module's own source for every bare
    sp.symbols()/sp.Symbol() token, grouped by which honorary-engine
    section used it.

    This is the dynamic half of the resolution policy: for any token NOT
    in KNOWN_COLLISIONS, this is what `describe_symbol` and the `symbol`
    CLI subcommand fall back to -- an honest "here is everywhere this
    spelling appears" instead of a guess about whether it is one
    physical quantity or several. (Best-effort: it does not unescape
    embedded quotes, so a token containing \\' is skipped rather than
    misparsed.)
    """
    section = None
    report = {}
    with open(__file__, encoding='utf-8') as f:
        for line in f:
            header = _SECTION_HEADER_RE.match(line.strip())
            if header:
                section = header.group(1).strip()
                continue
            for call_match in _TOKEN_RE.finditer(line):
                s = call_match.group(1) or call_match.group(2)
                for tok in _re.split(r'[,\s]+', s):
                    if tok:
                        report.setdefault(tok, set()).add(section)
    return {tok: sorted(s for s in secs if s) for tok, secs in report.items()}


# =====================================================================
# LAMBDIFY CACHE: turning equations into runnable, cacheable pieces
# =====================================================================
# What "saved" actually means here, checked directly rather than assumed:
# a lambdify()'d function's compiled object is NOT picklable by reference
# -- its __module__/__qualname__ do not resolve to anything re-importable,
# so stdlib pickle raises PicklingError on it. What sympy DOES give you is
# the generated source text: lambdify registers it with linecache so
# ``inspect.getsource`` reads it back whole. Caching THAT text and exec'ing
# it on a hit is what "saved" means below -- it skips sympy's printer/
# codegen work on a hit, which is the part that scales with equation count,
# not the act of calling the function itself.
#
# This mirrors turing/examples/chamber_dt_join.py's own law-module cache
# (``load_law_module_cached``: pickle the constructed SymPy trees, keyed on
# a source digest) -- same idea, applied one level lower, to one law's
# generated call instead of a whole module's tree construction.

import hashlib as _hashlib
import inspect as _inspect
import os as _os
from pathlib import Path as _Path
from typing import Dict as _Dict, Optional as _Optional, Sequence as _Sequence

_LAW_CACHE_DIR = _Path(__file__).resolve().parent / "__lawcache__"

# Coordinates and operator placeholders declared once at the top of this
# module -- scaffolding, never a physical field a piece should read or own.
_STRUCTURAL_NAMES = frozenset({
    "t", "x", "y", "z",
    "nabla", "nabla_op", "transpose", "det", "tr", "cross",
})


def _lambdify_cache_key(law_id: str, argument_names: tuple, expr) -> str:
    payload = f"{law_id}|{argument_names}|{sp.srepr(expr)}".encode("utf-8")
    return _hashlib.sha256(payload).hexdigest()[:24]


def lambdify_cached(law_id: str, argument_names: tuple, expr, *, modules="numpy",
                     cache_dir=None):
    """A lambdified callable for ``argument_names -> expr``, cached to disk
    by content hash (law id + argument order + ``sympy.srepr`` of the exact
    expression). A hit execs the saved source; a miss lambdifies, writes the
    generated source via ``inspect.getsource``, and returns the live
    function either way."""
    root = _Path(cache_dir) if cache_dir is not None else _LAW_CACHE_DIR
    key = _lambdify_cache_key(law_id, argument_names, expr)
    cache_file = root / f"{law_id}-{key}.py"
    if cache_file.exists():
        source = cache_file.read_text(encoding="utf-8")
        namespace: dict = {}
        exec(compile(source, str(cache_file), "exec"), namespace)
        return namespace["_lambdifygenerated"]
    symbols = tuple(sp.Symbol(name) for name in argument_names)
    fn = sp.lambdify(symbols, expr, modules=modules)
    source = _inspect.getsource(fn)
    root.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(f".{_os.getpid()}.tmp")
    tmp.write_text(source, encoding="utf-8")
    _os.replace(tmp, cache_file)
    return fn


@dataclass(frozen=True)
class Piece:
    """One law, lambdified and shaped like an ``LLVMPiece``
    (``argument_names``/``output_names``/a positional callable) so it can
    be experimented with against a real dt system deployment without
    inventing a second interface for the same idea."""

    entry: str
    argument_names: tuple
    output_names: tuple
    fn: object

    def __call__(self, *columns):
        result = self.fn(*columns)
        return result if isinstance(result, tuple) else (result,)


def _scalarize_applied_functions(expr):
    """Replace every applied function call (``G(x)``) with a plain Symbol
    of the same name (``G``), so a lambdified piece's parameter and its use
    inside the expression body agree -- lambdify treats an unsubstituted
    ``AppliedUndef`` as something to CALL, not a scalar input, and a plain
    parameter symbol of the same name is not the same sympy object as the
    applied call, which is exactly the 'float object is not callable' bug
    this fixes.

    Returns ``None`` (refuse, do not guess) if the same function name is
    applied to more than one distinct argument tuple in ``expr`` -- e.g.
    ``p_s(T_b) - p_s(T_0_b)`` -- because collapsing those to one scalar
    parameter would silently conflate two different physical values.
    """
    seen_args: dict = {}
    for app in expr.atoms(_AppliedUndef):
        name = app.func.__name__
        if name in seen_args and seen_args[name] != app.args:
            return None
        seen_args[name] = app.args
    substitution = {
        app: sp.Symbol(app.func.__name__)
        for app in expr.atoms(_AppliedUndef)
    }
    return expr.subs(substitution)


def law_piece(law_id: str, eq_obj, *, modules="numpy", cache_dir=None) -> _Optional["Piece"]:
    """One ``eq_XX_n`` as a ``Piece``: its read symbols (sorted, minus the
    structural placeholders) are ``argument_names``, its owned symbol is
    its one ``output_name``. Returns ``None`` for a constraint/relational
    (no single quantity to lambdify against), for a law that applies the
    same function name to more than one distinct argument set (see
    ``_scalarize_applied_functions``), or for a law whose RHS contains
    something sympy's numpy printer cannot turn into numeric code as-is
    (an unevaluated ``Derivative``/``Integral``, or one of this module's
    own operator placeholders like ``nabla``/``transpose`` -- those need a
    discretization choice before they are a numeric piece, which is
    modeling work this function does not invent on your behalf) rather
    than guessing one or crashing the whole batch.
    """
    if not isinstance(eq_obj, Equality):
        return None
    lhs, rhs = eq_obj.lhs, eq_obj.rhs
    owned_names, lhs_extra_reads = _owned_identities(lhs)
    owned_names -= _STRUCTURAL_NAMES
    if len(owned_names) != 1:
        return None  # compound/ambiguous owner: not a piece-shaped law
    output_name = next(iter(owned_names))
    read_names = ((_identities_in(lhs) | _identities_in(rhs)) - owned_names
                  - _STRUCTURAL_NAMES) | (lhs_extra_reads - _STRUCTURAL_NAMES)
    argument_names = tuple(sorted(read_names))
    scalar_rhs = _scalarize_applied_functions(rhs)
    if scalar_rhs is None:
        return None
    try:
        fn = lambdify_cached(law_id, argument_names, scalar_rhs, modules=modules, cache_dir=cache_dir)
    except Exception:
        return None
    return Piece(law_id, argument_names, (output_name,), fn)


def law_pieces(engines: _Optional[_Sequence[str]] = None, *, modules="numpy",
               cache_dir=None) -> _Dict[str, "Piece"]:
    """Every piece-shaped law across the requested engines, lambdified and
    cached. Constraint/relational laws are silently skipped (see
    ``law_piece``); a caller that needs to know which were skipped should
    compare its own equation set against this function's keys."""
    by_engine = _discover_equations()
    selected = list(engines) if engines is not None else sorted(by_engine)
    pieces: _Dict[str, Piece] = {}
    for engine in selected:
        for name, obj in by_engine.get(engine, {}).items():
            piece = law_piece(name, obj, modules=modules, cache_dir=cache_dir)
            if piece is not None:
                pieces[name] = piece
    return pieces


# The owned/read extraction that ``law_piece`` needs, mechanical over the
# sympy shape of an Eq's lhs (same heuristic §5-6 of
# DT_GRAPH_STATE_TENSOR_COMPOSITION_AUDIT.md describes: the lhs names what
# is owned, everything else in the equation is read).
from sympy.core.function import AppliedUndef as _AppliedUndef
from sympy.core.relational import Equality


def _identities_in(expr) -> set:
    names: set = set()
    if not hasattr(expr, "atoms"):
        return names
    for app in expr.atoms(_AppliedUndef):
        names.add(app.func.__name__)
    for sym in getattr(expr, "free_symbols", ()):
        names.add(sym.name)
    return names


def _owned_identities(lhs) -> tuple:
    if isinstance(lhs, _AppliedUndef):
        owned = {lhs.func.__name__}
        return owned, _identities_in(lhs) - owned
    if isinstance(lhs, sp.Derivative):
        inner = lhs.expr
        if isinstance(inner, _AppliedUndef):
            owned = {inner.func.__name__}
            return owned, _identities_in(inner) - owned
        if isinstance(inner, sp.Symbol):
            return {inner.name}, set()
        return _identities_in(inner), set()
    if isinstance(lhs, sp.Symbol):
        return {lhs.name}, set()
    return _identities_in(lhs), set()




def _format_identity(identity):
    lines = [f"  [{identity.key}] {identity.description}"]
    for u in identity.usages:
        note = f" -- {u.note}" if u.note else ""
        lines.append(f"      {u.engine}: {u.python_name}{note}")
    return "\n".join(lines)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog='honorary_engine_equation_catalogue',
        description='Query the honorary-engine equation catalogue: list '
                     'engines, inspect equations, resolve symbol collisions, '
                     'and build modular equation sets for a sim build.')
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('engines', help='list every honorary engine with an equation count')

    p_list = sub.add_parser('list', help='list the eq_* names for one engine')
    p_list.add_argument('engine')

    p_show = sub.add_parser('show', help='pretty-print one equation by its eq_ name')
    p_show.add_argument('name')

    p_symbol = sub.add_parser('symbol', help='resolve what a bare token like M or A means')
    p_symbol.add_argument('token')
    p_symbol.add_argument('--engine', default=None)

    sub.add_parser('collisions', help='list every curated symbol collision and name-vs-domain trap')

    p_set = sub.add_parser('set', help='build a named or ad hoc equation set')
    p_set.add_argument('--name', default=None, help='a name from ENGINE_SETS')
    p_set.add_argument('--engines', nargs='*', default=None, help='an ad hoc list of engine names')

    args = parser.parse_args(argv)

    try:
        _dispatch(args)
    except KeyError as exc:
        raise SystemExit(str(exc).strip('"'))


def _dispatch(args):
    if args.command == 'engines':
        by_engine = _discover_equations()
        for engine in available_engines():
            prefix = next((p for p, (e, _s) in ENGINE_PREFIXES.items() if e == engine), '?')
            section = ENGINE_PREFIXES.get(prefix, (None, None))[1]
            section_str = f"section {section}" if section else "not yet in the markdown catalogue"
            print(f"{engine:16s} ({prefix:>3s}, {section_str}): {len(by_engine[engine])} equations")

    elif args.command == 'list':
        for name in sorted(equations_by_engine(args.engine)):
            print(name)

    elif args.command == 'show':
        value = globals().get(args.name)
        if value is None:
            raise SystemExit(f"no such equation: {args.name}")
        print(sp.pretty(value))

    elif args.command == 'symbol':
        result = describe_symbol(args.token, engine=args.engine)
        if isinstance(result, dict):
            print(f"{args.token!r} is not curated. Raw occurrences by section:")
            for section in result['occurrences']:
                print(f"  - {section}")
        else:
            if len(result) > 1:
                print(f"{args.token!r} is ambiguous across {len(result)} identities; pass --engine to pick one:")
            for identity in result:
                print(_format_identity(identity))

    elif args.command == 'collisions':
        for token, identities in KNOWN_COLLISIONS.items():
            print(f"Token {token!r}: {len(identities)} distinct identities")
            for identity in identities:
                print(_format_identity(identity))
        print("\nName-vs-domain traps (not sympy collisions, just shared English words):")
        for a, b, note in NAME_VS_DOMAIN_TRAPS:
            print(f"  - {a}\n    vs {b}\n    {note}")

    elif args.command == 'set':
        if args.name:
            engines = ENGINE_SETS[args.name]
        elif args.engines:
            engines = tuple(args.engines)
        else:
            raise SystemExit('pass --name or --engines')
        eqs = equations_for(*engines)
        print(f"Set of {len(eqs)} equations from {engines}:")
        for name in sorted(eqs):
            print(f"  {name}")


if __name__ == '__main__':
    main()