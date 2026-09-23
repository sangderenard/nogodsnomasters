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
eq_N1_3 = sp.Eq(sp.Derivative(R(t), t), R(t) * sp.Function('skew')(omega_B(t)))  # dR/dt = R [omega_B]_x, skew(.) = cross-product matrix placeholder (a MatrixSymbol here made sympy evaluate the Eq to False)
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
eq_T5_2 = sp.Eq(transpose(sp.Symbol('phi_i')) * sp.Symbol('M') * sp.Symbol('phi_j'), sp.KroneckerDelta(sp.Symbol('i'), sp.Symbol('j'))) # mass-orthonormal modes phi_i^T M phi_j = delta_ij (symbolic indices so delta does not evaluate to 1)
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
eq_F3_6 = sp.Eq(T_tens, eps0_f(t)*(sp.Symbol('E_i')*sp.Symbol('E_j') - sp.Rational(1, 2)*sp.KroneckerDelta(sp.Symbol('i'), sp.Symbol('j'))*sp.Symbol('|E|^2')) + 1/mu0_f(t)*(sp.Symbol('B_i')*sp.Symbol('B_j') - sp.Rational(1, 2)*sp.KroneckerDelta(sp.Symbol('i'), sp.Symbol('j'))*sp.Symbol('|B|^2'))) # Maxwell stress T_ij = eps0(E_iE_j - delta_ij E^2/2) + (B_iB_j - delta_ij B^2/2)/mu0
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
eq_NS1_2 = sp.Eq(sp.Derivative(rho_n(t)*u_n(t), t) + nabla(rho_n(t)*sp.Function('outer')(u_n(t), u_n(t)) + p_n(t)*sp.Symbol('I_3')), nabla(tau_n(t)) + rho_n(t)*g_n_vec(t) + f_other(t) + S_p(t))  # momentum: div(rho u(x)u + p I) = div(tau) + body forces (outer = dyadic product, I_3 = identity)
eq_NS1_3 = sp.Eq(sp.Derivative(rho_n(t)*E_n(t), t) + nabla((rho_n(t)*E_n(t) + p_n(t))*u_n(t)), nabla(tau_n(t)*u_n(t) - q_n(t)) + rho_n(t)*g_n_vec(t)*u_n(t) + f_other(t)*u_n(t) + S_E(t))
eq_NS1_4 = sp.Eq(E_n(t), e_n(t) + 0.5*sp.Abs(u_n(t))**2)
eq_NS1_5 = sp.Eq(sp.Derivative(rho_n(t)*Y_s(t), t) + nabla(rho_n(t)*Y_s(t)*u_n(t) + j_s(t)), omega_s(t) + S_s(t))
eq_NS1_6 = sp.Eq(sp.Sum(omega_s(t), ('s', 1, sp.Symbol('N'))), 0)
eq_NS1_7 = sp.Eq(sp.Sum(S_s(t), ('s', 1, sp.Symbol('N'))), S_m(t))

# NS2
D_tens, mu_n, zeta_n, k_n, T_n, h_s, D_s, j_s0, x_s, c_tot, D_sr, J_s_mol = sp.symbols('D mu zeta k T h_s D_s j_s^0 x_s c_tot mathcal{D}_{sr} J_s')
eq_NS2_1 = sp.Eq(D_tens, sp.Rational(1, 2)*(nabla(u_n(t)) + transpose(nabla(u_n(t))))) # strain rate D = (grad u + grad u^T)/2
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
eq_BR1_2 = sp.Eq(a_i_vec * b_j_vec, 2*sp.pi*sp.KroneckerDelta(sp.Symbol('i'), sp.Symbol('j'))) # reciprocal lattice a_i . b_j = 2 pi delta_ij (dot product abstracted)
eq_BR1_3 = sp.Eq(G_hkl, h_idx*b1 + k_idx*b2 + l_idx*b3)
eq_BR1_4 = sp.Eq(2*d_hkl*sp.sin(theta_br), n_br*lambda_br)
eq_BR1_5 = sp.Eq(F_G, sp.Sum(f_alpha * sp.exp(sp.I * G_hkl * r_alpha), ('alpha', 1, sp.Symbol('N'))))

# BR2
eps_br, u_br, sig_br, C_br, eps_p_br, eps_th_br, alpha_br, T_br, T0_br, z_br, psi_br, F_br, x_br, X_br, Fe_br, Fp_br, P_br, Psi_br, J_br = sp.symbols('epsilon u sigma C epsilon^p epsilon^{th} alpha T T_0 z psi F x X F_e F_p P Psi J')
eq_BR2_1 = sp.Eq(eps_br, sp.Rational(1, 2)*(nabla(u_br) + transpose(nabla(u_br)))) # small strain eps = (grad u + grad u^T)/2
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
eq_BR8_2 = sp.Eq(v_nk, hbar**-1 * sp.Function('nabla_k')(E_n_band))  # band group velocity: gradient in k-space, not position space
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
eq_H5_1 = sp.Eq(F_I, -sp.Function('nabla_R')(E_BO)) # force on nucleus I: gradient w.r.t. nuclear coordinate R_I
eq_H5_2 = sp.Eq(M_I*sp.Derivative(R_I, t, 2), F_I)
eq_H5_3 = sp.Eq(sp.Function('nabla_R')(E_BO), 0) # equilibrium geometry R_*: nuclear-coordinate gradient vanishes
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
eq_H7_5 = sp.Eq(E_n_func, T_s_func + sp.Integral(v_ext*n_r, r_i) + 0.5*sp.Integral(n_r*sp.Symbol('n(r\')')/sp.Abs(r_i - sp.Symbol('r\'')), r_i, sp.Symbol('r\'')) + E_xc_func + V_NN)  # Kohn-Sham energy; the Hartree term is a DOUBLE volume integral over r and r' (was written as r_i with upper bound r')

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
eq_E3_7 = sp.Eq(eta_mu_nu, sp.Function('diag')(-1, 1, 1, 1))  # Minkowski metric eta = diag(-1,1,1,1), placeholder like det/tr (a scalar Symbol == Matrix made sympy evaluate the Eq to False)
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
eq_E6_3 = sp.Eq(sp.Function('nabla_mu')(xi_nu) + sp.Function('nabla_nu')(sp.Symbol('xi_mu')), 0) # Killing equation nabla_(mu xi_nu) = 0: symmetrized covariant derivative
eq_E6_4 = sp.Eq(sp.Derivative(sp.Symbol('p_mu xi^mu'), sp.Symbol('tau')), 0) # Killing-vector conserved quantity along a geodesic: d(p_mu xi^mu)/d tau = 0


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
eq_BO1_1 = sp.Eq(sp.Derivative(f_s(t), t) + v_bo(t)*nabla(f_s(t)) + (F_s(t)/m_s_bo(t))*sp.Function('nabla_v')(f_s(t)), C_s_bo(f_s(t)) + S_s_bo(t))  # Boltzmann: spatial gradient for streaming, VELOCITY-space gradient for the force term
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
eq_BO8_1 = sp.Eq((1/v_E(t))*sp.Derivative(psi_bo(t), t) + Omega_bo(t)*nabla(psi_bo(t)) + Sig_t(t)*psi_bo(t), sp.Integral(Sig_s(t)*psi_bo(t), sp.Symbol('E\''), sp.Symbol('Omega\'')) + S_bo_rad(t))


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
# EXPANSION (2026-09-22): every established system represented
# =====================================================================
# HONORARY_CATALOGUE_COVERAGE_AUDIT.md lists every law the established
# systems in engine_toy/ and turing/ use that the sections above did not
# yet state.  Each block below adds one engine's laws as further numbered
# sections (N8+, T8+, ...) or, for the six fields with no home above, as a
# new engine (26 Willis .. 31 Maxwell).
#
# Unlike the flat sections above, every block defines its symbols INSIDE a
# function and publishes only its eq_* names into module scope.  The flat
# sections reuse Python names across engines (A_f, S_g, w, gamma, z, ...),
# so a later section silently rebinds an earlier one's name; scoping the
# symbols is what keeps several hundred new laws from doing the same.  The
# registry below discovers them exactly like the flat ones.

# =====================================================================
# 1b. NEWTON -- expansion (2026-09-22 coverage audit)
# =====================================================================
def _expand_newton():
    eqs = {}

    # N8 tire and friction contact
    mu_v, mu_k, mu_s, v_slip, v_S = sp.symbols('mu(v) mu_k mu_s v_{slip} v_S')
    F_lim, F_req, F_fr, F_N, s_load, s_patch = sp.symbols('F_{lim} F_{req} F_{fr} F_N s_{load} s_{patch}')
    eps_L, F_ref, mu_0, F_z, F_z0 = sp.symbols('epsilon_L F_{ref} mu_0 F_z F_{z0}')
    k_sw, c_sw, delta_sw = sp.symbols('k_{sw} c_{sw} delta_{sw}')
    sigma_rl, v_x, F_ss = sp.symbols('sigma_{relax} v_x F_{ss}')
    F_tr = sp.Function('F_{tire}')(t)
    eqs['eq_N8_1'] = sp.Eq(mu_v, mu_k + (mu_s - mu_k) * sp.exp(-sp.Abs(v_slip / v_S)**sp.Symbol('delta_S')))  # Stribeck curve (Armstrong-Helouvry 1994; delta_S ~ 2 Gaussian, 1 exponential); turing abstract_ui_vehicles.py wheel contact writes the Lorentzian 1/(1+(v/v_S)^2) variant
    eqs['eq_N8_2'] = sp.Eq(F_lim, mu_v * s_load * s_patch * F_N)  # Coulomb friction limit with load and patch scaling; turing abstract_ui_vehicles.py wheel contact
    eqs['eq_N8_3'] = sp.Eq(F_fr, F_lim * sp.tanh(F_req / F_lim))  # bristle/brush saturation of the requested force at the Coulomb limit; turing abstract_ui_vehicles.py wheel contact, engine_harm.py
    eqs['eq_N8_4'] = sp.Eq(s_load, 1 - eps_L * sp.Max(0, F_N / F_ref - 1))  # tire load sensitivity, empirical linear derate (code clamps to [0.58,1]); turing abstract_ui_vehicles.py
    eqs['eq_N8_5'] = sp.Eq(sp.Symbol('mu(F_z)'), mu_0 * (1 - eps_L * (F_z - F_z0) / F_z0))  # Pacejka-style load sensitivity of peak friction, empirical; standard form
    eqs['eq_N8_6'] = sp.Eq(F_req, -(k_sw * delta_sw + c_sw * sp.Derivative(delta_sw, t)))  # sidewall shear spring-damper force demand; turing abstract_ui_vehicles.py wheel contact
    eqs['eq_N8_7'] = sp.Eq(sigma_rl / sp.Abs(v_x) * sp.Derivative(F_tr, t) + F_tr, F_ss)  # tire relaxation-length first-order filter (Pacejka); sidewall relaxation, turing abstract_ui_vehicles.py

    # N9 regularized drivetrain
    T_cl, T_max, k_cl, d_omega = sp.symbols('T_{clutch} T_{max} k_{clutch} Delta_omega')
    T_cap, mu_c, p_c, A_c, N_f, r_m, r_o, r_i = sp.symbols('T_{cap} mu_c p_c A_c N_f r_m r_o r_i')
    P_slip, T_bl, k_bl, dth, b_bl = sp.symbols('P_{slip} T_{backlash} k_{backlash} Delta_theta b_{backlash}')
    w_in, w_out, i_g, eta_g, T_in, T_out = sp.symbols('omega_{in} omega_{out} i_g eta_g T_{in} T_{out}')
    J_eq, J_k, w_k = sp.symbols('J_{eq} J_k omega_k')
    F_rr, C_rr, T_rr, r_w, w_wheel, v_rr = sp.symbols('F_{rr} C_{rr} T_{rr} r_w omega_{wheel} v_{rr}')
    eqs['eq_N9_1'] = sp.Eq(T_cl, T_max * sp.tanh(k_cl * d_omega / T_max))  # tanh-regularized Coulomb clutch torque; turing abstract_ui_vehicles.py powertrain, drivetrain_port.py, couplings.py
    eqs['eq_N9_2'] = sp.Eq(T_cap, mu_c * p_c * A_c * N_f * r_m)  # friction clutch torque capacity; couplings.py, drivetrain_port.py
    eqs['eq_N9_3'] = sp.Eq(r_m, (r_o + r_i) / 2)  # mean friction radius, uniform-wear assumption (Shigley)
    eqs['eq_N9_4'] = sp.Eq(r_m, sp.Rational(2, 3) * (r_o**3 - r_i**3) / (r_o**2 - r_i**2))  # mean friction radius, uniform-pressure assumption (Shigley)
    eqs['eq_N9_5'] = sp.Eq(P_slip, T_cl * d_omega)  # clutch slip heat; couplings.py, drivetrain_port.py
    eqs['eq_N9_6'] = sp.Eq(T_bl, sp.Piecewise((k_bl * (dth - b_bl / 2), dth > b_bl / 2), (k_bl * (dth + b_bl / 2), dth < -b_bl / 2), (0, True)))  # backlash dead-band torque; drivetrain_port.py
    eqs['eq_N9_7'] = sp.Eq(w_out, w_in / i_g)  # gear ratio kinematics; drivetrain_port.py, turing powertrain
    eqs['eq_N9_8'] = sp.Eq(T_out, eta_g * i_g * T_in)  # gear torque with mesh efficiency; drivetrain_port.py, turing powertrain
    eqs['eq_N9_9'] = sp.Eq(J_eq, sp.Sum(J_k * (w_k / w_in)**2, ('k', 1, sp.Symbol('N'))))  # reflected inertia; drivetrain_port.py, rotating_inertia.py
    eqs['eq_N9_10'] = sp.Eq(F_rr, C_rr * F_N)  # rolling resistance force, empirical C_rr; turing abstract_ui_vehicles.py
    eqs['eq_N9_11'] = sp.Eq(T_rr, C_rr * F_N * r_w * sp.tanh(w_wheel / v_rr))  # rolling resistance torque with smooth direction (code uses a fixed T_rr0 times smooth sign); turing abstract_ui_vehicles.py

    # N10 mass properties
    I_ij, m_p, r_p, r_pi, r_pj = sp.symbols('I_{ij} m_p |r_p| r_{p;i} r_{p;j}')
    i_idx, j_idx = sp.symbols('i j', integer=True)
    I_rot, k_shape, m_b, r_b = sp.symbols('I_{rot} k_{shape} m_b r_b')
    U_ub, m_ub, e_ub, e_per, w_op, G_bal = sp.symbols('U_{unb} m_{rotor} e_{cg} e_{per} Omega G_{grade}')
    eqs['eq_N10_1'] = sp.Eq(I_ij, sp.Sum(m_p * (r_p**2 * sp.KroneckerDelta(i_idx, j_idx) - r_pi * r_pj), ('p', 1, sp.Symbol('N'))))  # inertia tensor of point masses; engine_mass_properties.py
    eqs['eq_N10_2'] = sp.Eq(I_rot, k_shape * m_b * r_b**2)  # shape-factor inertia (k=1 ring, 1/2 disc, 2/5 sphere); rotating_inertia.py
    eqs['eq_N10_3'] = sp.Eq(U_ub, m_ub * e_ub)  # ISO 21940 unbalance; operating_states.py, rotating_inertia.py
    eqs['eq_N10_4'] = sp.Eq(G_bal, e_per * w_op)  # ISO 21940-11 balance quality grade G = e_per*Omega; standard form

    # N11 game-physics constraints
    F_pen, k_pen, pen, b_pen, v_n = sp.symbols('F_{pen} k_{pen} delta_{pen} b_{pen} v_n')
    F_rope, k_rope, l_rope, l0_rope, c_rope, ldot = sp.symbols('F_{rope} k_{rope} l l_0 c_{rope} dot{l}')
    F_dmp, c_bump, c_reb, v_dmp = sp.symbols('F_{damper} c_{bump} c_{rebound} v_{damper}')
    dlam, C_x, alpha_t, lam_x, gradC, M_inv, alpha_c, dt_s, dx = sp.symbols('Delta_lambda C(x) tilde{alpha} lambda nabla_C M^{-1} alpha Delta_t Delta_x')
    F_st, b_st, v_rel, mu_fl, R_sp = sp.symbols('F_{Stokes} b_{Stokes} v_{rel} mu_{fluid} R_{sphere}')
    eqs['eq_N11_1'] = sp.Eq(F_pen, sp.Max(0, k_pen * pen - b_pen * v_n))  # penalty contact, clamped non-adhesive (v_n>0 separating); turing dt_system engines, softbody
    eqs['eq_N11_2'] = sp.Eq(F_rope, sp.Piecewise((k_rope * (l_rope - l0_rope) + c_rope * ldot, l_rope > l0_rope), (0, True)))  # tension-only rope (slack carries nothing); turing dt_system engines
    eqs['eq_N11_3'] = sp.Eq(F_dmp, sp.Piecewise((-c_bump * v_dmp, v_dmp < 0), (-c_reb * v_dmp, True)))  # asymmetric bump/rebound damper; turing dt_system engines
    eqs['eq_N11_4'] = sp.Eq(alpha_t, alpha_c / dt_s**2)  # XPBD time-scaled compliance (Macklin 2016)
    eqs['eq_N11_5'] = sp.Eq(dlam, (-C_x - alpha_t * lam_x) / (gradC * M_inv * transpose(gradC) + alpha_t))  # XPBD compliant Lagrange multiplier update (Macklin 2016); turing softbody
    eqs['eq_N11_6'] = sp.Eq(dx, M_inv * transpose(gradC) * dlam)  # XPBD position correction (Macklin 2016); turing softbody
    eqs['eq_N11_7'] = sp.Eq(F_st, -b_st * v_rel)  # linear (Stokes) drag; turing dt_system engines
    eqs['eq_N11_8'] = sp.Eq(b_st, 6 * sp.pi * mu_fl * R_sp)  # Stokes sphere drag coefficient; standard form

    # N12 centrifugal governor
    m_g, w_g, r_g, F_0g, k_g, r_min, F_s, F_f = sp.symbols('m_{ball} omega_{gov} r_{ball} F_{preload} k_{gov} r_{min} F_{spring} F_{fric}')
    w_eq = sp.Symbol('omega_{eq}')
    r_gt = sp.Function('r_{ball}')(t)
    eqs['eq_N12_1'] = sp.Eq(F_s, F_0g + k_g * sp.Max(0, r_g - r_min))  # preloaded governor spring; governor.py
    eqs['eq_N12_2'] = sp.Eq(m_g * sp.Derivative(r_gt, t, 2), m_g * w_g**2 * r_gt - F_s + F_f)  # flyball radial dynamics (centrifugal vs spring, friction opposing motion); governor.py
    eqs['eq_N12_3'] = sp.Eq(w_eq, sp.sqrt((F_0g + k_g * (r_g - r_min)) / (m_g * r_g)))  # governor equilibrium (set) speed at radius r; governor.py

    return eqs


globals().update(_expand_newton())


# =====================================================================
# 2b. TIMOSHENKO -- expansion (2026-09-22 coverage audit)
# =====================================================================
def _expand_timoshenko():
    eqs = {}

    # T8 section properties
    I_z, A_s, y_s, I_c, d_pa, r_gy = sp.symbols('I_z A y I_c d_{pa} r_{gyr}')
    J_r, a_r, c_r = sp.symbols('J_{rect} a_{long} c_{short}')
    J_open, b_k, t_k = sp.symbols('J_{open} b_k t_k')
    J_B, A_m, s_arc, t_w, q_sf, T_q = sp.symbols('J_{Bredt} A_m s t_w q_{shear} T')
    S_tube, D_o, D_i = sp.symbols('S_{tube} D_o D_i')
    k_rect, k_circ, k_tube, nu = sp.symbols('kappa_{rect} kappa_{circ} kappa_{tube} nu')
    eqs['eq_T8_1'] = sp.Eq(I_z, sp.Integral(y_s**2, A_s))  # second moment of area; milspec.py, surfaces.py
    eqs['eq_T8_2'] = sp.Eq(I_z, I_c + A_s * d_pa**2)  # parallel-axis (Steiner) theorem; milspec.py, joints.py
    eqs['eq_T8_3'] = sp.Eq(r_gy, sp.sqrt(I_z / A_s))  # radius of gyration; milspec.py
    eqs['eq_T8_4'] = sp.Eq(J_r, a_r * c_r**3 * (sp.Rational(1, 3) - sp.Float('0.21') * c_r / a_r * (1 - c_r**4 / (12 * a_r**4))))  # St-Venant torsion constant of a rectangle, Roark approximation; surfaces.py
    eqs['eq_T8_5'] = sp.Eq(J_open, sp.Sum(b_k * t_k**3 / 3, ('k', 1, sp.Symbol('N'))))  # thin open-section St-Venant torsion constant; milspec.py
    eqs['eq_T8_6'] = sp.Eq(J_B, 4 * A_m**2 / sp.Integral(1 / t_w, s_arc))  # Bredt-Batho closed-section torsion constant (contour integral); surfaces.py
    eqs['eq_T8_7'] = sp.Eq(q_sf, T_q / (2 * A_m))  # Bredt-Batho shear flow; surfaces.py
    eqs['eq_T8_8'] = sp.Eq(S_tube, sp.pi * (D_o**4 - D_i**4) / (32 * D_o))  # circular tube elastic section modulus; milspec.py
    eqs['eq_T8_9'] = sp.Eq(k_rect, 10 * (1 + nu) / (12 + 11 * nu))  # Cowper shear coefficient, rectangle (defining); surfaces.py, milspec.py
    eqs['eq_T8_10'] = sp.Eq(k_circ, 6 * (1 + nu) / (7 + 6 * nu))  # Cowper shear coefficient, solid circle (defining); milspec.py
    eqs['eq_T8_11'] = sp.Eq(k_tube, 2 * (1 + nu) / (4 + 3 * nu))  # Cowper shear coefficient, thin-walled tube (defining); milspec.py

    # T9 stress and deflection
    sig_x, N_ax, M_b, y_f, E, I = sp.symbols('sigma_x N M y E I')
    d_tip, w_l, L = sp.symbols('delta_{tip} w L')
    kap_th, alpha, dT, d_depth = sp.symbols('kappa_{th} alpha Delta_T d')
    eqs['eq_T9_1'] = sp.Eq(sig_x, N_ax / A_s + M_b * y_f / I)  # combined axial + bending normal stress; ring_joints.py, muzzle_reference.py
    eqs['eq_T9_2'] = sp.Eq(d_tip, w_l * L**4 / (8 * E * I))  # cantilever tip droop under uniform load (Euler-Bernoulli); muzzle_reference.py
    eqs['eq_T9_3'] = sp.Eq(kap_th, alpha * dT / d_depth)  # thermal-bow curvature from through-depth temperature difference; barrel_thermal.py

    # T10 vibration
    X_a, F_0, k, r_w, zeta, phi_r, w, w_n, m = sp.symbols('X F_0 k r zeta phi omega omega_n m')
    F_u, U_ub = sp.symbols('F_{unb} U_{unb}')
    T_R = sp.Symbol('T_R')
    I_d, I_p, Om, k_th, w_f, w_b, dw_g = sp.symbols('I_d I_p Omega k_theta omega_f omega_b Delta_omega_{gyro}')
    x_b, x_i, q_m, T_CB, Phi_c, Phi_n, K_ii, K_ib, K_CB, M_CB, K_full, M_full = sp.symbols(
        'x_b x_i q_m T_{CB} Phi_c Phi_n K_{ii} K_{ib} K_{CB} M_{CB} K_{full} M_{full}')
    eqs['eq_T10_1'] = sp.Eq(r_w, w / w_n)  # frequency ratio; machine_shake.py
    eqs['eq_T10_2'] = sp.Eq(w_n, sp.sqrt(k / m))  # undamped natural frequency; machine_shake.py, engine_mounts.py
    eqs['eq_T10_3'] = sp.Eq(X_a, (F_0 / k) / sp.sqrt((1 - r_w**2)**2 + (2 * zeta * r_w)**2))  # harmonic response amplitude, SDOF (Rao); machine_shake.py
    eqs['eq_T10_4'] = sp.Eq(sp.tan(phi_r), 2 * zeta * r_w / (1 - r_w**2))  # harmonic response phase lag (Rao); machine_shake.py
    eqs['eq_T10_5'] = sp.Eq(F_u, U_ub * w**2)  # rotating unbalance force; machine_shake.py, engine_mounts.py
    eqs['eq_T10_6'] = sp.Eq(T_R, sp.sqrt((1 + (2 * zeta * r_w)**2) / ((1 - r_w**2)**2 + (2 * zeta * r_w)**2)))  # base-excitation (and force) transmissibility (Rao); engine_mounts.py
    eqs['eq_T10_7'] = sp.Eq(I_d * w_f**2 - I_p * Om * w_f - k_th, 0)  # gyroscopic whirl characteristic equation, forward whirl (backward: +I_p*Omega); standard rigid-disk rotor, component_mode_atlas.py
    eqs['eq_T10_8'] = sp.Eq(w_f, (I_p * Om + sp.sqrt((I_p * Om)**2 + 4 * I_d * k_th)) / (2 * I_d))  # forward whirl frequency; component_mode_atlas.py
    eqs['eq_T10_9'] = sp.Eq(w_b, (-I_p * Om + sp.sqrt((I_p * Om)**2 + 4 * I_d * k_th)) / (2 * I_d))  # backward whirl frequency; component_mode_atlas.py
    eqs['eq_T10_10'] = sp.Eq(dw_g, I_p * Om / I_d)  # gyroscopic mode splitting omega_f - omega_b; component_mode_atlas.py, wrench_paths.py
    eqs['eq_T10_11'] = sp.Eq(Phi_c, -K_ii**-1 * K_ib)  # Craig-Bampton constraint modes (static condensation); component_mode_atlas.py
    eqs['eq_T10_12'] = sp.Eq(sp.Matrix([x_b, x_i]), sp.Matrix([[1, 0], [Phi_c, Phi_n]]) * sp.Matrix([x_b, q_m]))  # Craig-Bampton transformation (1 = identity block, Phi_n fixed-interface modes); component_mode_atlas.py
    eqs['eq_T10_13'] = sp.Eq(K_CB, transpose(T_CB) * K_full * T_CB)  # Craig-Bampton reduced stiffness; component_mode_atlas.py
    eqs['eq_T10_14'] = sp.Eq(M_CB, transpose(T_CB) * M_full * T_CB)  # Craig-Bampton reduced mass; component_mode_atlas.py

    # T11 contact stiffness
    k_p, a_p, E_f, nu_f = sp.symbols('k_{punch} a_{punch} E_{floor} nu_{floor}')
    k_rk, k_foot, d_foot, I_rk, w_rk = sp.symbols('k_{rock} k_{foot} d_{foot} I_{axis} omega_{rock}')
    eqs['eq_T11_1'] = sp.Eq(k_p, 2 * a_p * E_f / (1 - nu_f**2))  # Boussinesq rigid circular punch stiffness; feet.py
    eqs['eq_T11_2'] = sp.Eq(k_rk, sp.Sum(k_foot * d_foot**2, ('f', 1, sp.Symbol('N'))))  # rocking stiffness of feet about the rocking axis; feet.py
    eqs['eq_T11_3'] = sp.Eq(w_rk, sp.sqrt(k_rk / I_rk))  # rocking natural frequency; feet.py

    # T12 springs
    k_s, G, d_wire, D_coil, n_a, k_wh, MR = sp.symbols('k_{spring} G d_{wire} D_{coil} n_a k_{wheel} MR')
    eqs['eq_T12_1'] = sp.Eq(k_s, G * d_wire**4 / (8 * D_coil**3 * n_a))  # helical compression spring rate; turing suspension
    eqs['eq_T12_2'] = sp.Eq(k_wh, k_s * MR**2)  # wheel rate through motion ratio; turing suspension
    return eqs


globals().update(_expand_timoshenko())


# =====================================================================
# 7b. BRAGG -- expansion (2026-09-22 coverage audit)
# =====================================================================
def _expand_bragg():
    eqs = {}

    # BR13 membranes and laminates
    I_i, J_i, K_i = sp.symbols('I J K', integer=True)
    E_IJ, F_kI, F_kJ, E_GL, F_def, a_met, A_met = sp.symbols('E_{IJ} F_{kI} F_{kJ} E_{GL} F a_{metric} A_{metric}')
    W_sv, mu_l, lam_l, S_IJ, E_KK = sp.symbols('W_{StVK} mu lambda S_{IJ} E_{KK}')
    Q11, Q22, Q12, Q66, Q16, Q26 = sp.symbols('Q_{11} Q_{22} Q_{12} Q_{66} Q_{16} Q_{26}')
    E1, E2, nu12, nu21, G12 = sp.symbols('E_1 E_2 nu_{12} nu_{21} G_{12}')
    W_Q, e1, e2, g12, E12 = sp.symbols('W_Q epsilon_1 epsilon_2 gamma_{12} E_{12}')
    R_ray, mu_d, lam_d, Ed, Q_d, qdot = sp.symbols('mathcal{R} mu_d lambda_d dot{E} Q_d dot{q}')
    eqs['eq_BR13_1'] = sp.Eq(E_IJ, sp.Rational(1, 2) * (sp.Sum(F_kI * F_kJ, ('k', 1, 3)) - sp.KroneckerDelta(I_i, J_i)))  # Green-Lagrange strain E = (F^T F - I)/2 in components; turing vehicle_balloon_tire.py
    eqs['eq_BR13_2'] = sp.Eq(E_GL, sp.Rational(1, 2) * (a_met - A_met))  # membrane Green-Lagrange strain from deformed vs natural surface metric (code's form); turing vehicle_balloon_tire.py
    eqs['eq_BR13_3'] = sp.Eq(W_sv, mu_l * tr(E_GL * E_GL) + lam_l / 2 * tr(E_GL)**2)  # St Venant-Kirchhoff strain energy density; turing vehicle_balloon_tire.py
    eqs['eq_BR13_4'] = sp.Eq(S_IJ, lam_l * E_KK * sp.KroneckerDelta(I_i, J_i) + 2 * mu_l * E_IJ)  # StVK second Piola-Kirchhoff stress (E_KK = tr E); standard form
    eqs['eq_BR13_5'] = sp.Eq(nu21, nu12 * E2 / E1)  # orthotropic reciprocity; turing vehicle_balloon_tire.py (standard, code takes Q directly)
    eqs['eq_BR13_6'] = sp.Eq(Q11, E1 / (1 - nu12 * nu21))  # orthotropic reduced stiffness Q11 (plane stress); turing vehicle_balloon_tire.py
    eqs['eq_BR13_7'] = sp.Eq(Q22, E2 / (1 - nu12 * nu21))  # orthotropic reduced stiffness Q22; turing vehicle_balloon_tire.py
    eqs['eq_BR13_8'] = sp.Eq(Q12, nu12 * E2 / (1 - nu12 * nu21))  # orthotropic reduced stiffness Q12; turing vehicle_balloon_tire.py
    eqs['eq_BR13_9'] = sp.Eq(Q66, G12)  # orthotropic reduced stiffness Q66; turing vehicle_balloon_tire.py
    eqs['eq_BR13_10'] = sp.Eq(g12, 2 * E12)  # engineering shear strain; turing vehicle_balloon_tire.py
    eqs['eq_BR13_11'] = sp.Eq(W_Q, sp.Rational(1, 2) * (Q11 * e1**2 + Q22 * e2**2 + 2 * Q12 * e1 * e2 + Q66 * g12**2 + 2 * Q16 * e1 * g12 + 2 * Q26 * e2 * g12))  # orthotropic (rotated, Q-bar) laminate energy density; turing vehicle_balloon_tire.py
    eqs['eq_BR13_12'] = sp.Eq(R_ray, mu_d * tr(Ed * Ed) + lam_d / 2 * tr(Ed)**2)  # Rayleigh dissipation function on strain rate (Kelvin-Voigt membrane); turing vehicle_balloon_tire.py
    eqs['eq_BR13_13'] = sp.Eq(Q_d, -sp.Derivative(R_ray, qdot))  # generalized damping force from the Rayleigh function; turing vehicle_balloon_tire.py

    # BR14 plasticity and fracture
    sig_y, sig_y0, H, epb = sp.symbols('sigma_y sigma_{y0} H bar{epsilon}^p')
    sig_eq, sig_n, tau_s = sp.symbols('sigma_{eq} sigma tau')
    dlam, sig_tr, E, G_sh = sp.symbols('Delta_lambda sigma_{eq}^{trial} E G')
    D_duc, eps_f, eps_frem, beta_f, h_w = sp.symbols('D_{duct} epsilon_f epsilon_f^{rem} beta_{frag} h_{work}')
    sig_u = sp.Symbol('sigma_u')
    P_v, V_m, eta_a, eta_b, eta_s, ea_d, eb_d, gs_d, W_p = sp.symbols(
        'P_{visc} V_m eta_a eta_b eta_s dot{epsilon}_a dot{epsilon}_b dot{gamma}_s Delta_W_p')
    eqs['eq_BR14_1'] = sp.Eq(sig_y, sig_y0 + H * epb)  # linear isotropic hardening; turing vehicle_mechanical_material.py
    eqs['eq_BR14_2'] = sp.Eq(sig_eq, sp.sqrt(sig_n**2 + 3 * tau_s**2))  # reduced von Mises (beam: one normal + one shear); turing vehicle_mechanical_material.py
    eqs['eq_BR14_3'] = sp.Eq(dlam, sp.Max(0, sig_tr - sig_y) / (E + H))  # 1-D radial-return plastic multiplier (3-D J2 uses 3G+H); turing vehicle_mechanical_material.py
    eqs['eq_BR14_4'] = sp.Eq(h_w, H * epb / sig_y0)  # normalized work hardening (code's); turing vehicle_mechanical_material.py
    eqs['eq_BR14_5'] = sp.Eq(eps_frem, eps_f / (1 + beta_f * h_w))  # remaining ductility after work hardening, empirical; turing vehicle_mechanical_material.py
    eqs['eq_BR14_6'] = sp.Eq(D_duc, sp.Max(sig_eq / sig_u, epb / eps_frem))  # ductility-exhaustion / ultimate-stress damage demand; turing vehicle_mechanical_material.py
    eqs['eq_BR14_7'] = D_duc >= 1  # fracture criterion (ductility exhausted or ultimate exceeded); turing vehicle_mechanical_material.py
    eqs['eq_BR14_8'] = sp.Eq(P_v, V_m * (eta_a * ea_d**2 + eta_b * eb_d**2 + eta_s * gs_d**2))  # viscous strain-rate dissipation power; turing vehicle_mechanical_material.py
    eqs['eq_BR14_9'] = sp.Eq(W_p, sig_y * dlam * V_m)  # plastic work increment; turing vehicle_mechanical_material.py
    return eqs


globals().update(_expand_bragg())


# =====================================================================
# 3b. FARADAY -- expansion (2026-09-22 coverage audit)
# =====================================================================
def _expand_faraday():
    eqs = {}

    # F8 transformers (engine_toy/transformers.py)
    V_rms, f_line, N_turns, A_core, B_pk = sp.symbols('V_{rms} f N A_{core} B_{pk}')
    X_l, L_leak, I_sc, V_oc, R_w, V_2, I_2, Z_w = sp.symbols('X_{leak} L_{leak} I_{sc} V_{oc} R_w V_2 I_2 Z_w')
    Reg, V_nl, V_fl = sp.symbols('Reg V_{nl} V_{fl}')
    P_core, k_st, alpha_st, beta_st, m_core = sp.symbols('P_{core} k_{St} alpha_{St} beta_{St} m_{core}')
    p_ref, B_ref, f_ref = sp.symbols('p_{ref} B_{ref} f_{ref}')
    P_cu, I_1, R_1, R_2, a_tr, N_1, N_2 = sp.symbols('P_{cu} I_1 R_1 R_2 a N_1 N_2')
    eqs['eq_F8_1'] = sp.Eq(V_rms, 2*sp.pi/sp.sqrt(2) * f_line * N_turns * A_core * B_pk)  # transformer EMF equation, 2pi/sqrt2 = 4.44; transformers.py
    eqs['eq_F8_2'] = sp.Eq(X_l, 2*sp.pi*f_line*L_leak)  # leakage reactance (referred to secondary); transformers.py
    eqs['eq_F8_3'] = sp.Eq(Z_w, sp.sqrt(X_l**2 + R_w**2))  # series winding impedance magnitude; transformers.py
    eqs['eq_F8_4'] = sp.Eq(I_sc, V_oc/Z_w)  # short-circuit secondary current; transformers.py
    eqs['eq_F8_5'] = sp.Eq(V_2, sp.sqrt(V_oc**2 - (Z_w*I_2)**2))  # quadrature-drop regulation circle (valid while X dominates R); transformers.py
    eqs['eq_F8_6'] = sp.Eq(Reg, (V_nl - V_fl)/V_fl)  # voltage regulation, textbook definition; transformers.py
    eqs['eq_F8_7'] = sp.Eq(P_core, k_st * f_line**alpha_st * B_pk**beta_st * m_core)  # Steinmetz core loss (general); transformers.py
    eqs['eq_F8_8'] = sp.Eq(P_core, p_ref*m_core*(B_pk/B_ref)**2*(f_line/f_ref)**sp.Rational(3, 2))  # Steinmetz scaled from a quoted point (beta=2, alpha=1.5 as in code); transformers.py
    eqs['eq_F8_9'] = sp.Eq(a_tr, N_2/N_1)  # turns ratio; transformers.py
    eqs['eq_F8_10'] = sp.Eq(I_1, a_tr*I_2)  # primary current reflected from secondary (ideal); transformers.py
    eqs['eq_F8_11'] = sp.Eq(P_cu, I_1**2*R_1 + I_2**2*R_2)  # copper (I^2 R) loss in both windings; transformers.py

    # F9 rotating machines and transducers (starter.py, dyno_brakes.py, turing symbolic_fluid_source.VoiceCoil)
    V_m, I_m, R_m, K_s, omega_m, T_m = sp.symbols('V_m I_m R_m K_s omega_m T_m')
    T_eddy, T_pk, c_fld, omega_c = sp.symbols('T_{eddy} T_{pk} c_{field} omega_c')
    F_vc, Bl, i_vc, v_bemf = sp.symbols('F_{vc} Bl i_{vc} v_{bemf}')
    x_c = sp.Function('x_c')(t)
    k_ms, M_ms, f_s, R_ms, Q_ms, Q_es, Q_ts, R_e, V_as, rho_air, c_air, S_d, C_ms = sp.symbols(
        'k_{ms} M_{ms} f_s R_{ms} Q_{ms} Q_{es} Q_{ts} R_e V_{as} rho_{air} c_{air} S_d C_{ms}')
    eqs['eq_F9_1'] = sp.Eq(V_m, I_m*R_m + K_s*I_m*omega_m)  # series-wound DC motor voltage (back-EMF ~ I*omega); starter.py
    eqs['eq_F9_2'] = sp.Eq(T_m, K_s*I_m**2)  # series-wound DC motor torque; starter.py
    eqs['eq_F9_3'] = sp.Eq(T_eddy, T_pk*c_fld**2*2*(omega_m/omega_c)/(1 + (omega_m/omega_c)**2))  # eddy-current brake characteristic (peak at critical speed, ~1/omega above, B^2 in excitation); dyno_brakes.py
    eqs['eq_F9_4'] = sp.Eq(F_vc, Bl*i_vc)  # voice-coil Lorentz force F = Bl i; VoiceCoil
    eqs['eq_F9_5'] = sp.Eq(v_bemf, Bl*sp.Derivative(x_c, t))  # voice-coil motional back-EMF (standard reciprocal of F = Bl i)
    eqs['eq_F9_6'] = sp.Eq(M_ms*sp.Derivative(x_c, t, 2), Bl*i_vc - R_ms*sp.Derivative(x_c, t) - k_ms*x_c)  # driver cone equation of motion; VoiceCoil.step
    eqs['eq_F9_7'] = sp.Eq(k_ms, M_ms*(2*sp.pi*f_s)**2)  # Thiele-Small suspension stiffness from Fs, Mms; VoiceCoil
    eqs['eq_F9_8'] = sp.Eq(R_ms, 2*sp.pi*f_s*M_ms/Q_ms)  # Thiele-Small mechanical loss from Qms; VoiceCoil
    eqs['eq_F9_9'] = sp.Eq(Q_es, 2*sp.pi*f_s*M_ms*R_e/Bl**2)  # Thiele-Small electrical Q (standard; not in code)
    eqs['eq_F9_10'] = sp.Eq(Q_ts, Q_ms*Q_es/(Q_ms + Q_es))  # Thiele-Small total Q (standard; not in code)
    eqs['eq_F9_11'] = sp.Eq(V_as, rho_air*c_air**2*S_d**2*C_ms)  # Thiele-Small equivalent compliance volume, C_ms = 1/k_ms (standard; not in code)

    # F10 conductors and components (dc_power.py, traces.py, knees.py, rectifiers.py, battery_bank.py, electrical_network.py)
    R_c, rho_e, L_len, A_x, R_0, alpha_T, T_c, T_0 = sp.symbols('R rho_e L A R_0 alpha_R T T_0')
    L_p, mu_0, l_tr, w_tr, t_tr = sp.symbols('L_p mu_0 l w t_{tr}')
    I_max, k_ipc, dT_rise, A_mil2 = sp.symbols('I_{max} k_{IPC} Delta_T A_{mil^2}')
    I_d, I_s, V_d, n_id, V_T, k_B, q_e, T_j = sp.symbols('I_D I_S V_D n V_T k_B q T_j')
    V_out, V_pk, V_sec, V_f, V_PIV, Q_cyc, I_avg = sp.symbols('V_{out} V_{pk} V_{sec;rms} V_f V_{PIV} Q_{cycle} I_{avg}')
    V_ocv, N_s, N_p, V_empty, V_full, SOC, V_term, I_b, R_int, R_mod, eta_c, C_Ah = sp.symbols(
        'V_{OCV} N_s N_p V_{empty} V_{full} SOC V_{term} I_b R_{int} R_{module} eta_c C_{Ah}')
    SOC_t = sp.Function('SOC')(t)
    eqs['eq_F10_1'] = sp.Eq(R_c, rho_e*L_len/A_x)  # Pouillet's law; dc_power.py, traces.py
    eqs['eq_F10_2'] = sp.Eq(R_c, R_0*(1 + alpha_T*(T_c - T_0)))  # linear temperature coefficient of resistance (Cu 0.00393/K); dc_power.py, transformers.py
    eqs['eq_F10_3'] = sp.Eq(L_p, mu_0/(2*sp.pi)*l_tr*(sp.log(2*l_tr/(w_tr + t_tr)) + sp.Rational(1, 2) + sp.Float('0.2235')*(w_tr + t_tr)/l_tr))  # Rosa/Grover partial self-inductance of a flat straight conductor; traces.py
    eqs['eq_F10_4'] = sp.Eq(I_max, k_ipc*dT_rise**sp.Float('0.44')*A_mil2**sp.Float('0.725'))  # IPC-2221 ampacity, empirical, k=0.048 external / 0.024 internal, A in mil^2; traces.py
    eqs['eq_F10_5'] = sp.Eq(I_d, I_s*(sp.exp(V_d/(n_id*V_T)) - 1))  # Shockley diode equation; knees.py
    eqs['eq_F10_6'] = sp.Eq(V_T, k_B*T_j/q_e)  # thermal voltage; knees.py
    eqs['eq_F10_7'] = sp.Eq(V_pk, sp.sqrt(2)*V_sec)  # sinusoidal peak from rms; rectifiers.py
    eqs['eq_F10_8'] = sp.Eq(V_out, 2*V_pk - V_f)  # half-wave voltage doubler open-circuit output; rectifiers.py
    eqs['eq_F10_9'] = sp.Eq(V_PIV, 2*V_pk)  # doubler diode peak inverse voltage; rectifiers.py
    eqs['eq_F10_10'] = sp.Eq(Q_cyc, I_avg/f_line)  # charge delivered per supply cycle; rectifiers.py
    eqs['eq_F10_11'] = sp.Eq(V_ocv, N_s*(V_empty + (V_full - V_empty)*SOC))  # battery OCV(SOC), linear lead-acid model (11.8-12.7 V/module); electrical_network.py
    eqs['eq_F10_12'] = sp.Eq(R_int, R_mod*N_s/N_p)  # pack internal resistance of series/parallel modules; electrical_network.py, battery_bank.py
    eqs['eq_F10_13'] = sp.Eq(V_term, V_ocv + I_b*R_int)  # battery terminal voltage (charging current positive); electrical_network.py
    eqs['eq_F10_14'] = sp.Eq(sp.Derivative(SOC_t, t), eta_c*I_b/(3600*C_Ah))  # coulomb counting with charge efficiency (eta_c=1 on discharge); electrical_network.py

    # F11 guided waves and apertures (waveguides.py)
    f_c, chi_p_mn, chi_mn, c_med, a_r, m_i, n_i, a_w, b_w = sp.symbols("f_c chi'_{mn} chi_{mn} c_{medium} a m n a_w b_w")
    beta_g, alpha_ev, k_0, k_c, A_dB, l_g = sp.symbols('beta alpha_{ev} k k_c A_{dB} l_g')
    Z_TE, Z_TM, eta_w, mu_w, eps_w, f_w = sp.symbols('Z_{TE} Z_{TM} eta mu epsilon f')
    alpha_m, alpha_e, sigma_t, T_ap, SE_ap = sp.symbols('alpha_m alpha_e sigma_t T_{ap} SE_{ap}')
    eqs['eq_F11_1'] = sp.Eq(f_c, chi_p_mn*c_med/(2*sp.pi*a_r))  # circular guide TE_mn cutoff, chi'_mn = n-th zero of J'_m (TE11: 1.8412); waveguides.py
    eqs['eq_F11_2'] = sp.Eq(f_c, chi_mn*c_med/(2*sp.pi*a_r))  # circular guide TM_mn cutoff, chi_mn = n-th zero of J_m (TM01: 2.4048); waveguides.py
    eqs['eq_F11_3'] = sp.Eq(f_c, c_med/2*sp.sqrt((m_i/a_w)**2 + (n_i/b_w)**2))  # rectangular guide TE_mn/TM_mn cutoff (TM needs m,n>=1); waveguides.py
    eqs['eq_F11_4'] = sp.Eq(k_c, 2*sp.pi*f_c/c_med)  # cutoff wavenumber; waveguides.py
    eqs['eq_F11_5'] = sp.Eq(alpha_ev, sp.sqrt(k_c**2 - k_0**2))  # evanescent attenuation constant below cutoff; waveguides.py
    eqs['eq_F11_6'] = sp.Eq(A_dB, 20/sp.log(10)*alpha_ev*l_g)  # below-cutoff attenuation in dB of field amplitude (32 dB/diameter for TE11); waveguides.py
    eqs['eq_F11_7'] = sp.Eq(eta_w, sp.sqrt(mu_w/eps_w))  # intrinsic impedance of the fill; waveguides.py
    eqs['eq_F11_8'] = sp.Eq(Z_TE, eta_w/sp.sqrt(1 - (f_c/f_w)**2))  # TE wave impedance (imaginary/inductive below cutoff); waveguides.py
    eqs['eq_F11_9'] = sp.Eq(Z_TM, eta_w*sp.sqrt(1 - (f_c/f_w)**2))  # TM wave impedance (imaginary/capacitive below cutoff); waveguides.py
    eqs['eq_F11_10'] = sp.Eq(alpha_m, sp.Rational(4, 3)*a_r**3)  # Bethe magnetic polarizability of a small circular hole; waveguides.py
    eqs['eq_F11_11'] = sp.Eq(alpha_e, sp.Rational(2, 3)*a_r**3)  # Bethe electric polarizability of a small circular hole; waveguides.py
    eqs['eq_F11_12'] = sp.Eq(sigma_t, 64*k_0**4*a_r**6/(27*sp.pi**2))  # Bethe transmission cross-section, normal incidence, thin PEC screen; waveguides.py
    eqs['eq_F11_13'] = sp.Eq(T_ap, sigma_t/(sp.pi*a_r**2))  # aperture power transmission coefficient; waveguides.py
    eqs['eq_F11_14'] = sp.Eq(SE_ap, -10*sp.log(T_ap, 10))  # aperture shielding in dB; waveguides.py
    eqs['eq_F11_15'] = k_0*a_r < 1  # Bethe validity: hole small against wavelength (ka << 1; also screen thin vs hole); waveguides.py

    # F12 cavities and radiators (cavities.py, emitters.py, chambers.py, applicators.py, turing dec_cavity.py)
    E_c = sp.Function('E')(x, y, z)
    J_c = sp.Function('J')(x, y, z)
    mu_c, eps_c, omega, eps_cplx, Q_u, a_n, k_n, e_n, N_modes = sp.symbols('mu epsilon omega epsilon_c Q a_n k_n e_n N')
    V_cav, S_cav, f_cav, c_0, U_st, P_loss, delta_s, R_s = sp.symbols('V S f c U P_{loss} delta_s R_s')
    a_c, b_c, d_c, eta_c0 = sp.symbols('a b d eta')
    L_n, C_n, R_n, omega_n, M_lp, L_loop, k_cpl, A_loop, h_n = sp.symbols('L_n C_n R_n omega_n M L_{loop} k_{cpl} A_{loop} h_n')
    R_rad, A_lp, lam, P_rad, m_dip, I_lp, k_w = sp.symbols('R_{rad} A lambda P_{rad} m I k')
    P_av, V_emf, R_src, SE, P_del, P_radiated = sp.symbols('P_{av} V_{emf} R_{src} SE P_{delivered} P_{radiated}')
    df_f, eps_r_p, F_fill, eps_pp, tan_d, P_abs, eps_0, E_0, W_body, W_cav, Q_d, R_d, tol = sp.symbols(
        "Delta_f/f epsilon'_r F_{fill} epsilon'' tan_delta P_{abs} epsilon_0 E_0 W_{body} W_{cavity} Q_d R_d tol")
    k_d = sp.Symbol('k')
    eqs['eq_F12_1'] = sp.Eq(nabla(nabla(E_c)/mu_c) - omega**2*eps_c*E_c, -sp.I*omega*J_c)  # driven time-harmonic curl-curl source form (curl abstracted; e^{-i omega t} sign convention of the code); dec_cavity.py
    eqs['eq_F12_2'] = sp.Eq(eps_cplx, eps_c*(1 + sp.I/Q_u))  # complex-permittivity Q loss at one frequency; dec_cavity.py
    eqs['eq_F12_3'] = sp.Eq(E_c, sp.Sum(a_n*e_n, ('n', 1, N_modes)))  # modal expansion over M-orthonormal cavity modes; dec_cavity.py, cavities.py
    eqs['eq_F12_4'] = sp.Eq(a_n, -sp.I*omega*mu_c*sp.Integral(e_n*J_c, x, y, z)/(k_n**2 - k_d**2*(1 + sp.I/Q_u)))  # modal coefficient of the driven, lossy cavity (uniform mu; follows from F12_1 with curl curl e_n = k_n^2 e_n); dec_cavity.py
    eqs['eq_F12_5'] = sp.Eq(N_modes, 8*sp.pi*V_cav*f_cav**3/(3*c_0**3))  # Weyl's law, EM mode count below f (both polarizations); dec_cavity.py
    eqs['eq_F12_6'] = sp.Eq(Q_u, omega*U_st/P_loss)  # quality factor definition, perturbation (lossless-field) method; cavities.py
    eqs['eq_F12_7'] = sp.Eq(Q_u, (k_d*a_c*d_c)**3*b_c*eta_c0/(2*sp.pi**2*R_s*(2*a_c**3*b_c + 2*b_c*d_c**3 + a_c**3*d_c + a_c*d_c**3)))  # wall-loss Q of rectangular TE101 (Pozar 6.46); cavities.py
    eqs['eq_F12_8'] = sp.Eq(Q_u, 2*V_cav/(delta_s*S_cav))  # order-of-magnitude wall-loss Q (volume/surface in skin depths); cavities.py
    eqs['eq_F12_9'] = sp.Eq(C_n, 1/(omega_n**2*L_n))  # modal series-LC equivalent: capacitance from mode inductance; cavities.py
    eqs['eq_F12_10'] = sp.Eq(R_n, omega_n*L_n/Q_u)  # modal series loss giving the mode its linewidth; cavities.py
    eqs['eq_F12_11'] = sp.Eq(M_lp, mu_c*A_loop*h_n)  # loop-to-mode mutual inductance, h_n = mode H projected on loop normal (gauge cancels in M^2/L); cavities.py
    eqs['eq_F12_12'] = sp.Eq(k_cpl, M_lp/sp.sqrt(L_n*L_loop))  # coupling coefficient; cavities.py
    eqs['eq_F12_13'] = sp.Eq(P_rad, eta_c0*k_w**4*m_dip**2/(12*sp.pi))  # magnetic-dipole radiated power, m = I A; emitters.py
    eqs['eq_F12_14'] = sp.Eq(R_rad, 2*P_rad/I_lp**2)  # radiation resistance from radiated power (peak phasor); emitters.py
    eqs['eq_F12_15'] = sp.Eq(R_rad, 320*sp.pi**4*(A_lp/lam**2)**2)  # small-loop radiation resistance in free space, 320 pi^4 = 31171; emitters.py
    eqs['eq_F12_16'] = sp.Eq(P_av, sp.Abs(V_emf)**2/(8*R_src))  # available power into a conjugate match (peak phasors); emitters.py
    eqs['eq_F12_17'] = sp.Eq(SE, 10*sp.log(P_del/P_radiated, 10))  # shielding effectiveness as a power ratio; chambers.py
    eqs['eq_F12_18'] = sp.Eq(eps_pp, eps_r_p*tan_d)  # loss factor from loss tangent; applicators.py
    eqs['eq_F12_19'] = sp.Eq(P_abs, omega*eps_0*eps_pp/2*sp.Abs(E_0)**2*W_body)  # dielectric absorbed power, W_body = integral |e|^2 dV over the load; applicators.py
    eqs['eq_F12_20'] = sp.Eq(Q_d, omega*L_n/R_d)  # Q the dielectric load alone gives the mode; applicators.py
    eqs['eq_F12_21'] = sp.Eq(F_fill, W_body/W_cav)  # electric filling factor; applicators.py
    eqs['eq_F12_22'] = sp.Eq(df_f, -(eps_r_p - 1)/2*F_fill)  # cavity-perturbation frequency pull (small dielectric body); applicators.py
    eqs['eq_F12_23'] = sp.Abs(df_f) <= tol  # loss-only (perturbation) validity criterion; applicators.py

    # F13 waves and plasma (turing examples/symbolic_em_solvers.py)
    psi = sp.Function('psi')(t, x, y, z)
    c_w, gamma_d, s_src, u_w, p_w, omega_w, k_wv = sp.symbols('c_w gamma_d s_{src} u_{wave} p_{diss} omega k')
    lap_psi = sp.Derivative(psi, x, 2) + sp.Derivative(psi, y, 2) + sp.Derivative(psi, z, 2)
    alpha_Tn, A_T, B_T, p_gas, E_mag, E_bd = sp.symbols('alpha_T A_T B_T p_{gas} |E| E_{bd}')
    n_q = sp.Function('n_q')(t)
    G_ion, mu_q, z_q, e_ch, k_rec, n_o, k_wall, D_q, Gamma_q = sp.symbols('G_{ion} mu_q z_q e k_{rec} n_{other} k_{wall} D_q Gamma_q')
    omega_p, lam_D, sigma_q, eps_q, m_q, k_Bq, T_g, n_s = sp.symbols('omega_p lambda_D sigma_q epsilon m_q k_B T_g n')
    eqs['eq_F13_1'] = sp.Eq(sp.Derivative(psi, t, 2), c_w**2*lap_psi - gamma_d*sp.Derivative(psi, t) + s_src)  # damped, driven scalar wave (d'Alembert) equation; wave_step
    eqs['eq_F13_2'] = sp.Eq(u_w, sp.Rational(1, 2)*(sp.Derivative(psi, t)**2 + c_w**2*(sp.Derivative(psi, x)**2 + sp.Derivative(psi, y)**2 + sp.Derivative(psi, z)**2)))  # wave energy density; wave_step
    eqs['eq_F13_3'] = sp.Eq(p_w, gamma_d*sp.Derivative(psi, t)**2 - s_src*sp.Derivative(psi, t))  # wave power lost to damping net of source work; wave_step
    eqs['eq_F13_4'] = sp.Eq(omega_w, c_w*k_wv)  # non-dispersive dispersion relation; wave_step
    eqs['eq_F13_5'] = sp.Eq(alpha_Tn, A_T*p_gas*sp.exp(-B_T*p_gas/E_mag))  # Townsend first ionization coefficient, empirical A,B per gas; charged_species_step
    eqs['eq_F13_6'] = sp.Eq(Gamma_q, -D_q*nabla(n_q) + z_q*mu_q*E_mag*n_q)  # drift-diffusion flux (field direction abstracted); charged_species_step
    eqs['eq_F13_7'] = sp.Eq(sp.Derivative(n_q, t), G_ion + alpha_Tn*sp.Abs(z_q*mu_q)*E_mag*n_q - k_rec*n_q*n_o - k_wall*n_q - nabla(Gamma_q))  # charged-species balance: source + Townsend avalanche - two-body recombination - wall loss - divergence; charged_species_step
    eqs['eq_F13_8'] = E_mag > E_bd  # breakdown (avalanche) criterion; charged_species_step
    eqs['eq_F13_9'] = sp.Eq(omega_p, sp.sqrt(n_s*z_q**2*e_ch**2/(eps_q*m_q)))  # plasma frequency; charged_species_step
    eqs['eq_F13_10'] = sp.Eq(lam_D, sp.sqrt(eps_q*k_Bq*T_g/(n_s*z_q**2*e_ch**2)))  # Debye length; charged_species_step
    eqs['eq_F13_11'] = sp.Eq(sigma_q, n_s*sp.Abs(z_q)*e_ch*mu_q)  # species conductivity sigma = n|q|mu (textbook; code writes e^2 z^2 n mu, i.e. a per-charge mobility); charged_species_step
    eqs['eq_F13_12'] = sp.Eq(sp.Derivative(n_q, t), -k_rec*n_q*n_o)  # two-body recombination; charged_species_step
    return eqs


globals().update(_expand_faraday())


# =====================================================================
# 12b. FOURIER -- expansion (2026-09-22 coverage audit)
# =====================================================================
def _expand_fourier():
    eqs = {}

    # FO9 forced convection in tubes (barrel_thermal.py, interior_ballistics.py)
    Nu, Re, Pr, h_c, k_fl, D_h = sp.symbols('Nu Re Pr h_c k D_h')
    rho, u_m, mu, c_p = sp.symbols('rho u_m mu c_p')
    eqs['eq_FO9_1'] = sp.Eq(Nu, sp.Float('0.023')*Re**sp.Float('0.8')*Pr**sp.Float('0.4'))  # Dittus-Boelter, empirical, turbulent heating (n=0.4); barrel_thermal.py, interior_ballistics.py
    eqs['eq_FO9_2'] = sp.Eq(Nu, sp.Float('4.36'))  # laminar fully developed, constant wall flux (48/11); barrel_thermal.py
    eqs['eq_FO9_3'] = sp.Eq(h_c, Nu*k_fl/D_h)  # Nusselt definition; barrel_thermal.py
    eqs['eq_FO9_4'] = sp.Eq(Re, rho*u_m*D_h/mu)  # Reynolds number; barrel_thermal.py
    eqs['eq_FO9_5'] = sp.Eq(Pr, c_p*mu/k_fl)  # Prandtl number; barrel_thermal.py

    # FO10 phase change and cryogenic insulation (autoclave.py, cryogenics.py)
    Q_cond, m_dot, h_fg, Q_leak, m_boil, k_eff, A_w, T_h, T_cold, delta_w, BOR, m_liq = sp.symbols(
        'Qdot_{cond} mdot h_{fg} Qdot_{leak} mdot_{boil} k_{eff} A T_h T_c delta BOR m_{liq}')
    q_mli, sigma_SB, N_lay, eps_lay = sp.symbols("q''_{MLI} sigma_{SB} N epsilon")
    eqs['eq_FO10_1'] = sp.Eq(Q_cond, m_dot*h_fg)  # condensation heat release; autoclave.py
    eqs['eq_FO10_2'] = sp.Eq(m_boil, Q_leak/h_fg)  # cryogen boil-off rate; cryogenics.py
    eqs['eq_FO10_3'] = sp.Eq(Q_leak, k_eff*A_w*(T_h - T_cold)/delta_w)  # insulation wrap heat leak with effective conductivity (MLI 5e-5 W/mK in vacuum, 0.045 spoiled); cryogenics.py
    eqs['eq_FO10_4'] = sp.Eq(BOR, m_boil*86400/m_liq*100)  # boil-off rate in percent per day; cryogenics.py
    eqs['eq_FO10_5'] = sp.Eq(q_mli, sigma_SB*(T_h**4 - T_cold**4)/((N_lay + 1)*(2/eps_lay - 1)))  # radiative flux through N floating grey shields (standard MLI ideal; code uses a tabulated k_eff instead)

    # FO11 conduction shapes and series resistance (thermal_storage.py, turing surface_step)
    Q_s, k_s, r_s, dT, z_b, U_o, h_i, h_o, L_j, k_j, R_tot, A_s = sp.symbols('Qdot k r Delta_T z_b U h_i h_o L_j k_j R_{tot} A')
    eqs['eq_FO11_1'] = sp.Eq(Q_s, 4*sp.pi*k_s*r_s*dT)  # sphere in an infinite medium, S = 4 pi r; thermal_storage.py
    eqs['eq_FO11_2'] = sp.Eq(Q_s, 4*sp.pi*k_s*r_s*dT/(1 - r_s/(2*z_b)))  # sphere buried at centre depth z_b below an isothermal surface (standard shape factor; code uses the infinite-medium form)
    eqs['eq_FO11_3'] = sp.Eq(1/U_o, 1/h_i + sp.Sum(L_j/k_j, ('j', 1, sp.Symbol('N'))) + 1/h_o)  # overall coefficient of plane layers in series; thermal_storage.py, surface_step
    eqs['eq_FO11_4'] = sp.Eq(Q_s, dT/R_tot)  # thermal-resistance form, R_tot = 1/(U A); thermal_storage.py

    # FO12 solar irradiance and PV (solar.py)
    AM, alpha_s, G_b, h_km, G_bt, cos_th, beta_t, gam_s, gam_p, G_dt, G_d = sp.symbols(
        'AM alpha_s G_b h G_{b;tilt} cos_theta beta gamma_s gamma G_{d;tilt} G_d')
    T_cell, T_amb, NOCT, G_t, eta_pv, eta_ref, beta_pv, P_pv, A_pv = sp.symbols(
        'T_{cell} T_{amb} NOCT G eta_{PV} eta_{ref} beta_{PV} P_{PV} A_{PV}')
    eqs['eq_FO12_1'] = sp.Eq(AM, 1/(sp.sin(alpha_s*sp.pi/180) + sp.Float('0.50572')*(alpha_s + sp.Float('6.07995'))**sp.Float('-1.6364')))  # Kasten-Young air mass, alpha_s = solar elevation in degrees, empirical; solar.py
    eqs['eq_FO12_2'] = sp.Eq(G_b, 1353*((1 - sp.Float('0.14')*h_km)*sp.Float('0.7')**(AM**sp.Float('0.678')) + sp.Float('0.14')*h_km))  # Meinel/Laue beam irradiance W/m^2, h in km, empirical (code: 1353*0.7^(AM^0.678)*(1+1.4e-4 h[m])); solar.py
    eqs['eq_FO12_3'] = sp.Eq(cos_th, sp.sin(alpha_s)*sp.cos(beta_t) + sp.cos(alpha_s)*sp.sin(beta_t)*sp.cos(gam_s - gam_p))  # incidence cosine on a tilted surface (elevation alpha_s, tilt beta, azimuths; angles in radians here, unlike FO12_1); solar.py
    eqs['eq_FO12_4'] = sp.Eq(G_bt, G_b*sp.Max(0, cos_th))  # beam on the plane of array; solar.py
    eqs['eq_FO12_5'] = sp.Eq(G_dt, G_d*(1 + sp.cos(beta_t))/2)  # isotropic-sky diffuse view factor (Liu-Jordan); solar.py
    eqs['eq_FO12_6'] = sp.Eq(T_cell, T_amb + (NOCT - 20)/800*G_t)  # NOCT cell temperature, G in W/m^2, empirical; solar.py
    eqs['eq_FO12_7'] = sp.Eq(eta_pv, eta_ref*(1 - beta_pv*(T_cell - 25)))  # PV temperature derate (T in C); solar.py
    eqs['eq_FO12_8'] = sp.Eq(P_pv, A_pv*eta_pv*G_t)  # PV array power; solar.py
    return eqs


globals().update(_expand_fourier())


# =====================================================================
# 20b. ZELDOVICH -- expansion (2026-09-22 coverage audit)
# =====================================================================
def _expand_zeldovich():
    eqs = {}

    # Z5 blast scaling (ordnance.py)
    Z_s, R_r, W_tnt, dP_s, P_r, p_0, i_s, K_i, R_fb, k_fb = sp.symbols('Z R W Delta_P_s P_r p_0 i_s K_i R_{fb} k_{fb}')
    gam = sp.Symbol('gamma')
    eqs['eq_Z5_1'] = sp.Eq(Z_s, R_r/W_tnt**sp.Rational(1, 3))  # Hopkinson-Cranz scaled distance (m/kg^(1/3)); ordnance.py
    eqs['eq_Z5_2'] = sp.Eq(dP_s, sp.Float('0.085')/Z_s + sp.Float('0.3')/Z_s**2 + sp.Float('0.8')/Z_s**3)  # Sadovsky free-air peak overpressure in MPa, empirical (1 <= Z <= 15); ordnance.py
    eqs['eq_Z5_3'] = sp.Eq(P_r, 2*dP_s + 6*dP_s**2/(dP_s + 7*p_0))  # normal reflected overpressure, Rankine-Hugoniot ideal gas gamma=1.4; ordnance.py
    eqs['eq_Z5_4'] = sp.Eq(P_r, 2*dP_s + (gam + 1)*dP_s**2/((gam - 1)*dP_s + 2*gam*p_0))  # normal reflection, general gamma (reduces to Z5_3 at 1.4)
    eqs['eq_Z5_5'] = sp.Eq(i_s, K_i*W_tnt**sp.Rational(2, 3)/R_r)  # Sadovsky scaled specific impulse, empirical K (~200 Pa s m/kg^(2/3)); ordnance.py
    eqs['eq_Z5_6'] = sp.Eq(R_fb, k_fb*W_tnt**sp.Rational(1, 3))  # fireball radius cube-root scaling, empirical k ~ 3 m/kg^(1/3); ordnance.py

    # Z6 fragment and plate velocity (ordnance.py, burst.py)
    V_g, sqrt2E, M_c, C_e, E_f, M_f, v_f = sp.symbols('V_G sqrt(2E) M C E_f M_f v_f')
    eqs['eq_Z6_1'] = sp.Eq(V_g, sqrt2E/sp.sqrt(M_c/C_e + sp.Rational(1, 2)))  # Gurney cylinder (cased charge); ordnance.py
    eqs['eq_Z6_2'] = sp.Eq(V_g, sqrt2E/sp.sqrt(M_c/C_e + sp.Rational(3, 5)))  # Gurney sphere (standard)
    eqs['eq_Z6_3'] = sp.Eq(V_g, sqrt2E/sp.sqrt(2*M_c/C_e + sp.Rational(1, 3)))  # Gurney symmetric sandwich, M = mass of each plate (standard)
    eqs['eq_Z6_4'] = sp.Eq(V_g, sqrt2E/sp.sqrt(((1 + 2*M_c/C_e)**3 + 1)/(6*(1 + M_c/C_e)) + M_c/C_e))  # Gurney open-faced sandwich (single flyer plate, standard)
    eqs['eq_Z6_5'] = sp.Eq(v_f, sp.sqrt(2*E_f/M_f))  # equal-share fragment speed from the energy fraction given to fragments; burst.py

    # Z7 stored-energy bursts (burst.py)
    E_b, m_g, R_g, T_g, p_v, p_a, V_v, K_l, gam_g = sp.symbols('E_{burst} m_g R_s T p p_0 V K gamma')
    eqs['eq_Z7_1'] = sp.Eq(E_b, m_g*R_g*T_g*sp.log(p_v/p_a))  # stored-gas burst energy, isothermal expansion to ambient; burst.py
    eqs['eq_Z7_2'] = sp.Eq(E_b, (p_v - p_a)*V_v/(gam_g - 1))  # Brode stored-gas burst energy (standard; code uses the isothermal form Z7_1)
    eqs['eq_Z7_3'] = sp.Eq(E_b, p_v**2*V_v/(2*K_l))  # elastic energy of a pressurised liquid, bulk modulus K; burst.py
    return eqs


globals().update(_expand_zeldovich())


# =====================================================================
# 18b. TARTAGLIA -- expansion (2026-09-22 coverage audit)
# =====================================================================
def _expand_tartaglia():
    eqs = {}

    # TA5 flat fire (calibres.py)
    v_x, v_0, k_d, x_r, rho_a, C_D, A_ref, m_p, t_f = sp.symbols('v v_0 k x rho C_D A m_p t_f')
    eqs['eq_TA5_1'] = sp.Eq(v_x, v_0*sp.exp(-k_d*x_r))  # flat-fire drag decay v(x) with constant-C_D quadratic drag; calibres.py
    eqs['eq_TA5_2'] = sp.Eq(k_d, rho_a*C_D*A_ref/(2*m_p))  # flat-fire drag constant (standard; code tabulates k per calibre)
    eqs['eq_TA5_3'] = sp.Eq(t_f, (sp.exp(k_d*x_r) - 1)/(k_d*v_0))  # time of flight to range x, integral of dx/v (standard companion)

    # TA6 free recoil (calibres.py, tasked_fire.py)
    I_p, I_gas, I_vent, I_net, m_c, beta_g, v_mz, eta_b, f_v, V_r, M_gun, E_r, t_r, F_pk = sp.symbols(
        'I_p I_{gas} I_{vent} I_{net} m_c beta v_{mz} eta_b f_v V_r M_{gun} E_r t_r F_{pk}')
    eqs['eq_TA6_1'] = sp.Eq(I_p, m_p*v_mz)  # projectile momentum; calibres.py
    eqs['eq_TA6_2'] = sp.Eq(I_gas, beta_g*m_c*v_mz)  # propellant-gas momentum, gas velocity ratio beta ~1.5 (SAAMI-style); calibres.py
    eqs['eq_TA6_3'] = sp.Eq(I_vent, I_gas*(eta_b + 2*f_v))  # momentum turned by muzzle brake / rearward vent, as in code (factor 2 = reversed vent gas); calibres.py
    eqs['eq_TA6_4'] = sp.Eq(I_net, I_p + I_gas - I_vent)  # net recoil impulse; calibres.py
    eqs['eq_TA6_5'] = sp.Eq(V_r, I_net/M_gun)  # free-recoil velocity (standard)
    eqs['eq_TA6_6'] = sp.Eq(E_r, M_gun*V_r**2/2)  # free-recoil energy (standard)
    eqs['eq_TA6_7'] = sp.Eq(F_pk, sp.Abs(I_net)/t_r)  # mean unbraked recoil force over the ~2 ms impulse; calibres.py
    return eqs


globals().update(_expand_tartaglia())


# =====================================================================
# 4b. NAVIER-STOKES -- expansion (2026-09-22 coverage audit)
# =====================================================================
def _expand_navier_stokes():
    eqs = {}

    # NS9 shallow water (Saint-Venant) with viscosity, linear drag, Coriolis
    h = sp.Function('h')(t, x, y)
    u = sp.Function('u_sw')(t, x, y)
    v = sp.Function('v_sw')(t, x, y)
    g, nu, r_drag, f_cor, c_wave = sp.symbols('g nu_sw r_drag f_Coriolis c_gravity_wave')

    def lap(q):
        return sp.Derivative(q, x, 2) + sp.Derivative(q, y, 2)

    eqs['eq_NS9_1'] = sp.Eq(sp.Derivative(h, t), -sp.Derivative(h*u, x) - sp.Derivative(h*v, y))  # Saint-Venant mass; turing symbolic_fluid_model.py (Shoal)
    eqs['eq_NS9_2'] = sp.Eq(sp.Derivative(h*u, t),
                            -sp.Derivative(h*u**2 + g*h**2/2, x) - sp.Derivative(h*u*v, y)
                            + nu*lap(h*u) - r_drag*h*u + f_cor*h*v)  # Saint-Venant x-momentum (viscosity, drag, Coriolis); symbolic_fluid_model.py
    eqs['eq_NS9_3'] = sp.Eq(sp.Derivative(h*v, t),
                            -sp.Derivative(h*u*v, x) - sp.Derivative(h*v**2 + g*h**2/2, y)
                            + nu*lap(h*v) - r_drag*h*v - f_cor*h*u)  # Saint-Venant y-momentum; symbolic_fluid_model.py
    eqs['eq_NS9_4'] = sp.Eq(c_wave, sp.sqrt(g*h))  # shallow-water gravity wave speed; symbolic_fluid_model.py (Rusanov |u|+sqrt(gh))

    # NS10 orifices and leaks
    mdot, Cd, A, p0, pb, gam, Rs, T0 = sp.symbols('dot{m}_orifice C_d A_orifice p_0 p_b gamma R_s T_0')
    v_T, Q_T, g2, H_head = sp.symbols('v_Torricelli Q_Torricelli g h_head')
    m_drop, r_tip, sigma = sp.symbols('m_drop r_tip sigma')
    We, Oh, rho_l, U_rel, d_drop, mu_l, We_c, We_c0 = sp.symbols('We Oh rho_l U_rel d_drop mu_l We_c We_c0')
    eqs['eq_NS10_1'] = sp.Eq(mdot, Cd*A*p0*sp.sqrt(2*gam/((gam - 1)*Rs*T0)
                             * ((pb/p0)**(2/gam) - (pb/p0)**((gam + 1)/gam))))  # Saint-Venant-Wanzel subcritical compressible orifice; fluid_circuit_laws.py, hole_emitters.py
    eqs['eq_NS10_2'] = (pb/p0) > (2/(gam + 1))**(gam/(gam - 1))  # subcritical (unchoked) validity of NS10_1; complement of NS6_5
    eqs['eq_NS10_3'] = sp.Eq(v_T, sp.sqrt(2*g2*H_head))  # Torricelli efflux speed; hole_emitters.py
    eqs['eq_NS10_4'] = sp.Eq(Q_T, Cd*A*sp.sqrt(2*g2*H_head))  # Torricelli discharge with C_d; hole_emitters.py, fittings.py
    eqs['eq_NS10_5'] = sp.Eq(m_drop, 2*sp.pi*r_tip*sigma/g2)  # Tate's drop law (ideal; Harkins-Brown factor omitted); hole_emitters.py
    eqs['eq_NS10_6'] = sp.Eq(We, rho_l*U_rel**2*d_drop/sigma)  # Weber number; hole_emitters.py
    eqs['eq_NS10_7'] = sp.Eq(Oh, mu_l/sp.sqrt(rho_l*sigma*d_drop))  # Ohnesorge number; hole_emitters.py
    eqs['eq_NS10_8'] = sp.Eq(We_c, We_c0*(1 + 1.077*Oh**1.6))  # breakup-map critical We (Brodkey/Pilch-Erdman), empirical, We_c0~12
    eqs['eq_NS10_9'] = We > We_c  # secondary-breakup criterion; hole_emitters.py

    # NS11 turbomachinery similarity
    Q1, Q2, H1, H2, P1, P2, n1, n2, D1, D2 = sp.symbols('Q_1 Q_2 H_1 H_2 P_1 P_2 n_1 n_2 D_1 D_2')
    T_wb, K_wb, rho, D, omega = sp.symbols('T_waterbrake K_waterbrake rho D omega')
    T_fc, lam_fc, n_rev = sp.symbols('T_coupling lambda_coupling n_rev')
    mdot_ad, T_ad, P_ad, A_disk, v_i, w_wake = sp.symbols('dot{m}_disk T_thrust P_induced A_disk v_i w_wake')
    eqs['eq_NS11_1'] = sp.Eq(Q2, Q1*(n2/n1)*(D2/D1)**3)  # affinity law, flow; hydraulics.py, station_cooling.py, air_movers.py
    eqs['eq_NS11_2'] = sp.Eq(H2, H1*(n2/n1)**2*(D2/D1)**2)  # affinity law, head
    eqs['eq_NS11_3'] = sp.Eq(P2, P1*(n2/n1)**3*(D2/D1)**5)  # affinity law, power
    eqs['eq_NS11_4'] = sp.Eq(T_wb, K_wb*rho*D**5*omega**2)  # water-brake absorption torque; dyno_brakes.py
    eqs['eq_NS11_5'] = sp.Eq(T_fc, lam_fc*rho*D**5*n_rev**2)  # Foettinger fluid coupling, lambda = capacity factor(slip); couplings.py
    eqs['eq_NS11_6'] = sp.Eq(mdot_ad, rho*A_disk*v_i)  # actuator-disk mass flow through disk; air_movers.py
    eqs['eq_NS11_7'] = sp.Eq(T_ad, mdot_ad*w_wake)  # actuator-disk momentum thrust (far-wake velocity); air_movers.py
    eqs['eq_NS11_8'] = sp.Eq(w_wake, 2*v_i)  # Rankine-Froude: far wake = twice induced velocity (static)
    eqs['eq_NS11_9'] = sp.Eq(P_ad, T_ad*v_i)  # ideal induced power = mdot*w^2/2 = 2 rho A v_i^3 (audit's "T=mdot v_i, P=1/2 rho A v_i^3" corrected)
    eqs['eq_NS11_10'] = sp.Eq(v_i, sp.sqrt(T_ad/(2*rho*A_disk)))  # hover induced velocity from thrust

    # NS12 separation
    Q_s, v_g, Sigma, Sigma_t, Sigma_d, L_b, r1, r2 = sp.symbols('Q_sep v_g Sigma Sigma_tubular Sigma_disc L_bowl r_1 r_2')
    N_disc, theta_d, d_p, d_c, drho, mu = sp.symbols('N_disc theta_disc d_p d_cut Delta_rho mu')
    d50, W_in, N_e, V_in, rho_p, rho_g, eta_c = sp.symbols('d_50 W_inlet N_e V_inlet rho_p rho_g eta_cyclone')
    eqs['eq_NS12_1'] = sp.Eq(v_g, drho*g*d_p**2/(18*mu))  # Stokes gravity settling velocity; centrifuges.py
    eqs['eq_NS12_2'] = sp.Eq(Q_s, v_g*Sigma)  # Sigma theory (full-capture cut; Ambler 50% convention is Q=2 v_g Sigma); centrifuges.py
    eqs['eq_NS12_3'] = sp.Eq(Sigma_t, sp.pi*L_b*omega**2*(r2**2 - r1**2)/(g*sp.log(r2/r1)))  # tubular-bowl Sigma (plug flow); centrifuges.py
    eqs['eq_NS12_4'] = sp.Eq(Sigma_d, 2*sp.pi*N_disc*omega**2*(r2**3 - r1**3)/(3*g*sp.tan(theta_d)))  # disc-stack Sigma, theta = half-cone angle; centrifuges.py
    eqs['eq_NS12_5'] = sp.Eq(d_c, sp.sqrt(18*mu*Q_s/(drho*g*Sigma)))  # Stokes cut size from Sigma; centrifuges.py
    eqs['eq_NS12_6'] = sp.Eq(d50, sp.sqrt(9*mu*W_in/(2*sp.pi*N_e*V_in*(rho_p - rho_g))))  # Lapple cyclone cut diameter; centrifuges.py
    eqs['eq_NS12_7'] = sp.Eq(eta_c, 1/(1 + (d50/d_p)**2))  # Lapple grade-efficiency curve, empirical

    # NS13 hydraulic lines
    dp_v, Q_v, A_v = sp.symbols('Delta_p_valve Q_valve A_valve')
    F_f, F_c, F_st, v_rel, v_str, b_visc, F_app = sp.symbols('F_friction F_coulomb F_breakaway v_rel v_Stribeck b_viscous F_applied')
    w_n, beta, A_p, V_t, m_l, V_1, V_2 = sp.symbols('omega_n beta_bulk A_piston V_total m_load V_1 V_2')
    Q_slot, w_s, h_s, dp_s, L_s, U_s = sp.symbols('Q_slot w_slot h_gap Delta_p_slot L_slot U_wall')
    tau, tau_y, mu_p, gdot, F_MR, F_y, c_0 = sp.symbols('tau tau_y mu_plastic dot{gamma} F_MR F_yield c_0')
    eqs['eq_NS13_1'] = sp.Eq(dp_v, rho*Q_v*sp.Abs(Q_v)/(2*Cd**2*A_v**2))  # quadratic valve pressure drop (inverted orifice); hydraulic_losses.py
    eqs['eq_NS13_2'] = sp.Eq(F_f, (F_c + (F_st - F_c)*sp.exp(-(v_rel/v_str)**2))*sp.sign(v_rel) + b_visc*v_rel)  # seal friction, Stribeck breakaway->running; actuators.py (code: F_st = 1.8 F_c)
    eqs['eq_NS13_3'] = sp.Abs(F_app) > F_st  # breakaway (stiction release) criterion; actuators.py
    eqs['eq_NS13_4'] = sp.Eq(w_n, sp.sqrt(beta*A_p**2/m_l*(1/V_1 + 1/V_2)))  # hydraulic natural frequency, two chambers; actuators.py
    eqs['eq_NS13_5'] = sp.Eq(w_n, sp.sqrt(4*beta*A_p**2/(V_t*m_l)))  # same at mid-stroke V_1=V_2=V/2; actuators.py
    eqs['eq_NS13_6'] = sp.Eq(Q_slot, w_s*h_s**3*dp_s/(12*mu*L_s) + w_s*h_s*U_s/2)  # parallel-plate slot flow, Poiseuille + Couette; hydraulic_losses.py
    eqs['eq_NS13_7'] = sp.Eq(tau, tau_y*sp.sign(gdot) + mu_p*gdot)  # Bingham plastic (yielded, |tau|>tau_y); actuators.py MR damper
    eqs['eq_NS13_8'] = sp.Abs(tau) > tau_y  # Bingham yield criterion (else gdot = 0)
    eqs['eq_NS13_9'] = sp.Eq(F_MR, F_y*sp.sign(v_rel) + c_0*v_rel)  # Bingham MR damper force model (Stanway); actuators.py

    # NS14 fluid properties and particle fluids
    nu_cSt, A_W, B_W, T_K = sp.symbols('nu_cSt A_Walther B_Walther T_K')
    p_T, B_T, rho0, gam_T, c0 = sp.symbols('p_Tait B_Tait rho_0 gamma_Tait c_0')
    f_B, beta_T, T_f, T_ref, g_vec = sp.symbols('f_Boussinesq beta_T T T_ref g_vec')
    p_f = sp.Function('p')(t, x, y, z)
    u_star, u_new, dt = sp.symbols('u^* u^{n+1} Delta_t')
    c_col = sp.Function('c_color')(t, x, y, z)
    f_sv, kappa_c, n_hat = sp.symbols('f_sv kappa_curv hat{n}')
    eqs['eq_NS14_1'] = sp.Eq(sp.log(sp.log(nu_cSt + 0.7, 10), 10), A_W - B_W*sp.log(T_K, 10))  # Walther / ASTM D341 viscosity-temperature, empirical; hydraulics.py
    eqs['eq_NS14_2'] = sp.Eq(p_T, B_T*((rho/rho0)**gam_T - 1))  # Tait (Cole) EOS, weakly compressible; turing bath/*
    eqs['eq_NS14_3'] = sp.Eq(B_T, rho0*c0**2/gam_T)  # Tait stiffness from reference sound speed
    eqs['eq_NS14_4'] = sp.Eq(f_B, -rho0*beta_T*(T_f - T_ref)*g_vec)  # Boussinesq buoyancy force density; turing bath/*
    eqs['eq_NS14_5'] = sp.Eq(nabla(nabla(p_f)), rho/dt*nabla(u_star))  # pressure-projection Poisson (Chorin); turing bath/*
    eqs['eq_NS14_6'] = sp.Eq(u_new, u_star - dt/rho*nabla(p_f))  # projection velocity correction
    eqs['eq_NS14_7'] = sp.Eq(f_sv, sigma*kappa_c*nabla(c_col))  # CSF surface tension (Brackbill); turing bath/*
    eqs['eq_NS14_8'] = sp.Eq(n_hat, nabla(c_col)/sp.Abs(nabla(c_col)))  # CSF interface normal
    eqs['eq_NS14_9'] = sp.Eq(kappa_c, -nabla(n_hat))  # CSF curvature = -div(n_hat)

    # NS15 films and pools
    l_c, h_sess, theta_c = sp.symbols('l_c h_sessile theta_contact')
    u_film, h_film, alpha_i, yy, u_mean, q_film = sp.symbols('u_film h_film alpha_incline y_n u_mean_film q_film')
    Q_weir, w_weir, H_weir = sp.symbols('Q_weir w_weir H_weir')
    eqs['eq_NS15_1'] = sp.Eq(l_c, sp.sqrt(sigma/(rho*g)))  # capillary length; turing surface_step/pool_step, surface_wetting.py
    eqs['eq_NS15_2'] = sp.Eq(h_sess, 2*l_c*sp.sin(theta_c/2))  # large sessile puddle height; surface_wetting.py
    eqs['eq_NS15_3'] = sp.Eq(u_film, rho*g*sp.sin(alpha_i)/mu*(h_film*yy - yy**2/2))  # Nusselt film velocity profile; surface_step
    eqs['eq_NS15_4'] = sp.Eq(u_mean, rho*g*sp.sin(alpha_i)*h_film**2/(3*mu))  # Nusselt film mean velocity
    eqs['eq_NS15_5'] = sp.Eq(q_film, rho*g*sp.sin(alpha_i)*h_film**3/(3*mu))  # Nusselt film flow per unit width
    eqs['eq_NS15_6'] = sp.Eq(Q_weir, sp.Rational(2, 3)*Cd*w_weir*sp.sqrt(2*g)*H_weir**sp.Rational(3, 2))  # sharp-crested rectangular weir (audit's C = 2/3 C_d); pool_step

    # NS16 ventilation
    C_v = sp.Function('C')(t)
    Q_a, V_room, C_in, S_src, ACH, C_ss, tau_v = sp.symbols('Q_air V_room C_in S_source ACH C_ss tau_vent')
    eqs['eq_NS16_1'] = sp.Eq(sp.Derivative(C_v, t), Q_a/V_room*(C_in - C_v) + S_src/V_room)  # well-mixed CSTR concentration; air_volumes.py
    eqs['eq_NS16_2'] = sp.Eq(ACH, 3600*Q_a/V_room)  # air changes per hour (Q in m^3/s); air_volumes.py
    eqs['eq_NS16_3'] = sp.Eq(C_ss, C_in + S_src/Q_a)  # CSTR steady state
    eqs['eq_NS16_4'] = sp.Eq(tau_v, V_room/Q_a)  # CSTR residence / relaxation time
    return eqs


globals().update(_expand_navier_stokes())


# =====================================================================
# 5b. BJERKNES -- expansion (2026-09-22 coverage audit)
# =====================================================================
def _expand_bjerknes():
    eqs = {}

    # B11 Kessler bulk warm-rain microphysics
    P_auto, P_acc, E_rev, k1, k2, k3, q_c, q_c0, q_r, S_sat = sp.symbols(
        'P_auto P_accr E_revap k_auto k_accr k_revap q_c q_c0 q_r S_sat')
    eqs['eq_B11_1'] = sp.Eq(P_auto, k1*sp.Max(q_c - q_c0, 0))  # Kessler autoconversion (threshold); turing symbolic_chamber_solvers.py voxel_species_step
    eqs['eq_B11_2'] = sp.Eq(P_acc, k2*q_c*q_r**sp.Float(0.875))  # Kessler accretion q_c q_r^0.875; voxel_species_step
    eqs['eq_B11_3'] = sp.Eq(E_rev, k3*sp.Max(1 - S_sat, 0)*q_r**sp.Float(0.65))  # Kessler rain re-evaporation, (q_vs-q_v)/q_vs ~ 1-S, empirical; voxel_species_step

    # B12 aerosol
    Cc, lam, d_p, Kn_p = sp.symbols('C_c lambda_mfp d_p Kn_p')
    K12, K_mono, kB, T, mu, d1, d2, C1, C2 = sp.symbols('K_12 K_coag k_B T mu_air d_1 d_2 C_c1 C_c2')
    N = sp.Function('N')(t)
    D_B, v_s, rho_p, g = sp.symbols('D_B v_settle rho_p g')
    S_k, D_w, D_d, kappa, A_kel, M_w, sig_w, R_u, rho_w, S_crit = sp.symbols(
        'S_Koehler D_wet D_dry kappa A_Kelvin M_w sigma_w R_u rho_w S_crit')
    C_a = sp.Function('C_aerosol')(t)
    Lam, a_wash, b_wash, R_rain = sp.symbols('Lambda_wash a_wash b_wash R_rain')
    eqs['eq_B12_1'] = sp.Eq(Cc, 1 + 2.514*lam/d_p)  # Cunningham slip, small-Kn limit of B12_2 (Kn_p = 2 lambda/d, exp term -> 0); aerosol_step uses this form
    eqs['eq_B12_2'] = sp.Eq(Cc, 1 + Kn_p*(1.257 + 0.4*sp.exp(-1.1/Kn_p)))  # Cunningham slip correction, Davies (1945) coefficients, Kn_p = 2 lambda/d; valid across the continuum-to-free-molecular range
    eqs['eq_B12_3'] = sp.Eq(K12, 2*kB*T/(3*mu)*(C1/d1 + C2/d2)*(d1 + d2))  # Brownian coagulation kernel (continuum+slip); Seinfeld-Pandis
    eqs['eq_B12_4'] = sp.Eq(K_mono, 8*kB*T*Cc/(3*mu))  # monodisperse Brownian kernel; aerosol_step
    eqs['eq_B12_5'] = sp.Eq(sp.Derivative(N, t), -K_mono*N**2/2)  # monodisperse coagulation number loss; aerosol_step
    eqs['eq_B12_6'] = sp.Eq(D_B, kB*T*Cc/(3*sp.pi*mu*d_p))  # Stokes-Einstein(-Cunningham) diffusivity; aerosol_step
    eqs['eq_B12_7'] = sp.Eq(v_s, rho_p*g*d_p**2*Cc/(18*mu))  # slip-corrected Stokes settling (air density neglected); aerosol_step
    eqs['eq_B12_8'] = sp.Eq(S_k, (D_w**3 - D_d**3)/(D_w**3 - D_d**3*(1 - kappa))*sp.exp(A_kel/D_w))  # kappa-Koehler (Petters-Kreidenweis); droplet_step
    eqs['eq_B12_9'] = sp.Eq(A_kel, 4*M_w*sig_w/(R_u*T*rho_w))  # Kelvin coefficient (diameter form); aerosol_step
    eqs['eq_B12_10'] = sp.Eq(sp.log(S_crit), sp.sqrt(4*A_kel**3/(27*kappa*D_d**3)))  # kappa-Koehler critical supersaturation; aerosol_step
    eqs['eq_B12_11'] = sp.Eq(sp.Derivative(C_a, t), -Lam*C_a)  # below-cloud washout; aerosol_step
    eqs['eq_B12_12'] = sp.Eq(Lam, a_wash*R_rain**b_wash)  # washout coefficient power law, empirical (code: b = 0.8 on rain water content)

    # B13 drops
    r, S_amb, S_eq, F_k, F_d, L_v, R_v, k_a, D_v, e_s = sp.symbols('r S_ambient S_eq F_k F_d L_v R_v k_a D_v e_s')
    v_S, v_N, v_t, rho_a, C_D = sp.symbols('v_Stokes v_Newton v_t rho_air C_D')
    Nu, h_c = sp.symbols('Nu h_c')
    eqs['eq_B13_1'] = sp.Eq(r*sp.Derivative(r, t), (S_amb - S_eq)/(F_k + F_d))  # Mason droplet growth law; droplet_step (Rogers-Yau)
    eqs['eq_B13_2'] = sp.Eq(F_k, (L_v/(R_v*T) - 1)*L_v*rho_w/(k_a*T))  # Mason heat-conduction term
    eqs['eq_B13_3'] = sp.Eq(F_d, rho_w*R_v*T/(D_v*e_s))  # Mason vapour-diffusion term
    eqs['eq_B13_4'] = sp.Eq(v_S, 2*(rho_p - rho_a)*g*r**2/(9*mu))  # Stokes terminal velocity (code drops rho_air); droplet_step
    eqs['eq_B13_5'] = sp.Eq(v_N, sp.sqrt(8*(rho_p - rho_a)*g*r/(3*C_D*rho_a)))  # Newton-regime terminal velocity (code drops rho_air); droplet_step
    eqs['eq_B13_6'] = sp.Eq(v_t, v_S*v_N/sp.sqrt(v_S**2 + v_N**2))  # Stokes-Newton blend, empirical interpolation; droplet_step
    eqs['eq_B13_7'] = sp.Eq(Nu, 2)  # quiescent sphere conduction limit; droplet_step
    eqs['eq_B13_8'] = sp.Eq(h_c, Nu*k_a/(2*r))  # sphere film coefficient, = k_a/r at Nu=2; droplet_step

    # B14 psychrometrics
    T_c, p_v, p, w_hr, T_dc, gam_m = sp.symbols('T_C p_v p_air w_humidity T_dew_C gamma_Magnus')
    j_ev, rho_vs, rho_vinf, R_kin, R_diff, alpha_m, h_m, h_t, c_p, Le = sp.symbols(
        'j_evap rho_{v;s} rho_{v;infty} R_kinetic R_diffusive alpha_m h_m h_heat c_p Le')
    eqs['eq_B14_1'] = sp.Eq(e_s, 610.94*sp.exp(17.625*T_c/(T_c + 243.04)))  # Magnus-Tetens (Alduchov-Eskridge), Pa, T in C; air_treatment.py
    eqs['eq_B14_2'] = sp.Eq(w_hr, 0.622*p_v/(p - p_v))  # humidity ratio; air_treatment.py
    eqs['eq_B14_3'] = sp.Eq(p_v, p*w_hr/(0.622 + w_hr))  # vapour pressure from humidity ratio; air_treatment.py
    eqs['eq_B14_4'] = sp.Eq(gam_m, sp.log(p_v/610.94))  # Magnus dew-point intermediate; air_treatment.py
    eqs['eq_B14_5'] = sp.Eq(T_dc, 243.04*gam_m/(17.625 - gam_m))  # Magnus inverse: dew point, C; air_treatment.py
    eqs['eq_B14_6'] = sp.Eq(h_m, h_t/(rho_a*c_p*Le**sp.Rational(2, 3)))  # Chilton-Colburn heat/mass analogy; air_treatment.py, symbolic_atmosphere_model
    eqs['eq_B14_7'] = sp.Eq(R_diff, 1/h_m)  # diffusive (boundary-layer) evaporation resistance
    eqs['eq_B14_8'] = sp.Eq(R_kin, 1/(alpha_m*sp.sqrt(R_v*T/(2*sp.pi))))  # kinetic (Hertz-Knudsen) resistance, density form
    eqs['eq_B14_9'] = sp.Eq(j_ev, (rho_vs - rho_vinf)/(R_kin + R_diff))  # series kinetic+diffusive evaporation flux
    return eqs


globals().update(_expand_bjerknes())


# =====================================================================
# 25b. ARCHIMEDES -- expansion (2026-09-22 coverage audit), fluid power as Pascal
# =====================================================================
def _expand_archimedes():
    eqs = {}

    # AR3 cylinders
    F_cap, F_rod, F_net, p, p1, p2, A_b, A_r, v_ext, v_ret, Q = sp.symbols(
        'F_cap F_rod F_net p p_1 p_2 A_bore A_rod v_extend v_retract Q')
    eqs['eq_AR3_1'] = sp.Eq(F_cap, p*A_b)  # Pascal cylinder force, cap side; actuators.py, outriggers.py
    eqs['eq_AR3_2'] = sp.Eq(F_rod, p*(A_b - A_r))  # rod-side (annulus) force; actuators.py
    eqs['eq_AR3_3'] = sp.Eq(F_net, p1*A_b - p2*(A_b - A_r))  # net force, both chambers pressurised; actuators.py, symbolic_parts.py
    eqs['eq_AR3_4'] = sp.Eq(v_ext, Q/A_b)  # extend speed; actuators.py
    eqs['eq_AR3_5'] = sp.Eq(v_ret, Q/(A_b - A_r))  # retract speed; actuators.py

    # AR4 motors and relief valves
    T_m, dp, D_m, n_m, w_m, P_loss = sp.symbols('T_motor Delta_p D_displacement n_motor omega_motor P_relief')
    eqs['eq_AR4_1'] = sp.Eq(T_m, dp*D_m/(2*sp.pi))  # ideal hydraulic motor torque, D per rev; starter.py, symbolic_parts.py
    eqs['eq_AR4_2'] = sp.Eq(n_m, Q/D_m)  # ideal motor speed, rev/s; starter.py
    eqs['eq_AR4_3'] = sp.Eq(w_m, 2*sp.pi*Q/D_m)  # ideal motor angular speed
    eqs['eq_AR4_4'] = sp.Eq(P_loss, dp*Q)  # relief-valve (throttling) power loss; actuators.py
    return eqs


globals().update(_expand_archimedes())


# =====================================================================
# 21b. OTTO -- expansion (2026-09-22 coverage audit)
# =====================================================================
def _expand_otto():
    eqs = {}
    Sym = sp.Symbol

    # O6 ideal air-standard cycles and the Carnot bound
    r_c, gam, r_co, r_p = sp.symbols('r_c gamma r_{co} r_p')
    eta_Otto, eta_Diesel, eta_Brayton, eta_Brayton_reg, eta_Carnot, eta_th = sp.symbols(
        'eta_{Otto} eta_{Diesel} eta_{Brayton} eta_{Brayton;reg} eta_{Carnot} eta_{th}')
    T_H, T_C, T_1_cyc, T_3_cyc = sp.symbols('T_H T_C T_{1;cyc} T_{3;cyc}')
    eqs['eq_O6_1'] = sp.Eq(eta_Otto, 1 - r_c**(1 - gam))  # Otto air-standard efficiency; thermo_cycles, derived_torque, engines
    eqs['eq_O6_2'] = sp.Eq(eta_Diesel, 1 - r_c**(1 - gam)*(r_co**gam - 1)/(gam*(r_co - 1)))  # Diesel efficiency with cutoff ratio r_co; thermo_cycles
    eqs['eq_O6_3'] = sp.Eq(eta_Brayton, 1 - r_p**((1 - gam)/gam))  # Brayton efficiency on pressure ratio; gas_turbine, thermo_cycles
    eqs['eq_O6_4'] = sp.Eq(eta_Brayton_reg, 1 - (T_1_cyc/T_3_cyc)*r_p**((gam - 1)/gam))  # ideal-regenerator Brayton (T1 compressor inlet, T3 turbine inlet); gas_turbine
    eqs['eq_O6_5'] = sp.Eq(eta_Carnot, 1 - T_C/T_H)  # Carnot efficiency; thermo_cycles
    eqs['eq_O6_6'] = eta_th <= eta_Carnot  # Carnot bound on any cycle between T_H and T_C; thermo_cycles

    # O7 mean effective pressure, torque, friction
    IMEP, BMEP, FMEP, eta_v, rho_a, LHV, AFR, eta_fi = sp.symbols('IMEP BMEP FMEP eta_v rho_{a;i} Q_{LHV} AFR eta_{f;i}')
    T_brake, V_d, n_R, S_p, S_stroke, N_rev, P_brake, p_max = sp.symbols('T_b V_d n_R \\bar{S}_p S N P_b p_{max}')
    A_cf, B_cf, C_cf, D_cf, eta_mech = sp.symbols('A_{fmep} B_{fmep} C_{fmep} D_{fmep} eta_m')
    eqs['eq_O7_1'] = sp.Eq(IMEP, eta_fi*eta_v*rho_a*LHV/AFR)  # IMEP from breathing and fuel energy (Heywood 2.41); engines, torque_curve, engine_sim
    eqs['eq_O7_2'] = sp.Eq(BMEP, IMEP - FMEP)  # brake MEP = indicated minus friction MEP; engines
    eqs['eq_O7_3'] = sp.Eq(T_brake, BMEP*V_d/(2*sp.pi*n_R))  # torque from BMEP, n_R revs per cycle; engines, torque_curve
    eqs['eq_O7_4'] = sp.Eq(T_brake, BMEP*V_d/(4*sp.pi))  # four-stroke (n_R=2) torque T=BMEP*V_d/(4 pi); engines, starter, turing abstract_ui_vehicles
    eqs['eq_O7_5'] = sp.Eq(FMEP, A_cf + B_cf*p_max + C_cf*S_p + D_cf*S_p**2)  # Chen-Flynn friction MEP, empirical; engines, engine_sim. Code drops the B*p_max term (FMEP=A+B*S_p+C*S_p^2)
    eqs['eq_O7_6'] = sp.Eq(S_p, 2*S_stroke*N_rev)  # mean piston speed, N in rev/s; engines, engine_sim
    eqs['eq_O7_7'] = sp.Eq(P_brake, 2*sp.pi*N_rev*T_brake)  # brake power P = 2 pi N T; engines, torque_curve
    eqs['eq_O7_8'] = sp.Eq(eta_mech, BMEP/IMEP)  # mechanical efficiency; engines

    # O8 breathing and tuning
    f_H, c_snd, A_neck, L_neck, V_cav = sp.symbols('f_{Helmholtz} c A_{runner} L_{runner} V_{plenum}')
    f_qw, L_pipe = sp.symbols('f_{1/4} L_{pipe}')
    Z_mach, B_bore, D_valve, C_i = sp.symbols('Z B D_{v} C_i')
    phi_th, phi_max, A_frac = sp.symbols('phi_{throttle} phi_{throttle;max} A_{open}/A_{open;max}')
    theta_c = Sym('\\theta_{crank}')
    V_crank = sp.Function('V')
    theta_IVC, theta_EVO, V_clear, r_c_eff, r_e_eff = sp.symbols('\\theta_{IVC} \\theta_{EVO} V_{c;cyl} r_{c;eff} r_{e;eff}')
    eqs['eq_O8_1'] = sp.Eq(f_H, (c_snd/(2*sp.pi))*sp.sqrt(A_neck/(L_neck*V_cav)))  # Helmholtz resonator runner tuning; engines, port_flow
    eqs['eq_O8_2'] = sp.Eq(f_qw, c_snd/(4*L_pipe))  # quarter-wave open-pipe resonance; engines (exhaust tailpipe)
    eqs['eq_O8_3'] = sp.Eq(Z_mach, (B_bore/D_valve)**2*S_p/(C_i*c_snd))  # Taylor inlet Mach index (Heywood 6.10); port_flow uses Z = port gas speed / c, same ratio
    eqs['eq_O8_4'] = sp.Eq(A_frac, (1 - sp.cos(phi_th))/(1 - sp.cos(phi_max)))  # butterfly open-area fraction, single plate; throttle_body (code form; full Heywood form also carries the closed-plate angle)
    eqs['eq_O8_5'] = sp.Eq(r_c_eff, V_crank(theta_IVC)/V_clear)  # effective (Miller/Atkinson) compression ratio: volume at IVC over clearance; valve_control
    eqs['eq_O8_6'] = sp.Eq(r_e_eff, V_crank(theta_EVO)/V_clear)  # effective expansion ratio at EVO (Miller/Atkinson: r_e > r_c); valve_control

    # O9 combustion timing, knock, incompleteness, pumping
    p_man, R_air, T_man, V_man, mdot_th, mdot_cyl, tau_man, p_man_ss = sp.symbols(
        'p_m R_{air} T_m V_m \\dot{m}_{th} \\dot{m}_{cyl} tau_m p_{m;ss}')
    S_T, S_L, k_turb = sp.symbols('S_T S_L k_T')
    theta_ign, theta_50, theta_MBT50, r_50, t_50 = sp.symbols('\\theta_{ign} \\Delta\\theta_{50} \\theta_{50;MBT} r_{50} t_{50}')
    I_knock, t_ivc, t_knk, tau_id, A_knk, n_knk, B_knk = sp.symbols('I_{knock} t_{IVC} t_{knock} tau_{id} A_{id} n_{id} B_{id}')
    p_cyl_f = sp.Function('p')
    T_cyl_f = sp.Function('T_{gas}')
    t_dum = Sym('t\'')
    f_crev, V_crev, V_cyl, T_gas, T_wall = sp.symbols('f_{crevice} V_{crevice} V_{cyl} T_{gas} T_{wall}')
    d_q, Pe_q, alpha_th, f_quench, A_ch = sp.symbols('d_q Pe_q alpha_{th} f_{quench} A_{chamber}')
    PMEP, p_exh, p_int, W_pump_rec, n_act, n_cyl, PMEP_deact = sp.symbols(
        'PMEP p_{exh} p_{int} \\Delta{}W_{pump} n_{active} n_{cyl} PMEP_{deact}')
    eqs['eq_O9_1'] = sp.Eq(sp.Derivative(p_man, t), (R_air*T_man/V_man)*(mdot_th - mdot_cyl))  # manifold filling/emptying (isothermal plenum); engine_cycle_sim
    eqs['eq_O9_2'] = sp.Eq(sp.Derivative(p_man, t), (p_man_ss - p_man)/tau_man)  # first-order MAP lag used by the sim; engine_cycle_sim
    eqs['eq_O9_3'] = sp.Eq(S_T, S_L + k_turb*S_p)  # turbulent flame speed S_T = S_L + k*u', u' ~ mean piston speed (Damkohler-type); engine_cycle_sim (audit wrote S_L(1+k rpm))
    eqs['eq_O9_4'] = sp.Eq(t_50, r_50/S_T)  # time for flame to reach 50% burnt radius; engine_cycle_sim
    eqs['eq_O9_5'] = sp.Eq(r_50, (B_bore/2)*sp.Rational(1, 2)**sp.Rational(1, 3))  # radius enclosing half the volume of a sphere of radius B/2; engine_cycle_sim
    eqs['eq_O9_6'] = sp.Eq(theta_50, 360*N_rev*t_50)  # crank degrees in t_50, N in rev/s; engine_cycle_sim (code writes 6*rpm*t)
    eqs['eq_O9_7'] = sp.Eq(theta_ign, theta_50 - theta_MBT50)  # MBT spark advance: place 50% burn at ~8-10 deg ATDC; engine_cycle_sim
    eqs['eq_O9_8'] = sp.Eq(I_knock, sp.Integral(1/tau_id, (t_dum, t_ivc, t_knk)))  # Livengood-Wu knock integral; knock when I_knock reaches 1; engine_cycle_sim
    eqs['eq_O9_9'] = sp.Eq(tau_id, A_knk*p_cyl_f(t_dum)**(-n_knk)*sp.exp(B_knk/T_cyl_f(t_dum)))  # Douaud-Eyzat ignition delay, empirical; engine_cycle_sim uses a Livengood-Wu-shaped rate
    eqs['eq_O9_10'] = sp.Eq(f_crev, (V_crev/V_cyl)*(T_gas/T_wall))  # crevice mass fraction at equal pressure, crevice gas at wall T (Heywood 7.6); combustion_efficiency
    eqs['eq_O9_11'] = sp.Eq(d_q, Pe_q*alpha_th/S_L)  # Peclet quench distance, Pe_q ~ 50 hydrocarbon two-plate; combustion_efficiency
    eqs['eq_O9_12'] = sp.Eq(f_quench, A_ch*d_q/V_cyl)  # quenched-layer volume fraction; combustion_efficiency
    eqs['eq_O9_13'] = sp.Eq(PMEP, p_exh - p_int)  # pumping mean effective pressure; cylinder_deactivation
    eqs['eq_O9_14'] = sp.Eq(W_pump_rec, V_d*(PMEP - (n_act/n_cyl)*PMEP_deact))  # pumping work recovered by deactivating cylinders (standard form; code uses an empirical load fraction); cylinder_deactivation

    # O10 two-stroke scavenging
    Lam, eta_tr_disp, eta_tr_mix, eta_tr, eta_ch, eta_sc_mix, k_plug = sp.symbols(
        'Lambda eta_{tr;disp} eta_{tr;mix} eta_{tr} eta_{ch} eta_{sc;mix} k_{plug}')
    eqs['eq_O10_1'] = sp.Eq(eta_tr_disp, sp.Min(1, 1/Lam))  # Hopkinson perfect-displacement trapping bound; two_stroke
    eqs['eq_O10_2'] = sp.Eq(eta_tr_mix, (1 - sp.exp(-Lam))/Lam)  # Hopkinson perfect-mixing trapping bound; two_stroke
    eqs['eq_O10_3'] = sp.Eq(eta_sc_mix, 1 - sp.exp(-Lam))  # perfect-mixing scavenging efficiency; two_stroke, derived_torque
    eqs['eq_O10_4'] = sp.Eq(eta_tr, k_plug*eta_tr_disp + (1 - k_plug)*eta_tr_mix)  # blend between the bounds by plug-flow fraction; two_stroke (code form)
    eqs['eq_O10_5'] = sp.Eq(eta_ch, Lam*eta_tr)  # charging efficiency = delivery ratio x trapping efficiency; two_stroke, derived_torque

    # O11 positive-displacement compressors and expanders
    eta_vc, C_cl, p_1, p_2, n_poly, W_poly, V_1, T_1, T_2 = sp.symbols(
        'eta_{v;cl} C_{cl} p_1 p_2 n W_{poly} V_1 T_1 T_2')
    MEP_iso, MEP_poly, p_sup, p_back, c_cut = sp.symbols('MEP_{iso} MEP_{poly} p_{sup} p_{back} c_{cut}')
    eqs['eq_O11_1'] = sp.Eq(eta_vc, 1 - C_cl*((p_2/p_1)**(1/n_poly) - 1))  # clearance volumetric efficiency; compressors
    eqs['eq_O11_2'] = sp.Eq(W_poly, (n_poly/(n_poly - 1))*p_1*V_1*((p_2/p_1)**((n_poly - 1)/n_poly) - 1))  # polytropic compression work; compressors, turing accessories
    eqs['eq_O11_3'] = sp.Eq(T_2, T_1*(p_2/p_1)**((n_poly - 1)/n_poly))  # polytropic discharge temperature; compressors
    eqs['eq_O11_4'] = sp.Eq(MEP_iso, p_sup*c_cut*(1 + sp.log(1/c_cut)) - p_back)  # cutoff-expander MEP, isothermal expansion, no clearance; expander
    eqs['eq_O11_5'] = sp.Eq(MEP_poly, p_sup*c_cut*(1 + (1 - c_cut**(n_poly - 1))/(n_poly - 1)) - p_back)  # cutoff-expander MEP, polytropic pV^n expansion, no clearance; expander
    return eqs


globals().update(_expand_otto())


# =====================================================================
# 3b. GIBBS -- expansion (2026-09-22 coverage audit)
# =====================================================================
def _expand_gibbs():
    eqs = {}

    # G9 process relations
    p, V, T, n_p, C_pv, C_tv = sp.symbols('p_{proc} V_{proc} T_{proc} n C_{pV^n} C_{TV^{n-1}}')
    gam, p1, p2, T1, T2s, T2, eta_c, eta_t, T2t = sp.symbols('gamma p_1 p_2 T_1 T_{2s} T_{2;c} eta_c eta_t T_{2;t}')
    W_iso, m_g, R_g, N_st, W_iso_ms, T_k, r_k, r_stage = sp.symbols('W_{iso} m R N_{stage} W_{iso;multi} T_k r_k r_{stage}')
    T_out_tx, T_in_tx, p_in_tx, p_out_tx = sp.symbols('T_{out;tx} T_{in;tx} p_{in;tx} p_{out;tx}')
    eqs['eq_G9_1'] = sp.Eq(C_pv, p*V**n_p)  # polytropic process pV^n = const; turing balloon-tire gas, symbolic_parts
    eqs['eq_G9_2'] = sp.Eq(C_tv, T*V**(n_p - 1))  # polytropic T V^(n-1) = const; turing balloon-tire gas
    eqs['eq_G9_3'] = sp.Eq(T2s, T1*(p2/p1)**((gam - 1)/gam))  # isentropic outlet temperature; oxygen_service, gas_harvest, cryogenics
    eqs['eq_G9_4'] = sp.Eq(T2, T1 + (T2s - T1)/eta_c)  # compressor outlet T with isentropic efficiency; oxygen_service, gas_harvest
    eqs['eq_G9_5'] = sp.Eq(T2t, T1 - eta_t*(T1 - T2s))  # expansion outlet T with isentropic efficiency; cryogenics, thermal_emission
    eqs['eq_G9_6'] = sp.Eq(W_iso, m_g*R_g*T1*sp.log(p2/p1))  # isothermal compression work; gas_harvest, oxygen_service
    eqs['eq_G9_7'] = sp.Eq(W_iso_ms, sp.Sum(m_g*R_g*T_k*sp.log(r_k), ('k', 1, N_st)))  # multistage isothermal work, stage k at T_k and ratio r_k; gas_harvest
    eqs['eq_G9_8'] = sp.Eq(r_stage, (p2/p1)**(1/N_st))  # equal per-stage pressure ratio; gas_harvest
    eqs['eq_G9_9'] = sp.Eq(T_out_tx, T_in_tx*(1 - eta_t*(1 - (p_out_tx/p_in_tx)**((gam - 1)/gam))))  # turboexpander outlet temperature; cryogenics

    # G10 refrigeration
    COP_C, COP_R, T_cold, T_hot, eta_II = sp.symbols('COP_{Carnot} COP_R T_{cold} T_{hot} eta_{II}')
    eqs['eq_G10_1'] = sp.Eq(COP_C, T_cold/(T_hot - T_cold))  # Carnot refrigerator COP; refrigeration
    eqs['eq_G10_2'] = sp.Eq(COP_R, eta_II*COP_C)  # real COP = second-law efficiency x Carnot; refrigeration

    # G11 mixtures
    p_tot, p_i, y_i, x_i, p_sat_i, T_inv, a_v, b_v, R_u = sp.symbols('p_{tot} p_i y_i x_i p_i^{sat} T_{inv} a b R_u')
    T_g, v_g, p_fr, p_tp, T_tp, L_sub, T_fr = sp.symbols('T v p_{frost} p_{tp} T_{tp} L_{sub} T_{frost}')
    v_T = sp.Function('v')(T_g)
    eqs['eq_G11_1'] = sp.Eq(p_tot, sp.Sum(p_i, ('i', 1, sp.Symbol('N'))))  # Dalton's law of partial pressures; autoclave, cryogenics
    eqs['eq_G11_2'] = sp.Eq(p_i, y_i*p_tot)  # Dalton partial pressure from mole fraction; autoclave
    eqs['eq_G11_3'] = sp.Eq(p_i, x_i*p_sat_i)  # Raoult's law; cryogenics, autoclave
    eqs['eq_G11_4'] = sp.Eq(v_T, T_g*sp.Derivative(v_T, T_g))  # JT inversion curve: mu_JT = 0 <=> v = T (dv/dT)_p; cryogenics
    eqs['eq_G11_5'] = sp.Eq(T_inv, 2*a_v/(R_u*b_v))  # van der Waals low-pressure inversion temperature; cryogenics
    eqs['eq_G11_6'] = sp.Eq(p_fr, p_tp*sp.exp(-(L_sub/R_u)*(1/T_fr - 1/T_tp)))  # CO2 frost-point curve, integrated Clausius-Clapeyron on sublimation from the triple point (standard form; source not read); cryogenics, autoclave

    # G12 solutions and colligative properties
    dT_f, dT_b, K_f, K_b, i_vh, b_m = sp.symbols('\\Delta{}T_f \\Delta{}T_b K_f K_b i b')
    p_solv, p_star, M_solv = sp.symbols('p_{solv} p^*_{solv} M_{solv}')
    c_sat, c_sat0, s_T, T_s, T_s0 = sp.symbols('c_{sat} c_{sat;0} s_T T_{soln} T_{soln;0}')
    S_ss, c_sol, G_gr, k_g, g_gr = sp.symbols('S c G k_g g')
    J_nuc, A_nuc, gam_sl, v_m, k_B = sp.symbols('J A_{nuc} gamma_{sl} v_m k_B')
    m_cr, k_dis, A_cr = sp.symbols('m_{crystal} k_{dis} A_{crystal}')
    eqs['eq_G12_1'] = sp.Eq(dT_f, K_f*i_vh*b_m)  # freezing-point depression; turing salt_solution_step, thermal_storage
    eqs['eq_G12_2'] = sp.Eq(dT_b, K_b*i_vh*b_m)  # boiling-point elevation; turing salt_solution_step
    eqs['eq_G12_3'] = sp.Eq(p_solv, p_star/(1 + i_vh*b_m*M_solv))  # Raoult in molality form, x_solv = 1/(1+i b M_solv); turing salt_solution_step
    eqs['eq_G12_4'] = sp.Eq(c_sat, c_sat0 + s_T*(T_s - T_s0))  # linear solubility in T, empirical; turing salt_solution_step
    eqs['eq_G12_5'] = sp.Eq(S_ss, c_sol/c_sat)  # supersaturation ratio; turing salt_solution_step
    eqs['eq_G12_6'] = sp.Eq(G_gr, k_g*(S_ss - 1)**g_gr)  # crystal growth rate power law, empirical; turing salt_solution_step
    eqs['eq_G12_7'] = sp.Eq(J_nuc, A_nuc*sp.exp(-16*sp.pi*gam_sl**3*v_m**2/(3*k_B**3*T_s**3*sp.log(S_ss)**2)))  # classical homogeneous nucleation rate; turing salt_solution_step (code form not read)
    eqs['eq_G12_8'] = sp.Eq(sp.Derivative(m_cr, t), -k_dis*A_cr*(c_sat - c_sol))  # Noyes-Whitney dissolution (undersaturated, c<c_sat); turing salt_solution_step

    # G13 distillation shortcut design
    N_min, x_D, x_B, alpha_LK, theta_U, q_feed, R_min, R_refl, N_th = sp.symbols(
        'N_{min} x_D x_B alpha_{LK/HK} theta_U q R_{min} R N')
    alpha_i, x_Fi, x_Di = sp.symbols('alpha_i x_{F;i} x_{D;i}')
    X_G, Y_G = sp.symbols('X_{Gil} Y_{Gil}')
    eqs['eq_G13_1'] = sp.Eq(N_min, sp.log((x_D/(1 - x_D))*((1 - x_B)/x_B))/sp.log(alpha_LK))  # Fenske minimum stages (binary / light-heavy key); air_separation
    eqs['eq_G13_2'] = sp.Eq(1 - q_feed, sp.Sum(alpha_i*x_Fi/(alpha_i - theta_U), ('i', 1, sp.Symbol('N'))))  # Underwood first equation, root theta_U; air_separation
    eqs['eq_G13_3'] = sp.Eq(R_min, sp.Sum(alpha_i*x_Di/(alpha_i - theta_U), ('i', 1, sp.Symbol('N'))) - 1)  # Underwood minimum reflux; air_separation
    eqs['eq_G13_4'] = sp.Eq(X_G, (R_refl - R_min)/(R_refl + 1))  # Gilliland abscissa; air_separation
    eqs['eq_G13_5'] = sp.Eq(Y_G, (N_th - N_min)/(N_th + 1))  # Gilliland ordinate; air_separation
    eqs['eq_G13_6'] = sp.Eq(Y_G, sp.Float(0.75)*(1 - X_G**sp.Float(0.5668)))  # Gilliland correlation, Eduljee fit, empirical; air_separation

    # G14 casting
    t_s, C_M, V_cast, A_cast, n_ch = sp.symbols('t_{solid} C_M V_{cast} A_{cast} n_{Chv}')
    L_ratio, s_k = sp.symbols('L_{final}/L_{master} s_k')
    eqs['eq_G14_1'] = sp.Eq(t_s, C_M*(V_cast/A_cast)**n_ch)  # Chvorinov's rule, n ~ 2; foundry
    eqs['eq_G14_2'] = sp.Eq(L_ratio, sp.Product(1 - s_k, ('k', 1, sp.Symbol('N'))))  # compounding linear shrinkage through N replication stages (wax, metal, ...); moulding
    return eqs


globals().update(_expand_gibbs())


# =====================================================================
# 14b. LAVOISIER -- expansion (2026-09-22 coverage audit)
# =====================================================================
def _expand_lavoisier():
    eqs = {}
    Nsum = sp.Symbol('N')

    # L8 equilibrium in temperature
    K_T, K_0, T, T_0, dH_r, R_u = sp.symbols('K(T) K(T_0) T T_0 \\Delta_r{}H^circ R_u')
    lnK = sp.Function('lnK')(T)
    eqs['eq_L8_1'] = sp.Eq(sp.Derivative(lnK, T), dH_r/(R_u*T**2))  # van't Hoff equation; turing compendium EquilibriumLaw
    eqs['eq_L8_2'] = sp.Eq(K_T, K_0*sp.exp(-(dH_r/R_u)*(1/T - 1/T_0)))  # integrated van't Hoff, constant Delta_r H; turing compendium EquilibriumLaw

    # L9 activity and electrolyte conductivity
    I_ion, m_i, z_i, log_gam, A_D, b_mol, n_sol, m_solv, c_mol, V_soln = sp.symbols(
        'I m_i z_i log_{10}gamma_i A_D b n_{solute} m_{solvent} c V_{solution}')
    kappa_el, Lam_m, Lam_m0, K_K = sp.symbols('kappa Lambda_m Lambda_m^circ K_{Kohl}')
    eqs['eq_L9_1'] = sp.Eq(I_ion, sp.Rational(1, 2)*sp.Sum(m_i*z_i**2, ('i', 1, Nsum)))  # ionic strength; turing compendium
    eqs['eq_L9_2'] = sp.Eq(log_gam, -A_D*z_i**2*(sp.sqrt(I_ion)/(1 + sp.sqrt(I_ion)) - sp.Float(0.3)*I_ion))  # Davies activity coefficient; turing compendium
    eqs['eq_L9_3'] = sp.Eq(b_mol, n_sol/m_solv)  # molality; turing compendium
    eqs['eq_L9_4'] = sp.Eq(c_mol, n_sol/V_soln)  # molarity; turing compendium
    eqs['eq_L9_5'] = sp.Eq(kappa_el, Lam_m*c_mol)  # conductivity from molar conductivity (Kohlrausch sigma = Lambda_m c); turing compendium
    eqs['eq_L9_6'] = sp.Eq(Lam_m, Lam_m0 - K_K*sp.sqrt(c_mol))  # Kohlrausch square-root law (dilute strong electrolyte); turing compendium (standard companion form)

    # L10 combustion bookkeeping
    x_C, y_H, z_O, u_S, n_O2 = sp.symbols('x_C y_H z_O u_S n_{O_2}')
    AFR_st, M_O2, M_fuel, Y_O2 = sp.symbols('AFR_{st} M_{O_2} M_{fuel} Y_{O_2;air}')
    nu_CO2, nu_H2O, nu_SO2, nu_N2, psi_N2 = sp.symbols('nu_{CO_2} nu_{H_2O} nu_{SO_2} nu_{N_2} psi_{N_2/O_2}')
    phi_eq, FA, FA_st, x_fuel, x_LFL, x_UFL, x_st, f_ub = sp.symbols('phi F/A (F/A)_{st} x_{fuel} x_{LFL} x_{UFL} x_{st} f_{unburnt}')
    x_O2_air = sp.Symbol('x_{O_2;air}')
    eqs['eq_L10_1'] = sp.Eq(n_O2, x_C + y_H/4 - z_O/2 + u_S)  # O2 demand of C_xH_yO_zS_u; organic_species (audit's "w" is the sulphur count u)
    eqs['eq_L10_2'] = sp.Eq(AFR_st, n_O2*M_O2/(M_fuel*Y_O2))  # stoichiometric air/fuel mass ratio; organic_species
    eqs['eq_L10_3'] = sp.Eq(nu_CO2, x_C)  # complete-combustion CO2 yield per mol fuel; combustion_products
    eqs['eq_L10_4'] = sp.Eq(nu_H2O, y_H/2)  # complete-combustion H2O yield; combustion_products
    eqs['eq_L10_5'] = sp.Eq(nu_SO2, u_S)  # complete-combustion SO2 yield; combustion_products
    eqs['eq_L10_6'] = sp.Eq(nu_N2, psi_N2*n_O2)  # N2 carried by the air, psi ~ 3.76; combustion_products
    eqs['eq_L10_7'] = sp.Eq(phi_eq, FA/FA_st)  # equivalence ratio; combustion_products, otto_langen, spectacle
    eqs['eq_L10_8'] = sp.Eq(x_st, 1/(1 + n_O2/x_O2_air))  # stoichiometric fuel volume fraction in air; otto_langen
    eqs['eq_L10_9'] = sp.Eq(phi_eq, (x_fuel/(1 - x_fuel))/(x_st/(1 - x_st)))  # phi from fuel volume fraction; otto_langen
    eqs['eq_L10_10'] = sp.And(x_LFL <= x_fuel, x_fuel <= x_UFL)  # flammability window LFL <= x_fuel <= UFL; otto_langen, spectacle
    eqs['eq_L10_11'] = sp.Eq(f_ub, sp.Max(0, 1 - 1/phi_eq))  # rich unburnt fuel fraction (oxygen-limited); combustion_products

    # L11 deposition and ageing
    r_age, T_oil, T_ref, dT_half, L_life, L_ref, E_a_eq = sp.symbols('r_{age} T_{oil} T_{ref} \\Delta{}T_{1/2} L_{life} L_{ref} E_{a;eq}')
    b_frac, k_dep, T_surf, T_th, k_cake, Q_flow = sp.symbols('b_{blocked} k_{dep} T_{surf} T_{th} k_{cake} Q')
    m_film, mdot_dep, k_scr, omega_c, k_burn = sp.symbols('m_{film} \\dot{m}_{dep} k_{scrape} omega_{crank} k_{burn}')
    m_dil, k_dil, T_cold_th, T_boil, k_boil = sp.symbols('m_{dil} k_{dil} T_{cold} T_{boil} k_{boil}')
    eqs['eq_L11_1'] = sp.Eq(r_age, 2**((T_oil - T_ref)/dT_half))  # Arrhenius rule of thumb: rate doubles (life halves) per ~10 K; hydraulics, automatic_transmission
    eqs['eq_L11_2'] = sp.Eq(L_life, L_ref/r_age)  # remaining life scaled by the ageing rate; hydraulics, automatic_transmission
    eqs['eq_L11_3'] = sp.Eq(E_a_eq, R_u*T_ref**2*sp.log(2)/dT_half)  # activation energy equivalent to the halving rule, linearized Arrhenius near T_ref; hydraulics
    eqs['eq_L11_4'] = sp.Eq(sp.Derivative(b_frac, t), k_dep*sp.Heaviside(T_surf - T_th))  # threshold deposition (scale, coking, varnish): grows with time above threshold; fouling
    eqs['eq_L11_5'] = sp.Eq(sp.Derivative(b_frac, t), k_cake*Q_flow)  # filter-cake / core-debris: grows with throughput; fouling
    eqs['eq_L11_6'] = sp.Eq(sp.Derivative(m_film, t), mdot_dep - (k_scr*omega_c + k_burn)*m_film)  # bore oil-film deposit/scrape/burn balance; crankcase_state
    eqs['eq_L11_7'] = sp.Eq(sp.Derivative(m_dil, t), k_dil*sp.Max(0, phi_eq - 1)*sp.Heaviside(T_cold_th - T_oil) - k_boil*sp.Heaviside(T_oil - T_boil)*m_dil)  # fuel dilution in: rich+cold; boil-off: first-order above ~360 K; crankcase_state

    # L12 emissions
    y_CO, y_HC, y_NOx, a_CO, b_CO, a_HC, b_HC, c_HC, phi_HC0 = sp.symbols('y_{CO} y_{HC} y_{NO_x} a_{CO} b_{CO} a_{HC} b_{HC} c_{HC} phi_{HC;0}')
    y_NOx_pk, phi_NOx, w_NOx, L_load = sp.symbols('y_{NO_x;pk} phi_{NO_x;pk} w_{NO_x} L_{load}')
    eta_lit, T_brick, T_LO, dT_LO, eta_cat, eta_pk, W_lam, w_lam = sp.symbols(
        'eta_{lit} T_{brick} T_{light-off} \\Delta{}T_{LO} eta_{cat} eta_{peak} W_{lambda} w_{lambda}')
    COHb, k_St, n_St, ppm_CO, RMV, t_exp, t_half = sp.symbols('COHb k_{Stewart} n_{Stewart} ppm_{CO} RMV t_{exp} t_{1/2;COHb}')
    eqs['eq_L12_1'] = sp.Eq(y_CO, a_CO + b_CO*sp.Max(0, phi_eq - 1))  # engine-out CO rises linearly rich (Heywood phi-trend), empirical; emissions
    eqs['eq_L12_2'] = sp.Eq(y_HC, a_HC + b_HC*sp.Max(0, phi_eq - 1) + c_HC*sp.Max(0, 1 - phi_eq - phi_HC0)**sp.Rational(3, 2))  # engine-out HC: rich rise + lean-misfire rise, empirical; emissions
    eqs['eq_L12_3'] = sp.Eq(y_NOx, y_NOx_pk*sp.exp(-((phi_eq - phi_NOx)/w_NOx)**2)*L_load)  # engine-out NOx peaks slightly lean (~0.9), empirical; emissions
    eqs['eq_L12_4'] = sp.Eq(eta_lit, 1/(1 + sp.exp(-(T_brick - T_LO)/dT_LO)))  # catalyst light-off sigmoid, empirical; emissions
    eqs['eq_L12_5'] = sp.Eq(W_lam, sp.exp(-((phi_eq - 1)/w_lam)**2))  # three-way lambda window (NOx reduction, lean side), empirical; emissions
    eqs['eq_L12_6'] = sp.Eq(eta_cat, eta_pk*eta_lit*W_lam)  # catalyst conversion = peak x light-off x window; emissions
    eqs['eq_L12_7'] = sp.Eq(COHb, k_St*ppm_CO**n_St*RMV*t_exp)  # Stewart CO uptake, %COHb (k=3.317e-5, n=1.036), empirical physiology; emissions
    eqs['eq_L12_8'] = sp.Eq(sp.Derivative(COHb, t), -(sp.log(2)/t_half)*COHb)  # COHb first-order elimination in clean air (t_half ~ 320 min); emissions
    return eqs


globals().update(_expand_lavoisier())


# =====================================================================
# 26. WILLIS (Machine Elements: Gearing, Joints, Tribology) -- 2026-09-22
# =====================================================================
def _expand_willis():
    eqs = {}
    # WI1 gear forces and ratios
    F_t, F_r, F_n, T_gear, r_pitch, alpha_p = sp.symbols('F_t F_r F_n T_{gear} r_{pitch} \\alpha_p')
    m_mod, N_teeth = sp.symbols('m_{mod} N_{teeth}')
    eqs['eq_WI1_1'] = sp.Eq(F_t, T_gear / r_pitch)   # tangential tooth load; ring_joints.py, slew_drive.py
    eqs['eq_WI1_2'] = sp.Eq(F_r, F_t * sp.tan(alpha_p))   # radial (separating) load from pressure angle; ring_joints.py, slew_drive.py
    eqs['eq_WI1_3'] = sp.Eq(F_n, F_t / sp.cos(alpha_p))   # normal load along the line of action; ring_joints.py (Hertz input)
    eqs['eq_WI1_4'] = sp.Eq(r_pitch, m_mod * N_teeth / 2)   # metric pitch radius from module; ring_joints.py, slew_drive.py
    w_sun, w_ring, w_carr, N_sun, N_ring = sp.symbols('\\omega_{sun} \\omega_{ring} \\omega_{carrier} N_{sun} N_{ring}')
    e_train = sp.Symbol('e_{train}')
    eqs['eq_WI1_5'] = sp.Eq(e_train, (w_sun - w_carr) / (w_ring - w_carr))   # Willis train value relative to the carrier; ring_joints.py
    eqs['eq_WI1_7'] = sp.Eq(e_train, -N_ring / N_sun)   # Willis planetary formula, simple sun/planet/ring train value; ring_joints.py
    i_diff, N_up, N_lo = sp.symbols('i_{diff} N_{upper} N_{lower}')
    eqs['eq_WI1_6'] = sp.Eq(i_diff, N_up / (N_up - N_lo))   # split-ring (differential) captive planetary reduction, carrier rev per upper-ring rev; ring_joints.py (infinite when N_up = N_lo)

    # WI2 gear strength and contact
    sigma_b, b_face, Y_lewis = sp.symbols('\\sigma_{b} b_{face} Y_{Lewis}')
    eqs['eq_WI2_1'] = sp.Eq(sigma_b, F_t / (b_face * m_mod * Y_lewis))   # Lewis bending stress (metric, module form); slew_drive.py tooth_capacity_nm
    p_H, W_line, L_line, E_star, R_eq = sp.symbols('p_{H} W L_{contact} E^{*}_{c} R\'_{eq}')
    eqs['eq_WI2_2'] = sp.Eq(p_H, sp.sqrt(W_line * E_star / (sp.pi * L_line * R_eq)))   # Hertz LINE contact peak pressure (Johnson); ring_joints.py
    E_1, E_2, nu_1, nu_2 = sp.symbols('E_1 E_2 \\nu_1 \\nu_2')
    eqs['eq_WI2_3'] = sp.Eq(E_star, 1 / ((1 - nu_1**2) / E_1 + (1 - nu_2**2) / E_2))   # Hertz contact modulus; ring_joints.py uses E/(2(1-nu^2)) for like materials
    r_c1, r_c2 = sp.symbols('\\rho_1 \\rho_2')
    eqs['eq_WI2_4'] = sp.Eq(R_eq, 1 / (1 / r_c1 + 1 / r_c2))   # relative curvature, external mesh / roller on flat (rho_2 -> oo); ring_joints.py
    eqs['eq_WI2_5'] = sp.Eq(R_eq, 1 / (1 / r_c1 - 1 / r_c2))   # relative curvature, INTERNAL mesh (concave mate); ring_joints.py contact_stress_pa
    eqs['eq_WI2_6'] = sp.Eq(r_c1, r_pitch * sp.sin(alpha_p))   # involute flank radius of curvature at the pitch point; ring_joints.py

    # WI3 threaded fasteners
    T_w, K_nut, d_maj, F_pre = sp.symbols('T_{wrench} K_{nut} d_{major} F_{preload}')
    eqs['eq_WI3_1'] = sp.Eq(T_w, K_nut * d_maj * F_pre)   # nut-factor torque-tension (Shigley); fasteners.py MachinedClamp.bolt_torque_nm
    A_t, p_thr = sp.symbols('A_t p_{thread}')
    eqs['eq_WI3_2'] = sp.Eq(A_t, sp.pi / 4 * (d_maj - sp.Rational(9382, 10000) * p_thr)**2)   # ISO metric tensile stress area; fasteners.py
    F_break, S_ut = sp.symbols('F_{break} S_{ut}')
    eqs['eq_WI3_3'] = sp.Eq(F_break, A_t * S_ut)   # bolt tensile failure load; fasteners.py bolt_break_n
    A_s, L_e, f_thr = sp.symbols('A_{strip} L_e f_{thread}')
    eqs['eq_WI3_4'] = sp.Eq(A_s, sp.pi * d_maj * L_e / 2 * f_thr)   # female thread stripping area, half-circumference simplification x percent thread; fasteners.py (Shigley uses ~0.75*pi*d*L_e for internal threads)
    tau_s, S_y = sp.symbols('\\tau_{s} S_y')
    eqs['eq_WI3_5'] = sp.Eq(tau_s, S_y / sp.sqrt(3))   # von Mises shear yield (0.577 S_y); fasteners.py, joints.py
    F_strip = sp.Symbol('F_{strip}')
    eqs['eq_WI3_6'] = sp.Eq(F_strip, A_s * tau_s)   # thread pull-out (stripping) force; fasteners.py pull_out_n
    eqs['eq_WI3_7'] = F_break <= F_strip   # design criterion: bolt breaks before the tapped part strips; fasteners.py bolt_breaks_first
    T_slip, mu_c, N_clamp, d_bore = sp.symbols('T_{slip} \\mu_{c} N_{clamp} d_{bore}')
    eqs['eq_WI3_8'] = sp.Eq(T_slip, mu_c * N_clamp * d_bore / 2)   # friction clamp slip torque; fasteners.py MachinedClamp.torque_capacity_nm
    d_drill, pct = sp.symbols('d_{drill} \\%_{thread}')
    eqs['eq_WI3_9'] = sp.Eq(d_drill, d_maj - pct / sp.Float('76.98') * p_thr)   # tap drill for percent thread (metric shop rule, empirical constant 76.98); fasteners.py

    # WI4 welds and pins
    t_throat, h_leg = sp.symbols('t_{throat} h_{leg}')
    eqs['eq_WI4_1'] = sp.Eq(t_throat, h_leg / sp.sqrt(2))   # equal-leg fillet weld throat; joints.py weld_throat_m
    tau_w, F_w, L_w = sp.symbols('\\tau_{weld} F_{weld} L_{weld}')
    eqs['eq_WI4_2'] = sp.Eq(tau_w, F_w / (t_throat * L_w))   # fillet weld throat shear stress (Shigley); joints.py
    M_w, d_tube, tau_wall = sp.symbols('M_{weld} d_{tube} \\tau_{allow}')
    eqs['eq_WI4_3'] = sp.Eq(M_w, sp.pi * d_tube**2 * t_throat / 4 * tau_wall)   # moment capacity of an annular fillet around a tube (thin-ring section modulus pi d^2 t/4); joints.py
    F_pin, d_pin = sp.symbols('F_{pin} d_{pin}')
    eqs['eq_WI4_4'] = sp.Eq(F_pin, 2 * sp.pi * d_pin**2 / 4 * tau_wall)   # pin in double shear; joints.py

    # WI5 lubrication
    h_film, k_R, mu_v, U_s, P_c = sp.symbols('h_{film} k_{R} \\mu_{visc} U_{slide} P_{contact}')
    eqs['eq_WI5_1'] = sp.Eq(h_film, k_R * sp.sqrt(mu_v * U_s / P_c))   # Reynolds-wedge entrained film grouping, empirical k_R; engine_harm.py entrained_film_m
    lam_f, sigma_r, R_q1, R_q2 = sp.symbols('\\Lambda_{film} \\sigma_{rough} R_{q1} R_{q2}')
    eqs['eq_WI5_2'] = sp.Eq(lam_f, h_film / sigma_r)   # specific film thickness (lambda ratio); engine_harm.py
    eqs['eq_WI5_3'] = sp.Eq(sigma_r, sp.sqrt(R_q1**2 + R_q2**2))   # composite RMS roughness (Hamrock); engine_harm.py SURFACE_ROUGHNESS_COMBINED_M
    eqs['eq_WI5_4'] = lam_f >= 3   # full-film (hydrodynamic) regime; engine_harm.py LAMBDA_FULL_FILM
    eqs['eq_WI5_5'] = lam_f <= 1   # boundary regime; engine_harm.py LAMBDA_BOUNDARY (1 < Lambda < 3 is mixed)
    m_film, rho_oil, A_wet = sp.symbols('m_{film} \\rho_{oil} A_{wet}')
    eqs['eq_WI5_6'] = sp.Eq(h_film, m_film / (rho_oil * A_wet))   # film thickness from film mass (supply ceiling); engine_harm.py lubrication
    P_fric, mu_f, F_N, v_sl = sp.symbols('P_{fric} \\mu_{f} F_N v_{slide}')
    eqs['eq_WI5_7'] = sp.Eq(P_fric, mu_f * F_N * v_sl)   # frictional heating at the contact; engine_harm.py
    tau_c, h_gap = sp.symbols('\\tau_{Couette} h_{gap}')
    eqs['eq_WI5_8'] = sp.Eq(tau_c, mu_v * U_s / h_gap)   # Couette film shear stress; pneumatic_gimbal.py friction_torque_nm, seals.py
    T_pet, w_j, R_j, L_j, c_r = sp.symbols('T_{Petroff} \\omega_j R_j L_j c_{r}')
    eqs['eq_WI5_9'] = sp.Eq(T_pet, 2 * sp.pi * mu_v * w_j * R_j**3 * L_j / c_r)   # Petroff journal (Couette) film torque; pneumatic_gimbal.py uses mu*omega*r^2*(2A)/gap for the aerostatic sphere

    # WI6 wear and filtration
    V_wear, K_arch, s_slide, H_hard = sp.symbols('V_{wear} K_{Archard} s_{slide} H_{hard}')
    eqs['eq_WI6_1'] = sp.Eq(V_wear, K_arch * F_N * s_slide / H_hard)   # Archard wear law (K dimensionless, empirical); wear.py, wear_debris.py
    beta_x, N_upst, N_dnst = sp.symbols('\\beta_x N_{upstream} N_{downstream}')
    eqs['eq_WI6_2'] = sp.Eq(beta_x, N_upst / N_dnst)   # filter beta ratio at particle size x (ISO 16889); wear_debris.py
    eta_cap = sp.Symbol('\\eta_{capture}')
    eqs['eq_WI6_3'] = sp.Eq(eta_cap, 1 - 1 / beta_x)   # single-pass capture efficiency from beta; wear_debris.py capture_efficiency
    D_miner, n_cyc, N_life = sp.symbols('D_{Miner} n_i N_i')
    eqs['eq_WI6_4'] = sp.Eq(D_miner, sp.Sum(n_cyc / N_life, ('i', 1, sp.Symbol('k'))))   # Palmgren-Miner damage accumulation for per-event wear; wear.py

    # WI7 pneumatic actuators
    F_mk, D_0, P_g, eps_c, theta_0 = sp.symbols('F_{McKibben} D_0 P_{gauge} \\varepsilon_c \\theta_0')
    eqs['eq_WI7_1'] = sp.Eq(F_mk, sp.pi * D_0**2 * P_g / 4 * (3 * (1 - eps_c)**2 / sp.tan(theta_0)**2 - 1 / sp.sin(theta_0)**2))   # McKibben muscle force (Gaylord / Chou-Hannaford); pneumatic_gimbal.py
    theta_lock = sp.Symbol('\theta_{lock}')
    eqs['eq_WI7_2'] = sp.Eq(theta_lock, sp.atan(sp.sqrt(2)))   # braid locking (zero-force) angle, 54.7 deg, where eq_WI7_1 vanishes at eps = 0; pneumatic_gimbal.py
    return eqs
globals().update(_expand_willis())


# =====================================================================
# 27. HODGKIN (Membrane Transport and Biophysics) -- 2026-09-22
# =====================================================================
def _expand_hodgkin():
    eqs = {}
    # HO1 electrodiffusion
    J_ghk, P_perm, z_ion, F_c, V_m, R_g, T_k, C_in, C_out = sp.symbols(
        'J_{GHK} P_{perm} z_{ion} F_{Far} V_m R_{gas} T_K C_{in} C_{out}')
    u_ghk = z_ion * F_c * V_m / (R_g * T_k)
    eqs['eq_HO1_1'] = sp.Eq(J_ghk, P_perm * u_ghk * (C_in - C_out * sp.exp(-u_ghk)) / (1 - sp.exp(-u_ghk)))   # Goldman-Hodgkin-Katz molar flux (outward positive); cellsim/transport/ghk.py
    I_ghk = sp.Symbol('I_{GHK}')
    eqs['eq_HO1_2'] = sp.Eq(I_ghk, z_ion * F_c * J_ghk)   # GHK current density from the molar flux (Hille); ghk.py
    V_rev, P_K, P_Na, P_Cl = sp.symbols('V_{rev} P_K P_{Na} P_{Cl}')
    K_o, K_i, Na_o, Na_i, Cl_o, Cl_i = sp.symbols('[K]_o [K]_i [Na]_o [Na]_i [Cl]_o [Cl]_i')
    eqs['eq_HO1_3'] = sp.Eq(V_rev, R_g * T_k / F_c * sp.ln((P_K * K_o + P_Na * Na_o + P_Cl * Cl_i) / (P_K * K_i + P_Na * Na_i + P_Cl * Cl_o)))   # GHK voltage equation (zero-net-current potential); ghk.py (standard form, not coded there)
    E_ion = sp.Symbol('E_{ion}')
    eqs['eq_HO1_4'] = sp.Eq(E_ion, R_g * T_k / (z_ion * F_c) * sp.ln(C_out / C_in))   # Nernst potential, single-ion zero-flux limit of eq_HO1_1; ghk.py

    # HO2 Kedem-Katchalsky and osmosis
    J_v, L_p, A_m, dP, sigma_r, dC = sp.symbols('J_v L_p A_m \\Delta{P} \\sigma_{refl} \\Delta{C}')
    Pi_osm = sp.Symbol('\\Pi_{osm}')
    eqs['eq_HO2_1'] = sp.Eq(Pi_osm, sigma_r * R_g * T_k * dC)   # van 't Hoff osmotic pressure with Staverman reflection coefficient; kedem_katchalsky.py (summed over species)
    eqs['eq_HO2_2'] = sp.Eq(J_v, L_p * A_m * (dP - Pi_osm))   # Kedem-Katchalsky volume flux; kedem_katchalsky.py fluxes
    J_s, P_s, C_bar = sp.symbols('J_s P_{s} \\bar{C}')
    eqs['eq_HO2_3'] = sp.Eq(J_s, P_s * A_m * dC + (1 - sigma_r) * C_bar * J_v)   # Kedem-Katchalsky solute flux; kedem_katchalsky.py uses the upstream C_L for C-bar (textbook: log-mean)
    P_arr, P_0a, E_a = sp.symbols('P_{T} P_{0} E_a')
    eqs['eq_HO2_4'] = sp.Eq(P_arr, P_0a * sp.exp(-E_a / (R_g * T_k)))   # Arrhenius permeability; kedem_katchalsky.py arrhenius

    # HO3 pumps and carrier saturation
    J_pump, J_max, Km_Na, Km_K, Na_ci, K_ce = sp.symbols('J_{pump} J_{max} K_{m;Na} K_{m;K} [Na]_i^{p} [K]_o^{p}')
    eqs['eq_HO3_1'] = sp.Eq(J_pump, J_max * (Na_ci / (Km_Na + Na_ci))**3 * (K_ce / (Km_K + K_ce))**2)   # Na/K-ATPase 3Na:2K saturating kinetics (Michaelis-Menten per site); cellsim/transport/pumps.py
    v_h, V_max, S_sub, K_half, n_hill = sp.symbols('v_{Hill} V_{max} S K_{1/2} n_H')
    eqs['eq_HO3_2'] = sp.Eq(v_h, V_max * S_sub**n_hill / (K_half**n_hill + S_sub**n_hill))   # Hill saturation (Keener-Sneyd); pumps.py generalization
    v_mm, K_m = sp.symbols('v_{MM} K_m')
    eqs['eq_HO3_3'] = sp.Eq(v_mm, V_max * S_sub / (K_m + S_sub))   # Michaelis-Menten (Hill n=1); pumps.py

    # HO4 membrane mechanics
    f_hel, kappa_b, H_mean, C_0, kappa_G, K_gauss = sp.symbols('f_{Helfrich} \\kappa_b H_{mean} C_0 \\kappa_G K_{Gauss}')
    eqs['eq_HO4_1'] = sp.Eq(f_hel, kappa_b / 2 * (2 * H_mean - C_0)**2 + kappa_G * K_gauss)   # Helfrich bending energy density; cellsim/membranes/membrane.py (kappa_G term omitted there: topological constant)
    E_hel, dA_m = sp.symbols('E_{Helfrich} dA_m')
    eqs['eq_HO4_2'] = sp.Eq(E_hel, sp.Integral(f_hel, dA_m))   # total Helfrich energy over the membrane surface; membrane.py (cotan-Laplacian discretization)
    dp_m, T_mem, R_1m, R_2m = sp.symbols('\\Delta{p}_m T_{mem} R_{1;m} R_{2;m}')
    eqs['eq_HO4_3'] = sp.Eq(dp_m, T_mem * (1 / R_1m + 1 / R_2m))   # Young-Laplace membrane tension, consolidates eq_NS5_6 (gamma_s -> T_mem); membrane.py
    R_cell = sp.Symbol('R_{cell}')
    eqs['eq_HO4_4'] = sp.Eq(dp_m, 2 * T_mem / R_cell)   # Laplace law for a spherical cell (R1 = R2); membrane.py

    # HO5 membrane electrical circuit
    C_mem, I_ion, I_app = sp.symbols('C_{mem} I_{ion;k} I_{app}')
    V_mt = sp.Function('V_m')(t)
    eqs['eq_HO5_1'] = sp.Eq(sp.Derivative(V_mt, t), (-sp.Sum(I_ion, ('k', 1, sp.Symbol('N_{ch}'))) + I_app) / C_mem)   # membrane capacitance current balance (Hodgkin-Huxley), consolidates eq_F6_4 (i_C = C dv_C/dt); ghk.py currents
    return eqs
globals().update(_expand_hodgkin())


# =====================================================================
# 28. JANSSEN (Granular and Soil Mechanics) -- 2026-09-22
# =====================================================================
def _expand_janssen():
    eqs = {}
    # JA1 silo stress
    sig_v, rho_b, g_acc, D_silo, mu_w, K_J, z_d = sp.symbols('\\sigma_v \\rho_b g D_{silo} \\mu_w K_J z_{depth}')
    eqs['eq_JA1_1'] = sp.Eq(sig_v, rho_b * g_acc * D_silo / (4 * mu_w * K_J) * (1 - sp.exp(-4 * mu_w * K_J * z_d / D_silo)))   # Janssen (1895) vertical stress, hydraulic radius D/4; materials_handling.py uses R (radius) in place of D/4, i.e. twice the textbook hydraulic radius
    sig_h = sp.Symbol('\\sigma_h')
    eqs['eq_JA1_2'] = sp.Eq(sig_h, K_J * sig_v)   # Janssen lateral stress ratio; materials_handling.py JANSSEN_K
    phi_i, delta_w = sp.symbols('\\phi_i \\delta_w')
    eqs['eq_JA1_3'] = sp.Eq(K_J, 1 - sp.sin(phi_i))   # Jaky at-rest coefficient, one standard choice for K (Nedergaard); materials_handling.py uses a fixed constant
    eqs['eq_JA1_4'] = sp.Eq(mu_w, sp.tan(delta_w))   # wall friction coefficient from wall friction angle; materials_handling.py
    # JA2 arching and discharge
    B_crit, H_theta, sig_c = sp.symbols('B_{crit} H(\\theta) \\sigma_c')
    eqs['eq_JA2_1'] = sp.Eq(B_crit, H_theta * sig_c / (rho_b * g_acc))   # Jenike critical arching outlet; materials_handling.py takes H(theta) = 2
    B_out, d_lump = sp.symbols('B_{out} d_{lump}')
    eqs['eq_JA2_2'] = B_out >= 6 * d_lump   # mechanical interlocking limit (empirical ~6 particle diameters); materials_handling.py interlock_limit_m
    W_dis, C_bev, k_bev = sp.symbols('W_{dis} C_{Bev} k_{Bev}')
    eqs['eq_JA2_3'] = sp.Eq(W_dis, C_bev * rho_b * sp.sqrt(g_acc) * (B_out - k_bev * d_lump)**sp.Rational(5, 2))   # Beverloo discharge (head-independent), empirical C, k; materials_handling.py
    # JA3 soil strength and bearing
    tau_f, c_coh, sig_n = sp.symbols('\\tau_f c_{coh} \\sigma_n')
    eqs['eq_JA3_1'] = sp.Eq(tau_f, c_coh + sig_n * sp.tan(phi_i))   # Mohr-Coulomb shear strength; earthworks.py, outriggers.py
    q_ult, N_c, N_q, N_g, q_sur, gam_s, B_f = sp.symbols('q_{ult} N_c N_q N_\\gamma q_{sur} \\gamma_{soil} B_f')
    eqs['eq_JA3_2'] = sp.Eq(q_ult, c_coh * N_c + q_sur * N_q + sp.Rational(1, 2) * gam_s * B_f * N_g)   # Terzaghi strip-footing bearing capacity; outriggers.py (code uses a declared ground_bearing_pa instead)
    eqs['eq_JA3_3'] = sp.Eq(N_q, sp.exp(sp.pi * sp.tan(phi_i)) * sp.tan(sp.pi / 4 + phi_i / 2)**2)   # Prandtl-Reissner N_q; outriggers.py (standard form)
    eqs['eq_JA3_4'] = sp.Eq(N_c, (N_q - 1) * sp.cot(phi_i))   # Prandtl N_c; outriggers.py (standard form)
    p_pad, F_leg, A_pad, q_allow = sp.symbols('p_{pad} F_{leg} A_{pad} q_{allow}')
    eqs['eq_JA3_5'] = sp.Eq(p_pad, F_leg / A_pad)   # outrigger pad bearing pressure; outriggers.py
    eqs['eq_JA3_6'] = p_pad > q_allow   # pad sinks (bearing failure) criterion; outriggers.py
    # JA4 earthworks
    V_loose, s_sw, V_insitu = sp.symbols('V_{loose} s_{swell} V_{insitu}')
    eqs['eq_JA4_1'] = sp.Eq(V_loose, s_sw * V_insitu)   # bulking (swell) factor; earthworks.py
    V_cone, r_cone, h_cone, phi_rep = sp.symbols('V_{cone} r_{cone} h_{cone} \\phi_{repose}')
    eqs['eq_JA4_2'] = sp.Eq(V_cone, sp.pi * r_cone**2 * h_cone / 3)   # heap as a cone; earthworks.py heap_height_m
    eqs['eq_JA4_3'] = sp.Eq(h_cone, r_cone * sp.tan(phi_rep))   # angle of repose sets the cone slope; earthworks.py
    # JA5 comminution
    E_R, K_Rit, x_p, x_f = sp.symbols('E_{Rittinger} K_{Rittinger} x_{product} x_{feed}')
    eqs['eq_JA5_1'] = sp.Eq(E_R, K_Rit * (1 / x_p - 1 / x_f))   # Rittinger comminution energy (new surface), empirical K; materials_handling.py Shredder.energy_per_kg_j (6/x surface per volume)
    F_sh, tau_m, b_bite, t_mat = sp.symbols('F_{shear} \\tau_{mat} b_{bite} t_{mat}')
    eqs['eq_JA5_2'] = sp.Eq(F_sh, tau_m * b_bite * t_mat)   # shear force of one cutter bite; materials_handling.py shear_force_n
    return eqs
globals().update(_expand_janssen())


# =====================================================================
# 29. PONCELET (Terminal Ballistics) -- 2026-09-22
# =====================================================================
def _expand_poncelet():
    eqs = {}
    # PO1 penetration and perforation
    x_p = sp.Function('x_p')(t)
    m_p, a_P, b_P = sp.symbols('m_p a_{P} b_{P}')
    eqs['eq_PO1_1'] = sp.Eq(sp.Derivative(x_p, t, 2), -(a_P + b_P * sp.Derivative(x_p, t)**2) / m_p)   # Poncelet resistance law (static strength + inertial term); ballistics.py (energy form used instead)
    X_pen, v_0 = sp.symbols('X_{pen} v_0')
    eqs['eq_PO1_2'] = sp.Eq(X_pen, m_p / (2 * b_P) * sp.ln(1 + b_P * v_0**2 / a_P))   # Poncelet penetration depth (Zukas); ballistics.py
    W_perf, U_tough, A_pres, t_plate, theta_i, k_def = sp.symbols('W_{perf} U_{tough} A_{pres} t_{plate} \\theta_i k_{def}')
    eqs['eq_PO1_3'] = sp.Eq(W_perf, U_tough * A_pres * t_plate / sp.cos(theta_i) * k_def)   # energy perforation: toughness x presented area x line-of-sight thickness (k_def mushroom factor, empirical); ballistics.py
    E_k = sp.Symbol('E_{k}')
    eqs['eq_PO1_4'] = E_k >= W_perf   # perforation criterion; ballistics.py
    X_part = sp.Symbol('X_{partial}')
    eqs['eq_PO1_5'] = sp.Eq(X_part, E_k / (U_tough * A_pres * k_def))   # partial penetration depth when E_k < W_perf; ballistics.py
    A_0, s_len, psi_y = sp.symbols('A_0 s_{L/D} \\psi_{yaw}')
    eqs['eq_PO1_6'] = sp.Eq(A_pres, A_0 * (sp.Abs(sp.cos(psi_y)) + s_len * sp.Abs(sp.sin(psi_y))))   # yawed presented area (code clamps factor to [1, 4]); ballistics.py presented_area_m2
    # PO2 ricochet
    E_n, theta_c = sp.symbols('E_{n} \\theta_{c}')
    eqs['eq_PO2_1'] = sp.Eq(E_n, E_k * sp.cos(theta_i)**2)   # normal component of impact energy; ballistics.py
    eqs['eq_PO2_2'] = theta_i >= theta_c   # ricochet critical obliquity (material-declared, empirical); ballistics.py (code also requires E_n < 1.5 W_perf)
    theta_c0, H_p, H_t, k_h = sp.symbols('\\theta_{c0} H_{proj} H_{target} k_{hard}')
    eqs['eq_PO2_3'] = sp.Eq(theta_c, theta_c0 + k_h * sp.log(H_p / H_t, 2))   # hardness-ratio shift of the ricochet angle, empirical (code k_h = 8 deg, minus a yaw term); ballistics.py
    # PO3 mushrooming
    D_f, D_p0, k_m, Pi_def, Pi_0, v_n, v_ref = sp.symbols('D_f D_{p0} k_{mush} \\Pi_{def} \\Pi_0 v_n v_{ref}')
    eqs['eq_PO3_1'] = sp.Eq(Pi_def, (H_t / H_p) * (v_n / v_ref)**2)   # deformation drive, hardness ratio x normalized normal-speed squared (a Johnson-damage-number surrogate), empirical; ballistics.py
    eqs['eq_PO3_2'] = sp.Eq(D_f, D_p0 * (1 + k_m * sp.Max(0, Pi_def - Pi_0)))   # mushroomed diameter ratio, empirical (code k_m = 0.28, Pi_0 = 0.15, capped at 1.8); ballistics.py
    # PO4 fluid-layer drag
    F_dl, rho_l, C_dl, v_pr = sp.symbols('F_{drag;l} \\rho_{l} C_{D;l} v_{proj}')
    eqs['eq_PO4_1'] = sp.Eq(F_dl, sp.Rational(1, 2) * rho_l * C_dl * A_pres * v_pr**2)   # quadratic drag in a contained fluid; ballistics.py
    W_fl, L_fl, p_fl, p_atm = sp.symbols('W_{fluid} L_{fluid} p_{fluid} p_{atm}')
    eqs['eq_PO4_2'] = sp.Eq(W_fl, F_dl * L_fl + (p_fl - p_atm) * A_pres * L_fl)   # energy spent crossing a pressurized fluid layer; ballistics.py
    v_pt = sp.Function('v_p')(t)
    eqs['eq_PO4_3'] = sp.Eq(sp.Derivative(v_pt, t), -sp.Rational(1, 2) * rho_l * C_dl * A_pres * v_pt**2 / m_p)   # drag deceleration (code integrates exactly: v/(1 + k v dt)); ballistics.py advance_projectile
    return eqs
globals().update(_expand_poncelet())


# =====================================================================
# 30. EMMONS (Fire Engineering) -- 2026-09-22
# =====================================================================
def _expand_emmons():
    eqs = {}
    # EM1 burning rate
    m_b, m_pp, A_pool = sp.symbols('\\dot{m}_{b} \\dot{m}\'\'_{b} A_{pool}')
    eqs['eq_EM1_1'] = sp.Eq(m_b, m_pp * A_pool)   # pool burning rate from fuel mass flux; fire.py burn_rate_kg_s
    m_inf, kbeta, D_pool = sp.symbols('\\dot{m}\'\'_{\\infty} k\\beta D_{pool}')
    eqs['eq_EM1_2'] = sp.Eq(m_pp, m_inf * (1 - sp.exp(-kbeta * D_pool)))   # Babrauskas large-pool burning flux (empirical k*beta); fire.py uses the m_inf plateau per fuel
    h_c, c_p, B_sp = sp.symbols('h_c c_{p} B_{Spalding}')
    eqs['eq_EM1_3'] = sp.Eq(m_pp, h_c / c_p * sp.ln(1 + B_sp))   # Emmons/Spalding mass-transfer burning rate; fire.py (standard form, not coded)
    Y_O, r_st, dH_c, T_inf, T_s, L_v = sp.symbols('Y_{O;\\infty} r_{st} \\Delta{H}_c T_{\\infty} T_{s} L_v')
    eqs['eq_EM1_4'] = sp.Eq(B_sp, (Y_O * dH_c / r_st + c_p * (T_inf - T_s)) / L_v)   # Spalding transfer number (Drysdale); fire.py
    # EM2 heat release and air demand
    Q_hrr, chi_c = sp.symbols('\\dot{Q}_{HRR} \\chi_{comb}')
    eqs['eq_EM2_1'] = sp.Eq(Q_hrr, m_b * dH_c * chi_c)   # heat release rate Q = m_dot * LHV * combustion efficiency; fire.py heat_release_w
    m_air, s_air = sp.symbols('\\dot{m}_{air} s_{air}')
    eqs['eq_EM2_2'] = sp.Eq(m_air, s_air * m_b)   # stoichiometric air demand (fire.py ~3.1 * 4.76 kg/kg); fire.py, air_volumes.py
    # EM3 radiation and ignition
    q_rad, chi_r, r_obs = sp.symbols('\\dot{q}\'\'_{rad} \\chi_{rad} r_{obs}')
    eqs['eq_EM3_1'] = sp.Eq(q_rad, chi_r * Q_hrr / (4 * sp.pi * r_obs**2))   # point-source radiation model (Drysdale/Modak); fire.py radiant_flux_at
    q_ig = sp.Symbol('\\dot{q}\'\'_{ig}')
    eqs['eq_EM3_2'] = q_rad >= q_ig   # piloted ignition criterion (~10 kW/m^2 liquid fuels, empirical); fire.py spread_targets
    # EM4 suppression
    m_wc, dh_w, eta_w = sp.symbols('\\dot{m}_{w;crit} \\Delta{h}_{w} \\eta_{w}')
    eqs['eq_EM4_1'] = sp.Eq(m_wc, Q_hrr / (dh_w * eta_w))   # critical water flow rate (heat absorbed = heat released); fire.py critical_water_kg_s
    m_w, tau_sup = sp.symbols('\\dot{m}_{w} \\tau_{sup}')
    chi_s = sp.Function('\\chi_{sup}')(t)
    eqs['eq_EM4_2'] = sp.Eq(sp.Derivative(chi_s, t), (sp.Min(1, m_w / m_wc) - chi_s) / tau_sup)   # first-order knock-down lag toward applied/critical (code tau = 3 s), empirical; fire.py apply_water
    m_bs = sp.Symbol('\\dot{m}_{b;sup}')
    eqs['eq_EM4_3'] = sp.Eq(m_bs, m_pp * A_pool * (1 - chi_s))   # suppressed burning rate; fire.py burn_rate_kg_s
    return eqs
globals().update(_expand_emmons())


# =====================================================================
# 31. MAXWELL (Regulation and Feedback Control; "On Governors", 1868) -- 2026-09-22
# =====================================================================
def _expand_maxwell():
    eqs = {}
    # MX1 PID / PI
    e_t = sp.Function('e')(t)
    u_t = sp.Function('u')(t)
    r_set = sp.Function('r_{set}')(t)
    y_meas = sp.Function('y')(t)
    K_p, K_i, K_d, u_0 = sp.symbols('K_p K_i K_d u_0')
    tau = sp.Symbol('\\tau')
    eqs['eq_MX1_1'] = sp.Eq(e_t, r_set - y_meas)   # control error; regulator.py, ecu.py
    eqs['eq_MX1_2'] = sp.Eq(u_t, u_0 + K_p * e_t + K_i * sp.Integral(sp.Function('e')(tau), (tau, 0, t)))   # PI law with feedforward base; regulator.py ClosedLoopRegulator, ecu.py, aftermarket_idle_controller.py
    eqs['eq_MX1_3'] = sp.Eq(u_t, K_p * e_t + K_i * sp.Integral(sp.Function('e')(tau), (tau, 0, t)) + K_d * sp.Derivative(e_t, t))   # PID law (Astrom-Hagglund parallel form); dyno_controller.py, hcu.py
    I_int, I_max = sp.symbols('I_{int} I_{max}')
    eqs['eq_MX1_4'] = sp.Abs(I_int) <= I_max   # integrator clamp (anti-windup); regulator.py integral_clamp, ecu.py
    u_min, u_max, u_raw = sp.symbols('u_{min} u_{max} u_{raw}')
    u_sat = sp.Symbol('u_{sat}')
    eqs['eq_MX1_5'] = sp.Eq(u_sat, sp.Min(u_max, sp.Max(u_min, u_raw)))   # actuator saturation; regulator.py _clamp
    # MX2 linearized plant and pole placement
    J_c, K_t, b_s, s_l = sp.symbols('J_{c} K_{t} b_{slope} s')
    dw = sp.Function('\\delta\\omega')(t)
    du = sp.Function('\\delta u')(t)
    eqs['eq_MX2_1'] = sp.Eq(sp.Derivative(dw, t), (K_t * du + b_s * dw) / J_c)   # linearized idle plant, b = dT/domega > 0 is destabilizing; ecu.py IdlePlant
    G_p = sp.Symbol('G_{plant}(s)')
    eqs['eq_MX2_2'] = sp.Eq(G_p, K_t / (J_c * s_l - b_s))   # plant transfer function; ecu.py
    p_u = sp.Symbol('p_{unstable}')
    eqs['eq_MX2_3'] = sp.Eq(p_u, b_s / J_c)   # open-loop unstable pole; ecu.py
    w_n, zeta = sp.symbols('\\omega_n \\zeta')
    Delta_cl = sp.Symbol('\Delta_{cl}(s)')
    eqs['eq_MX2_4'] = sp.Eq(Delta_cl, J_c * s_l**2 + (K_p * K_t - b_s) * s_l + K_i * K_t)   # PI closed-loop characteristic polynomial on the linearized plant; ecu.py
    eqs['eq_MX2_8'] = sp.Eq(Delta_cl, J_c * (s_l**2 + 2 * zeta * w_n * s_l + w_n**2))   # second-order target it is matched to; ecu.py
    eqs['eq_MX2_5'] = sp.Eq(K_p, (2 * zeta * w_n * J_c + b_s) / K_t)   # pole-placement proportional gain; ecu.py
    eqs['eq_MX2_6'] = sp.Eq(K_i, w_n**2 * J_c / K_t)   # pole-placement integral gain; ecu.py
    M_pole = sp.Symbol('M_{pole}')
    eqs['eq_MX2_7'] = w_n >= M_pole * p_u   # bandwidth must dominate the unstable pole (code M = 3); ecu.py idle_loop_natural_freq_rad_s
    # MX3 relays and limits
    S_prev, x_sig, x_on, x_off = sp.symbols('S_{prev} x_{sig} x_{on} x_{off}')
    S_rel = sp.Symbol('S_{relay}')
    eqs['eq_MX3_1'] = sp.Eq(S_rel, sp.Piecewise((1, x_sig >= x_on), (0, x_sig <= x_off), (S_prev, True)))   # hysteresis (latching) relay, x_off < x_on; governor.py trip lever, ignition_driver.py stages
    c_cut, n_rpm, n_red, f_st, f_bd, c_max = sp.symbols('c_{cut} n_{rpm} n_{red} f_{start} f_{band} c_{max}')
    eqs['eq_MX3_2'] = sp.Eq(c_cut, c_max * sp.Min(1, sp.Max(0, (n_rpm / n_red - f_st) / f_bd)))   # soft-taper rev limiter cut severity; ignition_driver.py cut_severity
    u_prog, u_rat, s_sig, s_on, s_full = sp.symbols('u_{prog} u_{rated} s_{sig} s_{onset} s_{full}')
    eqs['eq_MX3_3'] = sp.Eq(u_prog, u_rat * sp.Min(1, sp.Max(0, (s_sig - s_on) / (s_full - s_on))))   # open-loop progressive regulator ramp; regulator.py ProgressiveRegulator
    # MX4 centrifugal governor
    m_ball, w_g, i_g, w_e = sp.symbols('m_{ball} \\omega_{gov} i_{gov} \\omega_{eng}')
    r_b = sp.Function('r_{ball}')(t)
    F_0, k_spr, r_min, F_fr = sp.symbols('F_{0;spring} k_{spring} r_{min} F_{fric}')
    eqs['eq_MX4_1'] = sp.Eq(w_g, i_g * w_e)   # governor drive ratio; governor.py
    eqs['eq_MX4_2'] = sp.Eq(sp.Derivative(r_b, t, 2), (m_ball * w_g**2 * r_b - (F_0 + k_spr * (r_b - r_min)) - F_fr * sp.sign(sp.Derivative(r_b, t))) / m_ball)   # captive-ball governor radial dynamics (centrifugal - spring - Coulomb friction); governor.py step
    w_eq, r_eq = sp.symbols('\\omega_{eq} r_{eq}')
    eqs['eq_MX4_3'] = sp.Eq(w_eq, sp.sqrt((F_0 + k_spr * (r_eq - r_min)) / (m_ball * r_eq)))   # equilibrium (set) speed of the balls at radius r_eq; governor.py
    a_3, a_2, a_1, a_0 = sp.symbols('a_3 a_2 a_1 a_0')
    eqs['eq_MX4_4'] = sp.And(a_3 > 0, a_2 > 0, a_1 > 0, a_0 > 0, a_2 * a_1 > a_3 * a_0)   # Maxwell-Vyshnegradskii (3rd-order Routh-Hurwitz) stability of a governed engine a3 s^3 + a2 s^2 + a1 s + a0; governor.py (standard form)
    return eqs
globals().update(_expand_maxwell())


# =====================================================================
# 3c. FARADAY -- electrostatics, static charging, magnetism, eddy currents, fine details (2026-09-22)
# =====================================================================
# SI units throughout (Gaussian forms are NOT used). Vectors are plain
# Symbols; cross(a, b) is the cross product; dot(a, b) is a local
# placeholder Function for the scalar product; nabla(...) is grad/div/curl
# as commented; nabla_op**2 is the Laplacian.


def _expand_faraday_f14_electrostatics():
    eqs = {}
    dot = sp.Function('dot')
    eps0, eps, eps_r = sp.symbols('epsilon_0 epsilon epsilon_r', positive=True)
    q, q1, q2, Q, Q_enc = sp.symbols('q q_1 q_2 Q Q_enc')
    r, R, d, a, b, h, D, A, l, w, s_x = sp.symbols('r R d a b h D A l w x_slab', positive=True)
    F_C, E_r, V_r, r_hat = sp.symbols('F_C E_r V_r r_hat')
    E_vec, n_hat, dA, dV = sp.symbols('E_vec n_hat dA dV')
    rho_c, r_sep = sp.symbols('rho_c r_sep', positive=True)
    C, C_len, W_E, u_E, E_mag, P_es, sig, E_n = sp.symbols('C C_prime W_E u_E E P_es sigma E_n')
    V, F_x, chi_e, x_s = sp.symbols('V F_x chi_e x_s')
    alpha_sp = sp.Symbol('alpha_sp', positive=True)
    n = sp.Symbol('n', integer=True, positive=True)
    q_img, z_img, sig_ind, rho_p, F_img, b_img = sp.symbols('q_img z_img sigma_ind rho_perp F_img b_img')
    E_surf, sig_1, sig_2, r_1, r_2 = sp.symbols('E_surf sigma_1 sigma_2 r_1 r_2')
    Q_inner, Q_outer, E_in = sp.symbols('Q_inner Q_outer E_in')
    p, theta, V_dip, E_th, tau_p, U_p, F_p, p_vec = sp.symbols('p theta V_dip E_theta tau_p U_p F_p p_vec')
    alpha_pol, F_ind = sp.symbols('alpha_pol F_ind')
    m_b, g, L_s, theta_b = sp.symbols('m_b g L_string theta_b', positive=True)
    V_pot = sp.Symbol('V_pot')

    eqs['eq_F14_1'] = sp.Eq(F_C, q1*q2/(4*sp.pi*eps0*r**2)*r_hat)  # Coulomb's law, force on q2 from q1 along r_hat (Griffiths 2.1)
    eqs['eq_F14_2'] = sp.Eq(E_r, q/(4*sp.pi*eps0*r**2))  # point-charge field, radial component (Griffiths 2.7)
    eqs['eq_F14_3'] = sp.Eq(V_r, q/(4*sp.pi*eps0*r))  # point-charge potential, reference at infinity (Griffiths 2.26)
    eqs['eq_F14_4'] = sp.Eq(V_pot, sp.Integral(rho_c/(4*sp.pi*eps0*r_sep), dV))  # potential of a charge distribution, r_sep = |r - r'| (Griffiths 2.29)
    eqs['eq_F14_5'] = sp.Eq(sp.Integral(dot(E_vec, n_hat), dA), Q_enc/eps0)  # Gauss's law, integral form over a closed surface (integral of F1_3; Griffiths 2.13)
    eqs['eq_F14_6'] = sp.Eq(nabla_op**2*V_pot, 0)  # Laplace's equation in a charge-free region (F4_1 with rho=0); no interior extrema -> Earnshaw (Griffiths 3.1)
    eqs['eq_F14_7'] = sp.Eq(C, Q/V)  # capacitance definition (Griffiths 2.53)
    eqs['eq_F14_8'] = sp.Eq(C, eps*A/d)  # parallel-plate capacitor, fringing neglected (Griffiths 2.54)
    eqs['eq_F14_9'] = sp.Eq(C, 4*sp.pi*eps0*R)  # isolated conducting sphere (Purcell-Morin 3.4)
    eqs['eq_F14_10'] = sp.Eq(C, 4*sp.pi*eps0*a*b/(b - a))  # concentric spherical capacitor a < b (Griffiths ex. 2.11)
    eqs['eq_F14_11'] = sp.Eq(C_len, 2*sp.pi*eps/sp.log(b/a))  # coaxial capacitance per unit length (Griffiths prob. 2.43)
    eqs['eq_F14_12'] = sp.Eq(C_len, sp.pi*eps0/sp.acosh(D/(2*a)))  # two parallel wires radius a, centre spacing D, per unit length (Jackson/Smythe)
    eqs['eq_F14_13'] = sp.Eq(C, 4*sp.pi*eps0*R*sp.sinh(alpha_sp)*sp.Sum(1/sp.sinh(n*alpha_sp), (n, 1, sp.oo)))  # sphere radius R, centre height h over a grounded plane, cosh(alpha_sp) = h/R; -> 4 pi eps0 R (1 + R/(2h)) far away (Smythe 5.08)
    eqs['eq_F14_14'] = sp.Eq(sp.cosh(alpha_sp), h/R)  # bispherical parameter for F14_13 (Smythe 5.08)
    eqs['eq_F14_15'] = sp.Eq(C, 2*sp.pi*eps0*R*sp.sinh(alpha_sp)*sp.Sum(1/sp.sinh(n*alpha_sp), (n, 1, sp.oo)))  # two equal spheres at +Q/-Q, centre spacing D, cosh(alpha_sp) = D/(2R) (Smythe 5.08; half the sphere-plane value at h = D/2)
    eqs['eq_F14_16'] = sp.Eq(u_E, eps0*E_mag**2/2)  # electrostatic energy density in vacuum (Griffiths 2.45; electric part of F3_1)
    eqs['eq_F14_17'] = sp.Eq(W_E, eps0/2*sp.Integral(E_mag**2, dV))  # total electrostatic energy over all space (Griffiths 2.45)
    eqs['eq_F14_18'] = sp.Eq(W_E, sp.Integral(rho_c*V_pot, dV)/2)  # energy from charge and potential (Griffiths 2.43)
    eqs['eq_F14_19'] = sp.Eq(W_E, Q**2/(2*C))  # capacitor energy in terms of charge (Griffiths 2.55; cf. F6_6)
    eqs['eq_F14_20'] = sp.Eq(E_n, sig/eps0)  # field just outside a conductor, normal to its surface (Griffiths 2.48; F4_7 with D=0 inside)
    eqs['eq_F14_21'] = sp.Eq(P_es, sig**2/(2*eps0))  # electrostatic (outward) pressure on a conductor surface (Griffiths 2.52)
    eqs['eq_F14_22'] = sp.Eq(F_x, V**2/2*sp.Derivative(sp.Function('C')(x_s), x_s))  # force on a dielectric/electrode at constant voltage (Griffiths 4.64)
    eqs['eq_F14_23'] = sp.Eq(sp.Function('C')(x_s), eps0*w*(l + chi_e*x_s)/d)  # slab of chi_e inserted distance x_s into plates of length l, width w (Griffiths ex. 4.6) -> F = eps0 chi_e w V^2/(2d)
    eqs['eq_F14_24'] = sp.Eq(q_img, -q)  # image of q at height d above a grounded plane, placed at z = -d (Griffiths 3.2)
    eqs['eq_F14_25'] = sp.Eq(sig_ind, -q*d/(2*sp.pi*(rho_p**2 + d**2)**sp.Rational(3, 2)))  # induced surface charge on the grounded plane; integrates to -q (Griffiths 3.10)
    eqs['eq_F14_26'] = sp.Eq(F_img, -q**2/(4*sp.pi*eps0*(2*d)**2))  # attraction of q toward the grounded plane (Griffiths 3.2)
    eqs['eq_F14_27'] = sp.Eq(q_img, -q*R/a)  # grounded sphere radius R, charge q at distance a from centre: image charge (Griffiths prob. 3.8)
    eqs['eq_F14_28'] = sp.Eq(b_img, R**2/a)  # image position from the sphere centre (Griffiths prob. 3.8)
    eqs['eq_F14_29'] = sp.Eq(F_img, -q**2*R*a/(4*sp.pi*eps0*(a**2 - R**2)**2))  # force on q from a grounded sphere (Griffiths prob. 3.8)
    eqs['eq_F14_30'] = sp.Eq(q_img, -q*(eps_r - 1)/(eps_r + 1))  # image of q at distance d from a dielectric half-space (charged balloon on a wall; Griffiths 4.4.3)
    eqs['eq_F14_31'] = sp.Eq(F_img, -q**2*(eps_r - 1)/((eps_r + 1)*4*sp.pi*eps0*(2*d)**2))  # attraction of a charge to a dielectric half-space (Griffiths prob. 4.39)
    eqs['eq_F14_32'] = sp.Eq(E_surf, V/r)  # surface field of an isolated sphere at potential V: sharp points (small r) break down first (Purcell-Morin 3.1)
    eqs['eq_F14_33'] = sp.Eq(sig_1/sig_2, r_2/r_1)  # two spheres joined by a wire (equipotential), far apart: sigma ~ 1/r (Purcell-Morin 3.1, Griffiths prob. 2.48)
    eqs['eq_F14_34'] = sp.Eq(Q_inner, -Q_enc)  # Faraday ice pail: charge induced on the inner wall of a closed conductor (Gauss's law with E=0 in the metal)
    eqs['eq_F14_35'] = sp.Eq(Q_outer, Q_enc)  # Faraday ice pail: outer-surface charge of an initially neutral, insulated pail
    eqs['eq_F14_36'] = sp.Eq(E_in, 0)  # electrostatic field inside a conductor (conductor is an equipotential; tangential E = 0 at its surface, F4_9) (Griffiths 2.5.1)
    eqs['eq_F14_37'] = sp.Eq(V_dip, p*sp.cos(theta)/(4*sp.pi*eps0*r**2))  # electric dipole potential (Griffiths 3.99)
    eqs['eq_F14_38'] = sp.Eq(E_r, 2*p*sp.cos(theta)/(4*sp.pi*eps0*r**3))  # dipole field, radial component (Griffiths 3.103)
    eqs['eq_F14_39'] = sp.Eq(E_th, p*sp.sin(theta)/(4*sp.pi*eps0*r**3))  # dipole field, polar component (Griffiths 3.103)
    eqs['eq_F14_40'] = sp.Eq(tau_p, cross(p_vec, E_vec))  # torque on an electric dipole (Griffiths 4.4)
    eqs['eq_F14_41'] = sp.Eq(U_p, -dot(p_vec, E_vec))  # energy of a (rigid) electric dipole (Griffiths prob. 4.7)
    eqs['eq_F14_42'] = sp.Eq(F_p, nabla(dot(p_vec, E_vec)))  # force on a rigid dipole in an electrostatic (curl-free) field, grad (Griffiths 4.5)
    eqs['eq_F14_43'] = sp.Eq(alpha_pol, 4*sp.pi*eps0*R**3)  # polarizability of a conducting sphere, p = alpha E (Griffiths prob. 4.2 / 3.23)
    eqs['eq_F14_44'] = sp.Eq(alpha_pol, 4*sp.pi*eps0*R**3*(eps_r - 1)/(eps_r + 2))  # polarizability of a dielectric sphere (Griffiths ex. 4.7)
    eqs['eq_F14_45'] = sp.Eq(F_ind, alpha_pol/2*nabla(E_mag**2))  # force on an induced dipole: neutral scraps pulled up field gradients, grad (Jackson 4.9; Zangwill 6.7)
    eqs['eq_F14_46'] = sp.Eq(sp.tan(theta_b), q**2/(4*sp.pi*eps0*(2*L_s*sp.sin(theta_b))**2*m_b*g))  # pith-ball electroscope: two balls (charge q, mass m_b) on strings of length L_string (Griffiths/Purcell exercise)
    return eqs


def _expand_faraday_f15_static_discharge():
    eqs = {}
    eps0, eps, sigma_e, e = sp.symbols('epsilon_0 epsilon sigma_e e', positive=True)
    sig_c, phi_A, phi_B, z0 = sp.symbols('sigma_c phi_A phi_B z_0')
    tau_e, rho_c, q, q0 = sp.symbols('tau_e rho_c q q_0')
    Q, Q0, R_l, C = sp.symbols('Q Q_0 R_leak C', positive=True)
    i_H, V0, R_H, C_H, W_s, W_MIE = sp.symbols('i_HBM V_0 R_HBM C_HBM W_spark W_MIE', positive=True)
    V_b, A_P, B_P, p, d, gam = sp.symbols('V_b A_P B_P p d gamma_se', positive=True)
    pd_min, V_min, alpha_T = sp.symbols('pd_min V_min alpha_T', positive=True)
    E_c, E_0P, m_s, delta_a, K_P, r_w = sp.symbols('E_c E_0P m_s delta_air K_P r_w', positive=True)
    p0, T0, T = sp.symbols('p_0 T_0 T', positive=True)
    Q_max, R, E_bd = sp.symbols('Q_max R E_bd', positive=True)
    I_b, sig_b, w_b, v_b, V_t, I_leak, V_max, C_t = sp.symbols('I_belt sigma_b w_b v_b V_t I_leak V_max C_t')
    V_K, tau_K, C_i, nu_d, q_d, V_ring = sp.symbols('V_K tau_K C_i nu_drop q_drop V_ring')
    W_fl, Q_fl, dV_fl, AI, i_l, dT_c, rho_e, A_c, rho_m, c_m = sp.symbols('W_flash Q_flash Delta_V_flash AI i_lightning Delta_T_c rho_e A_c rho_m c_m')
    Q_pl, sig_d, A_pl, V_pl, C_pl, V_el, eps_r, d_el, sig_el, V_sep, z_sep = sp.symbols(
        'Q_plate sigma_d A_plate V_plate C_plate V_electret epsilon_r d_el sigma_el V_sep z_sep')

    eqs['eq_F15_1'] = sp.Eq(sig_c, eps0*(phi_A - phi_B)/(e*z0))  # contact electrification, condenser model: work-function difference (J) across the contact gap z0 (Harper 1967; Lowell & Rose-Innes 1980)
    eqs['eq_F15_2'] = sp.Eq(V_sep, sig_c*z_sep/eps0)  # voltage of two separated oppositely charged sheets: peeling tape reaches kV as z_sep grows (Gauss's law, parallel-plate field)
    eqs['eq_F15_3'] = sp.Eq(tau_e, eps/sigma_e)  # charge relaxation time of an ohmic medium (Haus-Melcher 7.2; Zangwill 9.6)
    eqs['eq_F15_4'] = sp.Eq(sp.Derivative(sp.Function('rho_c')(t), t), -(sigma_e/eps)*sp.Function('rho_c')(t))  # free-charge relaxation from F1_5 + F1_3 + F1_10 (Griffiths 7.1.1, Haus-Melcher 7.2)
    eqs['eq_F15_5'] = sp.Eq(sp.Function('q')(t), q0*sp.exp(-t/tau_e))  # solution of F15_4: net charge decays with tau_e (Haus-Melcher 7.2)
    eqs['eq_F15_6'] = sp.Eq(sp.Function('Q')(t), Q0*sp.exp(-t/(R_l*C)))  # electroscope leakage through insulator/air resistance R_leak (RC discharge)
    eqs['eq_F15_7'] = sp.Eq(sp.Function('i_HBM')(t), V0/R_H*sp.exp(-t/(R_H*C_H)))  # human-body-model ESD current, C_HBM = 100 pF, R_HBM = 1.5 kOhm (ANSI/ESDA/JEDEC JS-001)
    eqs['eq_F15_8'] = sp.Eq(W_s, C_H*V0**2/2)  # stored (spark) energy of a charged body (capacitor energy, F6_6)
    eqs['eq_F15_9'] = W_s >= W_MIE  # ignition criterion: spark energy vs minimum ignition energy of the mixture (NFPA 77; IEC 60079-32-1)
    eqs['eq_F15_10'] = sp.Eq(V_b, B_P*p*d/sp.log(A_P*p*d/sp.log(1 + 1/gam)))  # Paschen's law, uniform gap, A_P, B_P as in F13_5 (Raizer 7.2; Lieberman-Lichtenberg 14.3)
    eqs['eq_F15_11'] = sp.Eq(pd_min, sp.E*sp.log(1 + 1/gam)/A_P)  # Paschen minimum position (Raizer 7.2)
    eqs['eq_F15_12'] = sp.Eq(V_min, sp.E*B_P*sp.log(1 + 1/gam)/A_P)  # Paschen minimum breakdown voltage (Raizer 7.2)
    eqs['eq_F15_13'] = sp.Eq(gam*(sp.exp(alpha_T*d) - 1), 1)  # Townsend self-sustainment criterion, alpha_T from F13_5 (Raizer 7.1; Meek-Craggs)
    # CAVEAT: Peek's coefficients are in cm units (E_0P in kV/cm, K_P in cm^1/2, r_w in cm), so
    # they are kept symbolic rather than folded into SI literals; convert r_w before substituting.
    eqs['eq_F15_14'] = sp.Eq(E_c, E_0P*m_s*delta_a*(1 + K_P/sp.sqrt(delta_a*r_w)))  # Peek corona onset surface field; parallel wires E_0P = 30 kV/cm, K_P = 0.301 cm^1/2; coaxial 31 kV/cm, 0.308; r_w in cm (Peek 1929)
    eqs['eq_F15_15'] = sp.Eq(delta_a, (p/p0)*(T0/T))  # relative air density for F15_14, p0 = 101.3 kPa, T0 = 298 K (Peek 1929)
    eqs['eq_F15_16'] = sp.Eq(Q_max, 4*sp.pi*eps0*R**2*E_bd)  # largest charge an isolated sphere holds before surface breakdown, E_bd ~ 3 MV/m in air (F14_2 at r=R)
    eqs['eq_F15_17'] = sp.Eq(I_b, sig_b*w_b*v_b)  # Van de Graaff belt charging current (Van de Graaff et al. 1933)
    eqs['eq_F15_18'] = sp.Eq(C_t*sp.Derivative(sp.Function('V_t')(t), t), I_b - I_leak)  # terminal charging: belt current minus corona/leakage, C_t = 4 pi eps0 R (F14_9)
    eqs['eq_F15_19'] = sp.Eq(V_max, E_bd*R)  # Van de Graaff terminal voltage limit (F14_32 at E = E_bd)
    eqs['eq_F15_20'] = sp.Eq(q_d, -C_i*V_ring)  # Kelvin dropper: charge induced on a grounded drop leaving the inductor ring at V_ring, C_i = drop-ring coupling capacitance (Kelvin 1867; Zahn EM Field Theory)
    eqs['eq_F15_21'] = sp.Eq(sp.Derivative(sp.Function('V_K')(t), t), sp.Function('V_K')(t)/tau_K)  # Kelvin dropper cross-coupled positive feedback: exponential growth until breakdown (symmetric cans)
    # CAVEAT: tau_K = C/(C_i nu) is derived here from F15_20 for two identical cross-connected
    # cans and rings (each drop carries -C_i V_ring into the opposite can); it is not a quoted
    # textbook formula. Asymmetric builds, drop-size spread and leakage change it.
    eqs['eq_F15_22'] = sp.Eq(tau_K, C/(C_i*nu_d))  # Kelvin dropper growth time: C = each can's capacitance to ground, nu_drop = drops/s per stream (derived from F15_20 with cross-connected rings)
    # CAVEAT: W = Q dV is an UPPER BOUND on a stroke's electrical energy. The cloud-ground
    # potential is not a single number, the channel dissipates along its length, and most of
    # the energy goes into the channel and the thunder shock, not the attachment point.
    eqs['eq_F15_23'] = sp.Eq(W_fl, Q_fl*dV_fl)  # electrical work of a lightning stroke moving Q_flash through the cloud-ground potential difference (upper bound; Uman, The Lightning Discharge)
    eqs['eq_F15_24'] = sp.Eq(AI, sp.Integral(sp.Function('i_lightning')(t)**2, t))  # action integral (specific energy per ohm) of a stroke (IEC 62305-1)
    eqs['eq_F15_25'] = sp.Eq(dT_c, rho_e*AI/(A_c**2*rho_m*c_m))  # adiabatic heating of a down-conductor, constant resistivity (IEC 62305-1 Annex D form without temperature coefficient)
    eqs['eq_F15_26'] = sp.Eq(Q_pl, -sig_d*A_pl)  # electrophorus: charge left on the plate after grounding it on the charged dielectric (Volta 1775; Jefimenko, Electrostatic Motors)
    eqs['eq_F15_27'] = sp.Eq(V_pl, Q_pl/C_pl)  # electrophorus plate voltage once lifted, C_plate falls as it is raised (F14_7)
    eqs['eq_F15_28'] = sp.Eq(V_el, sig_el*d_el/(eps0*eps_r))  # surface potential of an electret of thickness d_el on a grounded back electrode (Sessler, Electrets)
    return eqs


def _expand_faraday_f16_magnetostatics():
    eqs = {}
    dot = sp.Function('dot')
    mu0, mu, I, I1, I2, R, s, z_a, n_t, N, L, A, r, d = sp.symbols('mu_0 mu I I_1 I_2 R s z n_t N L A r d', positive=True)
    B_vec, dl_vec, r_hat, dl, B_mag, I_enc = sp.symbols('B_vec dl_vec r_hat dl B I_enc')
    B_w, B_ax, B_sol, B_tor = sp.symbols('B_wire B_axis B_sol B_tor')
    m, m_vec, m1, m2, theta = sp.symbols('m m_vec m_1 m_2 theta')
    B_r, B_th, A_phi, tau_m, F_m, U_m, F_len, F_dd, F_im, F_w = sp.symbols('B_r B_theta A_phi tau_m F_m U_m F_per_len F_dd F_im F_wire')
    h = sp.Symbol('h', positive=True)
    p_B, T_B = sp.symbols('p_B T_B')
    mmf, Rel, Phi, l_c, A_c, mu_c, g, Rel_i, Rel_tot = sp.symbols('mmf Rel Phi l_c A_c mu_c g_gap Rel_i Rel_tot')
    i = sp.Symbol('i', integer=True)
    H_d, N_d, M, B_in, B_m, H_m, l_m, A_m, l_g, A_g, Pc = sp.symbols('H_d N_d M B_in B_m H_m l_m A_m l_g A_g P_c')
    F_pull, Br, Lm, Rm, zf = sp.symbols('F_pull B_rem L_mag R_mag z_f', positive=True)
    L_self, M_12, dl1, dl2, r12, F_x, x_c = sp.symbols('L_self M_12 dl_1 dl_2 r_12 F_x x_c')
    chi, V_b, U_dia, F_dia, rho_b, g_acc = sp.symbols('chi V_b U_dia F_dia rho_b g')
    U_pot, theta_w = sp.symbols('U_pot theta_w')

    eqs['eq_F16_1'] = sp.Eq(B_vec, mu0/(4*sp.pi)*sp.Integral(I*cross(dl_vec, r_hat)/r**2, dl))  # Biot-Savart law for a line current (Griffiths 5.34)
    eqs['eq_F16_2'] = sp.Eq(sp.Integral(dot(B_vec, dl_vec), dl), mu0*I_enc)  # Ampere's law, integral (magnetostatic) form around a closed loop (Griffiths 5.57)
    eqs['eq_F16_3'] = sp.Eq(B_w, mu0*I/(2*sp.pi*s))  # long straight wire at distance s (Griffiths 5.38)
    eqs['eq_F16_4'] = sp.Eq(B_ax, mu0*I*R**2/(2*(R**2 + z_a**2)**sp.Rational(3, 2)))  # circular loop, on axis at height z (Griffiths ex. 5.6)
    eqs['eq_F16_5'] = sp.Eq(B_sol, mu0*n_t*I)  # ideal long solenoid, n_t turns per length (Griffiths 5.59)
    eqs['eq_F16_6'] = sp.Eq(B_ax, mu0*n_t*I/2*((z_a + L/2)/sp.sqrt((z_a + L/2)**2 + R**2) - (z_a - L/2)/sp.sqrt((z_a - L/2)**2 + R**2)))  # finite solenoid of length L on its axis, z from centre (Griffiths prob. 5.11)
    eqs['eq_F16_7'] = sp.Eq(B_tor, mu0*N*I/(2*sp.pi*s))  # toroid interior, N total turns (Griffiths 5.60)
    eqs['eq_F16_8'] = sp.Eq(m, I*A)  # magnetic dipole moment of a planar loop (Griffiths 5.86)
    eqs['eq_F16_9'] = sp.Eq(B_r, mu0*2*m*sp.cos(theta)/(4*sp.pi*r**3))  # dipole field, radial component (Griffiths 5.86)
    eqs['eq_F16_10'] = sp.Eq(B_th, mu0*m*sp.sin(theta)/(4*sp.pi*r**3))  # dipole field, polar component (Griffiths 5.86)
    eqs['eq_F16_11'] = sp.Eq(A_phi, mu0*m*sp.sin(theta)/(4*sp.pi*r**2))  # dipole vector potential (Griffiths 5.85)
    eqs['eq_F16_12'] = sp.Eq(tau_m, cross(m_vec, B_vec))  # torque on a magnetic dipole (Griffiths 6.1)
    eqs['eq_F16_13'] = sp.Eq(F_m, nabla(dot(m_vec, B_vec)))  # force on a magnetic dipole, grad (Griffiths 6.3)
    eqs['eq_F16_14'] = sp.Eq(U_m, -dot(m_vec, B_vec))  # energy of a permanent magnetic dipole (Griffiths prob. 6.21)
    eqs['eq_F16_15'] = sp.Eq(F_dd, 3*mu0*m1*m2/(2*sp.pi*r**4))  # force between coaxial aligned dipoles (attractive) at spacing r (Griffiths prob. 6.3)
    eqs['eq_F16_16'] = sp.Eq(F_im, 3*mu0*m**2/(32*sp.pi*h**4))  # repulsion of a vertical dipole at height h above a perfect conductor / superconductor (antiparallel image at 2h, F16_15)
    eqs['eq_F16_17'] = sp.Eq(F_len, mu0*I1*I2/(2*sp.pi*d))  # force per length between parallel wires, attractive for like currents (Griffiths 5.40)
    eqs['eq_F16_18'] = sp.Eq(F_w, I*L*B_mag*sp.sin(theta_w))  # Ampere force on a straight wire of length L at angle theta_w to B (integral of F3_5; Griffiths 5.17)
    eqs['eq_F16_19'] = sp.Eq(p_B, B_mag**2/(2*mu0))  # magnetic pressure (normal to field lines) (Jackson 5.16; Zangwill 12.9)
    eqs['eq_F16_20'] = sp.Eq(T_B, B_mag**2/mu0)  # magnetic tension along field lines (Maxwell stress F3_6)
    eqs['eq_F16_21'] = sp.Eq(mmf, N*I)  # magnetomotive force (Fitzgerald-Kingsley ch. 1)
    eqs['eq_F16_22'] = sp.Eq(Rel, l_c/(mu_c*A_c))  # reluctance of a uniform limb (Fitzgerald-Kingsley 1.9)
    eqs['eq_F16_23'] = sp.Eq(mmf, Phi*Rel)  # Hopkinson's law (Fitzgerald-Kingsley 1.8)
    eqs['eq_F16_24'] = sp.Eq(Rel_tot, sp.Sum(sp.IndexedBase('Rel_i')[i], (i, 1, sp.Symbol('N_limbs'))))  # series reluctances add (Fitzgerald-Kingsley 1.12)
    eqs['eq_F16_25'] = sp.Eq(Phi, N*I/(l_c/(mu_c*A_c) + g/(mu0*A_c)))  # gapped core flux, fringing neglected (Fitzgerald-Kingsley ex. 1.1)
    eqs['eq_F16_26'] = sp.Eq(H_d, -N_d*M)  # demagnetizing field of a uniformly magnetized ellipsoid, N_x+N_y+N_z = 1 (Kittel 16; Coey 2.2)
    eqs['eq_F16_27'] = sp.Eq(B_in, sp.Rational(2, 3)*mu0*M)  # interior B of a uniformly magnetized sphere, N_d = 1/3 (Griffiths ex. 6.1)
    eqs['eq_F16_28'] = sp.Eq(Pc, -B_m/(mu0*H_m))  # permeance coefficient: slope of the permanent-magnet load line (Coey 13.1; Campbell, Permanent Magnet Materials)
    eqs['eq_F16_29'] = sp.Eq(Pc, (1 - N_d)/N_d)  # open-circuit magnet load line from B = mu0(H+M), H = -N_d M (Coey 13.1)
    eqs['eq_F16_30'] = sp.Eq(Pc, l_m*A_g/(l_g*A_m))  # magnet in a gapped circuit, leakage and iron reluctance neglected (Campbell 5.2)
    eqs['eq_F16_31'] = sp.Eq(F_pull, B_mag**2*A/(2*mu0))  # holding force of a magnet/electromagnet across a small gap of area A (Maxwell pull, F16_19)
    eqs['eq_F16_32'] = sp.Eq(B_ax, Br/2*((Lm + zf)/sp.sqrt(Rm**2 + (Lm + zf)**2) - zf/sp.sqrt(Rm**2 + zf**2)))  # on-axis field of a cylindrical magnet (length L_mag, radius R_mag, remanence B_rem), z_f from the pole face (equivalent-solenoid, F16_6)
    eqs['eq_F16_33'] = sp.Eq(L_self, mu0*N**2*A/L)  # long-solenoid self-inductance (Griffiths ex. 7.13)
    eqs['eq_F16_34'] = sp.Eq(M_12, mu0/(4*sp.pi)*sp.Integral(sp.Integral(dot(dl1, dl2)/r12, dl1), dl2))  # Neumann formula for mutual inductance (Griffiths 7.23)
    eqs['eq_F16_35'] = sp.Eq(F_x, I1*I2*sp.Derivative(sp.Function('M_12')(x_c), x_c))  # force between two circuits at fixed currents (virtual work on coenergy; Zangwill 12.4)
    eqs['eq_F16_36'] = sp.Eq(U_dia, -chi*V_b*B_mag**2/(2*mu0))  # magnetic energy of a small linear body, |chi| << 1 (Jackson 5.16; Simon-Geim 2000)
    eqs['eq_F16_37'] = sp.Eq(F_dia, chi*V_b/(2*mu0)*nabla(B_mag**2))  # force on a weakly magnetic body, grad; diamagnets (chi<0) pushed to low field (Simon-Geim, J. Appl. Phys. 87, 6200 (2000))
    eqs['eq_F16_38'] = chi/mu0*B_mag*sp.Derivative(sp.Function('B')(z), z) >= rho_b*g_acc  # diamagnetic levitation condition, z up: F16_37's vertical component must carry the weight; chi < 0 and B decreasing upward (dB/dz < 0) make the left side positive (Berry-Geim, Eur. J. Phys. 18, 307 (1997))
    eqs['eq_F16_39'] = sp.Eq(nabla_op**2*U_pot, 0)  # Earnshaw: potential of fixed charges/magnets in free space is harmonic -> no stable static minimum (Earnshaw 1842; Jackson 1.13)
    eqs['eq_F16_40'] = nabla_op**2*B_mag**2 >= 0  # Braunbek/Earnshaw loophole: |B|^2 has no interior maximum in free space, so chi<0 bodies can sit at a field minimum (Berry-Geim 1997)
    # Cartesian specialization of F16_9/F16_10 for a dipole m along +y.
    # Publishing it avoids manufacturing an angle merely to recover its sine
    # and cosine, and is algebraically the same Griffiths 5.86 field.
    dx_m, dy_m = sp.symbols('dx_m dy_m')
    B_x_dip, B_y_dip = sp.symbols('B_x_dip B_y_dip')
    r2_m = dx_m**2 + dy_m**2
    eqs['eq_F16_41'] = sp.Eq(B_x_dip, 3*mu0*m*dx_m*dy_m/(4*sp.pi*r2_m**sp.Rational(5, 2)))  # dipole field x component, m parallel +y (Griffiths 5.86)
    eqs['eq_F16_42'] = sp.Eq(B_y_dip, mu0*m*(3*dy_m**2-r2_m)/(4*sp.pi*r2_m**sp.Rational(5, 2)))  # dipole field y component, m parallel +y (Griffiths 5.86)
    return eqs


def _expand_faraday_f17_magnetic_materials():
    eqs = {}
    mu0, k_B, n, T, T_C, mu_B, g_J, J, S, Lq = sp.symbols('mu_0 k_B n T T_C mu_B g_J J S L_q', positive=True)
    chi, C_cu, mu_eff, lam_W, theta_N = sp.symbols('chi_m C_curie mu_eff lambda_W theta_N')
    M, mu_at, xi, B, x_B, M_s, e, m_e, hbar, Z, r2 = sp.symbols('M mu_at xi B x_B M_s e m_e hbar Z r2_mean')
    D_F, E_F, chi_P, chi_L, mu_r = sp.symbols('D_EF E_F chi_P chi_L mu_r')
    P_h, eta_S, f, B_max, W_h, H = sp.symbols('P_h eta_S f B_max W_h H')
    H_e, alpha_JA, a_JA, M_an, M_irr, M_rev, c_JA, k_JA, delta_JA = sp.symbols('H_e alpha_JA a_JA M_an M_irr M_rev c_JA k_JA delta_JA')
    lam_ms, lam_s, th = sp.symbols('lambda_ms lambda_s theta')
    M0, b_B = sp.symbols('M_s0 b_Bloch')
    dS_M, dT_ad, C_H, H0, H1 = sp.symbols('Delta_S_M Delta_T_ad C_H H_0 H_1')
    sig_w, delta_w, A_ex, K_u = sp.symbols('sigma_w delta_w A_ex K_u', positive=True)
    E_SW, psi, H_K = sp.symbols('E_SW psi H_K')

    def BJ(xx):
        return (2*J + 1)/(2*J)*sp.coth((2*J + 1)*xx/(2*J)) - 1/(2*J)*sp.coth(xx/(2*J))

    Mf = sp.Function('M')(H)
    eqs['eq_F17_1'] = sp.Eq(mu_r, 1 + chi)  # relative permeability, M = chi_m H (Griffiths 6.30-6.31; F1_7, F1_9)
    eqs['eq_F17_2'] = sp.Eq(chi, C_cu/T)  # Curie law of a paramagnet (Kittel 11.22)
    eqs['eq_F17_3'] = sp.Eq(C_cu, mu0*n*mu_eff**2/(3*k_B))  # Curie constant, SI dimensionless chi (Kittel 11; Blundell 2.6)
    eqs['eq_F17_4'] = sp.Eq(mu_eff, g_J*mu_B*sp.sqrt(J*(J + 1)))  # effective moment (Kittel 11.24)
    eqs['eq_F17_5'] = sp.Eq(g_J, 1 + (J*(J + 1) + S*(S + 1) - Lq*(Lq + 1))/(2*J*(J + 1)))  # Lande g-factor, g_s = 2 (Kittel 11.19)
    eqs['eq_F17_6'] = sp.Eq(mu_B, e*hbar/(2*m_e))  # Bohr magneton (Kittel 11.15)
    eqs['eq_F17_7'] = sp.Eq(chi, C_cu/(T - T_C))  # Curie-Weiss law, ferromagnet above T_C (Kittel 11.34) -- magnetism, not the Curie radioactivity engine
    eqs['eq_F17_8'] = sp.Eq(T_C, lam_W*C_cu)  # mean-field Curie temperature, molecular field B_E = mu0 lambda_W M (Kittel 11.33)
    eqs['eq_F17_9'] = sp.Eq(chi, C_cu/(T + theta_N))  # antiferromagnet above T_N, Curie-Weiss with negative intercept (Kittel 12.47)
    eqs['eq_F17_10'] = sp.Eq(M, n*mu_at*(sp.coth(xi) - 1/xi))  # Langevin paramagnetism (classical moments) (Kittel 11.22 note; Blundell 2.4)
    eqs['eq_F17_11'] = sp.Eq(xi, mu_at*B/(k_B*T))  # Langevin argument (Blundell 2.4)
    eqs['eq_F17_12'] = sp.Eq(M, n*g_J*mu_B*J*BJ(x_B))  # Brillouin paramagnetism (Kittel 11.20; Ashcroft-Mermin 31.44)
    eqs['eq_F17_13'] = sp.Eq(x_B, g_J*J*mu_B*B/(k_B*T))  # Brillouin argument (Kittel 11.20)
    eqs['eq_F17_14'] = sp.Eq(M_s, n*g_J*mu_B*J*BJ(g_J*J*mu_B*mu0*(H + lam_W*M_s)/(k_B*T)))  # Weiss mean-field self-consistency for the ferromagnet (Kittel 11.32; Ashcroft-Mermin 33)
    # CAVEAT: sqrt(3(1 - T/T_C)) is the mean-field limit for S = 1/2 only; other S change the
    # prefactor, and real ferromagnets follow the critical exponent beta ~ 0.33-0.37, not 1/2.
    eqs['eq_F17_15'] = sp.Eq(M_s/M0, sp.sqrt(3*(1 - T/T_C)))  # mean-field order parameter just below T_C, S = 1/2 (Kittel 11.35 limit): hot-nail Curie-point demo
    # CAVEAT: b_Bloch is left symbolic -- it depends on the exchange stiffness and lattice;
    # the T^(3/2) form is exact only for T << T_C (spin-wave regime).
    eqs['eq_F17_16'] = sp.Eq(M_s, M0*(1 - b_B*T**sp.Rational(3, 2)))  # Bloch T^{3/2} law from spin-wave excitation, T << T_C (Kittel 11.60)
    eqs['eq_F17_17'] = sp.Eq(chi, -mu0*n*Z*e**2*r2/(6*m_e))  # Langevin/Larmor diamagnetism, r2_mean = <r^2> per electron (Kittel 11.7 SI)
    eqs['eq_F17_18'] = sp.Eq(chi_P, mu0*mu_B**2*D_F)  # Pauli spin paramagnetism, D_EF = density of states per volume at E_F, both spins (Ashcroft-Mermin 31.69)
    eqs['eq_F17_19'] = sp.Eq(chi_P, 3*n*mu0*mu_B**2/(2*E_F))  # free-electron Pauli susceptibility (Kittel 11.51)
    eqs['eq_F17_20'] = sp.Eq(chi_L, -chi_P/3)  # Landau diamagnetism of free electrons (Ashcroft-Mermin 31.70)
    eqs['eq_F17_21'] = sp.Eq(P_h, eta_S*f*B_max**sp.Float('1.6'))  # Steinmetz hysteresis loss per volume, original exponent 1.6 (Steinmetz 1892; general form F8_7)
    eqs['eq_F17_22'] = sp.Eq(W_h, sp.Integral(H, B))  # hysteresis energy per volume per cycle, integral over the closed B-H loop (Jiles, Intro. Magnetism 2.5)
    eqs['eq_F17_23'] = sp.Eq(H_e, H + alpha_JA*Mf)  # Jiles-Atherton effective field (Jiles-Atherton, JMMM 61, 48 (1986))
    eqs['eq_F17_24'] = sp.Eq(M_an, M_s*(sp.coth(H_e/a_JA) - a_JA/H_e))  # Jiles-Atherton anhysteretic (modified Langevin) (JMMM 1986)
    eqs['eq_F17_25'] = sp.Eq(sp.Derivative(sp.Function('M_irr')(H), H), (M_an - sp.Function('M_irr')(H))/(k_JA*delta_JA - alpha_JA*(M_an - sp.Function('M_irr')(H))))  # J-A irreversible (pinning) branch, delta_JA = sign(dH/dt) (JMMM 1986)
    eqs['eq_F17_26'] = sp.Eq(M_rev, c_JA*(M_an - M_irr))  # J-A reversible (wall-bowing) component, M = M_irr + M_rev (JMMM 1986)
    eqs['eq_F17_27'] = sp.Eq(sp.Derivative(Mf, H), (1 - c_JA)*(M_an - M_irr)/(k_JA*delta_JA - alpha_JA*(M_an - M_irr)) + c_JA*sp.Derivative(sp.Function('M_an')(H), H))  # J-A total differential susceptibility (Jiles-Atherton 1986)
    eqs['eq_F17_28'] = sp.Eq(lam_ms, sp.Rational(3, 2)*lam_s*(sp.cos(th)**2 - sp.Rational(1, 3)))  # isotropic magnetostriction strain at angle theta to M (Chikazumi 14; Cullity 8)
    eqs['eq_F17_29'] = sp.Eq(dS_M, mu0*sp.Integral(sp.Derivative(sp.Function('M')(T, H), T), (H, H0, H1)))  # magnetocaloric entropy change from the Maxwell relation (Tishin-Spichkin; Pecharsky-Gschneidner 1999)
    eqs['eq_F17_30'] = sp.Eq(dT_ad, -mu0*sp.Integral(T/C_H*sp.Derivative(sp.Function('M')(T, H), T), (H, H0, H1)))  # adiabatic magnetocaloric temperature change, C_H per volume (Pecharsky-Gschneidner 1999)
    eqs['eq_F17_31'] = sp.Eq(sig_w, 4*sp.sqrt(A_ex*K_u))  # Bloch wall energy per area, uniaxial anisotropy (Kittel 12.56 analog; Coey 7.1)
    eqs['eq_F17_32'] = sp.Eq(delta_w, sp.pi*sp.sqrt(A_ex/K_u))  # Bloch wall width (Coey 7.1)
    eqs['eq_F17_33'] = sp.Eq(E_SW, K_u*sp.sin(th)**2 - mu0*M_s*H*sp.cos(th - psi))  # Stoner-Wohlfarth single-domain energy per volume, field at psi to easy axis (Stoner-Wohlfarth 1948)
    eqs['eq_F17_34'] = sp.Eq(H_K, 2*K_u/(mu0*M_s))  # anisotropy field = SW switching field along the easy axis (Stoner-Wohlfarth 1948; Coey 8.2)
    return eqs


def _expand_faraday_f18_eddy_induction():
    eqs = {}
    dot = sp.Function('dot')
    mu0, mu, sigma, rho_e, L_ch, v, d, f, omega, a, w_p, A_p = sp.symbols('mu_0 mu sigma rho_e L_ch v d f omega a w_pipe A_p', positive=True)
    EMF, E_vec, v_vec, B_vec, dl, dA, n_hat, J_vec, B_n = sp.symbols('EMF E_vec v_vec B_vec dl dA n_hat J_vec B_n')
    l_b, B, B0 = sp.symbols('l_b B B_0', positive=True)
    tau_m, R_m, p_ed, P_ed, J2, dV, P_disc = sp.symbols('tau_m R_m p_eddy P_eddy J2 dV P_disc')
    F_d, k_g = sp.symbols('F_drag k_g')
    F_L, F_D, F_I, w_R = sp.symbols('F_L F_D F_I w_R')
    F_pipe, m_mag, M_mag, g, v_t = sp.symbols('F_pipe m_mag M_mag g v_t', positive=True)
    I0, M12, R_r, L_r, I_r, I_c, phi_r, F_avg, zc, i_r, tau_r = sp.symbols('I_0 M_12 R_ring L_ring I_ring I_coil phi_ring F_avg z_c i_ring tau_ring')
    T_ind, V_th, R_th, X_th, R2, X2, s_slip, om_s, om_r, s_max = sp.symbols('T_ind V_th R_th X_th R_2 X_2 s omega_s omega_r s_Tmax')
    P_A, H0, R_s, delta_s = sp.symbols('P_per_A H_0 R_s delta_s')
    dPhi = sp.Symbol('dPhi_dt')

    Bf = sp.Function('B_vec')(t)
    eqs['eq_F18_1'] = sp.Eq(sp.Integral(dot(E_vec, dl), dl), -sp.Derivative(sp.Integral(dot(B_vec, n_hat), dA), t))  # Faraday's law, integral form, stationary loop (integral of F1_1; Griffiths 7.14)
    eqs['eq_F18_2'] = sp.Eq(EMF, -sp.Derivative(sp.Function('Phi_B')(t), t))  # flux rule, valid also for moving loops (Griffiths 7.13)
    eqs['eq_F18_3'] = sp.Eq(EMF, sp.Integral(dot(E_vec + cross(v_vec, B_vec), dl), dl))  # EMF of a moving circuit: force per charge in the conductor frame (Griffiths 7.9; Zangwill 14.2)
    eqs['eq_F18_4'] = sp.Eq(EMF, B*l_b*v)  # motional EMF of a bar of length l_b sliding at v across B (Griffiths 7.1.3)
    eqs['eq_F18_5'] = sp.Eq(J_vec, sigma*(E_vec + cross(v_vec, B_vec)))  # Ohm's law in a moving conductor (Jackson 5.18; Haus-Melcher 6.2)
    eqs['eq_F18_6'] = sp.Eq(sp.Derivative(Bf, t), nabla(cross(v_vec, Bf)) - nabla(nabla(Bf))/(mu*sigma))  # magnetic induction equation: advection curl(v x B) + diffusion, uniform mu, sigma; laplacian B = -curl curl B since div B = 0 (Jackson 5.18; Davidson MHD 2.7)
    eqs['eq_F18_7'] = sp.Eq(tau_m, mu*sigma*L_ch**2)  # magnetic diffusion time over length L_ch (Jackson 5.18; Haus-Melcher 6.4)
    eqs['eq_F18_8'] = sp.Eq(R_m, mu*sigma*v*L_ch)  # magnetic Reynolds number (Davidson MHD 2.7)
    eqs['eq_F18_9'] = sp.Eq(P_ed, sp.Integral(J2/sigma, dV))  # eddy dissipation = drag power F v of a steadily moving conductor (Joule, F5_3)
    eqs['eq_F18_10'] = sp.Eq(p_ed, sp.pi**2*d**2*B**2*f**2/(6*rho_e))  # eddy loss per volume in a lamination d << delta, B peak sinusoidal (Fitzgerald-Kingsley; Bozorth)
    eqs['eq_F18_11'] = sp.Eq(P_disc, sp.pi*sigma*omega**2*B0**2*a**4*d/16)  # time-averaged eddy loss in a thin disc (radius a, thickness d) in a uniform axial field B0 sin(omega t), d, a << delta (E_phi = -(r/2) dB/dt)
    # CAVEAT: k_g is NOT closed-form. It depends on pole shape, plate size relative to the pole
    # and the eddy return path (k_g -> 1 only for a plate much wider than the pole). Only the
    # bound k_g <= 1 is exact; calibrate k_g per geometry (Wiederick 1987, Heald 1988).
    eqs['eq_F18_12'] = sp.Eq(F_d, k_g*sigma*B**2*d*A_p*v)  # low-speed eddy drag on a thin plate under a pole of area A_p; k_g <= 1 accounts for the eddy return path (Wiederick et al., AJP 55, 500 (1987); Heald, AJP 56, 521 (1988))
    # CAVEAT: this is Reitz's thin-sheet characteristic speed, used for the drag-curve peak.
    # Wouterse's (1991) eddy-brake critical-speed coefficient could not be confirmed and is
    # not stated; for a rotating disc brake w_R is a thin-sheet approximation.
    eqs['eq_F18_13'] = sp.Eq(w_R, 2/(mu0*sigma*d))  # characteristic speed of a thin conducting sheet (Reitz, J. Appl. Phys. 41, 2067 (1970)); peak of the drag curve, cf. F9_3
    eqs['eq_F18_14'] = sp.Eq(F_L, F_I*(1 - w_R/sp.sqrt(v**2 + w_R**2)))  # electrodynamic (maglev) lift over a thin sheet, F_I = perfect-conductor image force (F16_16) (Reitz 1970)
    eqs['eq_F18_15'] = sp.Eq(F_D, w_R/v*F_L)  # electrodynamic drag, so F_L/F_D = v/w_R (Reitz 1970)
    eqs['eq_F18_16'] = sp.Eq(F_pipe, 45*mu0**2*m_mag**2*sigma*w_p*v/(1024*a**4))  # drag on a dipole magnet falling in a thin-walled pipe (radius a, wall w_pipe << a) (Levin et al., AJP 74, 815 (2006))
    eqs['eq_F18_17'] = sp.Eq(v_t, 1024*a**4*M_mag*g/(45*mu0**2*m_mag**2*sigma*w_p))  # terminal speed of the magnet in the pipe, M_mag = its mass (Levin et al. 2006)
    eqs['eq_F18_18'] = sp.Eq(I_r, -sp.I*omega*M12*I_c/(R_r + sp.I*omega*L_r))  # Thomson ring current phasor from the coil current phasor (e^{i omega t} convention) (Barry-Rowe; Tjossem-Parker, AJP 73 (2005))
    eqs['eq_F18_19'] = sp.Eq(sp.tan(phi_r), omega*L_r/R_r)  # ring's inductive phase lag beyond the 90 deg of pure induction (Tjossem-Parker 2005)
    eqs['eq_F18_20'] = sp.Eq(F_avg, -I0**2/2*omega**2*L_r*M12/(R_r**2 + omega**2*L_r**2)*sp.Derivative(sp.Function('M_12')(zc), zc))  # time-averaged jumping-ring force, coil current I0 cos(omega t); up since dM/dz < 0 above the coil (F16_35)
    eqs['eq_F18_21'] = sp.Eq(sp.Function('i_ring')(t), -M12*I0/L_r*sp.exp(-t*R_r/L_r))  # ring current after a step of coil current to I_0: flux linkage conserved at t=0+, then L/R decay (DC-switched ring)
    eqs['eq_F18_22'] = sp.Eq(s_slip, (om_s - om_r)/om_s)  # slip of the rotor/disc behind the travelling field (Chapman 6.3)
    eqs['eq_F18_23'] = sp.Eq(T_ind, 3*V_th**2*R2/s_slip/(om_s*((R_th + R2/s_slip)**2 + (X_th + X2)**2)))  # induction-motor/Arago-disc torque-slip, per-phase Thevenin equivalent (Chapman 6.50)
    eqs['eq_F18_24'] = sp.Eq(s_max, R2/sp.sqrt(R_th**2 + (X_th + X2)**2))  # slip of pull-out torque (Chapman 6.53)
    eqs['eq_F18_25'] = sp.Eq(P_A, H0**2*R_s/2)  # induction-heating power per area of a thick conductor, H0 = peak tangential H, R_s = 1/(sigma delta) (F5_5, F5_6; Jackson 8.12)
    eqs['eq_F18_26'] = sp.Eq(sp.sign(EMF), -sp.sign(dPhi))  # Lenz's law: induced EMF opposes the change of linked flux (sign of F18_2)
    return eqs


def _expand_faraday_f19_fine_details():
    eqs = {}
    dot = sp.Function('dot')
    eps0, mu0, c, e, h, hbar, m_e, k_B = sp.symbols('epsilon_0 mu_0 c e h hbar m_e k_B', positive=True)
    I_d, Phi_E, g_vec, S_vec, L_f, r_vec, E_vec, B_vec, dV = sp.symbols('I_d Phi_E g_vec S_vec L_field r_vec E_vec B_vec dV')
    dL_disc, Q, Phi_s = sp.symbols('Delta_L_disc Q Phi_sol')
    p_hid, m_vec = sp.symbols('p_hidden m_vec')
    P_rad, I_int, R_refl, th_i = sp.symbols('P_rad I_int R_refl theta_i')
    P_L, q, a_acc, a_vec, v_vec, gam = sp.symbols('P_Larmor q a a_vec v_vec gamma')
    om_c, m, B, r_L, v_perp, v_par, om_L, gam_m, g_f = sp.symbols('omega_c m B r_L v_perp v_par omega_L gamma_m g_f')
    v_E, v_gB, v_R, gradB, R_c = sp.symbols('v_ExB v_gradB v_curv grad_B R_c_vec')
    R_cm = sp.Symbol('R_c', positive=True)
    mu_ad, th_lc, B_min, B_max = sp.symbols('mu_ad theta_lc B_min B_max')
    V_H, I, n, t_H, R_H, E_H, J_x = sp.symbols('V_H I n t_H R_H E_H J_x')
    th_F, V_ver, L = sp.symbols('theta_F V_Verdet L')
    dE_Z, g_J, m_J, mu_B, d_om = sp.symbols('Delta_E_Z g_J m_J mu_B Delta_omega')
    Phi_0, Phi_f, N_f, phi_AB, Phi_AB = sp.symbols('Phi_0 Phi_fluxoid N_f phi_AB Phi_AB')
    N_f = sp.Symbol('N_f', integer=True)
    lam_L, n_s, m_s, J_s, B_in, B0, H_c, H_c0, T, T_c, f_ns = sp.symbols('lambda_L n_s m_s J_s B_in B_0 H_c H_c0 T T_c Delta_f_ns')
    f_J, V_J, R_xy, nu_f = sp.symbols('f_J V_J R_xy nu_f')
    EMF_h, om, R_d, T_bar = sp.symbols('EMF_hom omega R_d T_Barlow')
    dL_body, dmu, M_B, chi, v_W, E_mag = sp.symbols('Delta_L_body Delta_mu M_Barnett chi v_Wien E')
    E_par, E_par_p, E_perp_p, B_par, B_par_p, B_perp_p, E_perp, B_perp, gam_v = sp.symbols(
        "E_par E_par_prime E_perp_prime B_par B_par_prime B_perp_prime E_perp B_perp gamma_v")
    xx = sp.Symbol('x_depth', positive=True)

    eqs['eq_F19_1'] = sp.Eq(I_d, eps0*sp.Derivative(sp.Function('Phi_E')(t), t))  # Maxwell displacement current through a surface (Griffiths 7.37; F1_2 term)
    eqs['eq_F19_2'] = sp.Eq(g_vec, S_vec/c**2)  # electromagnetic momentum density g = eps0 E x B (Griffiths 8.29)
    eqs['eq_F19_3'] = sp.Eq(L_f, eps0*sp.Integral(cross(r_vec, cross(E_vec, B_vec)), dV))  # field angular momentum (Griffiths 8.34)
    eqs['eq_F19_4'] = sp.Eq(dL_disc, Q*Phi_s/(2*sp.pi))  # Feynman disc paradox: angular momentum given to charge Q on a ring enclosing a solenoid when its flux Phi_sol collapses (Feynman II 17-4; Griffiths ex. 8.4)
    eqs['eq_F19_5'] = sp.Eq(p_hid, cross(m_vec, E_vec)/c**2)  # hidden momentum of a magnetic dipole in an electric field (Griffiths ex. 12.12; Babson et al., AJP 77, 826 (2009))
    eqs['eq_F19_6'] = sp.Eq(P_rad, (1 + R_refl)*I_int*sp.cos(th_i)**2/c)  # radiation pressure on a flat surface, reflectivity R_refl, incidence theta_i: I/c absorbing, 2I/c mirror at normal incidence (Griffiths 9.64 generalized; Jackson 7)
    eqs['eq_F19_7'] = sp.Eq(P_L, q**2*a_acc**2/(6*sp.pi*eps0*c**3))  # Larmor radiated power (Griffiths 11.70)
    eqs['eq_F19_8'] = sp.Eq(P_L, q**2*gam**6/(6*sp.pi*eps0*c**3)*(a_acc**2 - sp.Abs(cross(v_vec, a_vec))**2/c**2))  # Lienard generalization (Griffiths 11.73)
    eqs['eq_F19_9'] = sp.Eq(om_c, q*B/m)  # cyclotron frequency (Griffiths 5.3; Chen 2.2)
    eqs['eq_F19_10'] = sp.Eq(r_L, m*v_perp/(sp.Abs(q)*B))  # Larmor (gyro) radius (Chen 2.7)
    eqs['eq_F19_11'] = sp.Eq(om_L, gam_m*B)  # Larmor precession of a magnetic moment (Jackson 11.8; Kittel 13.1)
    eqs['eq_F19_12'] = sp.Eq(gam_m, g_f*q/(2*m))  # gyromagnetic ratio mu = gamma J (Jackson 11.8)
    eqs['eq_F19_13'] = sp.Eq(v_E, cross(E_vec, B_vec)/B**2)  # E x B drift (Chen 2.15)
    eqs['eq_F19_14'] = sp.Eq(v_gB, m*v_perp**2/(2*q*B**3)*cross(B_vec, gradB))  # grad-B drift (Chen 2.24)
    eqs['eq_F19_15'] = sp.Eq(v_R, m*v_par**2/(q*B**2)*cross(R_c, B_vec)/R_cm**2)  # curvature drift, R_c_vec from centre of curvature (Chen 2.26)
    eqs['eq_F19_16'] = sp.Eq(mu_ad, m*v_perp**2/(2*B))  # magnetic moment adiabatic invariant (Chen 2.36)
    eqs['eq_F19_17'] = sp.Eq(sp.sin(th_lc)**2, B_min/B_max)  # magnetic-mirror loss cone, 1/mirror ratio (Chen 2.39)
    eqs['eq_F19_18'] = sp.Eq(V_H, I*B/(n*q*t_H))  # Hall voltage across a strip of thickness t_H, carrier density n (Griffiths prob. 5.41; Kittel 6.8)
    eqs['eq_F19_19'] = sp.Eq(R_H, 1/(n*q))  # Hall coefficient, single carrier (Kittel 6.55; Ashcroft-Mermin 1.15)
    eqs['eq_F19_20'] = sp.Eq(E_H, R_H*J_x*B)  # Hall field definition (Ashcroft-Mermin 1.14)
    eqs['eq_F19_21'] = sp.Eq(R_xy, h/(nu_f*e**2))  # quantum Hall resistance plateaus (von Klitzing 1980; Kittel 19)
    eqs['eq_F19_22'] = sp.Eq(th_F, V_ver*B*L)  # Faraday rotation, B along propagation path L (Jackson 7.6; Hecht 8.11)
    eqs['eq_F19_23'] = sp.Eq(dE_Z, g_J*m_J*mu_B*B)  # Zeeman energy shift, weak field (Griffiths QM 6.76)
    eqs['eq_F19_24'] = sp.Eq(d_om, e*B/(2*m_e))  # normal (Lorentz) Zeeman splitting = Larmor frequency of the electron (Jackson 7; Feynman II 34)
    eqs['eq_F19_25'] = sp.Eq(Phi_0, h/(2*e))  # superconducting flux quantum (Kittel 10.4)
    eqs['eq_F19_26'] = sp.Eq(Phi_f, N_f*Phi_0)  # fluxoid quantization in a superconducting ring (Kittel 10.4)
    eqs['eq_F19_27'] = sp.Eq(phi_AB, q*Phi_AB/hbar)  # Aharonov-Bohm phase for a path enclosing flux (Aharonov-Bohm 1959; Griffiths QM 4.5)
    eqs['eq_F19_28'] = sp.Eq(f_J, 2*e*V_J/h)  # AC Josephson frequency (Kittel 10.7)
    eqs['eq_F19_29'] = sp.Eq(sp.Derivative(sp.Function('J_s')(t), t), n_s*e**2/m_s*E_vec)  # first London equation (London 1935; Kittel 10.2)
    eqs['eq_F19_30'] = sp.Eq(nabla(J_s), -n_s*e**2/m_s*B_vec)  # second London equation, curl (London 1935; Kittel 10.2)
    eqs['eq_F19_31'] = sp.Eq(lam_L, sp.sqrt(m_s/(mu0*n_s*e**2)))  # London penetration depth (Kittel 10.14)
    eqs['eq_F19_32'] = sp.Eq(sp.Function('B')(xx), B0*sp.exp(-xx/lam_L))  # field screening into a superconductor (Kittel 10.15)
    eqs['eq_F19_33'] = sp.Eq(B_in, 0)  # Meissner effect: flux expelled from the bulk of a type-I superconductor below H_c (Kittel 10.1)
    eqs['eq_F19_34'] = sp.Eq(H_c, H_c0*(1 - (T/T_c)**2))  # empirical parabolic critical-field curve (Kittel 10.1; Tinkham 1)
    eqs['eq_F19_35'] = sp.Eq(f_ns, mu0*H_c**2/2)  # condensation energy per volume F_n - F_s (Kittel 10.4)
    eqs['eq_F19_36'] = sp.Eq(EMF_h, B*om*R_d**2/2)  # Faraday disc (homopolar generator) EMF, axis to rim (Griffiths prob. 7.8)
    eqs['eq_F19_37'] = sp.Eq(T_bar, I*B*R_d**2/2)  # Barlow wheel torque, radial current I in axial B (Ampere force on the radius)
    eqs['eq_F19_38'] = sp.Eq(v_W, E_mag/B)  # Wien filter / crossed-field velocity selector (Griffiths ex. 5.2)
    # CAVEAT: sign convention -- the body's angular momentum change is OPPOSITE to the change of
    # the electrons' spin angular momentum; with Delta_mu the change of the magnetic moment and
    # the electron's negative charge the signs below follow Kittel ch. 14.
    eqs['eq_F19_39'] = sp.Eq(dL_body, 2*m_e*dmu/(g_f*e))  # Einstein-de Haas: body angular momentum from a change of total spin moment Delta_mu, g_f ~ 2 (Einstein-de Haas 1915; Kittel 14)
    eqs['eq_F19_40'] = sp.Eq(M_B, chi*om/(mu0*gam_m))  # Barnett effect: rotation acts as B_eq = omega/gamma (Barnett 1915)
    eqs['eq_F19_41'] = sp.Eq(E_par_p, E_par)  # Lorentz transformation of fields, component parallel to v (Griffiths 12.109)
    eqs['eq_F19_42'] = sp.Eq(E_perp_p, gam_v*(E_perp + cross(v_vec, B_vec)))  # transverse E in a frame moving at v: the moving-magnet / moving-conductor symmetry (Griffiths 12.109)
    eqs['eq_F19_43'] = sp.Eq(B_par_p, B_par)  # parallel B unchanged (Griffiths 12.109)
    eqs['eq_F19_44'] = sp.Eq(B_perp_p, gam_v*(B_perp - cross(v_vec, E_vec)/c**2))  # transverse B in the moving frame (Griffiths 12.109)
    return eqs


def _expand_faraday_electrostatics_magnetism():
    eqs = {}
    eqs.update(_expand_faraday_f14_electrostatics())
    eqs.update(_expand_faraday_f15_static_discharge())
    eqs.update(_expand_faraday_f16_magnetostatics())
    eqs.update(_expand_faraday_f17_magnetic_materials())
    eqs.update(_expand_faraday_f18_eddy_induction())
    eqs.update(_expand_faraday_f19_fine_details())
    return eqs


globals().update(_expand_faraday_electrostatics_magnetism())


# =====================================================================
# 32. LANGMUIR (Plasma: ionized gases, discharges, lightning, and what they do) -- 2026-09-22
# =====================================================================
# WHAT IS EMERGENT, WHAT IS DECLARED
#
# EMERGENT (solutions of laws already in the catalogue, not separate laws):
#   Given the kinetic equation (BO1_1 / LA3_1..3) or its moments (BO2, LA4
#   two-fluid, F13_6/F13_7 drift-diffusion) closed by Poisson/Maxwell (F-series)
#   and the source terms (ionization, attachment, recombination: F13_7, LA4_6..8,
#   LA9 rate laws), the following are OUTCOMES, not inputs:
#     - Debye shielding (LA3_21/22 are the linearised Poisson-Boltzmann
#       solution, i.e. a derived result, kept here for reference),
#     - plasma (Langmuir) oscillation and its Bohm-Gross dispersion (LA3_15),
#     - Landau damping (LA3_16/17 are the asymptotic evaluation of the
#       Vlasov-Poisson dielectric LA3_18; they add nothing the Vlasov
#       equation does not already contain),
#     - sheaths and the Bohm criterion, floating potential (LA6_10..14 are
#       analytic reductions of the same fluid+Poisson system),
#     - ambipolar fields and ambipolar diffusion (LA4_11..13),
#     - avalanche growth, space-charge field of the avalanche head, streamer
#       formation and propagation (ionisation waves) -- LA5 drift-diffusion +
#       Poisson + photoionisation source,
#     - glow-discharge structure (cathode fall, negative glow, positive column)
#       given electrode boundary conditions with a secondary-emission yield.
#
# DECLARED (closures/data; they do NOT emerge from the laws above and must be
# supplied per gas/material; each such item is marked 'declared' or 'empirical'):
#     - cross sections sigma_j(eps) and everything integrated from them:
#       alpha(E/N), eta(E/N), mobilities mu(E/N), diffusion D(E/N), rate
#       coefficients k_j(E/N) (LA3_9, LA5_3/4, LA9_1), energy-branching
#       fractions (LA9_17) -- a Boltzmann solver turns sigma into these, but
#       sigma itself is data;
#     - transport coefficients of a thermal plasma (kappa(T), sigma(T), mu(T));
#     - emission data: Einstein A_ul, g_u, E_u, Gaunt factors, Stark widths;
#     - Saha/Boltzmann partition functions U_z(T) and ionisation energies chi_z;
#     - surface yields: secondary emission gamma, work function W,
#       Richardson constant correction, field-enhancement factor beta_f;
#     - photoionisation constants (Zheleznyak xi, chi_min, chi_max, p_q);
#     - REDUCED MODELS, which replace the kinetic/fluid solution by a posited
#       closure and are therefore declarations, not consequences: ideal/resistive
#       MHD (LA4_15..29), the Ayrton and Elenbaas-Heller arc column, the
#       Meek-Raether constant K, lightning engineering return-stroke models
#       (Heidler, TL/MTLE), striking-distance fits, Braginskii channel
#       expansion coefficient, the DBM branching exponent eta_DBM, the
#       Troyon/Greenwald limits, Knight's relation and Chapman-layer shape.
# =====================================================================
def _expand_langmuir_parameters():
    eqs = {}
    e, m_e, k_B, eps0, h, hbar, c = sp.symbols('e m_e k_B epsilon_0 h hbar c', positive=True)

    # LA1 plasma parameters and "is it a plasma" (lambda_D = F13_10, omega_p = F13_9)
    N_D, n_e, lam_D, Lam, lnLam, b_0, T_e = sp.symbols('N_D n_e lambda_D Lambda ln\\Lambda b_0 T_e', positive=True)
    nu_ei, n_i, Z_i, nu_en, n_g, sig_en, vbar_e, lam_e = sp.symbols(
        'nu_{ei} n_i Z nu_{en} n_g sigma_{en} \\bar{v}_e lambda_e', positive=True)
    x_ion, L_p, om_pe, tau_n, Gam_c, a_ws, f_pe = sp.symbols('x_i L omega_{pe} tau_n Gamma_c a_{ws} f_{pe}', positive=True)
    eqs['eq_LA1_1'] = sp.Eq(N_D, sp.Rational(4, 3)*sp.pi*n_e*lam_D**3)  # number of particles in a Debye sphere, lambda_D from F13_10 (Chen 1.6)
    eqs['eq_LA1_2'] = sp.Eq(Lam, 12*sp.pi*n_e*lam_D**3)  # plasma parameter Lambda = 9 N_D = lambda_D/b_0 (Spitzer "Physics of Fully Ionized Gases" 5.2; Bittencourt 1.5)
    eqs['eq_LA1_3'] = sp.Eq(b_0, e**2/(12*sp.pi*eps0*k_B*T_e))  # 90-degree impact parameter at m v^2 = 3 k_B T_e (Boyd-Sanderson 1.4)
    eqs['eq_LA1_4'] = sp.Eq(lnLam, sp.log(lam_D/b_0))  # Coulomb logarithm, equals ln(12 pi n_e lambda_D^3) (Boyd-Sanderson 1.4; NRL Formulary)
    # CAVEAT: Z=1 electron-ion momentum-transfer rate (Braginskii 1/tau_e); for T_e > ~10 Z^2 eV quantum b_min replaces b_0 in lnLambda
    eqs['eq_LA1_5'] = sp.Eq(nu_ei, sp.sqrt(2)*n_i*Z_i**2*e**4*lnLam/(12*sp.pi**sp.Rational(3, 2)*eps0**2*sp.sqrt(m_e)*(k_B*T_e)**sp.Rational(3, 2)))  # electron-ion collision frequency 1/tau_e (Braginskii 1965, SI; Goldston-Rutherford 11)
    eqs['eq_LA1_6'] = sp.Eq(vbar_e, sp.sqrt(8*k_B*T_e/(sp.pi*m_e)))  # mean electron speed of a Maxwellian (Lieberman-Lichtenberg 2.4)
    eqs['eq_LA1_7'] = sp.Eq(nu_en, n_g*sig_en*vbar_e)  # electron-neutral momentum-transfer collision frequency, sigma_en declared (Lieberman-Lichtenberg 3.1)
    eqs['eq_LA1_8'] = sp.Eq(lam_e, 1/(n_g*sig_en))  # electron mean free path in a neutral gas (no sqrt 2: v_e >> v_g; cf BO5_1) (Raizer 2.1)
    eqs['eq_LA1_9'] = sp.Eq(x_ion, n_i/(n_i + n_g))  # ionization fraction (Chen 1.1; Raizer 1.1)
    # CAVEAT: the three criteria are strong inequalities (<<, >>); sympy has no '<<', so the bare inequality stands for it
    eqs['eq_LA1_10'] = sp.StrictLessThan(lam_D, L_p)  # quasineutrality: lambda_D << L, plasma criterion (1) (Chen 1.6)
    eqs['eq_LA1_11'] = sp.StrictGreaterThan(N_D, 1)  # collective behaviour / ideal plasma: N_D >> 1, criterion (2) (Chen 1.6)
    eqs['eq_LA1_12'] = sp.StrictGreaterThan(om_pe*tau_n, 1)  # electrostatic not collisional dynamics: omega_p tau_n > 1, criterion (3) (Chen 1.6)
    eqs['eq_LA1_13'] = sp.Eq(a_ws, (3/(4*sp.pi*n_e))**sp.Rational(1, 3))  # Wigner-Seitz interparticle spacing (Ichimaru "Statistical Plasma Physics" 1.1)
    eqs['eq_LA1_14'] = sp.Eq(Gam_c, e**2/(4*sp.pi*eps0*a_ws*k_B*T_e))  # Coulomb coupling parameter; Gamma << 1 weakly coupled (ideal), Gamma > 1 strongly coupled (Ichimaru 1.1)
    eqs['eq_LA1_15'] = sp.Eq(f_pe, sp.sqrt(n_e*e**2/(eps0*m_e))/(2*sp.pi))  # electron plasma frequency in Hz, omega_pe of F13_9 with z=1, m=m_e (Chen 4.3)

    # LA2 equilibrium ionization (Saha, Boltzmann, LTE)
    n_z, n_z1, U_z, U_z1, chi_z, dchi_z, T, n_k, g_k, E_k, n_j, g_j, E_j = sp.symbols(
        'n_z n_{z+1} U_z U_{z+1} chi_z Delta\\chi_z T n_k g_k E_k n_j g_j E_j', positive=True)
    z_s, k_i, n_zk, dE_ul, p_g, x_s, n_s_z = sp.symbols('z k n_{z;k} Delta\\!E p x S_{Saha}', positive=True)
    kk = sp.Symbol('k', integer=True)
    s_i, z_i = sp.symbols('s z_i', integer=True)
    n_sz = sp.Function('n_{s,z}')
    eqs['eq_LA2_1'] = sp.Eq(n_z1*n_e/n_z, 2*(U_z1/U_z)*(2*sp.pi*m_e*k_B*T/h**2)**sp.Rational(3, 2)*sp.exp(-(chi_z - dchi_z)/(k_B*T)))  # Saha equation with partition functions and lowered ionization energy; U_z declared (Mitchner-Kruger 2.9; Griem "Principles of Plasma Spectroscopy" 6.3)
    eqs['eq_LA2_2'] = sp.Eq(U_z, sp.Sum(g_k*sp.exp(-E_k/(k_B*T)), (kk, 0, sp.Symbol('k_max'))))  # internal partition function, level sum truncated at k_max consistent with Delta chi (Griem 6.3)
    # CAVEAT: Debye-Huckel lowering, valid for Gamma_c << 1 (LA1_14); stronger coupling needs Stewart-Pyatt
    eqs['eq_LA2_3'] = sp.Eq(dchi_z, (z_s + 1)*e**2/(4*sp.pi*eps0*lam_D))  # Debye-Huckel ionization-potential lowering for the z -> z+1 stage (Griem 6.4; Mitchner-Kruger 2.10)
    eqs['eq_LA2_4'] = sp.Eq(n_k/n_z, (g_k/U_z)*sp.exp(-E_k/(k_B*T)))  # Boltzmann excitation population of level k within stage z (Griem 6.2)
    eqs['eq_LA2_5'] = sp.Eq(n_k/n_j, (g_k/g_j)*sp.exp(-(E_k - E_j)/(k_B*T)))  # two-level Boltzmann ratio (Griem 6.2)
    eqs['eq_LA2_6'] = sp.Eq(n_zk, n_e*n_z1*(g_k/(2*U_z1))*(h**2/(2*sp.pi*m_e*k_B*T))**sp.Rational(3, 2)*sp.exp((chi_z - dchi_z - E_k)/(k_B*T)))  # Saha-Boltzmann population of level k of stage z from the next stage (Griem 6.3)
    # CAVEAT: necessary, not sufficient, condition for LTE; T in K, Delta E (largest gap) in eV, n_e in m^-3; coefficient 1.6e18 m^-3 is the McWhirter estimate
    eqs['eq_LA2_7'] = sp.GreaterThan(n_e, sp.Float('1.6e18')*sp.sqrt(T)*dE_ul**3)  # McWhirter criterion for collision-dominated (LTE) populations (McWhirter 1965; Fujimoto "Plasma Spectroscopy" 4)
    eqs['eq_LA2_8'] = sp.Eq(x_s**2/(1 - x_s**2), n_s_z*k_B*T/p_g)  # single-stage Saha for a weakly/strongly ionized gas at pressure p: x = n_i/(n_0+n_i), n_s_z = RHS of LA2_1 (Raizer 4.5 thermal ionization; Mitchner-Kruger 2.9)
    eqs['eq_LA2_9'] = sp.Eq(n_s_z, 2*(U_z1/U_z)*(2*sp.pi*m_e*k_B*T/h**2)**sp.Rational(3, 2)*sp.exp(-(chi_z - dchi_z)/(k_B*T)))  # Saha function S(T) used in LA2_8 (Mitchner-Kruger 2.9)
    eqs['eq_LA2_10'] = sp.Eq(n_e, sp.Sum(sp.Sum(z_i*n_sz(s_i, z_i), (z_i, 1, sp.Symbol('Z_s'))), (s_i, 1, sp.Symbol('N_s'))))  # charge neutrality closing the multi-species Saha set: in air/metal vapour mixtures the lowest-chi species (Na, K, Cu, NO) set n_e (Mitchner-Kruger 2.9; Boulos-Fauchais-Pfender "Thermal Plasmas" 5)
    return eqs


globals().update(_expand_langmuir_parameters())


def _expand_langmuir_kinetic():
    eqs = {}
    e, m_e, k_B, eps0, c = sp.symbols('e m_e k_B epsilon_0 c', positive=True)
    nabla_v = sp.Function('nabla_v')  # velocity-space gradient/divergence

    # LA3 kinetic description
    f_s = sp.Function('f_s')(t, x, sp.Symbol('v'))
    v_v, E_v, B_v, q_s, m_s = sp.symbols('\\mathbf{v} \\mathbf{E} \\mathbf{B} q_s m_s')
    Cc, H_R, G_R, Gam_ss, lnLam, m_sp, n_e, T_e = sp.symbols("C_s H_s G_s Gamma_{ss'} ln\\Lambda m_{s'} n_e T_e", positive=True)
    q_sp = sp.Symbol("q_{s'}")
    f_sp = sp.Function("f_{s'}")(sp.Symbol("v'"))
    vp = sp.Symbol("\\mathbf{v}'")
    eps_, f_eps, f0, f1, nu_m, E_f, M_g, lam_f, C_D, eps_mean = sp.symbols(
        'epsilon f(epsilon) f_0 f_1 nu_m E M lambda C_D \\langle\\epsilon\\rangle', positive=True)
    vv, theta, T_g, S_inel, k_r, sig_r = sp.symbols('v theta T_g S_{inel} k sigma(epsilon)', positive=True)
    f0v = sp.Function('f_0')(t, vv)
    nu_v = sp.Function('nu_m')(vv)
    gam_L = sp.Symbol('gamma_L', real=True)
    eta_S, alpha_0, tau_e, om, kw, lam_D, om_pe, gam_Lx, fhat, c_s, M_i, T_i, gam_e, gam_i, eps_L = sp.symbols(
        'eta_{Sp} alpha_0(Z) tau_e omega k lambda_D omega_{pe} gamma_L \\hat{f} c_s M T_i gamma_e gamma_i epsilon_L', positive=True)
    phi, q_t, r_r, Z_i, E_Dr, E_CH = sp.symbols('phi q r Z E_D E_{c;CH}', positive=True)
    fhat_s = sp.Function('\\hat{f}_s')(vv)
    ss = sp.Symbol('s', integer=True)
    om_ps = sp.Function('omega_{p,s}')(ss)

    eqs['eq_LA3_1'] = sp.Eq(sp.Derivative(f_s, t), -v_v*nabla(f_s) - (q_s/m_s)*(E_v + cross(v_v, B_v))*nabla_v(f_s))  # Vlasov equation; nabla = spatial gradient, nabla_v = velocity gradient, products are dot products (Chen 7.2; Bittencourt 5.3)
    eqs['eq_LA3_2'] = sp.Eq(sp.Derivative(f_s, t), -v_v*nabla(f_s) - (q_s/m_s)*(E_v + cross(v_v, B_v))*nabla_v(f_s) + sp.Sum(sp.Function("C_{ss'}")(f_s, f_sp), (sp.Symbol("s'", integer=True), 1, sp.Symbol('N_s'))))  # Boltzmann equation with Lorentz force and bilinear collision operators summed over partner species; neutral-collision C = BO1_2 (Bittencourt 5.4; Lieberman-Lichtenberg 18.1)
    eqs['eq_LA3_3'] = sp.Eq(sp.Function("C_{ss'}")(f_s, f_sp), Gam_ss*(-nabla_v(f_s*nabla_v(H_R)) + sp.Rational(1, 2)*nabla_v(nabla_v(f_s*nabla_v(nabla_v(G_R))))))  # Fokker-Planck (Landau) collision operator in Rosenbluth form; nabla_v nabla_v : tensor contraction (Rosenbluth-MacDonald-Judd 1957; Boyd-Sanderson 8.4)
    eqs['eq_LA3_4'] = sp.Eq(Gam_ss, q_s**2*q_sp**2*lnLam/(4*sp.pi*eps0**2*m_s**2))  # Fokker-Planck coefficient, SI (Boyd-Sanderson 8.4)
    eqs['eq_LA3_5'] = sp.Eq(H_R, (1 + m_s/m_sp)*sp.Integral(f_sp/sp.Abs(v_v - vp), vp))  # Rosenbluth potential H (Rosenbluth-MacDonald-Judd 1957)
    eqs['eq_LA3_6'] = sp.Eq(G_R, sp.Integral(f_sp*sp.Abs(v_v - vp), vp))  # Rosenbluth potential G (Rosenbluth-MacDonald-Judd 1957)
    eqs['eq_LA3_7'] = sp.Eq(f_eps, 2*sp.sqrt(eps_/sp.pi)*(k_B*T_e)**sp.Rational(-3, 2)*sp.exp(-eps_/(k_B*T_e)))  # Maxwellian EEDF normalised to int f d(eps) = 1 (Lieberman-Lichtenberg 2.4)
    # CAVEAT: Druyvesteyn assumes elastic losses only, constant mean free path lambda, T_g -> 0, DC field; C_D is the normalisation
    eqs['eq_LA3_8'] = sp.Eq(f_eps, C_D*sp.sqrt(eps_)*sp.exp(-3*m_e*eps_**2/(M_g*(e*E_f*lam_f)**2)))  # Druyvesteyn EEDF (Druyvesteyn 1930; Raizer 2.4)
    eqs['eq_LA3_9'] = sp.Eq(k_r, sp.Integral(sig_r*sp.sqrt(2*eps_/m_e)*f_eps, (eps_, 0, sp.oo)))  # rate coefficient k = <sigma v> over the EEDF; sigma(eps) declared (Lieberman-Lichtenberg 3.5)
    eqs['eq_LA3_10'] = sp.Eq(sp.Function('f')(vv, theta), f0v + f1*sp.cos(theta))  # two-term (Lorentz) expansion of the EEDF about the field direction (Raizer 2.3; Hagelaar-Pitchford 2005)
    # CAVEAT: DC or low-frequency field (omega << nu_m), electron charge -e, theta measured from E
    eqs['eq_LA3_11'] = sp.Eq(f1, (e*E_f/(m_e*nu_v))*sp.Derivative(f0v, vv))  # anisotropic part in steady state (Raizer 2.3; Lieberman-Lichtenberg 18.3)
    eqs['eq_LA3_12'] = sp.Eq(sp.Derivative(f0v, t), (e*E_f/m_e)**2/(3*vv**2)*sp.Derivative(vv**2/nu_v*sp.Derivative(f0v, vv), vv) + (m_e/M_g)/vv**2*sp.Derivative(nu_v*vv**3*(f0v + k_B*T_g/(m_e*vv)*sp.Derivative(f0v, vv)), vv) + S_inel)  # isotropic-part equation: field heating + elastic recoil + inelastic term S_inel (Lieberman-Lichtenberg 18.3; Raizer 2.4)
    # CAVEAT: Chen's order-of-magnitude Spitzer form; the Braginskii coefficient is in LA3_14
    eqs['eq_LA3_13'] = sp.Eq(eta_S, sp.pi*e**2*sp.sqrt(m_e)*lnLam/((4*sp.pi*eps0)**2*(k_B*T_e)**sp.Rational(3, 2)))  # Spitzer resistivity estimate (Chen 5.76)
    eqs['eq_LA3_14'] = sp.Eq(eta_S, alpha_0*m_e/(n_e*e**2*tau_e))  # parallel Spitzer resistivity, alpha_0(1) = 0.5129, tau_e = 1/nu_ei of LA1_5 (Braginskii 1965; Goldston-Rutherford 11)
    eqs['eq_LA3_15'] = sp.Eq(om**2, om_pe**2 + 3*kw**2*k_B*T_e/m_e)  # Bohm-Gross dispersion of Langmuir waves (Chen 4.30)
    eqs['eq_LA3_16'] = sp.Eq(gam_L, sp.pi/2*om_pe**3/kw**2*sp.Subs(sp.Derivative(fhat, vv), vv, om/kw))  # Landau growth/damping rate, fhat = 1D distribution normalised to 1 (Chen 7.10; Bittencourt 18)
    # CAVEAT: Maxwellian, k lambda_D << 1 asymptotic; negative = damping
    eqs['eq_LA3_17'] = sp.Eq(gam_L, -sp.sqrt(sp.pi/8)*om_pe/(kw*lam_D)**3*sp.exp(-1/(2*kw**2*lam_D**2) - sp.Rational(3, 2)))  # Landau damping of Langmuir waves in a Maxwellian (Krall-Trivelpiece 8.6; Bittencourt 18)
    eqs['eq_LA3_18'] = sp.Eq(eps_L, 1 + sp.Sum(om_ps**2/kw**2*sp.Integral(kw*sp.Derivative(fhat_s, vv)/(om - kw*vv), (vv, -sp.oo, sp.oo)), (ss, 1, sp.Symbol('N_s'))))  # longitudinal Vlasov-Poisson dielectric function; waves are the roots eps_L = 0, Landau contour below the pole (Chen 7.5; Krall-Trivelpiece 8)
    eqs['eq_LA3_19'] = sp.Eq(c_s, sp.sqrt((gam_e*k_B*T_e + gam_i*k_B*T_i)/M_i))  # ion-acoustic speed (Chen 4.41)
    eqs['eq_LA3_20'] = sp.Eq(om/kw, sp.sqrt(k_B*T_e/M_i)/sp.sqrt(1 + kw**2*lam_D**2))  # ion-acoustic dispersion with Debye correction, T_i = 0 (Chen 4.48)
    eqs['eq_LA3_21'] = sp.Eq(nabla(nabla(phi)), phi/lam_D**2)  # linearised Poisson-Boltzmann (div grad), EMERGENT from Poisson + Boltzmann electrons (Chen 1.4)
    eqs['eq_LA3_22'] = sp.Eq(phi, q_t/(4*sp.pi*eps0*r_r)*sp.exp(-r_r/lam_D))  # Debye-shielded potential, solution of LA3_21 (Chen 1.4)
    eqs['eq_LA3_23'] = sp.Eq(E_Dr, n_e*Z_i*e**3*lnLam/(4*sp.pi*eps0**2*k_B*T_e))  # Dreicer field: above it thermal electrons run away (Dreicer 1959; Helander-Sigmar 2002)
    eqs['eq_LA3_24'] = sp.Eq(E_CH, n_e*e**3*lnLam/(4*sp.pi*eps0**2*m_e*c**2))  # Connor-Hastie critical field for relativistic runaway (Connor-Hastie 1975)
    return eqs


globals().update(_expand_langmuir_kinetic())


def _expand_langmuir_fluid_mhd():
    eqs = {}
    e, m_e, k_B, eps0, mu0 = sp.symbols('e m_e k_B epsilon_0 mu_0', positive=True)

    # LA4 two-fluid, drift-diffusion with sources, ambipolar, generalized Ohm, MHD
    n_s, u_s, S_s, q_s, m_s, p_s, nu_sn, u_n, E_v, B_v = sp.symbols('n_s \\mathbf{u}_s S_s q_s m_s p_s nu_{sn} \\mathbf{u}_n \\mathbf{E} \\mathbf{B}')
    W_e, Q_e, J_e, P_el, P_inel, n_e, T_e, T_g, M_g, nu_m, n_g = sp.symbols('w_e \\mathbf{Q}_e \\mathbf{J}_e P_{el} P_{inel} n_e T_e T_g M nu_m n_g')
    eps_j, k_j = sp.Function('epsilon')(sp.Symbol('j')), sp.Function('k')(sp.Symbol('j'))
    jj = sp.Symbol('j', integer=True)
    n_p, n_n, G_e, G_p, G_n, D_e, mu_e, mu_p, mu_n, alpha, eta_a, beta_ep, beta_np, k_det, S_ph, phi = sp.symbols(
        'n_+ n_- \\mathbf{\\Gamma}_e \\mathbf{\\Gamma}_+ \\mathbf{\\Gamma}_- D_e mu_e mu_+ mu_- alpha eta beta_{e+} beta_{+-} k_{det} S_{ph} phi')
    E_mag, D_a, D_i, mu_i, T_i, E_a, n = sp.symbols('|\\mathbf{E}| D_a D_i mu_i T_i \\mathbf{E}_a n')
    D_s, mu_ss, T_s = sp.symbols('D_s mu_s T_s')
    u, J, eta, p_e, rho, p, gam = sp.symbols('\\mathbf{u} \\mathbf{J} eta p_e rho p gamma')
    Phi_B, v_A, c_s, v_f, v_sl, th, beta, S_L, L, v_in, dl, p_B, tau_R = sp.symbols(
        'Phi_B v_A c_s v_f v_{sl} theta beta S L v_{in} delta p_B tau_R')
    Bm = sp.Symbol('B', positive=True)

    eqs['eq_LA4_1'] = sp.Eq(sp.Derivative(n_s, t), -nabla(n_s*u_s) + S_s)  # species continuity (div), S_s = ionization - recombination - attachment + detachment (Chen 3.3; Bittencourt 8)
    eqs['eq_LA4_2'] = sp.Eq(sp.Derivative(u_s, t), -u_s*nabla(u_s) + (q_s/m_s)*(E_v + cross(u_s, B_v)) - nabla(p_s)/(m_s*n_s) - nu_sn*(u_s - u_n))  # two-fluid momentum with neutral drag (u.grad u, grad p) (Chen 3.3; Lieberman-Lichtenberg 2.3)
    eqs['eq_LA4_3'] = sp.Eq(sp.Derivative(sp.Rational(3, 2)*n_e*k_B*T_e, t), -nabla(Q_e) + J_e*E_v - P_el - P_inel)  # electron energy equation, Q_e = 5/2 n k T u + heat flux (div) (Lieberman-Lichtenberg 2.3; Hagelaar-Pitchford 2005)
    eqs['eq_LA4_4'] = sp.Eq(P_el, 3*(m_e/M_g)*n_e*nu_m*k_B*(T_e - T_g))  # elastic electron energy loss to the gas (Lieberman-Lichtenberg 2.3)
    eqs['eq_LA4_5'] = sp.Eq(P_inel, sp.Sum(eps_j*k_j*n_e*n_g, (jj, 1, sp.Symbol('N_r'))))  # inelastic loss, eps_j threshold and k_j of LA3_9 (Lieberman-Lichtenberg 3.5)
    # LA4_6..9: three-species streamer/corona fluid model; extends F13_6/F13_7 with attachment, detachment, ion-ion recombination, photoionization
    eqs['eq_LA4_6'] = sp.Eq(sp.Derivative(n_e, t), -nabla(G_e) + (alpha - eta_a)*mu_e*E_mag*n_e - beta_ep*n_e*n_p + k_det*n_n*n_g + S_ph)  # electrons (Morrow-Lowke 1997; Bourdon et al. 2007)
    eqs['eq_LA4_7'] = sp.Eq(sp.Derivative(n_p, t), -nabla(G_p) + alpha*mu_e*E_mag*n_e - beta_ep*n_e*n_p - beta_np*n_n*n_p + S_ph)  # positive ions (Morrow-Lowke 1997)
    eqs['eq_LA4_8'] = sp.Eq(sp.Derivative(n_n, t), -nabla(G_n) + eta_a*mu_e*E_mag*n_e - k_det*n_n*n_g - beta_np*n_n*n_p)  # negative ions (Morrow-Lowke 1997)
    eqs['eq_LA4_9'] = sp.Eq(nabla(nabla(phi)), -e*(n_p - n_e - n_n)/eps0)  # Poisson closure (div grad); space charge is what makes the streamer (Raizer 12.4)
    eqs['eq_LA4_10'] = sp.Eq(D_s, mu_ss*k_B*T_s/e)  # Einstein relation, singly charged (Chen 5.1)
    eqs['eq_LA4_11'] = sp.Eq(D_a, (mu_i*D_e + mu_e*D_i)/(mu_i + mu_e))  # ambipolar diffusion coefficient (Chen 5.2)
    eqs['eq_LA4_12'] = sp.Eq(D_a, D_i*(1 + T_e/T_i))  # ambipolar limit mu_e >> mu_i (Chen 5.2)
    eqs['eq_LA4_13'] = sp.Eq(E_a, (D_i - D_e)/(mu_i + mu_e)*nabla(n)/n)  # ambipolar field (grad), EMERGENT from quasineutral two-fluid flux balance (Chen 5.2)
    eqs['eq_LA4_14'] = sp.Eq(E_v, -cross(u, B_v) + eta*J + cross(J, B_v)/(n*e) - nabla(p_e)/(n*e) + m_e/(n*e**2)*sp.Derivative(J, t))  # generalized Ohm's law: resistive, Hall, electron pressure, electron inertia (Goldston-Rutherford 7; Bittencourt 12)
    # LA4_15..: MHD is a REDUCED model (declared), valid for L >> r_L,i and slow compared with omega_ci
    eqs['eq_LA4_15'] = sp.Eq(sp.Derivative(rho, t), -nabla(rho*u))  # MHD continuity (div) (Goldston-Rutherford 7; Chen 5.9)
    eqs['eq_LA4_16'] = sp.Eq(sp.Derivative(u, t), -u*nabla(u) + (cross(J, B_v) - nabla(p))/rho)  # MHD momentum with J x B (Goldston-Rutherford 7)
    eqs['eq_LA4_17'] = sp.Eq(J, nabla(B_v)/mu0)  # pre-Maxwell Ampere law, nabla = curl here (Goldston-Rutherford 7)
    eqs['eq_LA4_18'] = sp.Eq(sp.Derivative(B_v, t), nabla(cross(u, B_v)) + (eta/mu0)*nabla(nabla(B_v)))  # resistive induction equation, curl(u x B) + magnetic diffusion (vector Laplacian) (Goldston-Rutherford 7; Chen 5.9)
    eqs['eq_LA4_19'] = sp.Eq(sp.Derivative(Phi_B, t), 0)  # Alfven frozen-in flux through any comoving surface, eta -> 0 (Goldston-Rutherford 8)
    eqs['eq_LA4_20'] = sp.Eq(sp.Derivative(p, t), -u*nabla(p) - gam*p*nabla(u))  # adiabatic MHD energy closure, d/dt(p rho^-gamma) = 0 (grad, div) (Goldston-Rutherford 7)
    eqs['eq_LA4_21'] = sp.Eq(v_A, Bm/sp.sqrt(mu0*rho))  # Alfven speed (Chen 4.18)
    eqs['eq_LA4_22'] = sp.Eq(v_f**2, sp.Rational(1, 2)*((c_s**2 + v_A**2) + sp.sqrt((c_s**2 + v_A**2)**2 - 4*c_s**2*v_A**2*sp.cos(th)**2)))  # fast magnetosonic speed at angle theta to B (Goldston-Rutherford 20; Boyd-Sanderson 4)
    eqs['eq_LA4_23'] = sp.Eq(v_sl**2, sp.Rational(1, 2)*((c_s**2 + v_A**2) - sp.sqrt((c_s**2 + v_A**2)**2 - 4*c_s**2*v_A**2*sp.cos(th)**2)))  # slow magnetosonic speed (Goldston-Rutherford 20)
    eqs['eq_LA4_24'] = sp.Eq(beta, 2*mu0*p/Bm**2)  # plasma beta (Chen 3.5)
    eqs['eq_LA4_25'] = sp.Eq(p_B, Bm**2/(2*mu0))  # magnetic pressure (Chen 5.9)
    eqs['eq_LA4_26'] = sp.Eq(S_L, mu0*L*v_A/eta)  # Lundquist number (Goldston-Rutherford 20; Priest-Forbes 1)
    eqs['eq_LA4_27'] = sp.Eq(tau_R, mu0*L**2/eta)  # resistive diffusion time (Chen 5.9)
    eqs['eq_LA4_28'] = sp.Eq(v_in/v_A, S_L**sp.Rational(-1, 2))  # Sweet-Parker reconnection inflow rate (Priest-Forbes 4.2)
    eqs['eq_LA4_29'] = sp.Eq(dl/L, S_L**sp.Rational(-1, 2))  # Sweet-Parker current-sheet aspect ratio (Priest-Forbes 4.2)
    return eqs


globals().update(_expand_langmuir_fluid_mhd())


def _expand_langmuir_breakdown():
    eqs = {}
    e, m_e, k_B, eps0 = sp.symbols('e m_e k_B epsilon_0', positive=True)

    # LA5 gas breakdown and air plasma (Paschen: F15_10..12; Townsend criterion: F15_13; Townsend alpha: F13_5)
    E_N, E, N, Td, alpha, eta, alpha_eff, eta_2, eta_3 = sp.symbols('E/N E N Td alpha eta alpha_{eff} eta_2 eta_3', positive=True)
    F_a, F_e2, F_e3 = sp.Function('F_alpha'), sp.Function('F_{eta2}'), sp.Function('F_{eta3}')
    E_N_cr, E_cr, p, T = sp.symbols('(E/N)_{cr} E_{cr} p T', positive=True)
    n_av, n_0, xx, x_c, K_MR, r_D, D_e, v_d, E_sc, N_e = sp.symbols('n n_0 x x_c K_{MR} r_D D_e v_d E_{sc} N_e', positive=True)
    v_str, r_m, E_m, n_ch, n_seed, q_h, E_st = sp.symbols('v_{str} r_m E_m n_{ch} n_{seed} q_h E_{st}', positive=True)
    nu_i = sp.Function('nu_i')
    S_ph, xi, nu_u, p_q, chi_min, chi_max, p_O2, R_ = sp.symbols('S_{ph} xi nu_u/nu_i p_q chi_{min} chi_{max} p_{O2} R', positive=True)
    S_i = sp.Function('S_i')(sp.Symbol("\\mathbf{r}'"))
    Rr = sp.Symbol("|\\mathbf{r}-\\mathbf{r}'|", positive=True)
    g_Z = sp.Function('g_Z')
    q_L, v_L, i_L, r_ch, rho_g, c_p, T_ch, E_ch, T_crit = sp.symbols('q_L v_L i_L r_{ch} rho_g c_p T_{ch} E_{ch} T_{crit}', positive=True)
    t_lag, t_s, t_f, beta_e, P_b, N_t, N_0 = sp.symbols('t_{lag} t_s t_f \\dot{N}_e P_b N(t) N_0', positive=True)

    eqs['eq_LA5_1'] = sp.Eq(E_N, E/N)  # reduced electric field, the similarity variable of swarm data (Raizer 2.1; Lieberman-Lichtenberg 3)
    eqs['eq_LA5_2'] = sp.Eq(Td, sp.Float('1e-21'))  # 1 townsend = 1e-21 V m^2 (defining; Raizer 2)
    eqs['eq_LA5_3'] = sp.Eq(alpha/N, F_a(E_N))  # reduced first Townsend coefficient, declared data (BOLSIG+ / swarm tables); F13_5 is its empirical fit (Raizer 4.1)
    eqs['eq_LA5_4'] = sp.Eq(eta, N*F_e2(E_N) + N**2*F_e3(E_N))  # attachment coefficient: two-body dissociative (O2 -> O- + O) plus three-body (O2 + M -> O2- + M), declared data (Raizer 6.1; Bazelyan-Raizer 1)
    eqs['eq_LA5_5'] = sp.Eq(alpha_eff, alpha - eta)  # effective ionization coefficient (Raizer 12.1)
    eqs['eq_LA5_6'] = sp.Eq(F_a(E_N_cr), F_e2(E_N_cr) + N*F_e3(E_N_cr))  # critical reduced field defined by alpha = eta; ~120 Td for dry air (Raizer 12.1; Bazelyan-Raizer 1)
    eqs['eq_LA5_7'] = sp.Eq(E_cr, E_N_cr*p/(k_B*T))  # critical breakdown field, N = p/(k_B T); ~3 MV/m at 1 atm, 293 K (Raizer 12.1)
    eqs['eq_LA5_8'] = sp.Eq(n_av, n_0*sp.exp(sp.Integral(alpha_eff, (xx, 0, x_c))))  # electron avalanche growth in a nonuniform field; n_0 exp(alpha_eff x) if uniform (Raizer 12.2)
    eqs['eq_LA5_9'] = sp.Eq(r_D, sp.sqrt(4*D_e*x_c/v_d))  # diffusive radius of the avalanche head after drift distance x (Raizer 12.2)
    eqs['eq_LA5_10'] = sp.Eq(E_sc, e*N_e/(4*sp.pi*eps0*r_D**2))  # space-charge field of the avalanche head (Raizer 12.3)
    # CAVEAT: K_MR ~ 18-20 (N_e ~ 1e8) is an empirical constant, not derived
    eqs['eq_LA5_11'] = sp.GreaterThan(sp.Integral(alpha_eff, (xx, 0, x_c)), K_MR)  # Meek-Raether avalanche-to-streamer criterion (Raether 1964; Raizer 12.3)
    eqs['eq_LA5_12'] = sp.Eq(E_m, q_h/(4*sp.pi*eps0*r_m**2))  # streamer head field from head charge q_h, radius r_m (Raizer 12.5; Bazelyan-Raizer 2)
    # CAVEAT: ionization-wave estimate, not exact; nu_i = alpha_eff v_d evaluated at the head field
    eqs['eq_LA5_13'] = sp.Eq(v_str, nu_i(E_m)*r_m/sp.log(n_ch/n_seed))  # streamer velocity scaling (Raizer 12.5; Bazelyan-Raizer 2.2)
    # CAVEAT: empirical, E_st ~ 4.5-5 kV/cm positive, ~10-12.5 kV/cm negative, in atmospheric air
    eqs['eq_LA5_14'] = sp.GreaterThan(E, E_st)  # mean field required for streamer propagation (Bazelyan-Raizer 2.4)
    # CAVEAT: Zheleznyak constants for air: xi*nu_u/nu_i ~ 0.06, chi_min ~ 0.035, chi_max ~ 2 (Torr cm)^-1, p_q ~ 30 Torr
    eqs['eq_LA5_15'] = sp.Eq(S_ph, xi*p_q/(p + p_q)*sp.Integral(S_i*g_Z(Rr)/(4*sp.pi*Rr**2), sp.Symbol("V'")))  # photoionization source seeding the streamer: N2 (b,c) UV ionizing O2 (Zheleznyak et al. 1982; Bourdon et al. 2007)
    eqs['eq_LA5_16'] = sp.Eq(g_Z(R_), (sp.exp(-chi_min*p_O2*R_) - sp.exp(-chi_max*p_O2*R_))/(R_*sp.log(chi_max/chi_min)))  # Zheleznyak absorption function (Zheleznyak et al. 1982)
    eqs['eq_LA5_17'] = sp.Eq(i_L, q_L*v_L)  # leader current from line charge and speed (Bazelyan-Raizer 4.1)
    eqs['eq_LA5_18'] = sp.Eq(sp.pi*r_ch**2*rho_g*c_p*sp.Derivative(T_ch, t), i_L*E_ch)  # isobaric Joule heating of the stem per unit length (Bazelyan-Raizer 4.2)
    # CAVEAT: T_crit ~ 1500-2000 K (thermal detachment of O-/O2-, V-T release) is a declared threshold
    eqs['eq_LA5_19'] = sp.GreaterThan(T_ch, T_crit)  # streamer-to-leader transition criterion (Bazelyan-Raizer 4.2; Gallimberti 1979)
    eqs['eq_LA5_20'] = sp.Eq(t_lag, t_s + t_f)  # breakdown time lag = statistical + formative (Raizer 12.1; Meek-Craggs 'Electrical Breakdown of Gases')
    eqs['eq_LA5_21'] = sp.Eq(t_s, 1/(beta_e*P_b))  # mean statistical lag: rate of seed-electron appearance x probability it leads to breakdown (Raizer 7; Meek-Craggs)
    eqs['eq_LA5_22'] = sp.Eq(N_t, N_0*sp.exp(-t/t_s))  # Laue plot: fraction of gaps not yet broken down (Laue 1925; Raizer 7)
    return eqs


globals().update(_expand_langmuir_breakdown())


def _expand_langmuir_discharges():
    eqs = {}
    e, m_e, k_B, eps0, mu0, h = sp.symbols('e m_e k_B epsilon_0 mu_0 h', positive=True)

    # LA6 discharge regimes, sheaths, electrodes, RF, DBD, corona
    I, I_0, alpha, d, gam, j_n, K_j, p = sp.symbols('I I_0 alpha d gamma j_n K_j p', positive=True)
    V_n, mu_p, d_n, V_min, nu_iz, D_a, j01, R_t = sp.symbols('V_n mu_+ d_n V_{min} nu_{iz} D_a j_{01} R', positive=True)
    E_col, F_col, n_r, n_0, r = sp.symbols('E_{col} F_{col} n(r) n_0 r', positive=True)
    V_arc, A_a, B_a, C_a, D_a2 = sp.symbols('V_{arc} A_{Ay} B_{Ay} C_{Ay} D_{Ay}', positive=True)
    J_CL, M, V, u_s, u_B, T_e, Gam_i, n_s, V_f, V_p = sp.symbols('J_{CL} M V u_s u_B T_e Gamma_i n_s V_f V_p', real=True)
    I_e, I_es, n_e, vbar_e, A_p, I_is, gam_se, eps_iz, W = sp.symbols('I_e I_{es} n_e \\bar{v}_e A_p I_{is} gamma_{se} epsilon_{iz} W', positive=True)
    J_th, A_G, lam_R, T, dW, E_s = sp.symbols('J_{th} A_G lambda_R T Delta\\!W E_s', positive=True)
    J_FN, t_y, v_y = sp.symbols('J_{FN} t(y) v(y)', positive=True)
    kap = sp.Function('kappa')
    sig = sp.Function('sigma')
    Tr = sp.Function('T')(r)
    E_arc = sp.Symbol('E_{arc}', positive=True)
    C_sh, s_sh, A_el, I_rf, V_sh, Va_b, Aa_b, q_KM = sp.symbols('C_{sh} s A I_{rf} V_{sh} V_a/V_b A_b/A_a q_{KM}', positive=True)
    eps_p, sig_p = sp.symbols('epsilon_p sigma_p')  # complex-valued
    del_p, del_c, c, om, sig_dc, nu_m, om_pe, eps_px, sig_px, p_abs, E_0 = sp.symbols(
        'delta_p delta_c c omega sigma_{dc} nu_m omega_{pe} epsilon_p sigma_p p_{abs} |\\tilde{E}|', positive=True)
    P_M, f, C_d, C_g, U_b, U_0, V_g, V_a, Q_t = sp.symbols('P f C_d C_g U_b U_0 V_g V_a Q_t', positive=True)
    tau_mem, I_T, mu_i, V_0, R_o, r_0, E_tip, beta_f, E_0f = sp.symbols('tau_{mem} I/l mu_i V_0 R_o r_0 E_{tip} beta_f E_0', positive=True)
    R_b, dVdI = sp.symbols('R_b dV/dI', real=True)

    eqs['eq_LA6_1'] = sp.Eq(I, I_0*sp.exp(alpha*d)/(1 - gam*(sp.exp(alpha*d) - 1)))  # Townsend (dark) discharge current; denominator zero is F15_13 (Raizer 7.1)
    # CAVEAT: K_j = (j_n/p^2) is gas/cathode data (e.g. air-Fe ~ 2.5 uA cm^-2 Torr^-2)
    eqs['eq_LA6_2'] = sp.Eq(j_n, K_j*p**2)  # normal glow current-density similarity law j/p^2 = const, empirical (Raizer 8.3; von Engel)
    eqs['eq_LA6_3'] = sp.Eq(j_n/p**2, 4*eps0*(mu_p*p)*(1 + gam)*V_n**2/(p*d_n)**3)  # von Engel-Steenbeck cathode-fall current density, linear field in the fall (Raizer 8.3)
    eqs['eq_LA6_4'] = sp.Eq(V_n, V_min)  # normal cathode fall = Paschen minimum V_min of F15_12, (p d_n) = (p d)_min of F15_11 (Raizer 8.3)
    eqs['eq_LA6_5'] = sp.Eq(nu_iz, D_a*(j01/R_t)**2)  # Schottky positive column: ionization balances ambipolar wall loss, j01 = 2.405 (Raizer 8.5; Lieberman-Lichtenberg 5.2)
    eqs['eq_LA6_6'] = sp.Eq(n_r, n_0*sp.besselj(0, j01*r/R_t))  # Schottky radial profile of the column (Lieberman-Lichtenberg 5.2)
    eqs['eq_LA6_7'] = sp.Eq(E_col/p, sp.Function('F_{col}')(p*R_t))  # positive-column similarity E/p = F(pR), F declared per gas (Raizer 8.5; von Engel 'Ionized Gases')
    # CAVEAT: empirical; constants depend on electrode material and gas; valid for free-burning DC arcs above a few amperes
    eqs['eq_LA6_8'] = sp.Eq(V_arc, A_a + B_a*d + (C_a + D_a2*d)/I)  # Ayrton arc voltage-current law (Ayrton 1902; Raizer 10)
    eqs['eq_LA6_9'] = sp.Eq(sp.Derivative(r*kap(Tr)*sp.Derivative(Tr, r), r), -r*sig(Tr)*E_arc**2)  # Elenbaas-Heller arc-column energy balance, radiation neglected; kappa(T), sigma(T) declared (Raizer 10; Boulos-Fauchais-Pfender 8)
    eqs['eq_LA6_10'] = sp.Eq(J_CL, sp.Rational(4, 9)*eps0*sp.sqrt(2*e/M)*V**sp.Rational(3, 2)/d**2)  # Child-Langmuir space-charge-limited current (Lieberman-Lichtenberg 6.3; Chen 8.2)
    eqs['eq_LA6_11'] = sp.GreaterThan(u_s, u_B)  # Bohm sheath criterion (Lieberman-Lichtenberg 6.2; Chen 8.2)
    eqs['eq_LA6_12'] = sp.Eq(u_B, sp.sqrt(k_B*T_e/M))  # Bohm speed (Lieberman-Lichtenberg 6.2)
    eqs['eq_LA6_13'] = sp.Eq(Gam_i, n_s*u_B)  # ion flux to a wall, n_s ~ 0.61 n_0 at the sheath edge (Lieberman-Lichtenberg 6.2)
    eqs['eq_LA6_14'] = sp.Eq(V_f - V_p, -(k_B*T_e/(2*e))*sp.log(M/(2*sp.pi*m_e)))  # floating potential of a planar wall, Maxwellian electrons, cold ions (Lieberman-Lichtenberg 6.2)
    eqs['eq_LA6_15'] = sp.Eq(I_e, I_es*sp.exp(e*(V - V_p)/(k_B*T_e)))  # Langmuir probe electron-retardation region, V < V_p (Lieberman-Lichtenberg 6.6; Chen 8.3)
    eqs['eq_LA6_16'] = sp.Eq(I_es, sp.Rational(1, 4)*e*n_e*vbar_e*A_p)  # electron saturation current (Lieberman-Lichtenberg 6.6)
    eqs['eq_LA6_17'] = sp.Eq(I_is, sp.Float('0.61')*e*n_e*u_B*A_p)  # Bohm ion saturation current, 0.61 = exp(-1/2) (Lieberman-Lichtenberg 6.6; Chen 8.3)
    eqs['eq_LA6_18'] = sp.Eq(T_e, (e/k_B)/sp.Derivative(sp.log(I_e), V))  # electron temperature from the probe semilog slope (Lieberman-Lichtenberg 6.6)
    # CAVEAT: empirical potential-emission fit, energies in eV, clean metal surfaces; gamma_se is declared surface data
    eqs['eq_LA6_19'] = sp.Eq(gam_se, sp.Float('0.016')*(eps_iz - 2*W))  # ion-induced secondary emission yield (Baragiola 1979; Lieberman-Lichtenberg 9.3)
    eqs['eq_LA6_20'] = sp.Eq(A_G, lam_R*4*sp.pi*m_e*e*k_B**2/h**3)  # Richardson constant, lambda_R material correction (A_0 = 1.20e6 A m^-2 K^-2) (Ashcroft-Mermin 18; Raizer 10)
    eqs['eq_LA6_21'] = sp.Eq(J_th, A_G*T**2*sp.exp(-(W - dW)/(k_B*T)))  # Richardson-Dushman thermionic emission with Schottky lowering: hot-cathode arcs (Raizer 10; Lieberman-Lichtenberg 9)
    eqs['eq_LA6_22'] = sp.Eq(dW, sp.sqrt(e**3*E_s/(4*sp.pi*eps0)))  # Schottky barrier lowering by surface field (Raizer 10)
    # CAVEAT: W in J; t(y), v(y) are Nordheim image-charge functions of y = sqrt(e^3 E/(4 pi eps0))/W (t ~ 1, v ~ 1 - y^2 + y^2 ln(y)/3 approx)
    eqs['eq_LA6_23'] = sp.Eq(J_FN, e**3*E_s**2/(8*sp.pi*h*W*t_y**2)*sp.exp(-8*sp.pi*sp.sqrt(2*m_e)*W**sp.Rational(3, 2)*v_y/(3*e*h*E_s)))  # Fowler-Nordheim field emission: cold-cathode arc spots, vacuum breakdown (Fowler-Nordheim 1928; Raizer 10)
    eqs['eq_LA6_24'] = sp.Eq(C_sh, eps0*A_el/s_sh)  # RF sheath capacitance (Lieberman-Lichtenberg 11.1)
    eqs['eq_LA6_25'] = sp.Eq(I_rf, C_sh*sp.Derivative(V_sh, t))  # capacitive displacement current through the sheath (Lieberman-Lichtenberg 11.1)
    # CAVEAT: Koenig-Maissel ideal exponent q = 4; measured q ~ 1-2.5
    eqs['eq_LA6_26'] = sp.Eq(Va_b, Aa_b**q_KM)  # capacitive self-bias: sheath voltage ratio vs electrode area ratio (Koenig-Maissel 1970; Lieberman-Lichtenberg 11.4)
    eqs['eq_LA6_27'] = sp.Eq(sig_p, e**2*n_e/(m_e*(nu_m + sp.I*om)))  # RF plasma conductivity, exp(i omega t) convention (Lieberman-Lichtenberg 4.2)
    eqs['eq_LA6_28'] = sp.Eq(eps_p, 1 - om_pe**2/(om*(om - sp.I*nu_m)))  # cold collisional plasma permittivity (Lieberman-Lichtenberg 4.2)
    eqs['eq_LA6_29'] = sp.Eq(del_p, c/om_pe)  # collisionless skin depth, ICP heating layer when omega << omega_pe, nu_m << omega (Lieberman-Lichtenberg 12.1)
    eqs['eq_LA6_30'] = sp.Eq(del_c, sp.sqrt(2/(om*mu0*sig_dc)))  # collisional (ohmic) skin depth, sigma_dc = n e^2/(m nu_m), nu_m >> omega (Lieberman-Lichtenberg 12.1)
    eqs['eq_LA6_31'] = sp.Eq(p_abs, sp.Rational(1, 2)*sp.re(sig_p)*E_0**2)  # time-averaged ohmic power density absorbed by electrons (Lieberman-Lichtenberg 4.2)
    eqs['eq_LA6_32'] = sp.Eq(V_g, (C_d*V_a - Q_t)/(C_d + C_g))  # DBD gap voltage: transferred charge Q_t on the dielectric self-extinguishes each microdischarge (Kogelschatz 2003)
    eqs['eq_LA6_33'] = sp.Eq(P_M, 4*f*C_d*U_b*(U_0 - (C_d + C_g)/C_d*U_b))  # Manley DBD power, U_0 peak applied, U_b burning voltage (Manley 1943; Kogelschatz 2003)
    # CAVEAT: heuristic, not a textbook law: a plasma-globe (RF DBD in noble gas) filament re-ignites in the same channel when its residual ionization outlives the half period
    eqs['eq_LA6_34'] = sp.StrictGreaterThan(tau_mem, 1/(2*f))  # plasma-globe filament persistence (memory) condition, tau_mem from LA8 residual decay (Kogelschatz 2003 filament memory)
    eqs['eq_LA6_35'] = sp.Eq(I_T, 8*sp.pi*eps0*mu_i*V*(V - V_0)/(R_o**2*sp.log(R_o/r_0)))  # Townsend unipolar coaxial corona current per unit length (Townsend 1914; Raizer 12.7)
    # CAVEAT: beta_f ~ h/r_tip for a rod of height h, empirical; St Elmo's fire = point corona when E_tip exceeds the Peek onset F15_14
    eqs['eq_LA6_36'] = sp.Eq(E_tip, beta_f*E_0f)  # field enhancement at a tip in an ambient (thunderstorm) field (Rakov-Uman 16; Bazelyan-Raizer 7)
    eqs['eq_LA6_37'] = sp.StrictGreaterThan(R_b + dVdI, 0)  # Kaufmann stability of a discharge with falling V-I characteristic on a ballast R_b (neon signs, lamps, arcs) (Raizer 8.1)
    return eqs


globals().update(_expand_langmuir_discharges())


def _expand_langmuir_lightning():
    eqs = {}
    eps0, mu0, c, k_B, e = sp.symbols('epsilon_0 mu_0 c k_B e', positive=True)

    # LA7 lightning (Rakov-Uman "Lightning: Physics and Effects"); energy F15_23, action integral F15_24
    v_L, l_s, T_s, I_L, q_L = sp.symbols('v_L l_{step} T_{step} I_L q_L', positive=True)
    i0, I_0, eta_H, tau1, tau2, n_H, a_be, b_be = sp.symbols('i_0(t) I_0 eta_H tau_1 tau_2 n alpha beta', positive=True)
    z_p, v_rs, lam_M = sp.symbols("z' v lambda", positive=True)
    i_f = sp.Function('i')
    E_rad, D = sp.symbols('E_{rad} D', positive=True)
    W_L, R_L, sig_ch, n_e, m_e, nu_ei, nu_en = sp.symbols("W' R' sigma_{ch} n_e m_e nu_{ei} nu_{en}", positive=True)
    r_ch, C_B, rho_0, I_p, R_c, xi_c, E_L, R_0, b_F, p_0, f_m, c_0 = sp.symbols(
        'r_{ch} C_B rho_0 I R xi_c E_L R_0 b_F p_0 f_m c_0', positive=True)
    phi, Q, H, E_gr, dd = sp.symbols('phi Q H E_{ground} D', positive=True)
    r1, r2 = sp.symbols("|\\mathbf{r}-\\mathbf{r}_Q| |\\mathbf{r}-\\mathbf{r}_Q'|", positive=True)
    N_NO, Phi_NO, W_fl = sp.symbols('N_{NO} Phi_{NO} W', positive=True)
    r_s, A_s, b_s, P_I, I_med, k_I = sp.symbols('r_s A_s b_s P(I>I_0) I_{med} k_I', positive=True)
    V_ind, Z_0, h_l, y_l, beta = sp.symbols('U_{max} Z_0 h y beta', positive=True)
    V_loop, Phi_l, h_loop, D1, D2 = sp.symbols('V_{loop} Phi h D_1 D_2', positive=True)
    E_QE, E_k0, N_z, N_0, zz, H_s, E_th = sp.symbols('E_{QE}(z) E_{k0} N(z) N_0 z H_s E_{th}', positive=True)
    E_amb = sp.Symbol('E', positive=True)

    eqs['eq_LA7_1'] = sp.Eq(v_L, l_s/T_s)  # stepped-leader mean speed: steps ~ 3-200 m (typ. 50 m) every 5-100 us -> ~2e5 m/s (Rakov-Uman 4.4)
    eqs['eq_LA7_2'] = sp.Eq(I_L, q_L*v_L)  # leader current from line charge (~1e-3 C/m) and speed (Rakov-Uman 4.4)
    eqs['eq_LA7_3'] = sp.Eq(i0, (I_0/eta_H)*(t/tau1)**n_H/(1 + (t/tau1)**n_H)*sp.exp(-t/tau2))  # Heidler channel-base current function, declared engineering fit (Heidler 1985; IEC 62305-1 Annex B)
    eqs['eq_LA7_4'] = sp.Eq(eta_H, sp.exp(-(tau1/tau2)*(n_H*tau2/tau1)**(1/n_H)))  # Heidler peak-correction factor (Heidler 1985)
    eqs['eq_LA7_5'] = sp.Eq(i0, I_0*(sp.exp(-a_be*t) - sp.exp(-b_be*t)))  # Bruce-Golde double exponential (Bruce-Golde 1941; Rakov-Uman 12)
    # CAVEAT: engineering return-stroke model; v ~ c/3 - 2c/3 is declared; lambda ~ 2 km for MTLE
    eqs['eq_LA7_6'] = sp.Eq(i_f(z_p, t), sp.exp(-z_p/lam_M)*i_f(0, t - z_p/v_rs))  # MTLE return-stroke current along the channel (lambda -> oo gives TL, Uman-McLain 1969) (Nucci-Rachidi 1989; Rakov-Uman 12.2)
    eqs['eq_LA7_7'] = sp.Eq(E_rad, -v_rs/(2*sp.pi*eps0*c**2*D)*i_f(0, t - D/c))  # TL-model far (radiation) field over perfectly conducting ground (Uman-McLain-Krider 1975; Rakov-Uman 12.2)
    eqs['eq_LA7_8'] = sp.Eq(W_L, sp.Integral(R_L*i_f(0, t)**2, t))  # channel energy input per unit length, R' time-varying resistance per length; typ. 1e5 J/m (Rakov-Uman 12.1; Borovsky 1998)
    eqs['eq_LA7_9'] = sp.Eq(sig_ch, n_e*e**2/(m_e*(nu_ei + nu_en)))  # channel conductivity from n_e of the Saha set LA2 at T_ch ~ 30 kK (Mitchner-Kruger 3; Rakov-Uman 12.1)
    # CAVEAT: Braginskii strong-shock channel; C_B = 0.93 in cgs (r cm, rho0 g/cm^3, I A, t s), assumes constant conductivity
    eqs['eq_LA7_10'] = sp.Eq(r_ch, C_B*rho_0**sp.Rational(-1, 6)*I_p**sp.Rational(1, 3)*t**sp.Rational(1, 2))  # Braginskii spark-channel expansion (Braginskii 1958; Rakov-Uman 12.1)
    # CAVEAT: xi_c ~ 1 depends on gamma; strong-shock, instantaneous line release E_L (J/m)
    eqs['eq_LA7_11'] = sp.Eq(R_c, xi_c*(E_L/rho_0)**sp.Rational(1, 4)*t**sp.Rational(1, 2))  # Sedov-Taylor cylindrical blast wave; spherical analogue Z3_1 (Sedov 1959; Lin 1954)
    # CAVEAT: b_F is a gamma-dependent constant of order pi; the pressure pulse relaxes to acoustic beyond R_0
    eqs['eq_LA7_12'] = sp.Eq(R_0, sp.sqrt(E_L/(b_F*p_0)))  # thunder relaxation radius (Few 1969; Rakov-Uman 11)
    eqs['eq_LA7_13'] = sp.Eq(f_m, sp.Float('0.63')*c_0/R_0)  # dominant thunder frequency, empirical coefficient (Few 1969; Rakov-Uman 11)
    eqs['eq_LA7_14'] = sp.Eq(phi, Q/(4*sp.pi*eps0)*(1/r1 - 1/r2))  # potential of a cloud charge centre with its ground image; sum over the tripole (Rakov-Uman 3.2)
    eqs['eq_LA7_15'] = sp.Eq(E_gr, 2*Q*H/(4*sp.pi*eps0*(H**2 + dd**2)**sp.Rational(3, 2)))  # ground field of a charge Q at height H, horizontal distance D (Rakov-Uman 3.2)
    # CAVEAT: empirical yield, Phi_NO ~ 1e16-1e17 NO molecules per joule, large uncertainty
    eqs['eq_LA7_16'] = sp.Eq(N_NO, Phi_NO*W_fl)  # lightning NOx production (Price et al. 1997; Rakov-Uman 19)
    # CAVEAT: electrogeometric fit, A_s = 10, b_s = 0.65 (r m, I kA) in IEC 62305; others differ
    eqs['eq_LA7_17'] = sp.Eq(r_s, A_s*I_p**b_s)  # striking distance / rolling-sphere radius (Love 1973; IEC 62305-3)
    # CAVEAT: empirical, I_med = 31 kA, k_I = 2.6 for first negative strokes
    eqs['eq_LA7_18'] = sp.Eq(P_I, 1/(1 + (I_0/I_med)**k_I))  # peak-current exceedance probability (Anderson-Eriksson 1980; IEEE Std 1410)
    # CAVEAT: step current, perfect ground, line height h at distance y >> h; Z_0 = (1/4pi) sqrt(mu0/eps0) = 30 ohm
    eqs['eq_LA7_19'] = sp.Eq(V_ind, Z_0*I_0*h_l/y_l*(1 + beta/sp.sqrt(2)/sp.sqrt(1 - beta**2/2)))  # Rusck peak induced overvoltage on an overhead line, beta = v/c (Rusck 1958; IEEE Std 1410)
    eqs['eq_LA7_20'] = sp.Eq(Z_0, sp.sqrt(mu0/eps0)/(4*sp.pi))  # Rusck impedance (Rusck 1958)
    eqs['eq_LA7_21'] = sp.Eq(Phi_l, mu0*i_f(0, t)*h_loop/(2*sp.pi)*sp.log(D2/D1))  # magnetic flux of a nearby straight stroke through a rectangular loop (Griffiths 7; Rakov-Uman 18)
    eqs['eq_LA7_22'] = sp.Eq(V_loop, -sp.Derivative(Phi_l, t))  # induced loop voltage, Faraday (Griffiths 7)
    eqs['eq_LA7_23'] = sp.Eq(N_z, N_0*sp.exp(-zz/H_s))  # isothermal atmosphere density (scale height H_s ~ 7 km) (Pasko et al. 1997)
    eqs['eq_LA7_24'] = sp.StrictGreaterThan(E_QE, E_k0*N_z/N_0)  # sprite criterion: post-stroke quasi-electrostatic field exceeds the density-scaled breakdown field (Wilson 1925; Pasko et al. 1997)
    # CAVEAT: empirical, E_th ~ 2.8e5 V/m at STP scaled with N; threshold of relativistic runaway electron avalanches (TGFs)
    eqs['eq_LA7_25'] = sp.GreaterThan(E_amb, E_th*N_z/N_0)  # relativistic runaway breakdown threshold (Gurevich-Milikh-Roussel-Dupre 1992; Dwyer 2003)
    return eqs


globals().update(_expand_langmuir_lightning())


def _expand_langmuir_tracing():
    eqs = {}
    e, k_B, eps0, h, c, m_e = sp.symbols('e k_B epsilon_0 h c m_e', positive=True)

    # LA8 tracing: where it goes, what it leaves, what it emits
    phi, phi_ch, p_i, E_i, E_j, eta_DBM, E_c, s, phi_0, E_chs = sp.symbols(
        'phi phi_{ch} p_i |E_i| |E_j| eta_{DBM} E_c s phi_0 E_{ch}', positive=True)
    jj = sp.Symbol('j', integer=True)
    sig, n_e, n_p, n_n, mu_e, mu_p, mu_n = sp.symbols('sigma(\\mathbf{x}) n_e n_+ n_- mu_e mu_+ mu_-', positive=True)
    n_t, n_0, beta_r, nu_a, k_2, k_3, N_O2, N_ = sp.symbols('n_e(t) n_0 beta_r nu_a k_{a2} k_{a3} N_{O2} N', positive=True)
    rho_ch, rho_0, T_0, T_ch, dT, dT0, r_0, a_th = sp.symbols('rho_{ch} rho_0 T_0 T_{ch} Delta\\!T(t) Delta\\!T_0 r_0 a_{th}', positive=True)
    P_ff, T_e, Z, n_i, g_B, P_rec, chi = sp.symbols('P_{ff} T_e Z n_i \\bar{g}_B P_{fb} chi_{z-1}', positive=True)
    eps_ul, nu_ul, A_ul, n_u, B_ul, B_lu, g_u, g_l = sp.symbols('epsilon_{ul} nu_{ul} A_{ul} n_u B_{ul} B_{lu} g_u g_l', positive=True)
    I_nu, j_nu, kap_nu, B_nu, nu, T, tau_nu = sp.symbols('I_nu j_nu kappa_nu B_nu nu T tau_nu', positive=True)
    I_ul, lam_ul, E_u, T_exc, C_bp = sp.symbols('I_{ul} lambda_{ul} E_u T_{exc} C', positive=True)
    dlam_S, w_S, n_ref, dlam_D, lam_0, M_a = sp.symbols('Delta\\lambda_{S} w_S n_{ref} Delta\\lambda_D lambda_0 M', positive=True)
    p_J, J_v, E_v, w_dep = sp.symbols('p_J \\mathbf{J} \\mathbf{E} w_{dep}', real=True)
    E_mag = sp.Symbol('|\\mathbf{E}|', positive=True)

    eqs['eq_LA8_1'] = sp.Eq(nabla(nabla(phi)), 0)  # Laplace outside the channel (div grad), channel held equipotential phi_ch (Niemeyer-Pietronero-Wiesmann 1984)
    eqs['eq_LA8_2'] = sp.Eq(p_i, E_i**eta_DBM/sp.Sum(E_j**eta_DBM, (jj, 1, sp.Symbol('N_c'))))  # dielectric breakdown model growth probability over candidate bonds; eta_DBM declared (eta = 1 -> D ~ 1.7 in 2D, emergent) (Niemeyer-Pietronero-Wiesmann 1984)
    eqs['eq_LA8_3'] = sp.Eq(p_i, sp.Heaviside(E_i - E_c)*(E_i - E_c)**eta_DBM/sp.Sum(sp.Heaviside(E_j - E_c)*(E_j - E_c)**eta_DBM, (jj, 1, sp.Symbol('N_c'))))  # threshold DBM: only bonds above the propagation field E_c (e.g. LA5_14) can grow (Wiesmann-Zeller 1986)
    eqs['eq_LA8_4'] = sp.Eq(sp.Function('phi_{ch}')(s), phi_0 - sp.Integral(sp.Function('E_{ch}')(sp.Symbol("s'")), (sp.Symbol("s'"), 0, s)))  # channel with internal gradient (leader ~ 1e5 V/m, streamer ~ E_st) instead of a perfect conductor (Pasko et al. 2000; Bazelyan-Raizer 4)
    eqs['eq_LA8_5'] = sp.Eq(sig, e*(n_e*mu_e + n_p*mu_p + n_n*mu_n))  # conductivity field of the discharge from the LA4 species densities; F13_11 per species (Raizer 2)
    eqs['eq_LA8_6'] = sp.Eq(n_t, n_0/(1 + beta_r*n_0*t))  # residual ionization decay by recombination alone (Raizer 6.3)
    eqs['eq_LA8_7'] = sp.Eq(nu_a, k_2*N_O2 + k_3*N_O2*N_)  # electron attachment frequency in air (two- plus three-body), the cold-channel lifetime 1/nu_a ~ 10-100 ns at STP (Raizer 6.1; Kossyi et al. 1992)
    eqs['eq_LA8_8'] = sp.Eq(rho_ch/rho_0, T_0/T_ch)  # isobaric density hole left in a hot channel after pressure equilibration (Bazelyan-Raizer 4; Rakov-Uman 12.1)
    # CAVEAT: Gaussian initial profile, constant thermal diffusivity a_th, no convection
    eqs['eq_LA8_9'] = sp.Eq(dT, dT0*r_0**2/(r_0**2 + 4*a_th*t))  # axial temperature decay of a cooling channel (Carslaw-Jaeger 10; Bazelyan-Raizer 4)
    eqs['eq_LA8_10'] = sp.Eq(P_ff, sp.sqrt(2*sp.pi*k_B*T_e/(3*m_e))*32*sp.pi*e**6/(3*h*m_e*c**3*(4*sp.pi*eps0)**3)*Z**2*n_e*n_i*g_B)  # thermal bremsstrahlung power density, SI (Rybicki-Lightman 5.15b)
    # CAVEAT: hydrogenic Kramers estimate of the free-bound/free-free ratio for recombination into the ground stage
    eqs['eq_LA8_11'] = sp.Eq(P_rec/P_ff, chi/(k_B*T_e))  # recombination-continuum power relative to bremsstrahlung (NRL Plasma Formulary, radiation; Griem 5)
    eqs['eq_LA8_12'] = sp.Eq(eps_ul, h*nu_ul/(4*sp.pi)*A_ul*n_u)  # optically thin line emission coefficient (W m^-3 sr^-1), n_u from LA2_4/LA2_6 in LTE (Griem 4; Thorne "Spectrophysics")
    eqs['eq_LA8_13'] = sp.Eq(g_l*B_lu, g_u*B_ul)  # Einstein relation, absorption/stimulated emission (Rybicki-Lightman 1.6)
    eqs['eq_LA8_14'] = sp.Eq(A_ul, 8*sp.pi*h*nu_ul**3/c**3*B_ul)  # Einstein A from B (Rybicki-Lightman 1.6)
    eqs['eq_LA8_15'] = sp.Eq(sp.Derivative(I_nu, s), j_nu - kap_nu*I_nu)  # radiative transfer along a ray (Rybicki-Lightman 1.4)
    eqs['eq_LA8_16'] = sp.Eq(j_nu/kap_nu, B_nu)  # Kirchhoff's law in LTE (Rybicki-Lightman 1.5)
    eqs['eq_LA8_17'] = sp.Eq(B_nu, 2*h*nu**3/c**2/(sp.exp(h*nu/(k_B*T)) - 1))  # Planck function (Rybicki-Lightman 1.5)
    eqs['eq_LA8_18'] = sp.Eq(I_nu, B_nu*(1 - sp.exp(-tau_nu)))  # emergent intensity of a uniform LTE slab/channel; tau >> 1 -> blackbody (Rybicki-Lightman 1.4)
    eqs['eq_LA8_19'] = sp.Eq(tau_nu, sp.Integral(kap_nu, s))  # optical depth (Rybicki-Lightman 1.4)
    eqs['eq_LA8_20'] = sp.Eq(sp.log(I_ul*lam_ul/(g_u*A_ul)), -E_u/(k_B*T_exc) + C_bp)  # Boltzmann-plot excitation temperature diagnostic (Griem 13; Fujimoto 4)
    # CAVEAT: electron-impact (Lorentzian) width linear in n_e, ion broadening neglected; w_S tabulated at n_ref (Griem tables), weakly T-dependent
    eqs['eq_LA8_21'] = sp.Eq(dlam_S, 2*w_S*n_e/n_ref)  # Stark FWHM electron-density diagnostic (Griem "Spectral Line Broadening by Plasmas" 1974)
    eqs['eq_LA8_22'] = sp.Eq(dlam_D, lam_0*sp.sqrt(8*k_B*T*sp.log(2)/(M_a*c**2)))  # Doppler FWHM, heavy-particle temperature diagnostic (Thorne 9; Griem 4)
    eqs['eq_LA8_23'] = sp.Eq(p_J, J_v*E_v)  # local Joule power density J.E = sigma E^2 heating the channel (Raizer 10)
    eqs['eq_LA8_24'] = sp.Eq(w_dep, sp.Integral(sig*E_mag**2, t))  # deposited energy density, the input to LA9 branching and gas heating (Raizer 10; Fridman 4)
    return eqs


globals().update(_expand_langmuir_tracing())


def _expand_langmuir_chemistry():
    eqs = {}
    e, m_e, k_B = sp.symbols('e m_e k_B', positive=True)

    # LA9 plasma chemistry (rate laws in the L3_1 mass-action form)
    k_j, eps_, sig_j, E_N, T_e, K0, b_k, eps_th = sp.symbols('k_j epsilon sigma_j(epsilon) E/N T_e K_0 b epsilon_{th}', positive=True)
    f_eps = sp.Function('f')(eps_, E_N)
    n_e, N, n_p, beta_dr, beta_0, T_r, a_dr, k_3b, C_3, k_iz = sp.symbols('n_e N n_+ beta_{dr} beta_0 T_{ref} a k_{3b} C_3 k_{iz}', positive=True)
    nO2, nM, k_a3, k_a2, n_n, k_det, beta_ii, k_diss, nO, nO3, k_O3, k_O3_0, a_O3, T_g = sp.symbols(
        '[O_2] [M] k_{a3} k_{a2} n_- k_{det} beta_{ii} k_{diss} [O] [O_3] k_{O3} k_{O3;0} a_{O3} T_g', positive=True)
    NO, N2, Nat, OH, H, k1f, k1r, k2f, k2r, k3f, k3r, Oe, N2e = sp.symbols(
        '[NO] [N_2] [N] [OH] [H] k_{1f} k_{1r} k_{2f} k_{2r} k_{3f} k_{3r} [O]_e [N_2]_e', positive=True)
    nAs, nB, k_P, E_star, chi_B = sp.symbols('[A^*] [B] k_P E^*_A chi_B', positive=True)
    eta_j, eps_j, v_d, eta_el = sp.symbols('eta_j epsilon_j v_d eta_{el}', positive=True)
    jj = sp.Symbol('j', integer=True)
    eta_jf = sp.Function('eta')(jj)
    eps_v, eps_v_eq, tau_VT, G_val, N_prod, W_in, Y_E = sp.symbols('epsilon_v epsilon_v^{eq}(T) tau_{VT} G N_{prod} W Y_E', positive=True)

    eqs['eq_LA9_1'] = sp.Eq(k_j, sp.Integral(sig_j*sp.sqrt(2*eps_/m_e)*f_eps, (eps_, 0, sp.oo)))  # electron-impact rate coefficient in the local-field approximation k_j(E/N), EEDF from LA3_12 (Hagelaar-Pitchford 2005; Fridman 2)
    # CAVEAT: Arrhenius-form fit for Maxwellian EEDF, K0, b declared per process; generalizes L3_2 with T -> T_e
    eqs['eq_LA9_2'] = sp.Eq(k_j, K0*T_e**b_k*sp.exp(-eps_th/(k_B*T_e)))  # electron-impact ionization/excitation/dissociation rate fit (Lieberman-Lichtenberg 3.5)
    eqs['eq_LA9_3'] = sp.Eq(sp.Derivative(n_e, t), k_iz*n_e*N)  # e + A -> A+ + 2e ionization source (Lieberman-Lichtenberg 3.5)
    # CAVEAT: empirical, a ~ 0.5-0.7 per ion (e.g. O2+, N2+, NO+)
    eqs['eq_LA9_4'] = sp.Eq(beta_dr, beta_0*(T_r/T_e)**a_dr)  # dissociative recombination e + XY+ -> X + Y coefficient (Kossyi et al. 1992; Fridman 2.3)
    eqs['eq_LA9_5'] = sp.Eq(sp.Derivative(n_e, t), -beta_dr*n_e*n_p)  # dissociative recombination loss; dominant electron sink in molecular gas (Raizer 6.3)
    # CAVEAT: classical scaling T_e^(-9/2); C_3 declared
    eqs['eq_LA9_6'] = sp.Eq(k_3b, C_3*T_e**sp.Rational(-9, 2))  # three-body (collisional-radiative) recombination e + e + A+ -> e + A (Raizer 6.3; Fridman 2.3)
    eqs['eq_LA9_7'] = sp.Eq(sp.Derivative(n_e, t), -k_3b*n_e**2*n_p)  # three-body recombination loss, dominates in dense cool plasma (arc/lightning afterglow) (Raizer 6.3)
    eqs['eq_LA9_8'] = sp.Eq(sp.Derivative(n_e, t), -k_a3*n_e*nO2*nM - k_a2*n_e*nO2 + k_det*n_n*N)  # O2 three-body attachment (e + O2 + M -> O2- + M), dissociative attachment (e + O2 -> O- + O), and detachment (Raizer 6.1; Kossyi et al. 1992)
    eqs['eq_LA9_9'] = sp.Eq(sp.Derivative(n_p, t), -beta_ii*n_p*n_n)  # ion-ion recombination X+ + Y- -> neutrals (Raizer 6.3)
    eqs['eq_LA9_10'] = sp.Eq(sp.Derivative(nO, t), 2*k_diss*n_e*nO2 - k_O3*nO*nO2*nM)  # O production by electron-impact dissociation of O2 (6.1/8.4 eV) and loss to ozone (Eliasson-Kogelschatz 1991)
    eqs['eq_LA9_11'] = sp.Eq(sp.Derivative(nO3, t), k_O3*nO*nO2*nM)  # ozone formation O + O2 + M -> O3 + M in corona/DBD, destruction terms per mechanism (Eliasson-Kogelschatz 1991)
    # CAVEAT: empirical; k_O3,0 ~ 6.0e-46 m^6/s, a_O3 ~ 2.6 for M = air (JPL evaluation)
    eqs['eq_LA9_12'] = sp.Eq(k_O3, k_O3_0*(T_g/300)**(-a_O3))  # termolecular ozone rate coefficient (Sander et al. JPL 2011)
    eqs['eq_LA9_13'] = sp.Eq(sp.Derivative(NO, t), k1f*nO*N2 + k2f*Nat*nO2 + k3f*Nat*OH - k1r*NO*Nat - k2r*NO*nO - k3r*NO*H)  # extended Zeldovich thermal-NO mechanism N2+O<->NO+N, N+O2<->NO+O, N+OH<->NO+H; not previously in the catalogue (Zeldovich 1946; Turns 5.3; Heywood 11.2)
    eqs['eq_LA9_14'] = sp.Eq(sp.Derivative(NO, t), 2*k1f*Oe*N2e)  # initial NO formation rate, [N] steady state, equilibrium O and N2 (Heywood 11.8); O equilibrium from the G-series
    eqs['eq_LA9_15'] = sp.Eq(sp.Derivative(n_p, t), k_P*nAs*nB)  # Penning ionization A* + B -> A + B+ + e (neon-argon lamps) (Raizer 4.4; Lieberman-Lichtenberg 8)
    eqs['eq_LA9_16'] = sp.StrictGreaterThan(E_star, chi_B)  # Penning energy condition: metastable energy exceeds partner ionization energy (Raizer 4.4)
    eqs['eq_LA9_17'] = sp.Eq(eta_j, eps_j*k_j/(e*E_N*v_d))  # fraction of electron power into channel j (vibration, electronic, dissociation, ionization) vs E/N (Fridman 4.1; Raizer 2)
    eqs['eq_LA9_18'] = sp.Eq(eta_el, 1 - sp.Sum(eta_jf, (jj, 1, sp.Symbol('N_r'))))  # energy-branching closure incl. elastic fraction (Fridman 4.1)
    eqs['eq_LA9_19'] = sp.Eq(sp.Derivative(eps_v, t), (eps_v_eq - eps_v)/tau_VT)  # Landau-Teller vibrational-translational relaxation: delayed gas heating in afterglows (Landau-Teller 1936; Fridman 3.2)
    eqs['eq_LA9_20'] = sp.Eq(G_val, N_prod/(W_in/(100*e)))  # G-value: product molecules per 100 eV deposited, W in J (radiation/plasma chemistry convention; Fridman 1)
    eqs['eq_LA9_21'] = sp.Eq(Y_E, N_prod/W_in)  # energy yield (molecules per joule), cf LA7_16 (Fridman 1)
    return eqs


globals().update(_expand_langmuir_chemistry())


def _expand_langmuir_weak_plasmas():
    eqs = {}
    e, m_e, k_B, h, c = sp.symbols('e m_e k_B h c', positive=True)

    # LA10 flames, lamps, weak plasmas
    n_e, n_i, k_ci, CH, O, beta_f, sig_f, mu_e, nu_m = sp.symbols('n_e n_i k_{ci} [CH] [O] beta_f sigma_f mu_e nu_m', positive=True)
    I_FID, xi_FID, ndot_C, f_EHD, J_ion, mu_i, u_iw, eps0, rho_g, E_f = sp.symbols(
        'I_{FID} xi_{FID} \\dot{n}_C \\mathbf{f}_{EHD} \\mathbf{J}_{ion} mu_i u_{iw} epsilon_0 rho_g E', positive=True)
    eta_St, lam_exc, lam_em, eta_lamp, eta_UV, QE, eta_Stokes = sp.symbols(
        'eta_{Stokes} lambda_{exc} lambda_{em} eta_{lamp} eta_{UV} QE_{ph} eta_{St}', positive=True)
    tau_eff, tau_nat, k0, R_t, K_v, P_el, Phi_v = sp.symbols('tau_{eff} tau_{nat} k_0 R K_v P_{el} Phi_v', positive=True)

    eqs['eq_LA10_1'] = sp.Eq(sp.Derivative(n_e, t), k_ci*CH*O - beta_f*n_e*n_i)  # flame chemi-ionization CH + O -> CHO+ + e- balanced by dissociative recombination of H3O+ (after CHO+ + H2O -> H3O+ + CO) (Calcote 1963; Fialkov 1997)
    eqs['eq_LA10_2'] = sp.Eq(n_e, sp.sqrt(k_ci*CH*O/beta_f))  # steady flame-zone ion density, n_e = n_i (Lawton-Weinberg "Electrical Aspects of Combustion" 5)
    eqs['eq_LA10_3'] = sp.Eq(mu_e, e/(m_e*nu_m))  # electron mobility (Raizer 2.2)
    eqs['eq_LA10_4'] = sp.Eq(sig_f, e*n_e*mu_e)  # flame conductivity, electrons dominant; alkali seeding raises n_e via Saha LA2_8 (Lawton-Weinberg 5; Mitchner-Kruger 3)
    # CAVEAT: xi_FID (ions per carbon atom, ~1e-6..1e-5) is declared/calibrated; the proportionality holds for hydrocarbons
    eqs['eq_LA10_5'] = sp.Eq(I_FID, e*xi_FID*ndot_C)  # flame ionization detector current proportional to carbon-atom flux (McWilliam-Dewar 1958; Holm 1999)
    eqs['eq_LA10_6'] = sp.Eq(f_EHD, J_ion/mu_i)  # ionic-wind body force of a unipolar ion current (Robinson 1961; Lawton-Weinberg 7)
    # CAVEAT: order-of-magnitude velocity scale of the ionic wind
    eqs['eq_LA10_7'] = sp.Eq(u_iw, E_f*sp.sqrt(eps0/rho_g))  # ionic-wind velocity scale (Robinson 1961)
    eqs['eq_LA10_8'] = sp.Eq(eta_St, lam_exc/lam_em)  # phosphor Stokes (quantum-defect) efficiency, e.g. Hg 253.7 nm -> visible ~ 0.45 (Waymouth "Electric Discharge Lamps" 1971)
    eqs['eq_LA10_9'] = sp.Eq(eta_lamp, eta_UV*eta_St*QE)  # fluorescent lamp radiant efficiency chain: column UV efficiency x Stokes x phosphor QE (Waymouth 1971)
    # CAVEAT: infinite cylinder, Doppler line, k_0 R >> 1; Hg vapour pressure set by the lamp cold spot (~40 C)
    eqs['eq_LA10_10'] = sp.Eq(tau_eff, tau_nat*k0*R_t*sp.sqrt(sp.pi*sp.log(k0*R_t))/sp.Float('1.60'))  # Holstein radiation-imprisonment lifetime of the resonance line (Holstein 1951)
    eqs['eq_LA10_11'] = sp.Eq(K_v, Phi_v/P_el)  # luminous efficacy of a lamp (lm/W) (Waymouth 1971)
    return eqs


globals().update(_expand_langmuir_weak_plasmas())


def _expand_langmuir_space_fusion():
    eqs = {}
    e, m_e, k_B, eps0, mu0, m_p = sp.symbols('e m_e k_B epsilon_0 mu_0 m_p', positive=True)

    # LA11 space and fusion plasmas (gyroradius F19_10, cyclotron F19_9, loss cone F19_17)
    n, tau_E, T, sv, E_a, E_f, p_fus, p_al, W, P_loss, Q_f, P_fus, P_h = sp.symbols(
        'n tau_E T <sigma_v> E_alpha E_{fus} p_{fus} p_alpha W P_{loss} Q P_{fus} P_{heat}', positive=True)
    beta_t, p_av, B_T, beta_N, a, I_p, B_th, q, r, B_ph, R0, n_G, beta_max = sp.symbols(
        'beta_t <p> B_T beta_N a I_p B_theta q r B_phi R_0 n_G beta_{N;max}', positive=True)
    f_c, N_max, f_ob, f_v, th_i, n2, X, Y_T, Y_L, Zc = sp.symbols('f_c N_{max} f_{ob} f_v theta_i n^2 X Y_T Y_L Z', positive=True)
    n2 = sp.Symbol('n^2')  # complex-valued
    q_ch, q_m0, zp, chi, n_eq, a_rec = sp.symbols('q_{Ch} q_{m0} z_* chi n_e a_{rec}', positive=True)
    I_R, eta_em, zz, j_par, K_kn, Phi_par, n_m, T_m = sp.symbols('I_R eta_{em}(z) z j_par K_{Kn} Phi_par n_m T_m', positive=True)
    psi, Om_s, th, v_sw, B_r, B_0, r_0, B_ph2 = sp.symbols('psi Omega_sun theta v_{sw} B_r B_0 r_0 B_phi', positive=True)
    r_mp, R_E, f_cmp, rho_sw = sp.symbols('r_{mp} R_E f_{cmp} rho_{sw}', positive=True)
    H_c, T_c, g_s, P_rad, n_H = sp.symbols('H_c T_c g_sun P_{rad} n_H', positive=True)
    Lam_r = sp.Function('Lambda_{rad}')

    # CAVEAT: 50:50 D-T, T_e = T_i = T, W = 3 n k_B T; the 'triple product' form n T tau_E follows by multiplying by T
    eqs['eq_LA11_1'] = sp.StrictGreaterThan(n*tau_E, 12*k_B*T/(sv*E_a))  # Lawson ignition condition, alpha heating balances losses (Wesson "Tokamaks" 1.5; Goldston-Rutherford 1)
    eqs['eq_LA11_2'] = sp.StrictGreaterThan(n*k_B*T*tau_E, 12*(k_B*T)**2/(sv*E_a))  # triple-product ignition, ~3e21 keV s m^-3 near 14 keV (Wesson 1.5)
    eqs['eq_LA11_3'] = sp.Eq(p_fus, n**2/4*sv*E_f)  # D-T fusion power density, E_fus = 17.6 MeV (Wesson 1.2)
    eqs['eq_LA11_4'] = sp.Eq(p_al, n**2/4*sv*E_a)  # alpha heating power density, E_alpha = 3.5 MeV (Wesson 1.5)
    eqs['eq_LA11_5'] = sp.Eq(W, 3*n*k_B*T)  # plasma thermal energy density, electrons + ions (Wesson 1.5)
    eqs['eq_LA11_6'] = sp.Eq(tau_E, W/P_loss)  # energy confinement time (Wesson 1.5); P_loss includes bremsstrahlung LA8_10
    eqs['eq_LA11_7'] = sp.Eq(Q_f, P_fus/P_h)  # fusion gain (Wesson 1.5)
    eqs['eq_LA11_8'] = sp.Eq(beta_t, 2*mu0*p_av/B_T**2)  # toroidal beta, LA4_24 with vacuum toroidal field (Wesson 3; Goldston-Rutherford 9)
    eqs['eq_LA11_9'] = sp.Eq(beta_N, beta_t*100*a*B_T/I_p)  # normalized beta, beta in %, a m, B T, I_p MA (Troyon 1984)
    # CAVEAT: empirical Troyon limit, beta_N,max ~ 2.8-3.5
    eqs['eq_LA11_10'] = sp.LessThan(beta_N, beta_max)  # Troyon beta limit (Troyon et al. 1984; Wesson 6)
    eqs['eq_LA11_11'] = sp.Eq(q, r*B_ph/(R0*B_th))  # cylindrical (large-aspect-ratio) safety factor (Wesson 3; Goldston-Rutherford 9)
    eqs['eq_LA11_12'] = sp.Eq(B_th, mu0*I_p/(2*sp.pi*a))  # edge poloidal field of the plasma current (Goldston-Rutherford 9)
    eqs['eq_LA11_13'] = sp.StrictGreaterThan(q, 1)  # Kruskal-Shafranov kink stability q(a) > 1 (Wesson 6; Goldston-Rutherford 19)
    # CAVEAT: empirical; n_G in 1e20 m^-3, I_p in MA, a in m
    eqs['eq_LA11_14'] = sp.Eq(n_G, I_p/(sp.pi*a**2))  # Greenwald density limit (Greenwald et al. 1988)
    eqs['eq_LA11_15'] = sp.Eq(f_c, sp.sqrt(e**2/(4*sp.pi**2*eps0*m_e))*sp.sqrt(N_max))  # ionospheric critical frequency = peak plasma frequency, f_c ~ 8.98 sqrt(N_max) Hz (Budden "Propagation of Radio Waves" 3; Davies "Ionospheric Radio")
    eqs['eq_LA11_16'] = sp.Eq(f_ob, f_v/sp.cos(th_i))  # secant law for oblique reflection, flat ionosphere (Martyn; Davies 6)
    eqs['eq_LA11_17'] = sp.Eq(n2, 1 - X/(1 - sp.I*Zc - Y_T**2/(2*(1 - X - sp.I*Zc)) + sp.sqrt(Y_T**4/(4*(1 - X - sp.I*Zc)**2) + Y_L**2)))  # Appleton-Hartree refractive index, X = omega_p^2/omega^2, Y = omega_c/omega, Z = nu/omega; + root shown, - root is the other mode (Budden 4)
    eqs['eq_LA11_18'] = sp.Eq(q_ch, q_m0*sp.exp(1 - zp - sp.exp(-zp)/sp.cos(chi)))  # Chapman production layer, z_* = (z - z_m0)/H, solar zenith angle chi (Chapman 1931; Rishbeth-Garriott 3)
    eqs['eq_LA11_19'] = sp.Eq(n_eq, sp.sqrt(q_ch/a_rec))  # alpha-Chapman photochemical equilibrium electron density (Rishbeth-Garriott 3)
    eqs['eq_LA11_20'] = sp.Eq(I_R, sp.Float('1e-10')*sp.Integral(eta_em, zz))  # column emission rate in rayleighs, eta in photons m^-3 s^-1 (1 R = 1e10 photons m^-2 s^-1 column) (Hunten-Roach-Chamberlain 1956)
    # CAVEAT: Knight relation valid for 1 << e Phi_par/(k_B T_m) << B_ionosphere/B_source
    eqs['eq_LA11_21'] = sp.Eq(j_par, K_kn*Phi_par)  # Knight current-voltage relation of auroral acceleration (Knight 1973)
    eqs['eq_LA11_22'] = sp.Eq(K_kn, e**2*n_m/sp.sqrt(2*sp.pi*m_e*k_B*T_m))  # Knight conductance (Knight 1973; Lyons 1980)
    eqs['eq_LA11_23'] = sp.Eq(B_r, B_0*(r_0/r)**2)  # radial interplanetary field, frozen-in flux LA4_19 (Parker 1958)
    eqs['eq_LA11_24'] = sp.Eq(B_ph2, -B_r*Om_s*r*sp.sin(th)/v_sw)  # Parker spiral azimuthal field (Parker 1958; Kivelson-Russell 4)
    eqs['eq_LA11_25'] = sp.Eq(sp.tan(psi), Om_s*r*sp.sin(th)/v_sw)  # Parker spiral angle (Parker 1958)
    # CAVEAT: f_cmp ~ 2 (Chapman-Ferraro compression of the dipole field), cold solar wind, specular reflection
    eqs['eq_LA11_26'] = sp.Eq(r_mp/R_E, (f_cmp**2*B_0**2/(2*mu0*rho_sw*v_sw**2))**sp.Rational(1, 6))  # magnetopause standoff from pressure balance (Chapman-Ferraro 1931; Kivelson-Russell 9)
    eqs['eq_LA11_27'] = sp.Eq(H_c, 2*k_B*T_c/(m_p*g_s))  # coronal hydrostatic scale height, fully ionized H, T_e = T_i (Priest "Magnetohydrodynamics of the Sun" 3)
    eqs['eq_LA11_28'] = sp.Eq(P_rad, n_eq*n_H*Lam_r(T_c))  # optically thin radiative loss, Lambda(T) declared atomic data (Priest 2; Rosner-Tucker-Vaiana 1978)
    return eqs


globals().update(_expand_langmuir_space_fusion())


def _expand_langmuir_solid_state():
    eqs = {}
    e, m_e, eps0, hbar, k_B = sp.symbols('e m_e epsilon_0 hbar k_B', positive=True)

    # LA12 solid-state and degenerate electron plasmas
    sig0, n, tau, m_s, sig_w, om, eps_w, eps_inf, om_p, om_pt, R_refl, n_c = sp.symbols(
        'sigma_0 n tau m^* sigma(omega) omega epsilon(omega) epsilon_infty omega_p \\tilde{omega}_p R \\tilde{n}')
    k_TF, E_F, lam_TF, E_pl, om_sp, om_Fr, eps_r, lam_th, T, r_s, a_B, h = sp.symbols(
        'k_{TF} E_F lambda_{TF} E_{pl} omega_{sp} omega_{F} epsilon_r lambda_{th} T r_s a_B h', positive=True)

    eqs['eq_LA12_1'] = sp.Eq(sig0, n*e**2*tau/m_s)  # Drude DC conductivity (Ashcroft-Mermin 1; Kittel 6)
    eqs['eq_LA12_2'] = sp.Eq(sig_w, sig0/(1 - sp.I*om*tau))  # Drude AC conductivity, exp(-i omega t) convention (Ashcroft-Mermin 1)
    eqs['eq_LA12_3'] = sp.Eq(eps_w, eps_inf - om_p**2/(om**2 + sp.I*om/tau))  # Drude dielectric function with core polarization eps_inf (Kittel 14; Ashcroft-Mermin 1)
    eqs['eq_LA12_4'] = sp.Eq(om_p, sp.sqrt(n*e**2/(eps0*m_s)))  # electron plasma frequency with effective mass, F13_9 for band electrons (Kittel 14)
    eqs['eq_LA12_5'] = sp.Eq(om_pt, om_p/sp.sqrt(eps_inf))  # screened plasma edge (Kittel 14)
    eqs['eq_LA12_6'] = sp.Eq(R_refl, sp.Abs((n_c - 1)/(n_c + 1))**2)  # normal-incidence reflectivity, n_c = sqrt(eps(omega)); metals reflect below and transmit above the plasma edge (Ashcroft-Mermin 1; Kittel 14)
    eqs['eq_LA12_7'] = sp.Eq(om_p, sp.sqrt(n*e**2/(eps0*eps_r*m_s)))  # semiconductor free-carrier plasma frequency in a host of permittivity eps_r (Yu-Cardona 6)
    eqs['eq_LA12_8'] = sp.Eq(E_F, hbar**2*(3*sp.pi**2*n)**sp.Rational(2, 3)/(2*m_e))  # Fermi energy of the free-electron gas (Ashcroft-Mermin 2)
    eqs['eq_LA12_9'] = sp.Eq(k_TF**2, 3*n*e**2/(2*eps0*E_F))  # Thomas-Fermi screening wavevector, degenerate analogue of lambda_D (Ashcroft-Mermin 17)
    eqs['eq_LA12_10'] = sp.Eq(lam_TF, 1/k_TF)  # Thomas-Fermi screening length (Ashcroft-Mermin 17)
    eqs['eq_LA12_11'] = sp.Eq(E_pl, hbar*om_p)  # bulk plasmon energy (~15 eV for Al) (Kittel 14)
    eqs['eq_LA12_12'] = sp.Eq(om_sp, om_p/sp.sqrt(2))  # surface plasmon of a Drude metal-vacuum interface (Kittel 14)
    eqs['eq_LA12_13'] = sp.Eq(om_Fr, om_p/sp.sqrt(3))  # Frohlich (dipolar) plasmon of a small Drude sphere in vacuum (Bohren-Huffman 12)
    eqs['eq_LA12_14'] = sp.Eq(lam_th, h/sp.sqrt(2*sp.pi*m_e*k_B*T))  # thermal de Broglie wavelength (Kittel-Kroemer 3)
    eqs['eq_LA12_15'] = sp.StrictGreaterThan(n*lam_th**3, 1)  # degeneracy criterion (equivalently k_B T < E_F): Saha/Maxwell fail, Fermi-Dirac BO6_6 applies (Ichimaru 1.1)
    eqs['eq_LA12_16'] = sp.Eq(r_s, (3/(4*sp.pi*n))**sp.Rational(1, 3)/a_B)  # Wigner-Seitz density parameter, degenerate coupling measure (Ashcroft-Mermin 2)
    return eqs


globals().update(_expand_langmuir_solid_state())


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
    # new engines from the 2026-09-22 coverage audit; not yet in the markdown
    'WI': ('Willis', None),
    'HO': ('Hodgkin', None),
    'JA': ('Janssen', None),
    'PO': ('Poncelet', None),
    'EM': ('Emmons', None),
    'MX': ('Maxwell', None),
    'LA': ('Langmuir', None),
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
# SCALE BOOKKEEPING: where a law holds, and where a simulator sits
# =====================================================================
# Optional, information-only bookkeeping (2026-09-22).  Nothing here
# changes a law or chooses a solver; it records what is already true of
# the physics so a simulator can be told, per law, whether its own domain
# and resolution put that law in range.
#
#   LawScale        where a law is known to hold: validity predicates on
#                   dimensionless groups (never a docstring), the law's own
#                   intrinsic length if it has one, and the catalogue
#                   criterion equations that already state the limit.
#   SimulatorScale  a simulator's domain size L, resolution dx and window;
#                   classifies each declared law as resolved / subgrid /
#                   larger than the domain, or valid / below its averaging
#                   length, depending on which way the law's length runs.
#
# ``resolution`` says which way:
#   "resolve"  the phenomenon lives AT the length (skin depth, sheath,
#              boundary layer): a grid must have dx <= length / k to see it.
#   "average"  the law is an AVERAGE over the length (continuum over the
#              mean free path, quasineutrality over lambda_D): cells must be
#              >= k * length for the law to be the right description.
# Laws may be named individually (``eq_NS1_1``) or by section (``LA4``,
# meaning every eq_LA4_*).

from dataclasses import dataclass as _scale_dataclass, field as _scale_field


@_scale_dataclass(frozen=True)
class LawScale:
    regime: str
    laws: tuple
    valid_if: tuple = ()
    groups: dict = _scale_field(default_factory=dict)
    length: object = None            # the law's intrinsic length, a sympy expression
    resolution: str = "resolve"      # "resolve" | "average" (see above)
    criteria: tuple = ()             # catalogue eq ids stating the limit
    source: str = ""

    def covered(self):
        """Every eq_* id this regime applies to, sections expanded."""
        names = set()
        for item in self.laws:
            if item.startswith("eq_"):
                names.add(item)
            else:
                names.update(n for n in globals() if _EQ_NAME_RE.match(n)
                             and n.startswith(f"eq_{item}_"))
        return tuple(sorted(names))

    def validity(self, values):
        """Each predicate as True / False / None (a symbol it needs is
        missing from ``values``).  ``values`` maps symbol names to numbers;
        group names may be given directly or derived from ``groups``."""
        subs = {sp.Symbol(k): v for k, v in values.items()}
        for name, expr in self.groups.items():
            if sp.Symbol(name) not in subs:
                derived = expr.subs(subs)
                if not derived.free_symbols:
                    subs[sp.Symbol(name)] = derived
        out = []
        for predicate in self.valid_if:
            judged = predicate.subs(subs)
            out.append(bool(judged) if judged in (sp.true, sp.false) else None)
        return tuple(out)


@_scale_dataclass(frozen=True)
class SimulatorScale:
    """A simulator's own scale: domain size, resolution, time window."""
    domain_m: float
    resolution_m: float
    window_s: float = 0.0
    resolve_factor: float = 2.0      # cells per length needed to resolve it

    @property
    def cells_across(self):
        return self.domain_m / self.resolution_m

    def classify(self, law_scale, values):
        """Where this simulator sits against one law's intrinsic length."""
        if law_scale.length is None:
            return "no intrinsic length"
        ell = sp.sympify(law_scale.length).subs({sp.Symbol(k): v for k, v in values.items()})
        if ell.free_symbols:
            return "length unknown (missing: %s)" % sorted(map(str, ell.free_symbols))
        ell = float(ell)
        k = self.resolve_factor
        if law_scale.resolution == "average":
            if ell * k > self.domain_m:
                return "domain smaller than the averaging length"
            return "valid (averaged)" if self.resolution_m >= k * ell else "below its averaging length"
        if ell > self.domain_m:
            return "larger than the domain"
        return "resolved" if self.resolution_m <= ell / k else "subgrid"


def _law_scales():
    Kn, lam, L = sp.symbols('Kn lambda_mfp L')
    Re_p, rho_f, v_p, d_p, mu_f = sp.symbols('Re_p rho_f v_p d_p mu_f')
    lam_D, N_D = sp.symbols('lambda_D N_D')
    d_s, delta, omega, mu, sigma, v, w, a = sp.symbols('d delta omega mu sigma v w a')
    Rm = sp.Symbol('Rm')
    h, eps, a_c, R_c, Phi, c, v_b = sp.symbols('h epsilon a_contact R_contact Phi c v_body')
    skin = sp.sqrt(2 / (omega * mu * sigma))
    return {
        "continuum": LawScale(
            "continuum", ("NS1", "NS2"), (Kn < sp.Rational(1, 100),),
            {"Kn": lam / L}, length=lam, resolution="average",
            source="Kn < 0.01 continuum; 0.01-0.1 slip; >10 free molecular (Bird 1994; Karniadakis 2005)"),
        "stokes_drag": LawScale(
            "Stokes (creeping) drag", ("eq_B7_2", "eq_B13_4", "eq_NS12_1"), (Re_p < 1,),
            {"Re_p": rho_f * v_p * d_p / mu_f},
            source="particle Reynolds number < 1 (Clift, Grace & Weber 1978)"),
        "newton_drag": LawScale(
            "Newton-regime drag", ("eq_B13_5",), (Re_p > 1000, Re_p < 2 * 10**5),
            {"Re_p": rho_f * v_p * d_p / mu_f},
            source="C_D ~ 0.44 plateau, 1e3 < Re_p < 2e5 (Clift, Grace & Weber 1978)"),
        "quasineutral_plasma": LawScale(
            "quasineutral fluid plasma", ("LA4",), (lam_D / L < 1, N_D > 1),
            length=lam_D, resolution="average", criteria=("eq_LA1_10", "eq_LA1_11"),
            source="lambda_D << L and N_D >> 1 (Chen 1.6); sheaths need the resolved description"),
        "debye_sheath": LawScale(
            "sheath / non-neutral region",
            tuple(f"eq_LA6_{n}" for n in (10, 11, 12, 13, 14, 15, 17, 18, 24, 25, 26)), (),
            length=lam_D, resolution="resolve", criteria=("eq_LA1_10",),
            source="sheaths are a few lambda_D thick (Lieberman-Lichtenberg 6)"),
        "thin_lamination_eddy": LawScale(
            "thin-conductor eddy currents", ("eq_F18_10", "eq_F18_11", "eq_F18_12"), (d_s / delta < 1,),
            {"delta": skin}, length=skin, resolution="resolve", criteria=("eq_F5_5",),
            source="thickness << skin depth, field uniform through the sheet (Fitzgerald-Kingsley)"),
        "low_Rm_eddy_drag": LawScale(
            "low magnetic Reynolds number eddy drag", ("eq_F18_12",), (Rm < 1,),
            {"Rm": mu * sigma * v * d_s},
            source="eddy field does not distort the applied field (Wiederick 1987; Reitz 1970)"),
        "thin_wall_pipe": LawScale(
            "thin-walled pipe (magnet in a pipe)", ("eq_F18_16", "eq_F18_17"), (w / a < sp.Rational(1, 10),),
            source="wall w << radius a (Levin et al. 2006)"),
        "shallow_water": LawScale(
            "shallow water (Saint-Venant)", ("NS9",), (h / L < sp.Rational(1, 20),),
            source="depth << horizontal wavelength, hydrostatic pressure (Vreugdenhil 1994)"),
        "small_strain": LawScale(
            "small (infinitesimal) strain", ("eq_BR2_1",), (sp.Abs(eps) < sp.Rational(1, 100),),
            source="|eps| << 1; beyond it use Green-Lagrange (BR13) (Timoshenko & Goodier)"),
        "hertz_contact": LawScale(
            "Hertz contact", ("eq_N5_8",), (a_c / R_c < sp.Rational(1, 10),),
            source="contact radius << curvature radius, frictionless elastic (Johnson, Contact Mechanics)"),
        "weak_field_gravity": LawScale(
            "weak-field (Newtonian) gravity", ("eq_E6_1", "eq_E6_2"), (sp.Abs(Phi) / c**2 < sp.Rational(1, 100),),
            source="|Phi|/c^2 << 1 (Misner-Thorne-Wheeler 18)"),
        "non_relativistic": LawScale(
            "non-relativistic mechanics", ("N1",), (v_b / c < sp.Rational(1, 10),),
            source="v << c; gamma - 1 < 0.5% at v/c = 0.1"),
    }


LAW_SCALES = _law_scales()

for _regime, _scale in LAW_SCALES.items():   # a declaration must name real laws
    for _name in (*_scale.covered(), *_scale.criteria):
        assert _name in globals(), f"LAW_SCALES[{_regime!r}] names unknown law {_name}"
    assert _scale.covered(), f"LAW_SCALES[{_regime!r}] covers no law"


def scales_of(eq_name):
    """Every declared regime that covers ``eq_name``."""
    return tuple(s for s in LAW_SCALES.values() if eq_name in s.covered())


def scale_report(simulator, values, regimes=None):
    """Per declared regime: validity predicates and where ``simulator`` sits.

    ``values`` supplies the physical symbols (and/or group values) by name;
    ``L`` defaults to the simulator's domain.  Information only."""
    values = {"L": simulator.domain_m, **values}
    rows = []
    for name, scale in LAW_SCALES.items():
        if regimes is not None and name not in regimes:
            continue
        rows.append({"regime": name, "laws": len(scale.covered()),
                     "valid": scale.validity(values),
                     "grid": simulator.classify(scale, values)})
    return rows


# =====================================================================
# LAW DECLARATIONS: what SymPy's spelling cannot say
# =====================================================================
# The catalogue writes one placeholder, nabla, for grad, div and curl,
# and Integral(f, A) for an integral over a surface/volume domain A.
# These declarations say, per occurrence, which operator and which
# domain kind is meant -- the input turing/src/compiler/bitops.declare
# needs to complete a Manifold or Integral declaration.  Keys are the
# str() of the occurrence (nabla node) or of the bare integration symbol.
# nabla_k / nabla_R / nabla_v / nabla_mu,nu are gradients in wavevector,
# nuclear-coordinate, velocity and spacetime (covariant) space.

# 'nabla': str(occurrence) -> kind in {grad, div, curl, laplacian_0, laplacian_1, advective}.
#          A tuple value gives the kinds of repeated identical occurrences in sp.preorder_traversal order.
#          Nested scalar nabla(nabla(x)) is declared outer 'div', inner 'grad' unless noted.
# 'domain': str(bare integration symbol) -> {'kind': surface|volume|line|time|population, 'closed': bool (only when implied)}.
LAW_DECLARATIONS = {
    'eq_N4_3': {'nabla': {'nabla(Phi)': 'grad'}},   # gravitational force = -m grad(potential)
    'eq_T4_2': {'domain': {'x': {'kind': 'line'}}},   # element stiffness integrated along the element axis
    'eq_T7_3': {'nabla': {'nabla(nabla(w(x, y, t)))': 'laplacian_0', 'nabla(w(x, y, t))': 'laplacian_0'}},   # Kirchhoff plate D nabla^4 w: each placeholder is a scalar Laplacian (source comment: nabla^4)
    'eq_T8_1': {'domain': {'A': {'kind': 'surface', 'closed': False}}},   # second moment of area over the open cross-section
    'eq_T8_6': {'domain': {'s': {'kind': 'line', 'closed': True}}},   # Bredt: contour integral ds/t around the closed thin-wall cell
    'eq_F1_1': {'nabla': {'nabla(E(t))': 'curl'}},   # Faraday: dB/dt = -curl E
    'eq_F1_2': {'nabla': {'nabla(H(t))': 'curl'}},   # Ampere-Maxwell: dD/dt = curl H - J
    'eq_F1_3': {'nabla': {'nabla(D(t))': 'div'}},   # Gauss: div D = rho_f
    'eq_F1_4': {'nabla': {'nabla(B(t))': 'div'}},   # no monopoles: div B = 0
    'eq_F1_5': {'nabla': {'nabla(J_f(t))': 'div'}},   # charge continuity
    'eq_F3_3': {'nabla': {'nabla(S)': 'div'}},   # Poynting theorem: div S
    'eq_F3_7': {'domain': {'V': {'kind': 'volume'}, 'A': {'kind': 'surface', 'closed': True}}},   # field momentum in V; Maxwell stress over the closed boundary of V
    'eq_F4_1': {'nabla': {'nabla(epsilon*nabla(phi))': 'div', 'nabla(phi)': 'grad'}},   # Poisson: div(eps grad phi)
    'eq_F4_2': {'nabla': {'nabla(phi)': 'grad'}},   # electrostatic E = -grad phi
    'eq_F4_3': {'nabla': {'nabla(A)': 'curl'}},   # B = curl A
    'eq_F4_4': {'nabla': {'nabla(phi)': 'grad'}},   # E = -grad phi - dA/dt
    'eq_F7_1': {'nabla': {'nabla(nabla(E_m)/mu)': 'curl', 'nabla(E_m)': 'curl'}},   # cavity eigenproblem curl(curl E / mu)
    'eq_F12_1': {'nabla': {'nabla(nabla(E(x, y, z))/mu)': 'curl', 'nabla(E(x, y, z))': 'curl'}},   # driven vector wave equation curl(curl E / mu)
    'eq_F13_6': {'nabla': {'nabla(n_q(t))': 'grad'}},   # drift-diffusion flux: -D grad n
    'eq_F13_7': {'nabla': {'nabla(Gamma_q)': 'div'}},   # species balance: -div Gamma
    'eq_F14_4': {'domain': {'dV': {'kind': 'volume'}}},   # potential of a volume charge distribution
    'eq_F14_5': {'domain': {'dA': {'kind': 'surface', 'closed': True}}},   # Gauss's law integral form: closed Gaussian surface
    'eq_F14_17': {'domain': {'dV': {'kind': 'volume'}}},   # field energy over all space
    'eq_F14_18': {'domain': {'dV': {'kind': 'volume'}}},   # energy from charge x potential over the charge volume
    'eq_F14_42': {'nabla': {'nabla(dot(p_vec, E_vec))': 'grad'}},   # force on dipole = grad(p . E)
    'eq_F14_45': {'nabla': {'nabla(E**2)': 'grad'}},   # induced-dipole (dielectrophoretic) force ~ grad E^2
    'eq_F16_1': {'domain': {'dl': {'kind': 'line'}}},   # Biot-Savart along the current path
    'eq_F16_2': {'domain': {'dl': {'kind': 'line', 'closed': True}}},   # Ampere's law: closed Amperian loop
    'eq_F16_13': {'nabla': {'nabla(dot(m_vec, B_vec))': 'grad'}},   # force on magnetic dipole = grad(m . B)
    'eq_F16_37': {'nabla': {'nabla(B**2)': 'grad'}},   # diamagnetic force ~ grad B^2
    'eq_F17_22': {'domain': {'B': {'kind': 'line', 'closed': True}}},   # hysteresis loss: loop integral of H dB in the B-H state plane (not spatial)
    'eq_F18_1': {'domain': {'dA': {'kind': 'surface', 'closed': False}}},   # Faraday integral form: open surface bounded by the loop
    'eq_F18_6': {'nabla': {'nabla(cross(v_vec, B_vec(t)))': 'curl', 'nabla(nabla(B_vec(t)))': 'curl', 'nabla(B_vec(t))': 'curl'}},   # induction: curl(v x B) and diffusion -curl curl B / (mu sigma) (= laplacian B for div B = 0)
    'eq_F18_9': {'domain': {'dV': {'kind': 'volume'}}},   # eddy loss J^2/sigma over the conductor volume
    'eq_F19_3': {'domain': {'dV': {'kind': 'volume'}}},   # field angular momentum over all space
    'eq_F19_30': {'nabla': {'nabla(J_s)': 'curl'}},   # London equation: curl J_s = -n e^2 B / m
    'eq_NS1_1': {'nabla': {'nabla(rho(t)*u(t))': 'div'}},   # mass continuity: div(rho u)
    'eq_NS1_2': {'nabla': {'nabla(I_3*p(t) + outer(u(t), u(t))*rho(t))': 'div', 'nabla(tau(t))': 'div'}},   # momentum: div(rho u(x)u + p I) and div of viscous stress
    'eq_NS1_3': {'nabla': {'nabla((E(t)*rho(t) + p(t))*u(t))': 'div', 'nabla(-q(t) + tau(t)*u(t))': 'div'}},   # energy: div of enthalpy flux and of (tau.u - q)
    'eq_NS1_5': {'nabla': {'nabla(Y_s(t)*rho(t)*u(t) + j_s(t))': 'div'}},   # species: div of convective + diffusive flux
    'eq_NS2_1': {'nabla': {'nabla(u(t))': 'grad'}},   # strain rate: sym(grad u), both occurrences
    'eq_NS2_2': {'nabla': {'nabla(u(t))': 'div'}},   # Newtonian stress: (div u) I in deviatoric and bulk terms, both occurrences
    'eq_NS2_3': {'nabla': {'nabla(T)': 'grad'}},   # Fourier conduction -k grad T
    'eq_NS2_6': {'nabla': {'nabla(Y_s(t))': 'grad'}},   # Fickian diffusion flux -rho D grad Y
    'eq_NS2_8': {'nabla': {'nabla(x_s)': 'grad'}},   # Maxwell-Stefan: mole-fraction gradient
    'eq_NS4_1': {'nabla': {'nabla(u(t))': 'div'}},   # incompressibility: div u = 0
    'eq_NS4_2': {'nabla': {'nabla(u(t))': ('advective', 'grad'), 'nabla(p(t))': 'grad', 'nabla(nabla(u(t)))': 'div'}},   # incompressible NS: (u.grad)u advective; mu div(grad u) componentwise vector Laplacian; grad p
    'eq_NS7_1': {'nabla': {'nabla(u(t))': 'div'}},   # Boussinesq eddy viscosity: (div u) I term
    'eq_NS14_5': {'nabla': {'nabla(nabla(p(t, x, y, z)))': 'div', 'nabla(p(t, x, y, z))': 'grad', 'nabla(u^*)': 'div'}},   # pressure Poisson: div grad p = rho/dt div u*
    'eq_NS14_6': {'nabla': {'nabla(p(t, x, y, z))': 'grad'}},   # projection correction u = u* - dt/rho grad p
    'eq_NS14_7': {'nabla': {'nabla(c_color(t, x, y, z))': 'grad'}},   # CSF surface tension sigma kappa grad c
    'eq_NS14_8': {'nabla': {'nabla(c_color(t, x, y, z))': 'grad'}},   # interface normal grad c / |grad c|
    'eq_NS14_9': {'nabla': {'nabla(hat{n})': 'div'}},   # curvature = -div n_hat
    'eq_B1_4': {'nabla': {'nabla(p)': 'grad'}},   # geostrophic balance: pressure gradient
    'eq_B1_5': {'domain': {'p': {'kind': 'line', 'closed': True}}},   # Bjerknes: circuit integral of dp/rho around a closed material loop
    'eq_B1_6': {'domain': {'l': {'kind': 'line', 'closed': True}}},   # circulation around a closed loop
    'eq_B1_7': {'nabla': {'nabla(K*nabla(chi))': 'div', 'nabla(chi)': 'grad'}},   # eddy diffusion div(K grad chi)
    'eq_B8_1': {'nabla': {'nabla(v_p*n(m, x, t))': 'div'}},   # population balance: spatial flux divergence
    'eq_B10_2': {'domain': {'s': {'kind': 'line', 'closed': False}}},   # Beer-Lambert transmittance along the optical path
    'eq_BR2_1': {'nabla': {'nabla(u)': 'grad'}},   # small strain sym(grad u), both occurrences
    'eq_BR5_3': {'nabla': {'nabla(D*nabla(c))': 'div', 'nabla(c)': 'grad'}},   # Fickian diffusion div(D grad c)
    'eq_BR5_4': {'nabla': {'nabla(mu_c)': 'grad'}},   # chemical-potential-driven flux -M grad mu
    'eq_BR6_1': {'nabla': {'nabla(c)': 'grad', 'nabla(eta)': 'grad'}, 'domain': {'V': {'kind': 'volume'}}},   # Ginzburg-Landau free energy with |grad|^2 terms over the body
    'eq_BR6_2': {'nabla': {'nabla(M_c*nabla(Derivative(mathcal{F}, c)))': 'div', 'nabla(Derivative(mathcal{F}, c))': 'grad'}},   # Cahn-Hilliard div(M grad dF/dc)
    'eq_BR7_6': {'nabla': {'nabla(d)': 'grad'}, 'domain': {'V': {'kind': 'volume'}}},   # phase-field fracture energy with |grad d|^2 over the body
    'eq_BR8_2': {'nabla': {'nabla_k(E_n)': 'grad'}},   # band group velocity = grad_k E / hbar (wavevector space)
    'eq_BR9_1': {'nabla': {'nabla(epsilon*nabla(phi))': 'div', 'nabla(phi)': 'grad'}},   # semiconductor Poisson div(eps grad phi)
    'eq_BR9_2': {'nabla': {'nabla(phi)': 'grad'}},   # E = -grad phi
    'eq_BR9_3': {'nabla': {'nabla(n)': 'grad'}},   # electron drift-diffusion current: diffusion term
    'eq_BR9_4': {'nabla': {'nabla(p)': 'grad'}},   # hole drift-diffusion current: diffusion term
    'eq_BR9_5': {'nabla': {'nabla(J_n)': 'div'}},   # electron continuity div J_n
    'eq_BR9_6': {'nabla': {'nabla(J_p)': 'div'}},   # hole continuity div J_p
    'eq_H2_3': {'nabla': {'nabla(psi)': 'grad'}},   # probability current Im(psi* grad psi)
    'eq_H2_4': {'nabla': {'nabla(j_P)': 'div'}},   # probability continuity
    'eq_H5_1': {'nabla': {'nabla_R(E_{BO})': 'grad'}},   # Born-Oppenheimer force on nuclei = -grad_R E (nuclear coordinates)
    'eq_H5_3': {'nabla': {'nabla_R(E_{BO})': 'grad'}},   # equilibrium geometry: grad_R E = 0
    'eq_H7_5': {'domain': {'r_i': {'kind': 'volume'}, "r'": {'kind': 'volume'}}},   # Kohn-Sham: external-potential term over all space; Hartree term a double volume integral over r and r'
    'eq_H8_6': {'domain': {'x': {'kind': 'line', 'closed': False}}},   # WKB tunnelling exponent between turning points
    'eq_C3_2': {'domain': {'tau': {'kind': 'time'}}},   # survival probability: hazard integrated over time
    'eq_C5_1': {'domain': {'x': {'kind': 'line', 'closed': False}}},   # attenuation along the beam path
    'eq_C6_1': {'domain': {'E': {'kind': 'population'}}},   # reaction rate: cross-section x flux over the particle energy spectrum
    'eq_E5_10': {'nabla': {'nabla(T^{mu,nu})': 'div'}},   # stress-energy conservation: covariant divergence
    'eq_E6_2': {'nabla': {'nabla(nabla(Phi))': 'div', 'nabla(Phi)': 'grad'}},   # Newtonian gravity Poisson div grad Phi
    'eq_E6_3': {'nabla': {'nabla_mu(xi_nu)': 'covariant', 'nabla_nu(xi_mu)': 'covariant'}},   # Killing equation: symmetrized spacetime covariant derivative
    'eq_L5_1': {'nabla': {'nabla(c_s)': 'grad', 'nabla(phi)': 'grad'}},   # Nernst-Planck: diffusion + migration gradients
    'eq_FO1_1': {'nabla': {'nabla(T(t))': 'grad'}},   # Fourier's law
    'eq_FO1_2': {'nabla': {'nabla(k(t)*nabla(T(t)))': 'div', 'nabla(T(t))': 'grad'}},   # heat equation div(k grad T)
    'eq_FO1_3': {'nabla': {'nabla(T(t))': 'grad'}},   # entropy production k |grad T|^2 / T^2
    'eq_FO2_1': {'domain': {'T(t)': {'kind': 'line', 'closed': False}}},   # sensible enthalpy: c_p along the temperature state coordinate T0 -> T (not spatial)
    'eq_FO6_1': {'nabla': {'nabla(I_nu(t))': 'grad'}, 'domain': {"Omega'": {'kind': 'surface', 'closed': True}}},   # RTE: Omega . grad I streaming; in-scattering over the closed unit sphere of directions
    'eq_FO7_7': {'nabla': {'nabla(T(t))': 'grad'}},   # thermoelectric current: Seebeck term
    'eq_FO7_8': {'nabla': {'nabla(T(t))': 'grad'}},   # thermoelectric heat flux: Peltier + conduction
    'eq_BO1_1': {'nabla': {'nabla(f_s(t))': 'grad', 'nabla_v(f_s(t))': 'grad'}},   # Boltzmann: v . grad_x f streaming and (F/m) . grad_v f in velocity space
    'eq_BO2_1': {'domain': {'v(t)': {'kind': 'population'}}},   # number density = zeroth velocity moment
    'eq_BO2_3': {'domain': {'v(t)': {'kind': 'population'}}},   # momentum density = first velocity moment
    'eq_BO2_4': {'domain': {'v(t)': {'kind': 'population'}}},   # pressure = second central moment
    'eq_BO2_5': {'domain': {'v(t)': {'kind': 'population'}}},   # translational energy moment
    'eq_BO2_6': {'domain': {'v(t)': {'kind': 'population'}}},   # heat flux = third central moment
    'eq_BO4_1': {'domain': {'v(t)': {'kind': 'population'}}},   # collision invariants over velocity space
    'eq_BO8_1': {'nabla': {'nabla(psi(t))': 'grad'}, 'domain': {"E'": {'kind': 'population'}, "Omega'": {'kind': 'surface', 'closed': True}}},   # neutron transport: Omega . grad psi; scattering source over incoming energy E' and the unit sphere of directions Omega'
    'eq_NO1_4': {'nabla': {'nabla(j^mu)': 'div'}},   # conserved current: 4-divergence
    'eq_NO4_4': {'nabla': {'nabla(D)': 'div'}},   # Gauss-law residual div D - rho
    'eq_NO4_5': {'nabla': {'nabla(J_f)': 'div'}},   # charge continuity
    'eq_X2_1': {'domain': {'A': {'kind': 'surface'}}},   # mass flow through a (moving) control surface
    'eq_X2_2': {'domain': {'A': {'kind': 'surface'}}},   # species mass flow through a control surface
    'eq_Z2_1': {'nabla': {'nabla(G)': 'grad'}},   # G-equation: u . grad G and S_T |grad G|
    'eq_AR1_1': {'domain': {'partial V': {'kind': 'surface', 'closed': True}}},   # buoyancy: pressure over the closed wetted surface
    'eq_AR1_3': {'domain': {'V_{sub}': {'kind': 'volume'}}},   # buoyancy: weight of displaced fluid over submerged volume
    'eq_HO4_2': {'domain': {'dA_m': {'kind': 'surface'}}},   # Helfrich energy over the membrane surface
    'eq_LA3_1': {'nabla': {'nabla_v(f_s(t, x, v))': 'grad', 'nabla(f_s(t, x, v))': 'grad'}},   # Vlasov: v . grad_x f  [velocity-space term: grad_v f]
    'eq_LA3_2': {'nabla': {'nabla_v(f_s(t, x, v))': 'grad', 'nabla(f_s(t, x, v))': 'grad'}},   # Fokker-Planck/Boltzmann: v . grad_x f  [velocity-space term: grad_v f]
    'eq_LA3_21': {'nabla': {'nabla(nabla(phi))': 'div', 'nabla(phi)': 'grad'}},   # Debye screening div grad phi = phi / lambda_D^2
    'eq_LA4_1': {'nabla': {'nabla(\\mathbf{u}_s*n_s)': 'div'}},   # fluid species continuity
    'eq_LA4_2': {'nabla': {'nabla(\\mathbf{u}_s)': 'advective', 'nabla(p_s)': 'grad'}},   # species momentum: (u.grad)u and pressure gradient
    'eq_LA4_3': {'nabla': {'nabla(\\mathbf{Q}_e)': 'div'}},   # electron energy: div of energy flux
    'eq_LA4_6': {'nabla': {'nabla(\\mathbf{\\Gamma}_e)': 'div'}},   # electron continuity
    'eq_LA4_7': {'nabla': {'nabla(\\mathbf{\\Gamma}_+)': 'div'}},   # positive-ion continuity
    'eq_LA4_8': {'nabla': {'nabla(\\mathbf{\\Gamma}_-)': 'div'}},   # negative-ion continuity
    'eq_LA4_9': {'nabla': {'nabla(nabla(phi))': 'div', 'nabla(phi)': 'grad'}},   # plasma Poisson div grad phi
    'eq_LA4_13': {'nabla': {'nabla(n)': 'grad'}},   # ambipolar field ~ grad n / n
    'eq_LA4_14': {'nabla': {'nabla(p_e)': 'grad'}},   # generalized Ohm: electron pressure gradient
    'eq_LA4_15': {'nabla': {'nabla(\\mathbf{u}*rho)': 'div'}},   # MHD continuity
    'eq_LA4_16': {'nabla': {'nabla(p)': 'grad', 'nabla(\\mathbf{u})': 'advective'}},   # MHD momentum: grad p and (u.grad)u
    'eq_LA4_17': {'nabla': {'nabla(\\mathbf{B})': 'curl'}},   # Ampere (MHD): J = curl B / mu0
    'eq_LA4_18': {'nabla': {'nabla(nabla(\\mathbf{B}))': 'div', 'nabla(\\mathbf{B})': 'grad', 'nabla(cross(\\mathbf{u}, \\mathbf{B}))': 'curl'}},   # induction: eta/mu0 componentwise div grad B + curl(u x B)
    'eq_LA4_20': {'nabla': {'nabla(p)': 'grad', 'nabla(\\mathbf{u})': 'div'}},   # adiabatic pressure: -u.grad p - gamma p div u
    'eq_LA5_15': {'domain': {"V'": {'kind': 'volume'}}},   # photoionization source: kernel over emitting volume
    'eq_LA8_1': {'nabla': {'nabla(nabla(phi))': 'div', 'nabla(phi)': 'grad'}},   # Laplace equation div grad phi = 0
    'eq_LA8_19': {'domain': {'s': {'kind': 'line', 'closed': False}}},   # optical depth along the ray
    'eq_LA8_24': {'domain': {'t': {'kind': 'time'}}},   # deposited energy density: power integrated over time
    'eq_LA11_20': {'domain': {'z': {'kind': 'line', 'closed': False}}},   # line-of-sight emission integral
    'eq_LA3_3': {'nabla': {'nabla_v(H_s)': 'grad', 'nabla_v(G_s)': 'grad', 'nabla_v(nabla_v(G_s))': 'grad', 'nabla_v(f_s(t, x, v)*nabla_v(H_s))': 'div', 'nabla_v(f_s(t, x, v)*nabla_v(nabla_v(G_s)))': 'div', 'nabla_v(nabla_v(f_s(t, x, v)*nabla_v(nabla_v(G_s))))': 'div'}},   # Fokker-Planck (Rosenbluth form), all velocity space: grad H, grad G, Hessian of G (grad of grad), div of the friction flux, and the double divergence of the diffusion tensor (tensor -> vector -> scalar)
}

#: Operator kinds a LAW_DECLARATIONS 'nabla' entry may name, and domain kinds
#: a 'domain' entry may name (bitops.MANIFOLD_OPERATORS / Domain meanings).
DECLARED_OPERATOR_KINDS = frozenset({"grad", "div", "curl", "laplacian_0", "laplacian_1",
                                     "advective", "covariant"})
DECLARED_DOMAIN_KINDS = frozenset({"surface", "volume", "line", "time", "population"})


def check_law_declarations():
    """Every nabla-family occurrence and every domain-measure integral in the
    catalogue has a declaration, and no declaration is stale.  Returns the
    list of problems (empty when consistent)."""
    from sympy.core.function import AppliedUndef

    problems = []
    for engine in _discover_equations().values():
        for name, law in engine.items():
            decl = LAW_DECLARATIONS.get(name, {})
            declared_nabla = decl.get("nabla", {})
            declared_domain = decl.get("domain", {})
            seen_nabla, seen_domain = set(), set()
            for node in sp.preorder_traversal(law):
                if isinstance(node, AppliedUndef) and str(node.func).split("_")[0] == "nabla":
                    key = str(node)
                    seen_nabla.add(key)
                    kinds = declared_nabla.get(key)
                    kinds = kinds if isinstance(kinds, tuple) else (kinds,)
                    if kinds == (None,):
                        problems.append(f"{name}: undeclared {key}")
                    elif not set(kinds) <= DECLARED_OPERATOR_KINDS:
                        problems.append(f"{name}: unknown operator kind {kinds} for {key}")
                if isinstance(node, sp.Integral):
                    for limit in node.limits:
                        if len(limit) == 1 and (limit[0] not in node.function.free_symbols
                                                or str(limit[0]) in declared_domain):
                            key = str(limit[0])
                            seen_domain.add(key)
                            kind = declared_domain.get(key, {}).get("kind")
                            if kind not in DECLARED_DOMAIN_KINDS:
                                problems.append(f"{name}: domain {key} kind {kind!r}")
            problems += [f"{name}: stale nabla key {k}" for k in set(declared_nabla) - seen_nabla]
            problems += [f"{name}: stale domain key {k}" for k in set(declared_domain) - seen_domain]
    return problems


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
