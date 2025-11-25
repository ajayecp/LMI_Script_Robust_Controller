import numpy as np
import cvxpy as cp
from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import warnings
import itertools
from scipy.optimize import fsolve

print("Iniciando o script de projeto e validação do controlador robusto H-INFINITO (Versão 2).")

print("\n--- Parte 1: Definindo os parâmetros do modelo ---")
# Parâmetros Físicos do Reator 
Vr = 0.23      # Volume do reator [m^3]
Vc = 0.21      # Volume do fluido de arrefecimento [m^3]
qr_s = 0.015   # Vazão de reagente (estado estacionário) [m^3/min]
qc_s = 0.004   # Vazão de arrefecimento (estado estacionário) [m^3/min]
rho_r = 1020   # Densidade do reagente [kg/m^3]
rho_c = 998    # Densidade do fluido de arrefecimento [kg/m^3]
cpr = 4.02     # Capacidade calorífica do reagente [kJ/kg/K]
cpc = 4.182    # Capacidade calorífica do fluido de arrefecimento [kJ/kg/K]
Ah = 1.51      # Área de transferência de calor [m^2]
U = 42.8       # Coeficiente de transferência de calor [kJ/m^2/min/K]
g1 = 9850      # Energia de ativação 1 (E1/R) [K]
g2 = 22019     # Energia de ativação 2 (E2/R) [K]
cAf = 4.22     # Concentração de A na alimentação [kmol/m^3]
Trf = 310      # Temperatura da alimentação [K]
Tcf = 288      # Temperatura do fluido de arrefecimento na alimentação [K]
h1 = -8.6e4    # Entalpia da reação 1 [kJ/kmol]
h2 = -5.5e4    # Entalpia da reação 2 [kJ/kmol]
k10 = 1.55e11  # Fator pré-exponencial 1 [min^-1]
k20 = 8.55e26  # Fator pré-exponencial 2 [min^-1]
x_op = np.array([1.8614, 1.0113, 338.41, 328.06])
C0 = np.array([[0, 0, 1, 0]])

print("\n--- Parte 2: Gerando os 16 vértices do sistema ---")
h1_range = [-8.8e4, -8.4e4]
h2_range = [-5.7e4, -5.3e4]
k10_range = [1.5e11, 1.6e11]
k20_range = [4.95e26, 12.15e26]
param_ranges = [h1_range, h2_range, k10_range, k20_range]
param_combinations = list(itertools.product(*param_ranges))
def cstr_steady_state(x, params):
    cA, cB, Tr, Tc = x
    h1_p, h2_p, k10_p, k20_p = params
    k1 = k10_p * np.exp(-g1 / Tr)
    k2 = k20_p * np.exp(-g2 / Tr)
    eq1 = (qr_s / Vr) * (cAf - cA) - k1 * cA - k2 * cA
    eq2 = -(qr_s / Vr) * cB + k1 * cA
    eq3 = (qr_s / Vr) * (Trf - Tr) + ((-h1_p * k1 - h2_p * k2) / (rho_r * cpr)) * cA + ((U * Ah) / (Vr * rho_r * cpr)) * (Tc - Tr)
    eq4 = (qc_s / Vc) * (Tcf - Tc) + ((U * Ah) / (Vc * rho_c * cpc)) * (Tr - Tc)
    return [eq1, eq2, eq3, eq4]
def calculate_jacobians(x_ss, params):
    cA, cB, Tr, Tc = x_ss
    h1_p, h2_p, k10_p, k20_p = params
    k1 = k10_p * np.exp(-g1 / Tr)
    k2 = k20_p * np.exp(-g2 / Tr)
    dk1_dTr = k1 * g1 / (Tr ** 2)
    dk2_dTr = k2 * g2 / (Tr ** 2)
    A = np.zeros((4, 4)); B = np.zeros((4, 2))
    A[0, 0] = -qr_s / Vr - k1 - k2; A[0, 2] = -cA * (dk1_dTr + dk2_dTr); A[1, 0] = k1; A[1, 1] = -qr_s / Vr; A[1, 2] = cA * dk1_dTr
    A[2, 0] = (-h1_p * k1 - h2_p * k2) / (rho_r * cpr); A[2, 2] = -qr_s / Vr + ((-h1_p * dk1_dTr - h2_p * dk2_dTr) / (rho_r * cpr)) * cA - (U * Ah) / (Vr * rho_r * cpr)
    A[2, 3] = (U * Ah) / (Vr * rho_r * cpr); A[3, 2] = (U * Ah) / (Vc * rho_c * cpc); A[3, 3] = -qc_s / Vc - (U * Ah) / (Vc * rho_c * cpc)
    B[0, 0] = (cAf - cA) / Vr; B[1, 0] = -cB / Vr; B[2, 0] = (Trf - Tr) / Vr; B[3, 1] = (Tcf - Tc) / Vc
    return A, B
A_vertices = []; B_vertices = []; initial_guess = x_op
print("Iniciando o cálculo dos 16 vértices...")
for i, current_params in enumerate(param_combinations):
    x_ss_i, _, ier, _ = fsolve(cstr_steady_state, initial_guess, args=(current_params,), full_output=True)
    if ier != 1: print(f"Atenção: Vértice {i + 1} pode não ter convergido.")
    Ai, Bi = calculate_jacobians(x_ss_i, current_params)
    A_vertices.append(Ai); B_vertices.append(Bi)
    initial_guess = x_ss_i
print(f"Cálculo finalizado. {len(A_vertices)} matrizes de vértice geradas.\n")


print("\n--- Parte 3: Projetando o controlador H-Infinito com LMIs (Revisado) ---")

# --- 3.1 Definições do Problema  ---

q_const = 1.0
r_const = 0.1 # Valor drasticamente reduzido
print(f"Usando novos pesos: q_const={q_const}, r_const={r_const}")

Q = q_const * np.diag([1, 1, 1, 1]) # Ponderando todos os estados igualmente
R = r_const * np.diag([1, 1])      # Penalidade de controle muito menor
I_4 = np.eye(4)
I_2 = np.eye(2)

Cz = sqrtm(Q)
Dzu = sqrtm(R)
Cz_aug = np.vstack([Cz, np.zeros((2, 4))])
Dzu_aug = np.vstack([np.zeros((4, 2)), Dzu])
Dzw = np.zeros((6, 2))

# Adicionando um decaimento mínimo para robustez
alpha = 0.01
print(f"Exigindo um decaimento mínimo (margem de estabilidade) alpha = {alpha}")

# --- 3.2 Resolvendo a LMI H-Infinito ---
X = cp.Variable((4, 4), symmetric=True)
Y = cp.Variable((2, 4))
gamma = cp.Variable()

print("Resolvendo a LMI H-Infinito para encontrar X, Y e gamma...")
constraints = []

for Ai, Bi in zip(A_vertices, B_vertices):
    Bw_i = Bi 
    
    # LMI_11 inclui o termo de decaimento alpha
    LMI_11 = Ai @ X + X @ Ai.T + Bi @ Y + Y.T @ Bi.T + 2 * alpha * X
    
    LMI_12 = Bw_i
    LMI_13 = X @ Cz_aug.T + Y.T @ Dzu_aug.T
    LMI_21 = Bw_i.T
    LMI_22 = -gamma * I_2
    LMI_23 = Dzw.T
    LMI_31 = Cz_aug @ X + Dzu_aug @ Y
    LMI_32 = Dzw
    LMI_33 = -gamma * np.eye(6)
    
    LMI = cp.bmat([
        [LMI_11, LMI_12, LMI_13],
        [LMI_21, LMI_22, LMI_23],
        [LMI_31, LMI_32, LMI_33]
    ])
    
    constraints.append(LMI << 0)

constraints.append(X >> 1e-6 * I_4)

problem = cp.Problem(cp.Minimize(gamma), constraints)
problem.solve(solver=cp.SCS, verbose=True, eps=1e-8, max_iters=250000) # Tolerância um pouco mais rígida

# --- 3.3 Obtendo o Ganho do Controlador ---
if "optimal" in problem.status and X.value is not None:
    X_val = X.value
    Y_val = Y.value
    gamma_val = gamma.value
    
    K_val = Y_val @ np.linalg.inv(X_val) 
    
    print("\nSolução LMI H-Infinito encontrada!")
    print(f"Valor ótimo de gamma (ganho H-infinito): {gamma_val}")
    print("\nControlador robusto K (realimentação de estados) encontrado:")
    print(K_val)

    print("\n--- Parte 4: Validando o controlador ---")

    # --- 4.1 Verificação da Estabilidade ---
    print("\nVerificando a estabilidade dos vértices em malha fechada...")
    all_stable = True
    for i, (Ai, Bi) in enumerate(zip(A_vertices, B_vertices)):
        A_cl = Ai + Bi @ K_val
        eigenvalues = np.linalg.eigvals(A_cl)
        
        # Checa se TODOS os autovalores têm parte real negativa
        if np.all(np.real(eigenvalues) < 0):
            print(f"Vértice {i + 1}: Estável. (max Re(eig) = {np.max(np.real(eigenvalues)):.4f})")
        else:
            print(f"Vértice {i + 1}: INSTÁVEL. (max Re(eig) = {np.max(np.real(eigenvalues)):.4f})")
            all_stable = False

    if not all_stable:
        warnings.warn("Atenção: Nem todos os vértices do sistema são estáveis com o controlador encontrado.")

    # --- 4.2 Simulação do Sistema Não Linear ---
    print("\nIniciando a simulação do modelo não linear...")

    def cstr_closed_loop(t, x, x_op, K, qr_ss, qc_ss):
        cA, cB, Tr, Tc = x
        x_dev = x - x_op
        u = K @ x_dev
        qr = qr_ss + u[0]
        qc = qc_ss + u[1]
        qr = max(0, qr)
        qc = max(0, qc)
        k1 = k10 * np.exp(-g1 / Tr)
        k2 = k20 * np.exp(-g2 / Tr)
        dcA_dt = (qr / Vr) * (cAf - cA) - k1 * cA - k2 * cA
        dcB_dt = -(qr / Vr) * cB + k1 * cA
        dTr_dt = (qr / Vr) * (Trf - Tr) + ((-h1 * k1 - h2 * k2) / (rho_r * cpr)) * cA + ((U * Ah) / (Vr * rho_r * cpr)) * (Tc - Tr)
        dTc_dt = (qc / Vc) * (Tcf - Tc) + ((U * Ah) / (Vc * rho_c * cpc)) * (Tr - Tc)
        return [dcA_dt, dcB_dt, dTr_dt, dTc_dt]

    x0 = np.array([x_op[0], x_op[1], 335.0, x_op[3]])
    t_span = [0, 100]
    t_eval = np.linspace(t_span[0], t_span[1], 500)
    
    sol = solve_ivp(
        fun=cstr_closed_loop, t_span=t_span, y0=x0, t_eval=t_eval,
        args=(x_op, K_val, qr_s, qc_s) 
    )

    # --- 4.3 Plotagem dos Resultados ---
    print("Simulação concluída. Gerando gráficos...")
    t = sol.t; Tr = sol.y[2]; cB = sol.y[1]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Resultados da Simulação com Controlador H-Infinito (Realimentação de Estados - Revisado)')
    axs[0].plot(t, Tr, label='Resposta $T_r$'); axs[0].axhline(y=x_op[2], color='r', linestyle='--', label=f'Setpoint $T_r$ ({x_op[2]:.2f} K)')
    axs[0].set_title('Temperatura do Reator ($T_r$)'); axs[0].set_xlabel('t (min)'); axs[0].set_ylabel('$T_r$ (K)'); axs[0].legend(); axs[0].grid(True)
    axs[1].plot(t, cB, label='Resposta $c_B$'); axs[1].axhline(y=x_op[1], color='r', linestyle='--', label=f'Setpoint $c_B$ ({x_op[1]:.2f})')
    axs[1].set_title('Concentração do Produto ($c_B$)'); axs[1].set_xlabel('t (min)'); axs[1].set_ylabel('$c_B$ (kmol m⁻³)'); axs[1].legend(); axs[1].grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

else:
    print("\nERRO: Não foi possível encontrar uma solução viável para a LMI H-Infinito. O problema pode ser infactível com os pesos e restrições atuais.")