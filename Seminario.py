import numpy as np
import cvxpy as cp
from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import warnings
import itertools
from scipy.optimize import fsolve

print("Iniciando o script de projeto e validação do controlador robusto.")

# =============================================================================
# PARTE 1: DEFINIÇÃO DE PARÂMETROS E MODELO
# =============================================================================
print("\n--- Parte 1: Definindo os parâmetros do modelo ---")

# Parâmetros Físicos do Reator (Tabela 1 do artigo)
# Estes são os valores físicos constantes que descrevem o sistema do reator químico.
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

# Parâmetros nominais para as incertezas (valores médios)
# Estes são os valores médios dos parâmetros incertos, usados para a simulação nominal.
h1 = -8.6e4    # Entalpia da reação 1 [kJ/kmol]
h2 = -5.5e4    # Entalpia da reação 2 [kJ/kmol]
k10 = 1.55e11  # Fator pré-exponencial 1 [min^-1]
k20 = 8.55e26  # Fator pré-exponencial 2 [min^-1]

# Ponto de operação instável (estado estacionário nominal)
# Este é o estado estacionário alvo que o controlador deve manter.
x_op = np.array([1.8614, 1.0113, 338.41, 328.06])  # Vetor de estado: [cA, cB, Tr, Tc]

# A saída controlada é a temperatura do reator, Tr
# Esta matriz seleciona a terceira variável de estado (Tr) como a saída y para o controlador.
C0 = np.array([[0, 0, 1, 0]])

# =============================================================================
# PARTE 2: GERAÇÃO DOS VÉRTICES DO SISTEMA
# =============================================================================
print("\n--- Parte 2: Gerando os 16 vértices do sistema ---")

# Define os intervalos para os quatro parâmetros físicos incertos.
h1_range = [-8.8e4, -8.4e4]
h2_range = [-5.7e4, -5.3e4]
k10_range = [1.5e11, 1.6e11]
k20_range = [4.95e26, 12.15e26]

param_ranges = [h1_range, h2_range, k10_range, k20_range]
# Cria todas as 2^4 = 16 combinações dos valores mínimos/máximos dos parâmetros incertos.
param_combinations = list(itertools.product(*param_ranges))
print(f"Geradas {len(param_combinations)} combinações de parâmetros incertos.")


# Função que define as equações do sistema não linear para encontrar o estado estacionário.
def cstr_steady_state(x, params):
    cA, cB, Tr, Tc = x
    h1_p, h2_p, k10_p, k20_p = params
    k1 = k10_p * np.exp(-g1 / Tr)
    k2 = k20_p * np.exp(-g2 / Tr)
    eq1 = (qr_s / Vr) * (cAf - cA) - k1 * cA - k2 * cA
    eq2 = -(qr_s / Vr) * cB + k1 * cA
    eq3 = (qr_s / Vr) * (Trf - Tr) + ((-h1_p * k1 - h2_p * k2) / (rho_r * cpr)) * cA + (
                (U * Ah) / (Vr * rho_r * cpr)) * (Tc - Tr)
    eq4 = (qc_s / Vc) * (Tcf - Tc) + ((U * Ah) / (Vc * rho_c * cpc)) * (Tr - Tc)
    return [eq1, eq2, eq3, eq4]


# Função para linearizar o sistema calculando as matrizes Jacobianas (A e B) em um dado estado estacionário.
def calculate_jacobians(x_ss, params):
    cA, cB, Tr, Tc = x_ss
    h1_p, h2_p, k10_p, k20_p = params
    k1 = k10_p * np.exp(-g1 / Tr)
    k2 = k20_p * np.exp(-g2 / Tr)
    dk1_dTr = k1 * g1 / (Tr ** 2)
    dk2_dTr = k2 * g2 / (Tr ** 2)
    A = np.zeros((4, 4))
    A[0, 0] = -qr_s / Vr - k1 - k2
    A[0, 2] = -cA * (dk1_dTr + dk2_dTr)
    A[1, 0] = k1
    A[1, 1] = -qr_s / Vr
    A[1, 2] = cA * dk1_dTr
    A[2, 0] = (-h1_p * k1 - h2_p * k2) / (rho_r * cpr)
    A[2, 2] = -qr_s / Vr + ((-h1_p * dk1_dTr - h2_p * dk2_dTr) / (rho_r * cpr)) * cA - (U * Ah) / (Vr * rho_r * cpr)
    A[2, 3] = (U * Ah) / (Vr * rho_r * cpr)
    A[3, 2] = (U * Ah) / (Vc * rho_c * cpc)
    A[3, 3] = -qc_s / Vc - (U * Ah) / (Vc * rho_c * cpc)
    B = np.zeros((4, 2))
    B[0, 0] = (cAf - cA) / Vr
    B[1, 0] = -cB / Vr
    B[2, 0] = (Trf - Tr) / Vr
    B[3, 1] = (Tcf - Tc) / Vc
    return A, B


# Itera sobre cada uma das 16 combinações de parâmetros para gerar as matrizes A e B para cada vértice.
A_vertices = []
B_vertices = []
initial_guess = x_op
print("Iniciando o cálculo dos 16 vértices (pode levar um momento)...")
for i, current_params in enumerate(param_combinations):
    # Encontra o estado estacionário específico para esta combinação de parâmetros.
    x_ss_i, info, ier, msg = fsolve(cstr_steady_state, initial_guess, args=(current_params,), full_output=True)
    if ier != 1:
        print(f"Atenção: O solver de estado estacionário pode não ter convergido para o vértice {i + 1}.")
        print(f"   Mensagem: {msg}")
    # Lineariza o modelo neste novo estado estacionário para obter as matrizes do vértice.
    Ai, Bi = calculate_jacobians(x_ss_i, current_params)
    A_vertices.append(Ai)
    B_vertices.append(Bi)
    initial_guess = x_ss_i
    print(f"Vértice {i + 1}/{len(param_combinations)} calculado.")
print(
    f"\nCálculo finalizado. {len(A_vertices)} matrizes de vértice foram geradas e estão prontas para o projeto do controlador.")

# =============================================================================
# PARTE 3: PROJETO DO CONTROLADOR VIA LMI (CVXPY)
# =============================================================================
print("\n--- Parte 3: Projetando o controlador com LMIs (Modo de Depuração) ---")

# Define as matrizes de peso Q (para desvio de estado) e R (para esforço de controle).
q_const = 0.1
r_const = 500.0
print(f"Tentando nova combinação de pesos: q_const={q_const}, r_const={r_const}")

Q = q_const * np.diag([1, 1, 1e-5, 1e-5])
R = r_const * np.diag([1e3, 1e3])
I_4 = np.eye(4)
R_inv = np.linalg.inv(R)

# Etapa 1 do projeto LMI: Resolver para uma matriz S e gamma que satisfaçam a condição de estabilidade.
print("Resolvendo a LMI para encontrar S e gamma...")
S = cp.Variable((4, 4), symmetric=True)
gamma = cp.Variable()
constraints1 = []

for Ai, Bi in zip(A_vertices, B_vertices):
    LMI1 = cp.bmat([
        [S @ Ai.T + Ai @ S - Bi @ R_inv @ Bi.T, S @ sqrtm(Q)],
        [(S @ sqrtm(Q)).T, -gamma * np.eye(4)]
    ])
    constraints1.append(LMI1 << 0)

constraints1.append(S >> 1e-6 * I_4)

problem1 = cp.Problem(cp.Minimize(gamma), constraints1)
# Resolve o problema de otimização LMI.
problem1.solve(solver=cp.SCS, verbose=True, eps=1e-7, max_iters=200000)

# Se a primeira LMI foi resolvida com sucesso, prossegue para a segunda etapa.
if "optimal" in problem1.status:
    S_val = S.value
    gamma_val = gamma.value
    print("\nEtapa 1: Solução encontrada!")
    print(f"Valor ótimo de gamma: {gamma_val}")

    P_val = np.linalg.inv(S_val) # P é a inversa de S

    # Etapa 2 do projeto LMI: Com P fixo, resolve para a matriz de ganho do controlador F.
    print("Resolvendo a segunda LMI para encontrar o ganho F...")
    F = cp.Variable((2, 1))
    constraints2 = []

    for Ai, Bi in zip(A_vertices, B_vertices):
        Phi_i = -(Ai.T @ P_val + P_val @ Ai - P_val @ Bi @ R_inv @ Bi.T @ P_val + Q)
        LMI2 = cp.bmat([
            [-R, (Bi.T @ P_val + R @ F @ C0)],
            [(Bi.T @ P_val + R @ F @ C0).T, -Phi_i]
        ])
        constraints2.append(LMI2 << 0)

    problem2 = cp.Problem(cp.Minimize(0), constraints2)
    problem2.solve(solver=cp.SCS, verbose=True, eps=1e-7, max_iters=200000)

    if "optimal" in problem2.status:
        F_val = F.value # Armazena o ganho do controlador calculado.
        print("\nEtapa 2: Solução encontrada!")
        print("\nControlador robusto F encontrado:")
        print(F_val)

        # =============================================================================
        # PARTE 4: VALIDAÇÃO E SIMULAÇÃO
        # =============================================================================
        print("\n--- Parte 4: Validando o controlador ---")

        # --- 4.1 Verificação da Estabilidade ---
        # Checa se o controlador encontrado estabiliza todos os 16 vértices do sistema.
        print("\nVerificando a estabilidade dos vértices em malha fechada...")
        all_stable = True
        for i, (Ai, Bi) in enumerate(zip(A_vertices, B_vertices)):
            A_cl = Ai + Bi @ F_val @ C0
            eigenvalues = np.linalg.eigvals(A_cl)
            # Um sistema é estável se todos os autovalores têm parte real negativa.
            if np.all(np.real(eigenvalues) < 0):
                print(f"Vértice {i + 1}: Estável.")
            else:
                print(f"Vértice {i + 1}: INSTÁVEL.")
                all_stable = False

        if not all_stable:
            warnings.warn("Atenção: Nem todos os vértices do sistema são estáveis com o controlador encontrado.")

        # --- 4.2 Simulação do Sistema Não Linear ---
        print("\nIniciando a simulação do modelo não linear...")


        # Esta função define a dinâmica de malha fechada do sistema não linear.
        def cstr_closed_loop(t, x, x_op, F, C, qr_ss, qc_ss):
            # Variáveis de estado atuais
            cA, cB, Tr, Tc = x
            # Calcula o desvio do ponto de operação
            x_dev = x - x_op
            y_dev = C @ x_dev
            # Calcula a ação de controle com base no desvio da saída e no ganho F
            u = F @ y_dev
            # Aplica a ação de controle às variáveis manipuladas (vazões)
            qr = qr_ss + u[0]
            qc = qc_ss + u[1]
            
            # Calcula as derivadas usando as equações completas do modelo não linear
            k1 = k10 * np.exp(-g1 / Tr)
            k2 = k20 * np.exp(-g2 / Tr)
            dcA_dt = (qr / Vr) * (cAf - cA) - k1 * cA - k2 * cA
            dcB_dt = -(qr / Vr) * cB + k1 * cA
            dTr_dt = (qr / Vr) * (Trf - Tr) + ((-h1 * k1 - h2 * k2) / (rho_r * cpr)) * cA + (
                        (U * Ah) / (Vr * rho_r * cpr)) * (Tc - Tr)
            dTc_dt = (qc / Vc) * (Tcf - Tc) + ((U * Ah) / (Vc * rho_c * cpc)) * (Tr - Tc)
            return [dcA_dt, dcB_dt, dTr_dt, dTc_dt]

        # Define a condição inicial, com uma perturbação na temperatura do reator.
        x0 = np.array([x_op[0], x_op[1], 335.0, x_op[3]])
        # Define o intervalo de tempo da simulação.
        t_span = [0, 100]
        t_eval = np.linspace(t_span[0], t_span[1], 500)
        
        # Resolve a EDO (Equação Diferencial Ordinária) para simular a resposta do sistema.
        sol = solve_ivp(
            fun=cstr_closed_loop, t_span=t_span, y0=x0, t_eval=t_eval,
            args=(x_op, F_val, C0, qr_s, qc_s)
        )

        # --- 4.3 Plotagem dos Resultados ---
        # Extrai os resultados da simulação para plotagem.
        print("Simulação concluída. Gerando gráficos...")
        t = sol.t
        Tr = sol.y[2]
        cB = sol.y[1]
        
        # Cria a figura e os eixos para os gráficos.
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Resultados da Simulação com Controlador Robusto')
        
        # Plota a temperatura do reator ao longo do tempo.
        axs[0].plot(t, Tr)
        axs[0].set_title('Temperatura do Reator ($T_r$)')
        axs[0].set_xlabel('t (min)')
        axs[0].set_ylabel('$T_r$ (K)')
        axs[0].grid(True)
        
        # Plota a concentração do produto ao longo do tempo.
        axs[1].plot(t, cB)
        axs[1].set_title('Concentração do Produto ($c_B$)')
        axs[1].set_xlabel('t (min)')
        axs[1].set_ylabel('$c_B$ (kmol m⁻³)')
        axs[1].grid(True)
        
        # Ajusta o layout e exibe os gráficos.
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    else:
        print("ERRO: Não foi possível encontrar uma solução viável para o ganho F.")

else:
    print("ERRO: Não foi possível encontrar uma solução viável para a primeira LMI. O problema é infactível.")