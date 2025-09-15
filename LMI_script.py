
import numpy as np
import cvxpy as cp
from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import warnings
import itertools
from scipy.optimize import fsolve

print("Iniciando o script de projeto e validação do controlador robusto.")


# 2. DEFINIÇÃO DE PARÂMETROS E MODELO

print("\n--- Parte 1: Definindo os parâmetros do modelo ---")

# Parâmetros Físicos do Reator (Tabela 1) [cite: 109]
Vr = 0.23         # Volume do reator [m^3] [cite: 109]
Vc = 0.21         # Volume do fluido de arrefecimento [m^3] [cite: 109]
qr_s = 0.015      # Vazão de reagente (estado estacionário) [m^3/min] [cite: 109]
qc_s = 0.004      # Vazão de arrefecimento (estado estacionário) [m^3/min] [cite: 109]
rho_r = 1020      # Densidade do reagente [kg/m^3] [cite: 109]
rho_c = 998       # Densidade do fluido de arrefecimento [kg/m^3] [cite: 109]
cpr = 4.02        # Capacidade calorífica do reagente [kJ/kg/K] [cite: 109]
cpc = 4.182       # Capacidade calorífica do fluido de arrefecimento [kJ/kg/K] [cite: 109]
Ah = 1.51         # Área de transferência de calor [m^2] [cite: 109]
U = 42.8          # Coeficiente de transferência de calor [kJ/m^2/min/K] [cite: 109]
g1 = 9850         # Energia de ativação 1 (E1/R) [K] [cite: 109]
g2 = 22019        # Energia de ativação 2 (E2/R) [K] [cite: 109]
cAf = 4.22        # Concentração de A na alimentação [kmol/m^3] [cite: 109]
Trf = 310         # Temperatura da alimentação [K] [cite: 109]
Tcf = 288         # Temperatura do fluido de arrefecimento na alimentação [K] [cite: 109]

# Parâmetros nominais para as incertezas (valores médios) [cite: 111]
h1 = -8.6e4       # Entalpia da reação 1 [kJ/kmol]
h2 = -5.5e4       # Entalpia da reação 2 [kJ/kmol]
k10 = 1.55e11     # Fator pré-exponencial 1 [min^-1]
k20 = 8.55e26     # Fator pré-exponencial 2 [min^-1]

# Ponto de operação instável (estado estacionário nominal) [cite: 117]
x_op = np.array([1.8614, 1.0113, 338.41, 328.06]) # [cA, cB, Tr, Tc]

# Matrizes do modelo linearizado nominal (A0, B0, C0) 
# A0 = np.array([
#     [-0.1479, 0, -0.0226, 0],
#     [0.0354, -0.0652, 0.0057, 0],
#     [1.3763, 0,  0.2118, 0.0685],
#     [0, 0, 0.0737, -0.0928]
# ])


# B0 = np.array([
#     [10.2546, 0],
#     [-4.3968, 0],
#     [-123.5131, 0],
#     [0, -190.7612]
# ])
A0 = np.array([
    [-0.1479, 0, -0.0226, 0],
    [0.0354, -0.0652, 10.2546, 0],
    [0.0057, 0, 1.3763, -4.3968],
    [0, 0, 0.2118, -190.7612]
])

B0 = np.array([
    [-123.5131, 0],
    [0, 0],
    [-0.0928, 0.0685],
    [0, 0.0737]
])
C0 = np.array([[0, 0, 1, 0]]) # A saída controlada é a temperatura do reator, Tr [cite: 143]


# =============================================================================
# 3. GERAÇÃO DOS VÉRTICES (CÓDIGO COMPLETO)
# =============================================================================
print("\n--- Parte 2: Gerando os 16 vértices do sistema ---")

# Importações adicionais necessárias para esta seção

# --- PASSO 3.1: Definir os limites dos parâmetros incertos ---
# [cite_start]Conforme especificado no artigo [cite: 110]
h1_range = [-8.8e4, -8.4e4]
h2_range = [-5.7e4, -5.3e4]
k10_range = [1.5e11, 1.6e11]
k20_range = [4.95e26, 12.15e26]

# Gerar todas as 16 combinações de parâmetros usando itertools.product
param_ranges = [h1_range, h2_range, k10_range, k20_range]
param_combinations = list(itertools.product(*param_ranges))
print(f"Geradas {len(param_combinations)} combinações de parâmetros incertos.")

# --- PASSO 3.2: Definir as funções para encontrar o estado estacionário e linearizar ---

# Função das equações de estado estacionário (derivadas = 0) para usar com fsolve
def cstr_steady_state(x, params):
    cA, cB, Tr, Tc = x
    h1_p, h2_p, k10_p, k20_p = params
    
    # Reutilizando parâmetros globais definidos anteriormente
    k1 = k10_p * np.exp(-g1 / Tr)
    k2 = k20_p * np.exp(-g2 / Tr)
    
    # [cite_start]Equações (14-17) com derivadas igualadas a zero [cite: 94, 95, 96, 97]
    eq1 = (qr_s / Vr) * (cAf - cA) - k1 * cA - k2 * cA
    eq2 = -(qr_s / Vr) * cB + k1 * cA
    eq3 = (qr_s / Vr) * (Trf - Tr) + ((-h1_p * k1 - h2_p * k2) / (rho_r * cpr)) * cA + \
          ((U * Ah) / (Vr * rho_r * cpr)) * (Tc - Tr)
    eq4 = (qc_s / Vc) * (Tcf - Tc) + ((U * Ah) / (Vc * rho_c * cpc)) * (Tr - Tc)
    
    return [eq1, eq2, eq3, eq4]

# Função para calcular as matrizes Jacobianas (A e B)
# ATENÇÃO: As derivadas parciais foram calculadas analiticamente a partir
# das equações (14-17) e implementadas abaixo.
def calculate_jacobians(x_ss, params):
    cA, cB, Tr, Tc = x_ss
    h1_p, h2_p, k10_p, k20_p = params

    # Termos recorrentes
    k1 = k10_p * np.exp(-g1 / Tr)
    k2 = k20_p * np.exp(-g2 / Tr)
    dk1_dTr = k1 * g1 / (Tr**2)
    dk2_dTr = k2 * g2 / (Tr**2)

    # --- Matriz A: Jacobiana em relação aos estados x = [cA, cB, Tr, Tc] ---
    A = np.zeros((4, 4))

    # Linha 1: d(f1)/dx
    A[0,0] = -qr_s/Vr - k1 - k2      # df1/dcA
    A[0,2] = -cA * (dk1_dTr + dk2_dTr) # df1/dTr

    # Linha 2: d(f2)/dx
    A[1,0] = k1                       # df2/dcA
    A[1,1] = -qr_s/Vr                 # df2/dcB
    A[1,2] = cA * dk1_dTr             # df2/dTr

    # Linha 3: d(f3)/dx
    A[2,0] = (-h1_p * k1 - h2_p * k2) / (rho_r * cpr) # df3/dcA
    A[2,2] = -qr_s/Vr + ((-h1_p * dk1_dTr - h2_p * dk2_dTr) / (rho_r * cpr)) * cA - \
             (U * Ah) / (Vr * rho_r * cpr)           # df3/dTr
    A[2,3] = (U * Ah) / (Vr * rho_r * cpr)          # df3/dTc

    # Linha 4: d(f4)/dx
    A[3,2] = (U * Ah) / (Vc * rho_c * cpc)          # df4/dTr
    A[3,3] = -qc_s/Vc - (U * Ah) / (Vc * rho_c * cpc) # df4/dTc

    # --- Matriz B: Jacobiana em relação às entradas u = [qr, qc] ---
    # [cite_start]As entradas de controle são qr e qc [cite: 143]
    B = np.zeros((4, 2))

    # Linha 1: d(f1)/du
    B[0,0] = (cAf - cA) / Vr          # df1/dqr
    
    # Linha 2: d(f2)/du
    B[1,0] = -cB / Vr                 # df2/dqr

    # Linha 3: d(f3)/du
    B[2,0] = (Trf - Tr) / Vr          # df3/dqr

    # Linha 4: d(f4)/du
    B[3,1] = (Tcf - Tc) / Vc          # df4/dqc
    
    return A, B

# --- PASSO 3.3: Loop principal para gerar as 16 matrizes ---
A_vertices = []
B_vertices = []

# Usar o ponto de operação nominal como um bom chute inicial para o solver
initial_guess = x_op 

print("Iniciando o cálculo dos 16 vértices (pode levar um momento)...")
for i, current_params in enumerate(param_combinations):
    # Encontrar o estado estacionário para a combinação de parâmetros atual
    # A função fsolve encontra a raiz da função cstr_steady_state
    x_ss_i, info, ier, msg = fsolve(cstr_steady_state, initial_guess, args=(current_params,), full_output=True)
    
    if ier != 1: # Checar se o solver convergiu
        print(f"Atenção: O solver de estado estacionário pode não ter convergido para o vértice {i+1}.")
        print(f"   Mensagem: {msg}")

    # Calcular as matrizes A e B linearizadas em torno do estado estacionário encontrado
    Ai, Bi = calculate_jacobians(x_ss_i, current_params)
    
    # Adicionar as matrizes resultantes às listas
    A_vertices.append(Ai)
    B_vertices.append(Bi)
    
    # Atualizar o chute inicial para o próximo loop pode ajudar na convergência
    initial_guess = x_ss_i
    
    print(f"Vértice {i+1}/{len(param_combinations)} calculado.")

print(f"\nCálculo finalizado. {len(A_vertices)} matrizes de vértice foram geradas e estão prontas para o projeto do controlador.")


# 4. PROJETO DO CONTROLADOR VIA LMI (CVXPY)

print("\n--- Parte 3: Projetando o controlador com LMIs (Modo de Depuração) ---")

# Parâmetros de ponderação Q e R para a função de custo garantido
# AÇÃO: Testando uma nova combinação mais "relaxada" de pesos.
q_const = 0.1   # DIMINUÍDO: Exige menos desempenho dos estados.
r_const = 500.0 # AUMENTADO SIGNIFICATIVAMENTE: Penaliza fortemente o uso de controle.

print(f"Tentando nova combinação de pesos: q_const={q_const}, r_const={r_const}")

Q = q_const * np.diag([1, 1, 1e-5, 1e-5])
R = r_const * np.diag([1e3, 1e3])
gamma = 1e-6 
I_4 = np.eye(4)
R_inv = np.linalg.inv(R)

# --- Etapa 1 da LMI: Resolver para S = P^-1 (Equação 12) ---
print("Resolvendo a primeira LMI para encontrar a matriz S (com saída detalhada)...")
S = cp.Variable((4, 4), symmetric=True)
constraints1 = []

for Ai, Bi in zip(A_vertices, B_vertices):
    LMI1_row1 = cp.hstack([S @ Ai.T + Ai @ S - Bi @ R_inv @ Bi.T, S @ sqrtm(Q)])
    LMI1_row2 = cp.hstack([sqrtm(Q).T @ S, -np.eye(4)])
    LMI1 = cp.bmat([[LMI1_row1], [LMI1_row2]])
    constraints1.append(LMI1 << 0)

constraints1.append(S >> gamma * I_4)

problem1 = cp.Problem(cp.Minimize(0), constraints1)
problem1.solve(solver=cp.SCS, verbose=True, eps=1e-8)
# --- Etapa 2 da LMI: Resolver para F (Equação 13) ---
if problem1.status == 'optimal':
    S_val = S.value
    P_val = np.linalg.inv(S_val)
    print("Primeira LMI resolvida com sucesso. Matriz P encontrada.")
    print("Resolvendo a segunda LMI para encontrar o ganho F...")

    F = cp.Variable((2, 1))
    constraints2 = []

    for Ai, Bi in zip(A_vertices, B_vertices):
        # Calcular Phi_i da Equação (11) [cite: 77]
        Phi_i = -(Ai.T @ P_val + P_val @ Ai - P_val @ Bi @ R_inv @ Bi.T @ P_val + Q)
        
        # Construção da LMI para o cálculo de F [cite: 85]
        LMI2_row1 = cp.hstack([-R, (Bi.T @ P_val + R @ F @ C0)])
        LMI2_row2 = cp.hstack([(Bi.T @ P_val + R @ F @ C0).T, -Phi_i])
        LMI2 = cp.bmat([[LMI2_row1], [LMI2_row2]])
        constraints2.append(LMI2 << 0)

    problem2 = cp.Problem(cp.Minimize(0), constraints2)
    problem2.solve(solver=cp.SCS, verbose=False)

    if problem2.status == 'optimal':
        F_val = F.value
        print("Segunda LMI resolvida com sucesso.")
        print("\nControlador robusto F encontrado:")
        print(F_val)
        # O valor do artigo é F = [0.0023, 0.0186]^T[cite: 203]. O valor obtido pode
        # diferir por usar apenas 1 vértice e diferentes parâmetros q_const/r_const.
    else:
        F_val = None
        print("ERRO: Não foi possível encontrar uma solução viável para o ganho F.")
else:
    F_val = None
    print("ERRO: O problema de estabilização (LMI 1) não é viável.")
    print("O sistema pode não ser estabilizável com os pesos Q e R escolhidos.")


# 5. VALIDAÇÃO E SIMULAÇÃO

if F_val is not None:
    print("\n--- Parte 4: Validando o controlador ---")
    
    # --- 5.1 Verificação da Estabilidade ---
    print("\nVerificando a estabilidade dos vértices em malha fechada...")
    all_stable = True
    for i, (Ai, Bi) in enumerate(zip(A_vertices, B_vertices)):
        A_cl = Ai + Bi @ F_val @ C0
        eigenvalues = np.linalg.eigvals(A_cl)
        if np.all(np.real(eigenvalues) < 0):
            print(f"Vértice {i+1}: Estável.")
        else:
            print(f"Vértice {i+1}: INSTÁVEL.")
            all_stable = False
    
    if not all_stable:
        warnings.warn("Atenção: Nem todos os vértices do sistema são estáveis com o controlador encontrado.")

    # --- 5.2 Simulação do Sistema Não Linear ---
    print("\nIniciando a simulação do modelo não linear para reproduzir a Figura 2...")

    # Função que define as EDOs do sistema em malha fechada
    def cstr_closed_loop(t, x, x_op, F, C, qr_ss, qc_ss):
        cA, cB, Tr, Tc = x
        
        # Lógica do Controlador
        x_dev = x - x_op
        y_dev = C @ x_dev
        u = F @ y_dev
        qr = qr_ss + u[0]
        qc = qc_ss + u[1]
        
        # Equações do Modelo Não Linear (14-17) [cite: 94, 95, 96, 99]
        k1 = k10 * np.exp(-g1 / Tr) # [cite: 103]
        k2 = k20 * np.exp(-g2 / Tr) # [cite: 103]
        
        dcA_dt = (qr / Vr) * (cAf - cA) - k1 * cA - k2 * cA # [cite: 94]
        dcB_dt = -(qr / Vr) * cB + k1 * cA # [cite: 95]
        dTr_dt = (qr / Vr) * (Trf - Tr) + ((-h1 * k1 - h2 * k2) / (rho_r * cpr)) * cA + \
                 ((U * Ah) / (Vr * rho_r * cpr)) * (Tc - Tr) # [cite: 96]
        dTc_dt = (qc / Vc) * (Tcf - Tc) + ((U * Ah) / (Vc * rho_c * cpc)) * (Tr - Tc) # [cite: 99]
        
        return [dcA_dt, dcB_dt, dTr_dt, dTc_dt]

    # Condições da simulação para se assemelhar à Figura 2
    x0 = np.array([x_op[0], x_op[1], 335.0, x_op[3]]) # Tr começa em 335 K [cite: 179]
    t_span = [0, 100] # Simulação por 100 minutos [cite: 195]
    t_eval = np.linspace(t_span[0], t_span[1], 500)

    # Resolvendo as EDOs
    sol = solve_ivp(
        fun=cstr_closed_loop, t_span=t_span, y0=x0, t_eval=t_eval,
        args=(x_op, F_val, C0, qr_s, qc_s)
    )

    # --- 5.3 Plotagem dos Resultados ---
    print("Simulação concluída. Gerando gráficos...")
    t = sol.t
    Tr = sol.y[2] # Temperatura do reator é o 3º estado
    cB = sol.y[1] # Concentração de B é o 2º estado

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Resultados da Simulação com Controlador Robusto')

    # Gráfico da Temperatura do Reator (Tr)
    axs[0].plot(t, Tr)
    axs[0].set_title('Temperatura do Reator ($T_r$)') # Baseado na Figura 2 [cite: 202]
    axs[0].set_xlabel('t (min)')
    axs[0].set_ylabel('$T_r$ (K)')
    axs[0].grid(True)

    # Gráfico da Concentração do Produto (cB)
    axs[1].plot(t, cB)
    axs[1].set_title('Concentração do Produto ($c_B$)') # Baseado na Figura 2 [cite: 202]
    axs[1].set_xlabel('t (min)')
    axs[1].set_ylabel('$c_B$ (kmol m⁻³)')
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

else:
    print("\nScript finalizado sem sucesso. O controlador não pôde ser projetado.")