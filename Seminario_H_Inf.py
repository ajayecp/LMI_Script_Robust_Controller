import numpy as np
import cvxpy as cp
from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import fsolve

print("Iniciando o script de projeto do controlador robusto H-INFINITO.")

# --- Parte 1: Parâmetros Físicos e Ponto de Operação ---
# Definimos as constantes físicas do reator CSTR (Volumes, densidades, calores específicos, etc.)
Vr, Vc = 0.23, 0.21
qr_s, qc_s = 0.015, 0.004
rho_r, rho_c = 1020, 998
cpr, cpc = 4.02, 4.182
Ah, U = 1.51, 42.8
g1, g2 = 9850, 22019  # Energias de ativação
cAf, Trf, Tcf = 4.22, 310, 288
h1, h2 = -8.6e4, -5.5e4 # Entalpias nominais
k10, k20 = 1.55e11, 8.55e26 # Fatores pré-exponenciais nominais

# Ponto de Operação (Equilíbrio): [Concentração A, Concentração B, Temp Reator, Temp Camisa]
x_op = np.array([1.8614, 1.0113, 338.41, 328.06])

# --- Parte 2: Modelagem Politópica (Incertezas) ---
# Definimos os intervalos de incerteza para 4 parâmetros críticos.
# O controle robusto garantirá estabilidade para qualquer valor dentro destes limites.
h1_range, h2_range = [-8.8e4, -8.4e4], [-5.7e4, -5.3e4]
k10_range, k20_range = [1.5e11, 1.6e11], [4.95e26, 12.15e26]

# itertools.product gera todas as 16 combinações possíveis dos extremos (2^4 = 16 vértices)
param_combinations = list(itertools.product(h1_range, h2_range, k10_range, k20_range))

def calculate_jacobians(x_ss, params):
    """
    Calcula as matrizes A e B do espaço de estados linearizando o modelo não-linear
    em torno do ponto de operação para um dado conjunto de parâmetros.
    """
    cA, cB, Tr, Tc = x_ss
    h1_p, h2_p, k10_p, k20_p = params
    
    # Arrhenius: constante de velocidade da reação depende da temperatura
    k1, k2 = k10_p * np.exp(-g1 / Tr), k20_p * np.exp(-g2 / Tr)
    dk1_dTr, dk2_dTr = k1 * g1 / (Tr ** 2), k2 * g2 / (Tr ** 2)
    
    # Construção da Matriz A (Jacobiana dos estados)
    A = np.zeros((4, 4)); B = np.zeros((4, 2))
    A[0, 0] = -qr_s / Vr - k1 - k2; A[0, 2] = -cA * (dk1_dTr + dk2_dTr)
    A[1, 0] = k1; A[1, 1] = -qr_s / Vr; A[1, 2] = cA * dk1_dTr
    A[2, 0] = (-h1_p * k1 - h2_p * k2) / (rho_r * cpr)
    A[2, 2] = -qr_s / Vr + ((-h1_p * dk1_dTr - h2_p * dk2_dTr) / (rho_r * cpr)) * cA - (U * Ah) / (Vr * rho_r * cpr)
    A[2, 3] = (U * Ah) / (Vr * rho_r * cpr); A[3, 2] = (U * Ah) / (Vc * rho_c * cpc)
    A[3, 3] = -qc_s / Vc - (U * Ah) / (Vc * rho_c * cpc)
    
    # Construção da Matriz B (Jacobiana das entradas: vazões qr e qc)
    B[0, 0] = (cAf - cA) / Vr; B[1, 0] = -cB / Vr; B[2, 0] = (Trf - Tr) / Vr; B[3, 1] = (Tcf - Tc) / Vc
    return A, B

# Criamos uma lista de matrizes A e B para cada um dos 16 vértices do polítopo
A_vertices, B_vertices = [], []
for p in param_combinations:
    Ai, Bi = calculate_jacobians(x_op, p)
    A_vertices.append(Ai); B_vertices.append(Bi)

# --- Parte 3: Síntese do Controlador via LMIs (H-infinito) ---
# Q e R são matrizes de peso. Q penaliza o erro de estado, R penaliza o esforço de controle.
# Nota: Temperaturas (330+) têm escalas diferentes de concentrações (1.0), por isso os pesos variam.
Q = np.diag([10, 10, 0.1, 0.1]) 
R = np.diag([0.1, 0.1]) 

# Decomposição para formar as saídas de performance Z
Cz = sqrtm(Q)
Dzu = sqrtm(R)
Cz_aug = np.vstack([Cz, np.zeros((2, 4))])
Dzu_aug = np.vstack([np.zeros((4, 2)), Dzu])
Dzw = np.zeros((6, 2))

# Variáveis de decisão para a Otimização Convexa
X = cp.Variable((4, 4), symmetric=True) # Matriz de Lyapunov (X = P^-1)
Y = cp.Variable((2, 4))                # Variável auxiliar para o Ganho (Y = K*X)
gamma = cp.Variable()                  # Nível de atenuação H-infinito (a ser minimizado)
alpha = 0.05                           # Grau de estabilidade (força autovalores para a esquerda)

# Restrições: X deve ser positiva definida
constraints = [X >> 1e-6 * np.eye(4)]

# Para cada vértice do sistema incerto, adicionamos uma LMI baseada no "Bounded Real Lemma"
for Ai, Bi in zip(A_vertices, B_vertices):
    Bw_i = Bi * 0.1 # Modelamos a incerteza de entrada como um distúrbio w
    
    # LMI_11: Termo de estabilidade de Lyapunov com decaimento alpha
    LMI_11 = Ai @ X + X @ Ai.T + Bi @ Y + Y.T @ Bi.T + 2 * alpha * X
    
    # Montagem da matriz de blocos da LMI H-infinito
    LMI = cp.bmat([
        [LMI_11, Bw_i, (Cz_aug @ X + Dzu_aug @ Y).T],
        [Bw_i.T, -gamma * np.eye(2), Dzw.T],
        [(Cz_aug @ X + Dzu_aug @ Y), Dzw, -gamma * np.eye(6)]
    ])
    constraints.append(LMI << 0) # A matriz deve ser negativa definida

# Resolvemos o problema: encontrar X e Y que minimizam gama
prob = cp.Problem(cp.Minimize(gamma), constraints)
prob.solve(solver=cp.SCS, max_iters=50000)

if X.value is not None:
    # Recuperamos o ganho do controlador: K = Y * X^-1
    K_val = Y.value @ np.linalg.inv(X.value)
    print(f"Ganho K encontrado. Gamma ótimo: {gamma.value:.4f}")

    # --- Parte 4: Simulação do Sistema Não-Linear em Malha Fechada ---
    def cstr_loop(t, x):
        cA, cB, Tr, Tc = x
        # Lei de controle linear: u = K * (x - x_op)
        u = K_val @ (x - x_op)
        
        # Saturação: As vazões físicas não podem ser negativas ou infinitas
        qr = np.clip(qr_s + u[0], 0, 0.05)
        qc = np.clip(qc_s + u[1], 0, 0.05)
        
        # Equações diferenciais originais do CSTR (não-lineares)
        k1 = k10 * np.exp(-g1 / Tr)
        k2 = k20 * np.exp(-g2 / Tr)
        
        return [
            (qr / Vr) * (cAf - cA) - k1 * cA - k2 * cA,
            -(qr / Vr) * cB + k1 * cA,
            (qr / Vr) * (Trf - Tr) + ((-h1*k1 - h2*k2)/(rho_r*cpr))*cA + (U*Ah/(Vr*rho_r*cpr))*(Tc - Tr),
            (qc / Vc) * (Tcf - Tc) + (U*Ah/(Vc*rho_c*cpc))*(Tr - Tc)
        ]

    # Condição inicial: introduzimos um erro proposital de 5 Kelvin na temperatura
    x0 = x_op.copy()
    x0[2] -= 5.0 
    
    # Integração numérica das EDOs
    sol = solve_ivp(cstr_loop, [0, 60], x0, t_eval=np.linspace(0, 60, 1000))

    # --- Parte 5: Visualização dos Resultados ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Gráfico de Temperatura: Deve convergir para o Setpoint (linha vermelha)
    ax[0].plot(sol.t, sol.y[2], 'b', label='Tr (Reator)')
    ax[0].axhline(x_op[2], color='r', linestyle='--', label='Setpoint')
    ax[0].set_title('Temperatura do Reator [K]'); ax[0].grid(True); ax[0].legend()
    
    # Gráfico de Concentração: Mostra o impacto da correção de temperatura no produto
    ax[1].plot(sol.t, sol.y[1], 'g', label='cB (Produto)')
    ax[1].axhline(x_op[1], color='r', linestyle='--', label='Setpoint')
    ax[1].set_title('Concentração de B [kmol/m³]'); ax[1].grid(True); ax[1].legend()
    
    plt.tight_layout(); plt.show()
else:
    print("Infactível: Os pesos ou restrições impedem uma solução estável.")