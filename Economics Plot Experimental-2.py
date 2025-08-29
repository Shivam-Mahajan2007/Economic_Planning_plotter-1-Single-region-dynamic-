import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# PARAMETERS
# -----------------------------
np.random.seed(42)
n_industries = 10000
n_steps = 60  # 5 years (60 months)
k_iterations = 50

# Target demand growth (50% in 5 years)
total_growth_factor = 1.0 + np.random.uniform(0.4, 0.5, n_industries)

# Elasticities
true_epsilon = np.random.uniform(-0.9, -0.2, n_industries)
measured_epsilon = true_epsilon * (1.0 + np.random.uniform(-0.1, 0.1, n_industries))

print("making paramenters...")
# -----------------------------
# MATRICES A, B
# -----------------------------
print("making the matricies")
I = np.eye(n_industries)

# Technical coefficients (A)
A = np.zeros((n_industries, n_industries))
for i in range(n_industries):
    connections = np.random.choice(n_industries, size=int(0.1 * n_industries), replace=False)
    A[i, connections] = np.random.uniform(0.01, 0.9, size=len(connections))

rho = np.max(np.abs(np.linalg.eigvals(A)))
if rho >= 1:
    A *= 0.9 / rho

L = I - A

print("lrontief matrix built!")

# Capital coefficients (B)
B = np.zeros((n_industries, n_industries))
cap_providers = list(range(10, 20))
for i in range(n_industries):
    providers = np.random.choice(cap_providers, size=min(5, len(cap_providers)), replace=False)
    B[i, providers] = np.random.uniform(0.2, 0.9, size=len(providers))

print("Capital matrix built!")

# -----------------------------
# NEUMANN SERIES APPROXIMATION
# -----------------------------
def neumann_approx(A, k, vec=None):
    if vec is None:
        S, P = I.copy(), I.copy()
        for _ in range(1, k+1):
            P = P @ A
            S += P
        return S
    else:
        res, term = vec.copy(), vec.copy()
        for _ in range(1, k+1):
            term = A @ term
            res += term
        return res


# -----------------------------
# INITIAL STATES
# -----------------------------
d = np.random.uniform(200, 500, n_industries)
X = neumann_approx(A, k_iterations, d)
C = X * 1.20

C_target = C * total_growth_factor

wL = np.random.uniform(5, 20, n_industries)
P0 = neumann_approx(A.T, k_iterations, wL)

print("inital states set!")
# -----------------------------
# STORAGE
# -----------------------------
AD, AS = [], []
CAPACITY_TARGET = []
GAP, UNEMPLOYMENT = [], []
GDP, GDP_GROWTH = [], []
INFLATION, PRICE_LEVEL = [], []
MULTIPLIER, ACCELERATOR = [], []
CONS_SHARE, INV_SHARE = [], []

INVESTMENT = []

sector_gaps = []

# -----------------------------
# SIMULATION LOOP
# -----------------------------
print("Simulation started!")
for t in range(n_steps):
    # 1. Autonomous growth in demand
    growth = 1.0 + np.random.uniform(0.004, 0.005, n_industries)
    d_base = d * growth

    # 2. Price adjustment
    price_shock = 1.0 + np.random.uniform(-0.01, 0.01, n_industries)
    P = P0 * price_shock
    delta_P = P - P0

    # 3. Demand response to prices
    delta_d = d_base * (delta_P / (P0 + 1e-12)) * true_epsilon
    d_new = d_base + delta_d
    delta_d_est = d_base * (delta_P / (P0 + 1e-12)) * measured_epsilon

    # 4. Investment
    delta_X_short = neumann_approx(A, k_iterations, delta_d_est)
    steps_left = max(1, n_steps - t)
    required_growth_factor = (C_target / np.maximum(C, 1e-12)) ** (1.0 / steps_left) - 1.0
    required_growth_factor = np.clip(required_growth_factor, 0.0, 0.02)
    delta_X_long = required_growth_factor * C
    delta_C = delta_X_short + delta_X_long
    C += delta_C

    I_total = B @ delta_C
    INVESTMENT.append(I_total.sum())

    # 5. Production
    prod_input = d_new + I_total
    X = np.clip(neumann_approx(A, k_iterations, prod_input), 0.0, None)

    AS_vec = L @ X
    AD_val = (d_new + I_total).sum()
    AS_val = AS_vec.sum()

    AD.append(AD_val)
    AS.append(AS_val)
    CAPACITY_TARGET.append((L @ C).sum())  # potential output

    # Gaps and unemployment
    gap = (AS_val - AD_val) / (AD_val + 1e-12) * 100
    GAP.append(gap)
    UNEMPLOYMENT.append(max(0, 5 + 0.5 * gap))  # Okun-style

    # GDP and growth
    GDP.append(AS_val)
    if t > 0:
        GDP_GROWTH.append((AS_val / GDP[-2] - 1) * 100)
    else:
        GDP_GROWTH.append(0)

    # Inflation (demand pull)
    INFLATION.append((P.sum() / P0.sum() - 1) * 100)

    # Price level (weighted average)
    price_level_t = (P @ X) / (X.sum() + 1e-12)
    PRICE_LEVEL.append(price_level_t)

    # Keynesian ratios
    C_share = d_new.sum() / (AS_val + 1e-12) * 100
    I_share = I_total.sum() / (AS_val + 1e-12) * 100
    CONS_SHARE.append(C_share)
    INV_SHARE.append(I_share)

    if t > 0:
        ΔY = GDP[-1] - GDP[-2]
        ΔI = INVESTMENT[-1] - INVESTMENT[-2]
        if abs(ΔI) > 1e-8:
            MULTIPLIER.append(ΔY / ΔI)
        else:
            MULTIPLIER.append(0)
        if abs(ΔY) > 1e-8:
            ACCELERATOR.append(ΔI / ΔY)
        else:
            ACCELERATOR.append(0)
    else:
        MULTIPLIER.append(0)
        ACCELERATOR.append(0)

    # Sector gaps
    sector_gap_t = (AS_vec - (d_new + I_total)) / ((d_new + I_total) + 1e-12) * 100
    sector_gaps.append(sector_gap_t)

    d = d_new.copy()
    
    print(t+1,"/",n_steps,"copmpleted")

print("simulation complete!")

# -----------------------------
# COMPREHENSIVE SUBPLOT DASHBOARD
# -----------------------------
T = np.arange(n_steps)

# Create a 3x2 grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(16, 12))

# 1. AD vs AS with capacity target
axes[0,0].plot(T, AD, label="Aggregate Demand", linewidth=2)
axes[0,0].plot(T, AS, label="Aggregate Supply", linewidth=2)
axes[0,0].plot(T, CAPACITY_TARGET, linestyle=":", linewidth=2, label="Potential production")
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)
axes[0,0].set_title("Keynesian Cross: AD vs AS")
axes[0,0].set_xlabel("Month")
axes[0,0].set_ylabel("Output Units")

# 2. Output gap & unemployment
axes[0,1].plot(T, GAP, label="Output Gap (%)", linewidth=2, color='blue')
axes[0,1].plot(T, UNEMPLOYMENT, label="Unemployment (%)", linewidth=2, color='red')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)
axes[0,1].set_title("Output Gap and Unemployment")
axes[0,1].set_xlabel("Month")
axes[0,1].set_ylabel("Percent")

# 3. Keynesian expenditure shares
axes[1,0].plot(T, CONS_SHARE, label="Consumption Share", linewidth=2, color='green')
axes[1,0].plot(T, INV_SHARE, label="Investment Share", linewidth=2, color='orange')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)
axes[1,0].set_title("Expenditure Shares of GDP")
axes[1,0].set_xlabel("Month")
axes[1,0].set_ylabel("Percent")

# 4. Multiplier & Accelerator effects
axes[1,1].plot(T, MULTIPLIER, label="Multiplier (ΔY/ΔI)", linewidth=2, color='purple')
axes[1,1].plot(T, ACCELERATOR, label="Accelerator (ΔI/ΔY)", linewidth=2, color='brown')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)
axes[1,1].set_title("Multiplier and Accelerator Effects")
axes[1,1].set_xlabel("Month")
axes[1,1].set_ylabel("Ratio")

# 5. Inflation dynamics
axes[2,0].plot(T, INFLATION, label="Inflation (%)", linewidth=2, color='red')
axes[2,0].legend()
axes[2,0].grid(True, alpha=0.3)
axes[2,0].set_title("Demand-Pull Inflation")
axes[2,0].set_xlabel("Month")
axes[2,0].set_ylabel("Percent")

# 6. Price level
axes[2,1].plot(T, PRICE_LEVEL, label="Price Level", linewidth=2, color='darkblue')
axes[2,1].legend()
axes[2,1].grid(True, alpha=0.3)
axes[2,1].set_title("General Price Level")
axes[2,1].set_xlabel("Month")
axes[2,1].set_ylabel("Index")

plt.tight_layout()
plt.show()
