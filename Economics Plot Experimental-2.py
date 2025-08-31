import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# PARAMETERS
# -----------------------------
np.random.seed(42)
n_industries = 1000    # reduce if too slow
n_steps = 60           # time steps
k_iterations = 50      # Neumann series depth

# Elasticities
true_epsilon = np.random.uniform(-2, -0.2, n_industries)
measured_epsilon = true_epsilon * (1.0 + np.random.uniform(-0.1, 0.1, n_industries))

total_growth_factor = 1.0 + np.random.uniform(0.4, 0.5, n_industries)

print("Making parameters...")

# -----------------------------
# MATRICES A, B
# -----------------------------
I = np.eye(n_industries)

# Technical coefficients (A)
A = np.zeros((n_industries, n_industries))
for i in range(n_industries):
    connections = np.random.choice(n_industries, size=int(0.01 * n_industries), replace=False)
    A[i, connections] = np.random.uniform(0.5, 0.9, size=len(connections))

# Stabilize A (ρ(A) < 1)
rho = np.max(np.abs(np.linalg.eigvals(A)))
if rho >= 1:
    A *= 0.9 / rho

L = I - A

# Capital coefficients (B)
B = np.zeros((n_industries, n_industries))
cap_providers = list(range(10, 30))
for i in range(n_industries):
    providers = np.random.choice(cap_providers, size=min(3, len(cap_providers)), replace=False)
    B[i, providers] = np.random.uniform(0.1, 0.9, size=len(providers))

print("Matrices built!")

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
d_real = np.random.uniform(200, 500, n_industries)
d_estimate = d_real.copy()  # Start with perfect estimation

# Initial production
X = neumann_approx(A, k_iterations, d_real)

# Equilibrium price
wL = np.random.uniform(5, 20, n_industries)
P0 = neumann_approx(A.T, k_iterations, wL)
P = P0.copy()

X = neumann_approx(A, k_iterations, d_real)
C = X * 1.25
C_target = C * total_growth_factor

print("Initial states set!")

# -----------------------------
# STORAGE
# -----------------------------
AD, AS = [], []
GAP, UNEMPLOYMENT = [], []
final_CAPACITY = []
GDP, GDP_GROWTH = [], []
INFLATION, PRICE_LEVEL = [], []
CONS_SHARE, INV_SHARE = [], []
INVESTMENT = []
DEMAND_GAP, PRICE_CHANGES = [], []
sector_gaps = []


# -----------------------------
# SIMULATION LOOP
# -----------------------------
print("Simulation started!")

for t in range(n_steps):
    # 1. Real demand grows exogenously
    growth = 1.0 + np.random.uniform(0.005, 0.01, n_industries)
    d_real = d_real * growth

    # 2. Demand gap
    delta_d = d_real - d_estimate

    # 3. Price adjustment
    Delta_P = (delta_d / (d_real + 1e-12)) * (P0 / true_epsilon)
    P = P0 + Delta_P

    # 4. Update estimated demand
    Delta_d = d_estimate * (Delta_P / (P0 + 1e-12)) * measured_epsilon
    d_estimate = d_estimate + Delta_d

    # 5a. Short-run investment (driven by Δd)
    I_short = B @ neumann_approx(A, k_iterations, Delta_d)
    steps_left = max(1, n_steps - t)
    required_growth_factor = (C_target / np.maximum(C, 1e-12)) ** (1.0 / steps_left) - 1.0
    required_growth_factor = np.clip(required_growth_factor, 0.0, 0.02)
    delta_X_long = required_growth_factor * C
    I_long = B @ delta_X_long

    # 5c. Total investment
    I_total = I_short + I_long 

    # 6. Production
    prod_input = d_estimate + I_total
    X = neumann_approx(A, k_iterations, prod_input)
    C += delta_X_long + neumann_approx(A, k_iterations, Delta_d)

    # 7. Aggregate demand & supply
    AD_val = (d_real + I_total).sum()
    AS_vec = L @ X
    AS_val = AS_vec.sum()
    LRAS = L @ C
    LRAS_val = LRAS.sum()

    AD.append(AD_val)
    AS.append(AS_val)
    final_CAPACITY.append(LRAS_val)  # potential output


    # 8. Gaps and unemployment
    gap = (AS_val - AD_val) / (AD_val + 1e-12) * 100
    GAP.append(gap)
    UNEMPLOYMENT.append(max(0, 5 + 0.5 * gap))

    # 9. GDP and growth
    GDP.append(AS_val)
    if t > 0:
        GDP_GROWTH.append((AS_val / GDP[-2] - 1) * 100)
    else:
        GDP_GROWTH.append(0)

    # 10. Inflation and price tracking
    INFLATION.append((P.sum() / P0.sum() - 1) * 100)
    price_level_t = (P @ X) / (X.sum() + 1e-12)
    PRICE_LEVEL.append(price_level_t)

    # 11. Expenditure shares
    C_share = d_real.sum() / (AS_val + 1e-12) * 100
    I_share = I_total.sum() / (AS_val + 1e-12) * 100
    CONS_SHARE.append(C_share)
    INV_SHARE.append(I_share)

    # 12. Sector-level gaps
    sector_gap_t = (AS_vec - (d_real + I_total)) / ((d_real + I_total) + 1e-12) * 100
    sector_gaps.append(sector_gap_t)

    # 13. Tracking accuracy
    demand_gap = np.mean(np.abs(delta_d) / (d_real + 1e-12)) * 100
    DEMAND_GAP.append(demand_gap)
    PRICE_CHANGES.append(np.mean(np.abs(Delta_P) / (P0 + 1e-12)) * 100)

    # Save investment
    INVESTMENT.append(I_total.sum())

    print(t+1, "/", n_steps, "completed")

print("Simulation complete!")

# -----------------------------
# PLOTTING
# -----------------------------
T = np.arange(n_steps)
fig, axes = plt.subplots(4, 2, figsize=(16, 16))

# 1. AD vs AS
axes[0,0].plot(T, AD, label="Aggregate Demand", linewidth=2)
axes[0,0].plot(T, AS, label="(SR) Aggregate Supply", linewidth=2)
axes[0,0].plot(T, final_CAPACITY, label="(LR) Aggregate Supply", linewidth=2)
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
axes[0,0].set_title("AD vs AS"); axes[0,0].set_xlabel("Month"); axes[0,0].set_ylabel("Output Units")

# 2. Output gap & unemployment
axes[0,1].plot(T, GAP, label="Output Gap (%)", linewidth=2, color='blue')
axes[0,1].plot(T, UNEMPLOYMENT, label="Unemployment (%)", linewidth=2, color='red')
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)
axes[0,1].set_title("Output Gap and Unemployment"); axes[0,1].set_xlabel("Month"); axes[0,1].set_ylabel("Percent")

# 3. Expenditure shares
axes[1,0].plot(T, CONS_SHARE, label="Consumption Share", linewidth=2, color='green')
axes[1,0].plot(T, INV_SHARE, label="Investment Share", linewidth=2, color='orange')
axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)
axes[1,0].set_title("Expenditure Shares of GDP"); axes[1,0].set_xlabel("Month"); axes[1,0].set_ylabel("Percent")

# 4. GDP growth
axes[1,1].plot(T, GDP_GROWTH, label="GDP Growth (%)", linewidth=2, color='purple')
axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)
axes[1,1].set_title("GDP Growth Rate"); axes[1,1].set_xlabel("Month"); axes[1,1].set_ylabel("Percent")

# 5. Inflation
axes[2,0].plot(T, INFLATION, label="Inflation (%)", linewidth=2, color='red')
axes[2,0].legend(); axes[2,0].grid(True, alpha=0.3)
axes[2,0].set_title("Demand-Pull Inflation"); axes[2,0].set_xlabel("Month"); axes[2,0].set_ylabel("Percent")

# 6. Price level
axes[2,1].plot(T, PRICE_LEVEL, label="Price Level", linewidth=2, color='darkblue')
axes[2,1].legend(); axes[2,1].grid(True, alpha=0.3)
axes[2,1].set_title("General Price Level"); axes[2,1].set_xlabel("Month"); axes[2,1].set_ylabel("Index")

# 7. Demand estimation gap
axes[3,0].plot(T, DEMAND_GAP, label="Demand Estimation Gap (%)", linewidth=2, color='orange')
axes[3,0].legend(); axes[3,0].grid(True, alpha=0.3)
axes[3,0].set_title("Demand Estimation Accuracy"); axes[3,0].set_xlabel("Month"); axes[3,0].set_ylabel("Percent Gap")

# 8. Price changes
axes[3,1].plot(T, PRICE_CHANGES, label="Price Changes (%)", linewidth=2, color='green')
axes[3,1].legend(); axes[3,1].grid(True, alpha=0.3)
axes[3,1].set_title("Price Adjustments"); axes[3,1].set_xlabel("Month"); axes[3,1].set_ylabel("Percent Change")

plt.tight_layout()
plt.show()
