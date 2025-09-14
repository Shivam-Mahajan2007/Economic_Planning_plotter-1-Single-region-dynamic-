import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# -----------------------------
# PARAMETERS
# -----------------------------
n_industries = 1000      # reduce for speed; increase to test scalability
n_steps = 60             # number of discrete time steps
k_iterations = 50        # depth of Neumann approximation (paper suggests ~50)
connectivity = 0.01      # fraction of nonzero columns per row in A
target_growth_min = 0.6  # target total growth factor lower bound (50%)
target_growth_max = 0.7  # upper bound (60%)
delta = np.random.uniform(0.002, 0.004, n_industries)

def neumann_approx(A, v, k):
    res = v.copy().astype(float)
    term = v.copy().astype(float)
    for _ in range(1, k+1):
        term = A @ term
        res += term
    return res

print("parameters set.")
# -----------------------------
# BUILD MATRICES
# -----------------------------
I = np.eye(n_industries)

# Construct a sparse-like random technical coefficient matrix A
A = np.zeros((n_industries, n_industries), dtype=float)
for i in range(n_industries):
    k_conn = max(1, int(connectivity * n_industries))
    choices = np.random.choice(n_industries, size=k_conn, replace=False)
    A[i, choices] = np.random.uniform(0.05, 0.25, size=len(choices))

# Ensure spectral radius < 1 (stabilize)
eigvals = np.linalg.eigvals(A)
rho = max(np.abs(eigvals))
if rho >= 1 - 1e-6:
    A *= 0.9 / rho

L = I - A
print("leontif matrix L built.")

# Capital coefficients B (sparse-like)
B = np.zeros((n_industries, n_industries))
cap_providers = np.arange(int(0.1*n_industries), int(0.3*n_industries))
for i in range(n_industries):
    providers = np.random.choice(cap_providers, size=min(3, len(cap_providers)), replace=False)
    B[i, providers] = np.random.uniform(0.2, 1, size=len(providers))
print("Capital matrix B built.")

# -----------------------------
# INITIAL STATES
# -----------------------------
d_real = np.random.uniform(200.0, 500.0, n_industries)
d_est = d_real.copy()
X = neumann_approx(A, d_real, k_iterations)
C = X * 1.3
total_growth_factor = 1.0 + np.random.uniform(target_growth_min, target_growth_max, n_industries)
C_target = C * total_growth_factor

wL = np.random.uniform(5.0, 20.0, n_industries)
P0 = neumann_approx(A.T, wL, k_iterations)
P = P0.copy()

true_epsilon = np.random.uniform(-2.0, -0.2, n_industries)
measured_epsilon = true_epsilon * (1.0 + np.random.uniform(-0.2, 0.2, n_industries))
print("Initial states set.")

# -----------------------------
# STORAGE FOR TIME SERIES
# -----------------------------
AD_series = []
AS_series = []
GDP = []
GDP_growth = []
INFLATION = []
CPI_SERIES = []
PRICE_LEVEL = []
PRICE_INDEX = 1.0
INVESTMENT_SERIES = []
INVESTMENT_SHORT_SERIES = []
INVESTMENT_LONG_SERIES = []
DEMAND_GAP = []
PRICE_CHANGES = []
CAPACITY_TARGET_SERIES = []

# Track prices in selected industries
selected_inds = [0, 1, 2, 3, 4]  # first 5 industries
PRICE_SERIES_INDS = {i: [] for i in selected_inds}

X_prev = X.copy()
P_prev = P0.copy()
base_basket_value = X_prev @ P_prev

# -----------------------------
# SIMULATION LOOP 
# -----------------------------
print("Starting simulation...")
for t in range(n_steps):
    growth = 1.0 + np.random.uniform(0.002, 0.008, n_industries)
    d_real = d_real * growth
    delta_d = d_real - d_est
    frac = np.divide(delta_d, d_real) 
    Delta_P = frac * (P0 / true_epsilon)
    ratio = np.divide(Delta_P, P0)
    Delta_d = d_est * ratio * measured_epsilon
    d_est = d_est + Delta_d
    
    P = P0 + Delta_P

    # record selected prices
    for i in selected_inds:
        PRICE_SERIES_INDS[i].append(P[i])

    steps_left = max(1, n_steps - t)
    C = C * (1.0 - delta)
    growth_rate = np.power(np.divide(C_target, C), 1.0 / steps_left) - 1.0
    G_diag = growth_rate

    I_short = B @ neumann_approx(A, Delta_d, k_iterations)
    I_long = B @ (G_diag * C)
    I_total = I_short + I_long

    d_ag = d_est + I_total
    X = neumann_approx(A, d_ag, k_iterations)
    C = C + (G_diag * C) + neumann_approx(A, Delta_d, k_iterations)

    AD_val = (d_real + I_total).sum()
    AS_vec = L @ X
    AS_val = AS_vec.sum()

    AD_series.append(AD_val)
    AS_series.append(AS_val)
    GDP.append(AS_vec @ P)
    if t == 0:
        GDP_growth.append(0.0)
    else:
        GDP_growth.append((GDP[-1] / GDP[-2] - 1.0) * 100.0)

    current_basket_value = X @ P
    CPI_t = current_basket_value / base_basket_value
    CPI_SERIES.append(CPI_t)
    if t == 0:
        inflation_rate = (CPI_t - 1.0) / 1.0 
    else:
        inflation_rate =((CPI_t - CPI_SERIES[t-1]) / CPI_SERIES[t-1]) - GDP_growth[-1]/100.0
    INFLATION.append(inflation_rate * 100.0)
    PRICE_INDEX *= (1.0 + inflation_rate)
    PRICE_LEVEL.append(PRICE_INDEX)

    demand_gap_pct = np.mean(np.abs(delta_d) / d_real) * 100.0
    DEMAND_GAP.append(demand_gap_pct)
    PRICE_CHANGES.append(np.mean(np.abs(Delta_P) / P0) * 100.0)
    INVESTMENT_SERIES.append(I_total.sum())
    CAPACITY_TARGET_SERIES.append((L @ C).sum())

    P_prev = P.copy()

    INVESTMENT_SHORT_SERIES.append(I_short.sum())
    INVESTMENT_LONG_SERIES.append(I_long.sum())

    print("Step", t+1, "/", n_steps)

print("Simulation finished.")

# -----------------------------
# COMBINED SUBPLOTS
# -----------------------------
fig, axes = plt.subplots(4, 2, figsize=(14, 10))
T = np.arange(n_steps)

# (0,0) AD vs AS
ax = axes[0,0]
ax.plot(T, AD_series, label="Aggregate Demand")
ax.plot(T, AS_series, label="Aggregate Supply")
ax.plot(T, CAPACITY_TARGET_SERIES, label="Long run Aggregate Supply")
ax.axhline(y=CAPACITY_TARGET_SERIES[-1], color='k', linestyle='--', label="Target LRAS")
ax.set_title("AD vs AS")
ax.set_ylabel("Output Units")
ax.legend()
ax.grid(True)

# (0,1) Output gap and unemployment
ax = axes[0,1]
output_gap = (np.array(AD_series) - np.array(AS_series)) / AS_series * 100
unemployment = 5 + 0.2 * output_gap
ax.plot(T, output_gap, label="Output Gap (%)")
ax.plot(T, unemployment, label="Unemployment (%)")
ax.set_title("Output Gap and Unemployment")
ax.set_ylabel("Percent")
ax.legend()
ax.grid(True)

# (1,0) Expenditure shares
ax = axes[1,0]
consumption_share = np.array(AD_series) / (np.array(AD_series) + np.array(INVESTMENT_SERIES)) * 100
investment_share = np.array(INVESTMENT_SERIES) / (np.array(AD_series) + np.array(INVESTMENT_SERIES)) * 100
ax.plot(T, consumption_share, label="Consumption Share")
ax.plot(T, investment_share, label="Investment Share")
ax.set_title("Expenditure Shares of GDP")
ax.set_ylabel("Percent")
ax.legend()
ax.grid(True)

# (1,1) GDP growth
ax = axes[1,1]
ax.plot(T, GDP_growth, color="purple", label="GDP Growth (%)")
ax.set_title("GDP Growth Rate")
ax.set_ylabel("Percent")
ax.legend()
ax.grid(True)

# (2,0) Inflation
ax = axes[2,0]
ax.plot(T, INFLATION, color="red", label="Inflation (%)")
ax.set_title("CPI Inflation Rate")
ax.set_ylabel("Percent")
ax.legend()
ax.grid(True)

# (2,1) Price level
ax = axes[2,1]
ax.plot(T, PRICE_LEVEL, color="blue", label="Price Level Index")
ax.set_title("General Price Level (Index)")
ax.set_ylabel("Index (Base=1.0)")
ax.legend()
ax.grid(True)

# (3,0) Demand estimation accuracy
ax = axes[3,0]
ax.plot(T, DEMAND_GAP, color="orange", label="Demand Estimation Gap (%)")
ax.set_title("Demand Estimation Accuracy")
ax.set_ylabel("Percent Gap")
ax.legend()
ax.grid(True)

# (3,1) Price adjustments
ax = axes[3,1]
ax.plot(T, PRICE_CHANGES, color="green", label="Price Changes (%)")
ax.set_title("Price Adjustments (Algorithm)")
ax.set_ylabel("Percent Change")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

# -----------------------------
# Short vs Long Term Investment
# -----------------------------
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(T, INVESTMENT_SHORT_SERIES, label="Short-term Investment")
ax2.plot(T, INVESTMENT_LONG_SERIES, label="Long-term Investment")import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# -----------------------------
# PARAMETERS
# -----------------------------
n_industries = 1000      # reduce for speed; increase to test scalability
n_steps = 60             # number of discrete time steps
k_iterations = 50        # depth of Neumann approximation (paper suggests ~50)
connectivity = 0.01      # fraction of nonzero columns per row in A
target_growth_min = 0.6  # target total growth factor lower bound (50%)
target_growth_max = 0.7  # upper bound (60%)
delta = np.random.uniform(0.002, 0.004, n_industries)

def neumann_approx(A, v, k):
    res = v.copy().astype(float)
    term = v.copy().astype(float)
    for _ in range(1, k+1):
        term = A @ term
        res += term
    return res

print("parameters set.")
# -----------------------------
# BUILD MATRICES
# -----------------------------
I = np.eye(n_industries)

# Construct a sparse-like random technical coefficient matrix A
A = np.zeros((n_industries, n_industries), dtype=float)
for i in range(n_industries):
    k_conn = max(1, int(connectivity * n_industries))
    choices = np.random.choice(n_industries, size=k_conn, replace=False)
    A[i, choices] = np.random.uniform(0.05, 0.25, size=len(choices))

# Ensure spectral radius < 1 (stabilize)
eigvals = np.linalg.eigvals(A)
rho = max(np.abs(eigvals))
if rho >= 1 - 1e-6:
    A *= 0.9 / rho

L = I - A
print("leontif matrix L built.")

# Capital coefficients B (sparse-like)
B = np.zeros((n_industries, n_industries))
cap_providers = np.arange(int(0.1*n_industries), int(0.3*n_industries))
for i in range(n_industries):
    providers = np.random.choice(cap_providers, size=min(3, len(cap_providers)), replace=False)
    B[i, providers] = np.random.uniform(0.2, 1, size=len(providers))
print("Capital matrix B built.")

# -----------------------------
# INITIAL STATES
# -----------------------------
d_real = np.random.uniform(200.0, 500.0, n_industries)
d_est = d_real.copy()
X = neumann_approx(A, d_real, k_iterations)
C = X * 1.3
total_growth_factor = 1.0 + np.random.uniform(target_growth_min, target_growth_max, n_industries)
C_target = C * total_growth_factor

wL = np.random.uniform(5.0, 20.0, n_industries)
P0 = neumann_approx(A.T, wL, k_iterations)
P = P0.copy()

true_epsilon = np.random.uniform(-2.0, -0.2, n_industries)
measured_epsilon = true_epsilon * (1.0 + np.random.uniform(-0.2, 0.2, n_industries))
print("Initial states set.")

# -----------------------------
# STORAGE FOR TIME SERIES
# -----------------------------
AD_series = []
AS_series = []
GDP = []
GDP_growth = []
INFLATION = []
CPI_SERIES = []
PRICE_LEVEL = []
PRICE_INDEX = 1.0
INVESTMENT_SERIES = []
INVESTMENT_SHORT_SERIES = []
INVESTMENT_LONG_SERIES = []
DEMAND_GAP = []
PRICE_CHANGES = []
CAPACITY_TARGET_SERIES = []

# Track prices in selected industries
selected_inds = [0, 1, 2, 3, 4]  # first 5 industries
PRICE_SERIES_INDS = {i: [] for i in selected_inds}

X_prev = X.copy()
P_prev = P0.copy()
base_basket_value = X_prev @ P_prev

# -----------------------------
# SIMULATION LOOP 
# -----------------------------
print("Starting simulation...")
for t in range(n_steps):
    growth = 1.0 + np.random.uniform(0.002, 0.008, n_industries)
    d_real = d_real * growth
    delta_d = d_real - d_est
    frac = np.divide(delta_d, d_real) 
    Delta_P = frac * (P0 / true_epsilon)
    ratio = np.divide(Delta_P, P0)
    Delta_d = d_est * ratio * measured_epsilon
    d_est = d_est + Delta_d
    
    P = P0 + Delta_P

    # record selected prices
    for i in selected_inds:
        PRICE_SERIES_INDS[i].append(P[i])

    steps_left = max(1, n_steps - t)
    C = C * (1.0 - delta)
    growth_rate = np.power(np.divide(C_target, C), 1.0 / steps_left) - 1.0
    G_diag = growth_rate

    I_short = B @ neumann_approx(A, Delta_d, k_iterations)
    I_long = B @ (G_diag * C)
    I_total = I_short + I_long

    d_ag = d_est + I_total
    X = neumann_approx(A, d_ag, k_iterations)
    C = C + (G_diag * C) + neumann_approx(A, Delta_d, k_iterations)

    AD_val = (d_real + I_total).sum()
    AS_vec = L @ X
    AS_val = AS_vec.sum()

    AD_series.append(AD_val)
    AS_series.append(AS_val)
    GDP.append(AS_vec @ P)
    if t == 0:
        GDP_growth.append(0.0)
    else:
        GDP_growth.append((GDP[-1] / GDP[-2] - 1.0) * 100.0)

    current_basket_value = X @ P
    CPI_t = current_basket_value / base_basket_value
    CPI_SERIES.append(CPI_t)
    if t == 0:
        inflation_rate = (CPI_t - 1.0) / 1.0 
    else:
        inflation_rate =((CPI_t - CPI_SERIES[t-1]) / CPI_SERIES[t-1]) - GDP_growth[-1]/100.0
    INFLATION.append(inflation_rate * 100.0)
    PRICE_INDEX *= (1.0 + inflation_rate)
    PRICE_LEVEL.append(PRICE_INDEX)

    demand_gap_pct = np.mean(np.abs(delta_d) / d_real) * 100.0
    DEMAND_GAP.append(demand_gap_pct)
    PRICE_CHANGES.append(np.mean(np.abs(Delta_P) / P0) * 100.0)
    INVESTMENT_SERIES.append(I_total.sum())
    CAPACITY_TARGET_SERIES.append((L @ C).sum())

    P_prev = P.copy()

    INVESTMENT_SHORT_SERIES.append(I_short.sum())
    INVESTMENT_LONG_SERIES.append(I_long.sum())

    print("Step", t+1, "/", n_steps)

print("Simulation finished.")

# -----------------------------
# COMBINED SUBPLOTS
# -----------------------------
fig, axes = plt.subplots(4, 2, figsize=(14, 10))
T = np.arange(n_steps)

# (0,0) AD vs AS
ax = axes[0,0]
ax.plot(T, AD_series, label="Aggregate Demand")
ax.plot(T, AS_series, label="Aggregate Supply")
ax.plot(T, CAPACITY_TARGET_SERIES, label="Long run Aggregate Supply")
ax.axhline(y=CAPACITY_TARGET_SERIES[-1], color='k', linestyle='--', label="Target LRAS")
ax.set_title("AD vs AS")
ax.set_ylabel("Output Units")
ax.legend()
ax.grid(True)

# (0,1) Output gap and unemployment
ax = axes[0,1]
output_gap = (np.array(AD_series) - np.array(AS_series)) / AS_series * 100
unemployment = 5 + 0.2 * output_gap
ax.plot(T, output_gap, label="Output Gap (%)")
ax.plot(T, unemployment, label="Unemployment (%)")
ax.set_title("Output Gap and Unemployment")
ax.set_ylabel("Percent")
ax.legend()
ax.grid(True)

# (1,0) Expenditure shares
ax = axes[1,0]
consumption_share = np.array(AD_series) / (np.array(AD_series) + np.array(INVESTMENT_SERIES)) * 100
investment_share = np.array(INVESTMENT_SERIES) / (np.array(AD_series) + np.array(INVESTMENT_SERIES)) * 100
ax.plot(T, consumption_share, label="Consumption Share")
ax.plot(T, investment_share, label="Investment Share")
ax.set_title("Expenditure Shares of GDP")
ax.set_ylabel("Percent")
ax.legend()
ax.grid(True)

# (1,1) GDP growth
ax = axes[1,1]
ax.plot(T, GDP_growth, color="purple", label="GDP Growth (%)")
ax.set_title("GDP Growth Rate")
ax.set_ylabel("Percent")
ax.legend()
ax.grid(True)

# (2,0) Inflation
ax = axes[2,0]
ax.plot(T, INFLATION, color="red", label="Inflation (%)")
ax.set_title("CPI Inflation Rate")
ax.set_ylabel("Percent")
ax.legend()
ax.grid(True)

# (2,1) Price level
ax = axes[2,1]
ax.plot(T, PRICE_LEVEL, color="blue", label="Price Level Index")
ax.set_title("General Price Level (Index)")
ax.set_ylabel("Index (Base=1.0)")
ax.legend()
ax.grid(True)

# (3,0) Demand estimation accuracy
ax = axes[3,0]
ax.plot(T, DEMAND_GAP, color="orange", label="Demand Estimation Gap (%)")
ax.set_title("Demand Estimation Accuracy")
ax.set_ylabel("Percent Gap")
ax.legend()
ax.grid(True)

# (3,1) Price adjustments
ax = axes[3,1]
ax.plot(T, PRICE_CHANGES, color="green", label="Price Changes (%)")
ax.set_title("Price Adjustments (Algorithm)")
ax.set_ylabel("Percent Change")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

# -----------------------------
# Short vs Long Term Investment
# -----------------------------
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(T, INVESTMENT_SHORT_SERIES, label="Short-term Investment")
ax2.plot(T, INVESTMENT_LONG_SERIES, label="Long-term Investment")
ax2.plot(T, np.array(INVESTMENT_SHORT_SERIES) + np.array(INVESTMENT_LONG_SERIES),
         linestyle="--", color="gray", label="Total Investment")
ax2.set_title("Short-term vs Long-term Investment")
ax2.set_ylabel("Investment Units")
ax2.set_xlabel("Time Step")
ax2.legend()
ax2.grid(True)
plt.show()

# -----------------------------
#Prices in Selected Industries
# -----------------------------
fig3, ax3 = plt.subplots(figsize=(10, 6))
for i, series in PRICE_SERIES_INDS.items():
    ax3.plot(T, series, label=f"Industry {i}")
ax3.set_title("Prices in Selected Industries")
ax3.set_ylabel("Price")
ax3.set_xlabel("Time Step")
ax3.legend()
ax3.grid(True)
plt.show()

ax2.plot(T, np.array(INVESTMENT_SHORT_SERIES) + np.array(INVESTMENT_LONG_SERIES),
         linestyle="--", color="gray", label="Total Investment")
ax2.set_title("Short-term vs Long-term Investment")
ax2.set_ylabel("Investment Units")
ax2.set_xlabel("Time Step")
ax2.legend()
ax2.grid(True)
plt.show()

# -----------------------------
#Prices in Selected Industries
# -----------------------------
fig3, ax3 = plt.subplots(figsize=(10, 6))
for i, series in PRICE_SERIES_INDS.items():
    ax3.plot(T, series, label=f"Industry {i}")
ax3.set_title("Prices in Selected Industries")
ax3.set_ylabel("Price")
ax3.set_xlabel("Time Step")
ax3.legend()
ax3.grid(True)
plt.show()
