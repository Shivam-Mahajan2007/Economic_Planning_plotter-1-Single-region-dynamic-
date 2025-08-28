# Economic_Planning_plotter: Single-region-dynamic

Here’s a **clean README.md** draft for your project. I’ve written it in a way that makes it look like a small research simulation package, while being readable for someone who just wants to run your script:

---

# Economic Simulation Model

This project implements a **Keynesian-inspired macroeconomic simulation** using an **Input-Output (IO) framework** with investment dynamics, demand feedback, and price effects.
It generates a variety of **Keynesian economic indicators** (GDP, unemployment, multiplier, accelerator, inflation, expenditure shares, etc.) and visualizations over a 5-year horizon.

---

## 📌 Features

* **Input-Output Model** with Neumann series approximation
* **Keynesian demand-driven growth** with autonomous demand expansion and price sensitivity
* **Investment dynamics** via short-term demand signals and long-term capacity targets
* **Price adjustment mechanism** influencing demand through elasticities
* **Capacity planning** with dotted line for long-run target capacity (potential output)
* **Weighted price index** over time

## 📊 Indicators Generated

* **Aggregate Demand (AD)**, **Aggregate Supply (AS)**, and **Capacity Target (LRAS-like)**
* **Output Gap (%)** and **Unemployment proxy** (Okun’s Law style)
* **GDP level** and **GDP growth rate**
* **Demand-pull Inflation (%)**
* **Price Level index** (weighted by sectoral output)
* **Consumption share** and **Investment share** of GDP
* **Multiplier effect** ($ΔY/ΔI$) and **Accelerator effect** ($ΔI/ΔY$)
* **Sector-level output gaps** (distribution histogram)

---

## 📈 Plots Produced

1. **Keynesian Cross (AD vs AS with Capacity Target)**
2. **Output Gap and Unemployment (%)**
3. **Price Level Over Time (weighted index)**
4. **Consumption and Investment Shares of GDP (%)**
5. **Multiplier and Accelerator effects**
6. **Inflation dynamics (demand-pull)**
7. **Sector-level output gap distribution** (final month histogram)

---

## ⚙️ Requirements

This project uses **Python 3.9+** with the following packages:

```bash
pip install numpy matplotlib
```

---

## ▶️ Running the Simulation

Clone or download the project, then run:

```bash
python Economic_model.py
```

This will generate all plots sequentially.

---
---

## 🧠 Notes

* This is a **stylized model** — it is not intended to forecast real economies but to explore Keynesian dynamics in a controlled IO framework.
* Price dynamics and unemployment are modeled heuristically, for teaching/illustrative purposes.
* Parameters (elasticities, investment response, growth targets) can be modified in the script to test alternative scenarios.

---

## 🔮 Possible Extensions

* Add a **Phillips Curve** plot (Unemployment vs Inflation).
* Incorporate **fiscal/monetary policy shocks** (government spending, interest rate channels).
* Allow for **heterogeneous consumer preferences** across industries.

---

Would you like me to make the README more **formal/academic** (like a research project write-up), or **accessible/educational** (like a teaching tool for high school/undergrad students)?
