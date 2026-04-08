#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq
 
# ================================================
# 1. LOAD FUNCTIONS
# ================================================
 
def load_pjm_offers(path_offers_csv, price_cap=2000):
    """Load ALL PJM offers (do not filter by zone — competition is system-wide)."""
    df = pd.read_csv(path_offers_csv)
    df.columns = df.columns.str.strip().str.lower()
 
    df["bid_datetime_beginning_ept"] = pd.to_datetime(
        df["bid_datetime_beginning_ept"], errors="coerce"
    )
    df = df.dropna(subset=["bid_datetime_beginning_ept"])
    df = df.rename(columns={
        "bid_datetime_beginning_ept": "ts",
        "unit_code": "firm"
    })
 
    rows = []
    for k in range(1, 21):
        mw_col  = f"mw{k}"
        bid_col = f"bid{k}"
        if mw_col in df.columns and bid_col in df.columns:
            tmp = df[["ts", "firm", mw_col, bid_col]].copy()
            tmp = tmp.rename(columns={mw_col: "mw", bid_col: "price"})
            tmp["step"] = k
            rows.append(tmp)
 
    out = pd.concat(rows, ignore_index=True)
    out["mw"]    = pd.to_numeric(out["mw"],    errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["ts", "firm", "mw", "price"])
    out = out[out["mw"] > 0]
    if price_cap:
        out = out[out["price"] <= price_cap]
    out["ts"] = out["ts"].dt.tz_localize(None)
    return out[["ts", "firm", "step", "mw", "price"]]
 
def load_pjm_da_prices(path_prices_csv, zone="PSEG"):
    """Load congestion-adjusted day-ahead prices for a single representative node."""
    df = pd.read_csv(path_prices_csv)
    df.columns = df.columns.str.strip().str.lower()
 
    df["datetime_beginning_ept"] = pd.to_datetime(
        df["datetime_beginning_ept"], errors="coerce"
    )
    df = df.dropna(subset=["datetime_beginning_ept"])
    df = df[df["zone"] == zone].copy()
 
    df = df.rename(columns={
        "datetime_beginning_ept": "ts",
        "total_lmp_da":           "lmp",
        "congestion_price_da":    "congestion"
    })
    df["lmp"]        = pd.to_numeric(df["lmp"],        errors="coerce")
    df["congestion"] = pd.to_numeric(df["congestion"], errors="coerce")
    df = df.dropna(subset=["lmp", "congestion"])
 
    # Remove congestion to get system energy price (single-price assumption)
    df["price"] = df["lmp"] - df["congestion"]
    df["ts"]    = df["ts"].dt.tz_localize(None)
    return df[["ts", "price", "lmp"]].drop_duplicates("ts").sort_values("ts")
 
def load_pjm_hourly_load(path_load_csv, zone="PS"):
    """Load zonal hourly demand — used as price/demand anchor, not supply filter."""
    df = pd.read_csv(path_load_csv)
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
 
    df["ts"] = pd.to_datetime(
        df["datetime_beginning_ept"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce"
    )
    df = df.dropna(subset=["ts"])
    df["zone"] = df["zone"].str.strip().str.upper()
    zone       = zone.strip().upper()
    df = df[df["zone"] == zone].copy()
 
    df["Q"]  = pd.to_numeric(df["mw"], errors="coerce")
    df = df.dropna(subset=["Q"])
    df["ts"] = df["ts"].dt.floor("H")
    return df.set_index("ts")["Q"].sort_index()
 


# In[2]:


# ================================================
# 2. LOAD DATA
# ================================================
 
PATH_OFFERS = "energy_market_offers.csv"
PATH_PRICES_FILES = [
    "dahrl_lmps_sept1tosept15.csv",
    "dahrl_lmps_sept15-sept31.csv"
]

# Load and concatenate
prices = pd.concat(
    [load_pjm_da_prices(f, zone="PSEG") for f in PATH_PRICES_FILES],
    ignore_index=True
).drop_duplicates("ts").sort_values("ts")

print(f"Prices after concat: {len(prices)} rows, "
      f"{prices['ts'].min()} to {prices['ts'].max()}")
PATH_LOAD   = "hrl_load_metered (1).csv"
 
offers = load_pjm_offers(PATH_OFFERS)
load   = load_pjm_hourly_load(PATH_LOAD, zone="PS")
 
print(f"Offers: {len(offers):,} rows, {offers['ts'].nunique()} hours, {offers['firm'].nunique()} unique units")
print(f"Prices: {len(prices)} rows, {prices['ts'].min()} to {prices['ts'].max()}")
print(f"Load:   {len(load)} rows, {load.index.min()} to {load.index.max()}")


# In[5]:


# ================================================
# 3. PSEUDO-FIRM AGGREGATION
# ================================================
N_STRATEGIC = 5   # number of pseudo-firms you want
unit_totals = offers.groupby("firm")["mw"].sum().sort_values(ascending=False)

# Take top 50 units and divide them into 5 equal-sized groups
TOP_UNITS_TO_GROUP = 50   # how many units go into strategic firms
top_50 = unit_totals.head(TOP_UNITS_TO_GROUP).index.tolist()

# Assign units to firms in round-robin so each firm gets similar total MW
unit_to_firm = {}
for i, unit in enumerate(top_50):
    firm_label = f"PseudoFirm_{(i % N_STRATEGIC) + 1}"
    unit_to_firm[unit] = firm_label

# Everything outside top 50 is fringe
offers["pseudo_firm"] = offers["firm"].map(unit_to_firm).fillna("FRINGE")
strategic_firms = [f"PseudoFirm_{i+1}" for i in range(N_STRATEGIC)]

# Rebuild aggregated offers
offers_agg = (
    offers.groupby(["ts", "pseudo_firm", "price"])["mw"]
    .sum().reset_index()
    .rename(columns={"pseudo_firm": "firm"})
)
offers_agg["step"] = offers_agg.groupby(["ts","firm"]).cumcount() + 1
offers_trimmed = offers_agg[offers_agg["ts"].isin(common_ts)]
PS_SHARE = load_trimmed.mean() / offers_trimmed.groupby("ts")["mw"].sum().mean()
print(f"PS_SHARE = {PS_SHARE:.5f}")

# Scale ALL offer MW quantities to zonal scale ONCE here
offers_trimmed_scaled = offers_trimmed.copy()
offers_trimmed_scaled["mw"] = offers_trimmed_scaled["mw"] * PS_SHARE

print(f"Avg scaled total supply per hour: "
      f"{offers_trimmed_scaled.groupby('ts')['mw'].sum().mean():.0f} MW")
print(f"Avg zonal load:                   {load_trimmed.mean():.0f} MW")


# In[4]:


# ================================================
# 4. ALIGN TIMESTAMPS
# ================================================
 
price_series = prices.set_index("ts")["price"]
lmp_series   = prices.set_index("ts")["lmp"]
 
# Find hours where all three datasets overlap
common_ts = sorted(
    price_series.index
    .intersection(load.index)
    .intersection(offers_agg["ts"].unique())
)
print(f"\nOverlapping hours across all three datasets: {len(common_ts)}")
 
if len(common_ts) == 0:
    print("\nDEBUG: No overlapping hours found. Checking timestamp ranges:")
    print("  Price timestamps:", price_series.index.min(), "to", price_series.index.max())
    print("  Load timestamps: ", load.index.min(),         "to", load.index.max())
    print("  Offer timestamps:", offers_agg["ts"].min(),   "to", offers_agg["ts"].max())
    raise ValueError("No overlapping hours — check that your data files cover the same dates.")

load_trimmed   = load.reindex(common_ts).dropna()
offers_trimmed = offers_agg[offers_agg["ts"].isin(common_ts)]
P_series       = price_series.reindex(common_ts)
Q_series       = load_trimmed
 
# ================================================
# 5. GLOBAL DEMAND FIT
# ================================================
# FIX: rolling window=12 on 25 hours gives near-zero slope.
# Use the full sample for a single global demand estimate instead.
 
def fit_global_demand(P_series, Q_series):
    """
    Fit Q = A - B*P using all available hours.
    Returns (A, B) where B > 0 (downward sloping demand).
    """
    df = pd.DataFrame({"P": P_series, "Q": Q_series}).dropna()
    if len(df) < 3:
        raise ValueError("Not enough data points to fit demand.")
    X = np.column_stack([np.ones(len(df)), -df["P"].values])
    coef, *_ = np.linalg.lstsq(X, df["Q"].values, rcond=None)
    A, B = coef
    B = abs(B) + 1e-6   # enforce downward slope
    print(f"Global demand fit: A={A:.1f}, B={B:.4f}  (slope = -{B:.4f})")
    print(f"  Implied price range: [{(A - Q_series.max()) / B:.1f}, {A / B:.1f}] $/MWh")
    return A, B
 
A_global, B_global = fit_global_demand(P_series, Q_series)


# In[6]:


# ================================================
# 6. COST FUNCTIONS
# ================================================
 
def fit_quadratic_mc(offers_hour_scaled, firm):
    """Fit MC from scaled offer stack."""
    g = offers_hour_scaled[offers_hour_scaled["firm"] == firm].sort_values("price").copy()
    if len(g) < 2:
        return (20.0, 0.0, 0.0)
    g["cumq"] = g["mw"].cumsum()   # cumulative in zonal MW
    q = g["cumq"].values
    p = g["price"].values
    X = np.column_stack([np.ones_like(q), q, q**2])
    coef, *_ = np.linalg.lstsq(X, p, rcond=None)
    a, b, c = coef
    c = max(c, 0.0)
    return (a, b, c)
 
def mc_from_quadratic(params, q):
    a, b, c = params
    return a + b * q + c * q**2
 
# ================================================
# 7. FRINGE SUPPLY
# ================================================
 
avg_ps_load      = load_trimmed.mean()
avg_pjm_supply   = offers_trimmed.groupby("ts")["mw"].sum().mean()

def compute_fringe_supply(offers_hour_scaled, strategic_firms, p):
    """Fringe supply already in zonal MW — no additional scaling needed."""
    fringe = offers_hour_scaled[~offers_hour_scaled["firm"].isin(strategic_firms)]
    return fringe[fringe["price"] <= p]["mw"].sum()

def total_supply_ne(p):
    s = compute_fringe_supply(offers_hour, strategic_firms, p)
    for f in strategic_firms:
        a, b = ne_params[f]
        s += max((p - a) / b, 0.0)   
    return s


# In[7]:


import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq

# ================================================
# CONFIGURATION
# ================================================

N_VALUES       = [5, 10, 20, 40, 80]  # pseudo-firm counts to test
N_HOURS_SAMPLE = 10               # hours per experiment (keep small for speed)
MAX_SEC_FIRM   = 60               # per-firm timeout (seconds)
MAX_SEC_HOUR   = 500              # per-hour timeout (seconds)
# No iteration cap — convergence criterion (max_change < 1e-3) stops the loop.
# The only hard stop is MAX_SEC_HOUR.

# Use the same fixed hours for every N so runtime differences are apples-to-apples
hour_indices = np.linspace(0, len(common_ts) - 1, N_HOURS_SAMPLE, dtype=int)
scaling_ts   = [common_ts[i] for i in hour_indices]

print(f"Scaling experiment: N_VALUES={N_VALUES}, hours={N_HOURS_SAMPLE}")
print(f"Hours used: {[str(t) for t in scaling_ts]}\n")


# ================================================
# HELPER: build pseudo-firms for a given N
# ================================================

def build_pseudo_firms(n_strategic, top_units_per_firm=10):
    """
    Aggregate the top (n_strategic * top_units_per_firm) units by total MW
    into n_strategic pseudo-firms using round-robin assignment.
    Everything else goes to FRINGE.
    """
    top_n = unit_totals.head(n_strategic * top_units_per_firm).index.tolist()
    unit_to_firm = {
        unit: f"PseudoFirm_{(i % n_strategic) + 1}"
        for i, unit in enumerate(top_n)
    }
    return unit_to_firm, [f"PseudoFirm_{i+1}" for i in range(n_strategic)]


def rebuild_offers_for_n(n_strategic):
    """Re-aggregate offers_trimmed_scaled using a new pseudo-firm mapping."""
    unit_to_firm, strategic_firms = build_pseudo_firms(n_strategic)

    raw = offers_trimmed_scaled.copy()
    raw["pseudo_firm"] = raw["firm"].map(unit_to_firm).fillna("FRINGE")

    agg = (
        raw.groupby(["ts", "pseudo_firm", "price"])["mw"]
        .sum().reset_index()
        .rename(columns={"pseudo_firm": "firm"})
    )
    return agg, strategic_firms


# ================================================
# HELPER: run Nash for one hour (no iteration cap)
# ================================================

def compute_ne_hour_timed(offers_hour_scaled, strategic_firms, A, B):
    """
    Best-response loop with no iteration cap.
    Stops when max_change < 1e-3 (converged) or MAX_SEC_HOUR is exceeded (timeout).
    Returns (ne_params, n_iterations, elapsed_seconds, converged).
    """
    mc_params  = {f: fit_quadratic_mc(offers_hour_scaled, f) for f in strategic_firms}
    params     = {f: (15.0, 0.05) for f in strategic_firms}
    hour_start = time.time()
    iteration  = 0

    def demand_Q(p):
        return max(A - B * p, 0.0)

    while True:
        iteration += 1
        max_change = 0.0

        for f in strategic_firms:
            firm_start = time.time()

            def neg_profit(x, f=f):
                if time.time() - firm_start > MAX_SEC_FIRM:
                    return 1e9
                alpha, beta = x
                if beta <= 0:
                    return 1e9

                def q_i(p):
                    return max((p - alpha) / beta, 0.0)

                def total_supply(p):
                    s = compute_fringe_supply(offers_hour_scaled, strategic_firms, p)
                    for r in strategic_firms:
                        a_r, b_r = params[r]
                        s += q_i(p) if r == f else max((p - a_r) / b_r, 0.0)
                    return s

                def excess(p):
                    return total_supply(p) - demand_Q(p)

                try:
                    p_eq = brentq(excess, 0.0, 5000.0, maxiter=200)
                except Exception:
                    return 1e9

                q      = q_i(p_eq)
                mc     = mc_from_quadratic(mc_params[f], q)
                profit = (p_eq - mc) * q
                return -profit

            res = minimize(
                neg_profit,
                x0     = list(params[f]),
                bounds = [(0.0, 3000.0), (1e-4, 500.0)],
                method = "L-BFGS-B"
            )
            if res.success and res.fun < 1e8:
                old        = params[f]
                params[f]  = tuple(res.x)
                max_change = max(max_change,
                                 abs(params[f][0] - old[0]),
                                 abs(params[f][1] - old[1]))

            # Check hour-level timeout inside the firm loop too
            if time.time() - hour_start > MAX_SEC_HOUR:
                return params, iteration, time.time() - hour_start, False

        # Convergence check
        if max_change < 1e-3:
            return params, iteration, time.time() - hour_start, True

        # Hour-level timeout after completing a full iteration
        if time.time() - hour_start > MAX_SEC_HOUR:
            return params, iteration, time.time() - hour_start, False


# ================================================
# MAIN SCALING LOOP
# ================================================

scaling_records = []

for N in N_VALUES:
    print(f"\n{'='*55}")
    print(f"N = {N} pseudo-firms")
    print(f"{'='*55}")

    offers_n, strategic_n = rebuild_offers_for_n(N)
    hour_times   = []
    hour_iters   = []
    convergences = []

    for i, ts in enumerate(scaling_ts):
        oh = offers_n[offers_n["ts"] == ts]
        if oh.empty:
            print(f"  [{i+1}/{N_HOURS_SAMPLE}] {ts} — SKIP (no offers)")
            continue

        p_obs = float(price_series.loc[ts])
        Q_obs = float(load_trimmed.loc[ts])

        if p_obs <= 0 or Q_obs <= 0:
            print(f"  [{i+1}/{N_HOURS_SAMPLE}] {ts} — SKIP (bad data)")
            continue

        _, n_iters, elapsed, conv = compute_ne_hour_timed(
            oh, strategic_n, A_global, B_global
        )

        hour_times.append(elapsed)
        hour_iters.append(n_iters)
        convergences.append(conv)

        status = "converged" if conv else "TIMEOUT"
        print(f"  [{i+1}/{N_HOURS_SAMPLE}] {ts}  "
              f"{elapsed:.2f}s  iters={n_iters}  {status}")

    mean_time  = np.mean(hour_times)   if hour_times else np.nan
    total_time = np.sum(hour_times)    if hour_times else np.nan
    conv_rate  = np.mean(convergences) if convergences else np.nan

    print(f"  → Mean time/hour: {mean_time:.2f}s | "
          f"Total: {total_time:.1f}s | "
          f"Convergence rate: {conv_rate:.0%}")

    scaling_records.append({
        "N":               N,
        "mean_time_s":     mean_time,
        "total_time_s":    total_time,
        "conv_rate":       conv_rate,
        "n_hours":         len(hour_times),
        "extrap_720h_min": mean_time * 720 / 60,
    })

df_scaling = pd.DataFrame(scaling_records)

print(f"\n{'='*55}")
print("SCALING SUMMARY")
print(df_scaling.to_string(index=False))
print(f"{'='*55}")

df_scaling.to_csv("nash_scaling_results.csv", index=False)
print("Saved: nash_scaling_results.csv")


# ================================================
# PLOT: Runtime vs N
# ================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: mean time per hour
ax = axes[0]
ax.plot(df_scaling["N"], df_scaling["mean_time_s"],
        marker="o", linewidth=2, color="steelblue", markersize=8)
ax.set_xlabel("Number of pseudo-firms (N)", fontsize=12)
ax.set_ylabel("Mean solve time per hour (seconds)", fontsize=12)
ax.set_title("Nash Solve Time vs. Number of Firms", fontsize=13)
ax.set_xticks(N_VALUES)
ax.grid(True, alpha=0.3)

for _, row in df_scaling.iterrows():
    ax.annotate(f"{row['mean_time_s']:.1f}s",
                xy=(row["N"], row["mean_time_s"]),
                xytext=(5, 8), textcoords="offset points", fontsize=9)

# Right: extrapolated total runtime for 720 hours
ax2 = axes[1]
ax2.plot(df_scaling["N"], df_scaling["extrap_720h_min"],
         marker="s", linewidth=2, color="firebrick", markersize=8)
ax2.set_xlabel("Number of pseudo-firms (N)", fontsize=12)
ax2.set_ylabel("Extrapolated runtime for 720 hours (minutes)", fontsize=12)
ax2.set_title("Projected Runtime for Full Sample", fontsize=13)
ax2.set_xticks(N_VALUES)
ax2.grid(True, alpha=0.3)

ref = df_scaling[df_scaling["N"] == 5]["extrap_720h_min"].values
if len(ref):
    ax2.axhline(ref[0], linestyle="--", color="gray", alpha=0.6,
                label=f"N=5 baseline ({ref[0]:.1f} min)")
    ax2.legend(fontsize=9)

for _, row in df_scaling.iterrows():
    ax2.annotate(f"{row['extrap_720h_min']:.1f} min",
                 xy=(row["N"], row["extrap_720h_min"]),
                 xytext=(5, 8), textcoords="offset points", fontsize=9)

plt.tight_layout()
plt.savefig("nash_scaling_runtime.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: nash_scaling_runtime.png")
print("\n--- THESIS SUMMARY ---")
for _, row in df_scaling.iterrows():
    print(f"N={int(row['N']):3d}: "
          f"{row['mean_time_s']:.2f}s/hour, "
          f"~{row['extrap_720h_min']:.1f} min for full sample, "
          f"convergence={row['conv_rate']:.0%}")

if len(df_scaling) >= 2:
    n_low  = df_scaling.iloc[0]
    n_high = df_scaling.iloc[-1]
    ratio  = n_high["mean_time_s"] / n_low["mean_time_s"]
    print(f"\nRuntime scaling factor from N={int(n_low['N'])} to N={int(n_high['N'])}: {ratio:.1f}x")
    print(f"At N={int(n_high['N'])}, full-sample runtime would be "
          f"~{n_high['extrap_720h_min']:.0f} minutes.")
    print(f"True market has 1,207 units — Nash is computationally infeasible at that scale.")


# In[7]:


import time
import signal

# ================================================
# 8. NASH EQUILIBRIUM (HOUR-BY-HOUR)
# ================================================

MAX_SECONDS_PER_HOUR = 60   # if one hour takes longer than this, skip it
MAX_SECONDS_PER_FIRM = 20   # if one firm's best-response takes longer, abort it

def compute_ne_hour(offers_hour_scaled, strategic_firms, A, B,
                    n_iter=15, verbose=False):
    """
    All quantities already in zonal scale — no PS_SHARE needed here.
    """
    mc_params  = {f: fit_quadratic_mc(offers_hour_scaled, f) 
                  for f in strategic_firms}
    hour_start = time.time()

    def demand_Q(p):
        return max(A - B * p, 0.0)

    params = {f: (15.0, 0.05) for f in strategic_firms}

    for iteration in range(n_iter):
        iter_start = time.time()
        max_change = 0.0

        for f in strategic_firms:
            firm_start = time.time()

            def neg_profit(x, f=f):
                if time.time() - firm_start > MAX_SECONDS_PER_FIRM:
                    return 1e9
                alpha, beta = x
                if beta <= 0:
                    return 1e9

                def q_i(p):
                    return max((p - alpha) / beta, 0.0)

                def total_supply(p):
                    s = compute_fringe_supply(
                            offers_hour_scaled, strategic_firms, p)
                    for r in strategic_firms:
                        a_r, b_r = params[r]
                        if r == f:
                            s += q_i(p)
                        else:
                            s += max((p - a_r) / b_r, 0.0)
                    return s

                def excess(p):
                    return total_supply(p) - demand_Q(p)

                try:
                    p_eq = brentq(excess, 0.0, 5000.0, maxiter=200)
                except Exception:
                    return 1e9

                q      = q_i(p_eq)
                mc     = mc_from_quadratic(mc_params[f], q)
                profit = (p_eq - mc) * q
                return -profit

            res = minimize(
                neg_profit,
                x0     = list(params[f]),
                bounds = [(0.0, 3000.0), (1e-4, 500.0)],
                method = "L-BFGS-B"
            )
            if res.success and res.fun < 1e8:
                old        = params[f]
                params[f]  = tuple(res.x)
                max_change = max(max_change,
                                 abs(params[f][0] - old[0]),
                                 abs(params[f][1] - old[1]))

            if verbose and time.time() - firm_start > 5:
                print(f"      SLOW: firm {f} took "
                      f"{time.time()-firm_start:.1f}s on iter {iteration}")

        iter_elapsed  = time.time() - iter_start
        total_elapsed = time.time() - hour_start

        if verbose:
            print(f"    iter {iteration+1}/{n_iter}: "
                  f"max_change={max_change:.4f}, "
                  f"iter_time={iter_elapsed:.1f}s, "
                  f"total={total_elapsed:.1f}s")

        if max_change < 1e-3:
            if verbose:
                print(f"    Converged at iteration {iteration+1}")
            break

        if total_elapsed > MAX_SECONDS_PER_HOUR:
            print(f"    TIMEOUT after {iteration+1} iterations")
            break

    return params


# ================================================
# 9. RUN NASH MODEL ACROSS ALL HOURS
# ================================================

# Sample every 10th hour for Nash to avoid multi-hour runtime
sample_ts_nash = common_ts[::10]
print(f"\nRunning Nash on {len(sample_ts_nash)} sampled hours "
      f"(every 10th of {len(common_ts)} total)...")

nash_results = []
skipped      = 0
total_start  = time.time()

for i, ts in enumerate(sample_ts_nash):

    hour_start   = time.time()
    elapsed_total = time.time() - total_start

    print(f"  [{i+1:3d}/{len(sample_ts_nash)}] {ts}  "
          f"(total elapsed: {elapsed_total/60:.1f} min)", end="", flush=True)

    offers_hour = offers_trimmed_scaled[offers_trimmed_scaled["ts"] == ts]
    if offers_hour.empty:
        print("  → SKIP (no offers)")
        skipped += 1
        continue

    p_obs = float(price_series.loc[ts])
    Q_obs = float(load_trimmed.loc[ts])

    if p_obs <= 0 or Q_obs <= 0:
        print("  → SKIP (zero price/load)")
        skipped += 1
        continue

    # Set verbose=True for first 3 hours so you can see iteration timing
    verbose = (i < 3)
    ne_params = compute_ne_hour(
        offers_hour, strategic_firms, A_global, B_global,
        n_iter=15, verbose=verbose
    )

    def demand_Q(p):
        return max(A_global - B_global * p, 0.0)

    def total_supply_ne(p):
        s = compute_fringe_supply(offers_hour, strategic_firms, p)
        for f in strategic_firms:
            a, b = ne_params[f]
            s += max((p - a) / b, 0.0)
        return s

    def excess_ne(p):
        return total_supply_ne(p) - demand_Q(p)

    try:
        p_nash = brentq(excess_ne, 0.0, 5000.0, maxiter=200)
        q_nash = demand_Q(p_nash)
    except Exception as e:
        print(f"  → SKIP (brentq failed: {e})")
        skipped += 1
        continue

    mc_vals = []
    for f in strategic_firms:
        a, b = ne_params[f]
        q_f  = max((p_nash - a) / b, 0.0)
        mc_f = mc_from_quadratic(fit_quadratic_mc(offers_hour, f), q_f)
        mc_vals.append(mc_f)
    mc_avg = np.mean(mc_vals) if mc_vals else np.nan

    hour_elapsed = time.time() - hour_start
    print(f"  → p_obs={p_obs:.1f}  p_nash={p_nash:.1f}  "
          f"mc={mc_avg:.1f}  ({hour_elapsed:.1f}s)")

    nash_results.append({
        "ts":     ts,
        "p_obs":  p_obs,
        "lmp":    float(lmp_series.loc[ts]),
        "p_nash": p_nash,
        "Q_obs":  Q_obs,
        "q_nash": q_nash,
        "mc_avg": mc_avg,
        "markup": p_nash - mc_avg,
    })

df_nash = pd.DataFrame(nash_results)

print(f"\n{'='*55}")
print(f"Nash complete: {len(df_nash)} results, {skipped} skipped")
print(f"Total runtime: {(time.time()-total_start)/60:.1f} minutes")
print(f"{'='*55}")

if df_nash.empty:
    print("ERROR: df_nash is empty.")
else:
    print("\nNash vs Observed summary:")
    print(df_nash[["ts", "p_obs", "p_nash", "mc_avg", "markup"]].to_string())


# In[8]:


# ================================================
# CONDUCT FUNCTIONS
# ================================================

def compute_conduct_df(offers_trimmed_scaled, price_series, load_trimmed,
                       strategic_firms, A, B, common_ts):
    results = []
    for ts in common_ts:
        offers_hour = offers_trimmed_scaled[offers_trimmed_scaled["ts"] == ts]
        if offers_hour.empty:
            continue

        p_eq = float(price_series.loc[ts])
        Q_eq = float(load_trimmed.loc[ts])

        if p_eq <= 0 or Q_eq <= 0:
            continue
        dQdP = -B
        if abs(dQdP) < 1e-8:
            continue

        total_cap   = 0.0
        weighted_mc = 0.0

        for f in strategic_firms:
            firm_offers = offers_hour[offers_hour["firm"] == f]
            dispatched  = firm_offers[firm_offers["price"] <= p_eq]["mw"].sum()
            if dispatched <= 0:
                continue
            mc_params_f = fit_quadratic_mc(offers_hour, f)
            mc_f        = mc_from_quadratic(mc_params_f, dispatched)
            weighted_mc += mc_f * dispatched
            total_cap   += dispatched

        if total_cap <= 0:
            continue

        mc_avg = weighted_mc / total_cap
        theta  = (p_eq - mc_avg) * dQdP / (-Q_eq)
        theta  = float(np.clip(theta, 0, 1))

        results.append({
            "ts":    ts,
            "Q_eq":  Q_eq,
            "p_eq":  p_eq,
            "mc_avg": mc_avg,
            "theta": theta,
        })

    return pd.DataFrame(results)


def build_conduct_function(conduct_df, n_bins=10):
    if conduct_df.empty:
        print("Conduct DF is empty.")
        return pd.DataFrame()

    df = conduct_df.copy()
    df["q_norm"] = df["Q_eq"] / df["Q_eq"].max()

    bins       = np.linspace(df["q_norm"].min(), 1.0, n_bins + 1)
    df["bin"]  = pd.cut(df["q_norm"], bins, include_lowest=True)

    g   = df.groupby("bin")["theta"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["se"]       = out["std"] / np.sqrt(out["count"].clip(lower=1))
    out["q_center"] = out["bin"].apply(lambda x: (x.left + x.right) / 2)

    return out.dropna(subset=["mean"])


# ================================================
# RUN CONDUCT ON ALL 720 HOURS
# ================================================

print("Computing conduct function on all 720 hours...")
conduct_df = compute_conduct_df(
    offers_trimmed_scaled, price_series, load_trimmed,
    strategic_firms, A_global, B_global, common_ts
)
print(f"Conduct estimates: {len(conduct_df)} hours")
print(conduct_df[["ts", "p_eq", "mc_avg", "theta"]].head(10).to_string())

phi_df = build_conduct_function(conduct_df, n_bins=10)
print("\nConduct function bins:")
print(phi_df[["q_center", "mean", "se", "count"]].to_string())


# In[9]:


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ================================================
# PLOT 1: Nash vs Observed — time series + scatter
# ================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: time series
ax = axes[0]
ax.plot(df_nash["ts"], df_nash["p_obs"],
        label="Observed price", color="steelblue", linewidth=2)
ax.plot(df_nash["ts"], df_nash["p_nash"],
        label="Nash equilibrium price", color="darkorange",
        linewidth=2, linestyle="--")
ax.fill_between(df_nash["ts"], df_nash["p_nash"], df_nash["p_obs"],
                where=(df_nash["p_nash"] > df_nash["p_obs"]),
                alpha=0.12, color="red", label="Nash overpredicts")
ax.fill_between(df_nash["ts"], df_nash["p_nash"], df_nash["p_obs"],
                where=(df_nash["p_nash"] < df_nash["p_obs"]),
                alpha=0.12, color="green", label="Nash underpredicts")
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Price ($/MWh)", fontsize=11)
ax.set_title("PJM PSEG: Observed vs Nash Price\n(September 2025)", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# Right: scatter
ax2 = axes[1]
lim = max(df_nash["p_obs"].max(), df_nash["p_nash"].max()) * 1.05
ax2.scatter(df_nash["p_obs"], df_nash["p_nash"],
            alpha=0.8, edgecolors="steelblue",
            facecolors="lightblue", s=60, zorder=3)
ax2.plot([0, lim], [0, lim], "k--", linewidth=1, label="45° (perfect fit)")
ax2.axhline(df_nash["p_nash"].mean(), color="darkorange",
            linestyle=":", linewidth=1.5,
            label=f"Mean Nash = ${df_nash['p_nash'].mean():.1f}")
ax2.set_xlabel("Observed Price ($/MWh)", fontsize=11)
ax2.set_ylabel("Nash Equilibrium Price ($/MWh)", fontsize=11)
ax2.set_title("Nash vs Observed — PJM PSEG\n(congestion-adjusted)", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pjm_nash_vs_observed.png", dpi=150)
plt.show()

print("=== PJM Nash vs Observed Summary ===")
print(f"Hours computed:           {len(df_nash)}")
print(f"Mean observed price:      ${df_nash['p_obs'].mean():.2f}/MWh")
print(f"Mean Nash price:          ${df_nash['p_nash'].mean():.2f}/MWh")
print(f"Mean gap (Nash−observed): ${(df_nash['p_nash']-df_nash['p_obs']).mean():.2f}/MWh")
print(f"Nash > observed:          {(df_nash['p_nash'] > df_nash['p_obs']).mean():.0%} of hours")
print(f"Nash < observed:          {(df_nash['p_nash'] < df_nash['p_obs']).mean():.0%} of hours")


# ================================================
# PLOT 2: Markup vs Load
# ================================================

fig, ax = plt.subplots(figsize=(8, 5))

# Observed markup = p_obs - mc_avg
# Nash markup = p_nash - mc_avg
obs_markup  = df_nash["p_obs"]  - df_nash["mc_avg"]
nash_markup = df_nash["markup"]

sc = ax.scatter(df_nash["Q_obs"], obs_markup,
                c=df_nash["p_obs"], cmap="OrRd",
                s=70, alpha=0.85, zorder=3,
                label="Observed markup (p_obs − MC)")
ax.scatter(df_nash["Q_obs"], nash_markup,
           marker="^", color="steelblue", s=60,
           alpha=0.7, zorder=3,
           label="Nash markup (p_nash − MC)")
plt.colorbar(sc, ax=ax, label="Observed price ($/MWh)")

# Trend lines
for y_vals, color, label in [
    (obs_markup,  "red",       "Observed trend"),
    (nash_markup, "steelblue", "Nash trend"),
]:
    z     = np.polyfit(df_nash["Q_obs"], y_vals, 1)
    x_fit = np.linspace(df_nash["Q_obs"].min(), df_nash["Q_obs"].max(), 100)
    ax.plot(x_fit, np.polyval(z, x_fit), color=color,
            linewidth=2, linestyle="--", alpha=0.8, label=label)

ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Zonal load Q (MW)", fontsize=11)
ax.set_ylabel("Price − MC ($/MWh)", fontsize=11)
ax.set_title("PJM PSEG: Markup vs Load\n"
             "(observed markup rises with load; Nash markup relatively flat)",
             fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pjm_markup_vs_load.png", dpi=150)
plt.show()


# ================================================
# PLOT 3: Conduct function
# ================================================

# Run conduct on all 720 hours (fast — no optimizer)
conduct_df = compute_conduct_df(
    offers_trimmed_scaled, price_series, load_trimmed,
    strategic_firms, A_global, B_global, common_ts
)
print(f"\nConduct estimates: {len(conduct_df)} hours")

phi_df = build_conduct_function(conduct_df, n_bins=10)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(phi_df["q_center"], phi_df["mean"],
        marker="o", color="steelblue", linewidth=2, label="θ̂(q)")
ax.fill_between(
    phi_df["q_center"],
    (phi_df["mean"] - 1.96 * phi_df["se"]).clip(lower=0),
    (phi_df["mean"] + 1.96 * phi_df["se"]).clip(upper=1),
    alpha=0.2, color="steelblue", label="95% CI"
)
ax.axhline(0, color="gray", linestyle="--", linewidth=1,
           label="Perfect competition (θ=0)")
ax.axhline(1, color="red",  linestyle="--", linewidth=1,
           label="Full collusion (θ=1)")
ax.set_xlabel("Normalised zonal load q = Q / Q_max", fontsize=11)
ax.set_ylabel("Conduct parameter θ(q)", fontsize=11)
ax.set_title("PJM PSEG: Estimated Conduct Function\n"
             "(congestion-adjusted, pseudo-firm aggregation, September 2025)",
             fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(-0.05, 0.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pjm_conduct_function.png", dpi=150)
plt.show()


# ================================================
# PLOT 4: Offer stacks — 4 representative hours
# ================================================

# Pick hours spanning low, medium, high observed price
representative_hours = [
    df_nash.loc[df_nash["p_obs"].idxmin(), "ts"],   # lowest observed price
    df_nash.loc[(df_nash["p_obs"] - df_nash["p_obs"].median()).abs().idxmin(), "ts"],  # median
    df_nash.loc[df_nash["p_obs"].idxmax(), "ts"],   # highest observed price
    df_nash.loc[(df_nash["p_nash"] - df_nash["p_obs"]).abs().idxmin(), "ts"],  # closest fit
]

fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)

for ax, ts in zip(axes, representative_hours):
    row      = df_nash[df_nash["ts"] == ts].iloc[0]
    p_obs    = row["p_obs"]
    p_nash   = row["p_nash"]

    oh = offers_trimmed_scaled[offers_trimmed_scaled["ts"] == ts].sort_values("price")
    oh = oh[oh["price"] <= 150]
    oh["cum_mw"] = oh["mw"].cumsum()

    if oh.empty:
        continue

    xlim = oh["cum_mw"].quantile(0.95)

    ax.step(oh["cum_mw"], oh["price"], where="post",
            color="steelblue", linewidth=1.5, label="Offer stack")
    ax.axhline(p_obs,  color="navy",       linestyle="--", linewidth=2,
               label=f"Obs ${p_obs:.0f}")
    ax.axhline(p_nash, color="darkorange",  linestyle="--", linewidth=2,
               label=f"Nash ${p_nash:.0f}")

    ax.set_xlim(0, xlim)
    ax.set_ylim(0, min(150, max(p_obs, p_nash) * 1.6 + 10))
    ax.set_xlabel("Cumulative MW (zonal scale)", fontsize=9)
    ax.set_ylabel("$/MWh", fontsize=9)
    ax.set_title(f"{ts.strftime('%b %d %H:%M')}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("PJM PSEG Offer Stacks — Representative Hours\n"
             "(zonal-scaled supply, congestion-adjusted prices)",
             fontsize=12)
plt.tight_layout()
plt.savefig("pjm_offer_stacks.png", dpi=150)
plt.show()


# ================================================
# PLOT 5: Price gap vs load — key thesis figure
# ================================================

fig, ax = plt.subplots(figsize=(8, 5))

gap = df_nash["p_obs"] - df_nash["p_nash"]
sc  = ax.scatter(df_nash["Q_obs"], gap,
                 c=df_nash["p_obs"], cmap="RdYlGn",
                 s=70, alpha=0.85, zorder=3)
plt.colorbar(sc, ax=ax, label="Observed price ($/MWh)")

z     = np.polyfit(df_nash["Q_obs"], gap, 1)
x_fit = np.linspace(df_nash["Q_obs"].min(), df_nash["Q_obs"].max(), 100)
ax.plot(x_fit, np.polyval(z, x_fit), "k--",
        linewidth=2, label=f"Trend (slope={z[0]:.3f})")

ax.axhline(0, color="gray", linestyle="--", linewidth=1,
           label="Nash = Observed")
ax.set_xlabel("Zonal load Q (MW)", fontsize=11)
ax.set_ylabel("Observed − Nash price ($/MWh)", fontsize=11)
ax.set_title("PJM PSEG: Price Gap vs Load\n"
             "(positive = Nash underpredicts; negative = Nash overpredicts)",
             fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pjm_price_gap_vs_load.png", dpi=150)
plt.show()


# In[10]:


def plot_conduct(phi_df):

    if phi_df.empty:
        print("Nothing to plot.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(phi_df["q_center"], phi_df["mean"], marker="o", label="θ(q)")
    plt.fill_between(
        phi_df["q_center"],
        phi_df["mean"] - 1.96 * phi_df["se"],
        phi_df["mean"] + 1.96 * phi_df["se"],
        alpha=0.2,
        label="95% CI"
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Perfect competition")
    plt.axhline(1, color="red",  linestyle="--", linewidth=0.8, label="Full collusion")
    plt.xlabel("Normalized residual quantity q")
    plt.ylabel("Conduct θ(q)")
    plt.title("PJM PSEG Conduct Function (Congestion-Adjusted, Fringe-Netted)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# In[19]:


df_test = pd.read_csv("hrl_load_prelim.csv")
print(df_test.columns.tolist())


# In[11]:


# ================================================
# 10. MEAN FIELD GAME (MFG)
# ================================================

def demand_Q_linear(A, B, p):
    return max(A - B * p, 0.0)

def total_cost_from_mc(mc, q):
    a, b, c = mc
    return a*q + 0.5*b*q*q + (1/3)*c*q*q*q

def mc_from_quadratic(mc_params, q):
    a, b, c = mc_params
    return a + b*q + c*q*q

def representative_profit(p,m,demand_params,q_bar,beta,mc_params,eps_k, M):

    A, B = demand_params

    # Demand (scaled correctly)
    Dp = A - B * p

    # === Quantity terms ===
    q_plus  = (1.0 / M) * Dp + eps_k + (q_bar + beta * (m - p))
    q_minus = (1.0 / M) * Dp + eps_k - (q_bar + beta * (m - p))

    # enforce feasibility (very important)
    q_plus  = max(q_plus, 0.0)
    q_minus = max(q_minus, 0.0)

    # Revenue - Cost
    revenue = p * q_plus
    cost    = total_cost_from_mc(mc_params, q_minus)

    return revenue - cost

def solve_representative_firm(m,demand_params,q_bar,beta,mc_params,eps_grid,M,p_bounds=(0, 500)):

    p_star_k = []

    for eps_k in eps_grid:

        def obj(p):
            return -representative_profit(p[0],m,demand_params,q_bar,beta,mc_params,eps_k,M)

        res = minimize(obj,x0=[30.0],bounds=[p_bounds],method="L-BFGS-B")

        if res.success:
            p_star_k.append(float(res.x[0]))
        else:
            p_star_k.append(np.nan)

    return np.array(p_star_k)

def expected_price_given_m(m,demand_params,q_bar,beta,mc_params,eps_grid,M):

    p_star_k = solve_representative_firm(m,demand_params,q_bar,beta,mc_params,eps_grid,M)

    return np.nanmean(p_star_k)

def find_mfg_equilibrium(demand_params,q_bar,beta,mc_params,eps_grid,M,p_bounds=(0, 500),tol=1e-4):

    def g(m):
        return expected_price_given_m(m,demand_params,q_bar,beta,mc_params,eps_grid,M) - m

    try:
        return brentq(g, p_bounds[0], p_bounds[1], xtol=tol)
    except:
        return np.nan
    
def safe_mc_params(mc_list):
    avg = np.mean(mc_list, axis=0)
    a = max(avg[0], 15.0)   # floor on intercept
    b = max(avg[1], 0.0)    # force non-negative slope
    c = max(avg[2], 0.0)
    return (a, b, c)


# In[12]:


# 1. demand params 

# Fix 1: speed up the loop with a groupby lookup
offers_by_ts = {ts: grp for ts, grp in offers_trimmed_scaled.groupby("ts")}

# Fix 2: q_bar needs to reflect your 5 strategic firms, not all 1324 units
# Your model has 5 firms each representing 1324/5 units
M = 1324
n_strategic = len(strategic_firms)  # 5
q_bar = Q_series.mean() / n_strategic  # per strategic firm quantity
beta = np.mean([b for (_, b) in ne_params.values()])

print(f"M:            {M}")
print(f"n_strategic:  {n_strategic}")
print(f"q_bar:        {q_bar:.4f}")
print(f"beta:         {beta:.4f}")

mfg_dict = {}

for ts in Q_series.index:
    offers_hour = offers_by_ts.get(ts)
    if offers_hour is None:
        continue

    window = 48
    mask = (Q_series.index >= ts - pd.Timedelta(hours=window)) &            (Q_series.index <= ts + pd.Timedelta(hours=window))
    
    Q_local = Q_series[mask]
    P_local = P_series[mask]

    if len(Q_local) < 10:
        continue

    # Use global demand curve but center eps so mean is zero
    # This removes the level bias without needing local refit
    eps_grid = Q_local.values - (A_global - B_global * P_local.values)
    eps_grid = eps_grid - np.mean(eps_grid)  # ← KEY FIX: center the shocks

    mc_params_avg = safe_mc_params(
        [fit_quadratic_mc(offers_hour, f) for f in strategic_firms]
    )

    m_star = find_mfg_equilibrium(
        demand_params=(A_global, B_global),
        q_bar=q_bar,
        beta=beta,
        mc_params=mc_params_avg,
        eps_grid=eps_grid,
        M=M
    )

    mfg_dict[ts] = m_star

print(f"Done. Stored: {len(mfg_dict)}")
print(f"First 5: {list(mfg_dict.items())[:5]}")
# 2. pick representative firm parameters
# (use average or largest firm)
# representative parameters
m_star = find_mfg_equilibrium(
    demand_params=(A_global, B_global),
    q_bar=q_bar,
    beta=beta,
    mc_params=mc_params_avg,
    eps_grid=eps_grid,
    M=M
)

print("MFG equilibrium price:", m_star)


# In[13]:


ts_check = pd.Timestamp("2025-09-25 06:00:00")
offers_hour = offers_by_ts.get(ts_check)

raw_mc = [fit_quadratic_mc(offers_hour, f) for f in strategic_firms]
print("Raw MC params per firm:")
for f, mc in zip(strategic_firms, raw_mc):
    print(f"  {f}: a={mc[0]:.3f}, b={mc[1]:.3f}, c={mc[2]:.3f}")

safe = safe_mc_params(raw_mc)
print(f"\nSafe MC: {safe}")
print(f"MC intercept after floor: {safe[0]:.3f}  (should be >= 15)")


# In[14]:


ts_check = pd.Timestamp("2025-09-25 06:00:00")

window = 48
mask = (Q_series.index >= ts_check - pd.Timedelta(hours=window)) &        (Q_series.index <= ts_check + pd.Timedelta(hours=window))

Q_local = Q_series[mask]
P_local = P_series[mask]

eps_grid = Q_local.values - (A_global - B_global * P_local.values)

print(f"eps mean:  {np.mean(eps_grid):.2f}")
print(f"eps std:   {np.std(eps_grid):.2f}")
print(f"eps min:   {np.min(eps_grid):.2f}")
print(f"eps max:   {np.max(eps_grid):.2f}")
print(f"Q mean:    {Q_local.mean():.2f}")
print(f"P mean:    {P_local.mean():.2f}")
print(f"A - B*Pmean: {A_global - B_global * P_local.mean():.2f}")


# In[13]:


M = 1324
n_strategic = len(strategic_firms)
q_bar = Q_series.mean() / n_strategic
beta = np.mean([b for (_, b) in ne_params.values()])

mfg_dict = {}

for ts in Q_series.index:
    offers_hour = offers_by_ts.get(ts)
    if offers_hour is None:
        continue

    window = 48
    mask = (Q_series.index >= ts - pd.Timedelta(hours=window)) &            (Q_series.index <= ts + pd.Timedelta(hours=window))
    
    Q_local = Q_series[mask]
    P_local = P_series[mask]
    
    eps_grid = Q_local.values - (A_global - B_global * P_local.values)
    if len(eps_grid) == 0:
        continue

    mc_params_avg = tuple(np.mean(
        [fit_quadratic_mc(offers_hour, f) for f in strategic_firms], axis=0
    ))

    m_star = find_mfg_equilibrium(
        demand_params=(A_global, B_global),
        q_bar=q_bar,
        beta=beta,
        mc_params=mc_params_avg,
        eps_grid=eps_grid,
        M=M
    )

    mfg_dict[ts] = m_star

print(f"Done. Stored {len(mfg_dict)} results")
print(f"First 3: {list(mfg_dict.items())[:3]}")


# In[15]:


# Build aligned dataframe
mfg_series = pd.Series(mfg_dict).sort_index()

df_compare = pd.DataFrame({
    "mfg": mfg_series,
    "observed": P_series
}).dropna()

# Summary stats
print(f"Hours aligned:  {len(df_compare)}")
print(f"MFG mean:       {df_compare['mfg'].mean():.2f}")
print(f"Observed mean:  {df_compare['observed'].mean():.2f}")
print(f"MFG std:        {df_compare['mfg'].std():.2f}")
print(f"Observed std:   {df_compare['observed'].std():.2f}")
print(f"MFG min/max:    {df_compare['mfg'].min():.2f} / {df_compare['mfg'].max():.2f}")
print(f"Observed min/max: {df_compare['observed'].min():.2f} / {df_compare['observed'].max():.2f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: time series
ax = axes[0]
ax.plot(df_compare.index, df_compare["observed"], label="Observed", linewidth=1.5)
ax.plot(df_compare.index, df_compare["mfg"], label="MFG", linewidth=1.5, linestyle="--")
ax.fill_between(df_compare.index, df_compare["mfg"], df_compare["observed"],
                where=(df_compare["mfg"] > df_compare["observed"]),
                alpha=0.15, color="red", label="MFG overpredicts")
ax.fill_between(df_compare.index, df_compare["mfg"], df_compare["observed"],
                where=(df_compare["mfg"] < df_compare["observed"]),
                alpha=0.15, color="blue", label="MFG underpredicts")
ax.set_title("MFG vs Observed Prices")
ax.set_ylabel("Price ($/MWh)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# Right: scatter
ax2 = axes[1]
ax2.scatter(df_compare["observed"], df_compare["mfg"], alpha=0.4, s=10)
lim = max(df_compare["observed"].max(), df_compare["mfg"].max()) * 1.05
ax2.plot([0, lim], [0, lim], "k--", label="45° line")
ax2.set_xlabel("Observed Price ($/MWh)")
ax2.set_ylabel("MFG Price ($/MWh)")
ax2.set_title("MFG vs Observed Scatter")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[18]:


# Align all three on the same timestamps
nash_series = df_nash.set_index("ts")["p_nash"]

df_all = pd.DataFrame({
    "observed": P_series,
    "mfg": mfg_series,
    "nash": nash_series
}).dropna()

print(f"Hours aligned: {len(df_all)}")
print(f"Observed mean: {df_all['observed'].mean():.2f}")
print(f"Nash mean:     {df_all['nash'].mean():.2f}")
print(f"MFG mean:      {df_all['mfg'].mean():.2f}")

rmse_nash = np.sqrt(np.mean((df_all["observed"] - df_all["nash"])**2))
rmse_mfg  = np.sqrt(np.mean((df_all["observed"] - df_all["mfg"])**2))

print(f"\nRMSE Nash: {rmse_nash:.2f}")
print(f"RMSE MFG:  {rmse_mfg:.2f}")

# Main comparison figure
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df_all.index, df_all["observed"], label="Observed", linewidth=1.5, color="steelblue")
ax.plot(df_all.index, df_all["nash"], label="Nash", linewidth=1.5, linestyle="--", color="darkorange")
ax.plot(df_all.index, df_all["mfg"], label="MFG", linewidth=1.5, linestyle="-.", color="green")
ax.set_title("PJM PSEG: Observed vs Nash vs MFG Prices (September 2025)")
ax.set_ylabel("Price ($/MWh)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
plt.tight_layout()
plt.show()


# In[19]:


fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["Nash", "MFG"], [rmse_nash, rmse_mfg], color=["darkorange", "green"], alpha=0.8)
ax.set_ylabel("RMSE ($/MWh)")
ax.set_title("Prediction Error: Nash vs MFG\nPJM PSEG September 2025")
ax.set_ylim(0, 35)
for i, v in enumerate([rmse_nash, rmse_mfg]):
    ax.text(i, v + 0.5, f"{v:.2f}", ha="center", fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()


# In[20]:


print(f"Nash mean:     {df_all['nash'].mean():.2f}")
print(f"MFG mean:      {df_all['mfg'].mean():.2f}")
print(f"Observed mean: {df_all['observed'].mean():.2f}")
print(f"RMSE Nash:     {rmse_nash:.2f}")
print(f"RMSE MFG:      {rmse_mfg:.2f}")
print(f"Nash > observed: {(df_all['nash'] > df_all['observed']).mean():.1%}")
print(f"MFG > observed:  {(df_all['mfg'] > df_all['observed']).mean():.1%}")
print(f"MFG std:       {df_all['mfg'].std():.2f}")
print(f"Observed std:  {df_all['observed'].std():.2f}")
print(f"Nash std:      {df_all['nash'].std():.2f}")


# In[ ]:




