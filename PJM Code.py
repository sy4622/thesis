#!/usr/bin/env python
# coding: utf-8

# In[95]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, Callable, List, Optional
from scipy.optimize import minimize, brentq

# 1. SAFE HOUR-ENDING PARSER

def parse_hour_ending(series: pd.Series) -> pd.Series:
    """
    Handle Hour Ending in formats like "01:00", "1", "24:00", etc.
    Returns integer hours 0–23, with 24 mapped to 0 (midnight next day).
    """
    he = (
        series.astype(str)
        .str.strip()
        .str.split(':')
        .str[0]
        .replace('', '0')
        .astype(int)
        .replace(24, 0)
    )
    return he

PATH_OFFERS = "energy_market_offers.csv"
PATH_PRICES = "da_hrl_lmps (1).csv"
PATH_LOAD   = "hrl_load_metered (1).csv"

offers = load_pjm_offers(PATH_OFFERS)
prices = load_pjm_da_prices(PATH_PRICES, zone="PSEG")
load   = load_pjm_hourly_load(PATH_LOAD, zone="PS")

# Load full PJM load to get PSEG share (do this once)
df_all = pd.read_csv(PATH_LOAD)
df_all.columns = df_all.columns.str.strip().str.replace(" ", "_").str.lower()
df_all["ts"] = pd.to_datetime(df_all["datetime_beginning_ept"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
df_all["mw"] = pd.to_numeric(df_all["mw"], errors="coerce")
df_all["zone"] = df_all["zone"].str.strip().str.upper()

# PSEG share = PS load / total load across all zones
pseg_mw  = df_all[df_all["zone"] == "PS"]["mw"].sum()
total_mw = df_all["mw"].sum()
PSEG_SHARE = pseg_mw / total_mw

print(f"PSEG share of PJM load: {PSEG_SHARE:.4f}")


# In[96]:


def load_pjm_offers(path_offers_csv, price_cap=2000):

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
        mw_col = f"mw{k}"
        bid_col = f"bid{k}"

        if mw_col in df.columns and bid_col in df.columns:
            tmp = df[["ts", "firm", mw_col, bid_col]].copy()
            tmp = tmp.rename(columns={mw_col: "mw", bid_col: "price"})
            tmp["step"] = k
            rows.append(tmp)

    out = pd.concat(rows, ignore_index=True)

    out["mw"] = pd.to_numeric(out["mw"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")

    out = out.dropna(subset=["ts", "firm", "mw", "price"])
    out = out[out["mw"] > 0]

    if price_cap:
        out = out[out["price"] <= price_cap]

    out["ts"] = out["ts"].dt.tz_localize(None)

    return out[["ts", "firm", "step", "mw", "price"]]

def load_pjm_da_prices(path_prices_csv, zone="PSEG"):

    df = pd.read_csv(path_prices_csv)
    df.columns = df.columns.str.strip().str.lower()

    df["datetime_beginning_ept"] = pd.to_datetime(
        df["datetime_beginning_ept"], errors="coerce"
    )
    df = df.dropna(subset=["datetime_beginning_ept"])
    df = df[df["pnode_name"] == zone].copy()

    df = df.rename(columns={
        "datetime_beginning_ept": "ts",
        "total_lmp_da": "lmp",
        "congestion_price_da": "congestion"
    })

    df["lmp"] = pd.to_numeric(df["lmp"], errors="coerce")
    df["congestion"] = pd.to_numeric(df["congestion"], errors="coerce")
    df = df.dropna(subset=["lmp", "congestion"])

    # Isolate system energy price per equation 6.4
    df["price"] = df["lmp"] - df["congestion"]

    df["ts"] = df["ts"].dt.tz_localize(None)

    return df[["ts", "price", "lmp", "congestion"]].drop_duplicates("ts").sort_values("ts")
   

def load_pjm_hourly_load(path_load_csv, zone="PS"):

    df = pd.read_csv(path_load_csv)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.lower()
    )

    # Use EPT
    df["ts"] = pd.to_datetime(
        df["datetime_beginning_ept"],
        format="%m/%d/%Y %I:%M:%S %p",   # <-- explicit format for "9/1/2025 12:00:00 AM"
        errors="coerce"
    )

    df = df.dropna(subset=["ts"])

    df["zone"] = df["zone"].str.strip().str.upper()
    zone = zone.strip().upper()

    df = df[df["zone"] == zone].copy()

    df["Q"] = pd.to_numeric(df["mw"], errors="coerce")
    df = df.dropna(subset=["Q"])

    df["ts"] = df["ts"].dt.floor("H")

    result = df.set_index("ts")["Q"].sort_index()

    print("Load range:", result.index.min(), "→", result.index.max())
    print("Length:", len(result))

    return result



def compute_fringe_supply(offers_hour, strategic_firms, p, zone_share=PSEG_SHARE):
    fringe = offers_hour[~offers_hour["firm"].isin(strategic_firms)]
    raw = fringe[fringe["price"] <= p]["mw"].sum()
    return raw * zone_share


# In[97]:


# cost and demand functions
def fit_quadratic_mc_for_hour(offers_hour: pd.DataFrame):
    params = {}
    for firm, g in offers_hour.groupby("firm"):
        g = g.sort_values("price").copy()
        g["cumq"] = g["mw"].cumsum()

        q = g["cumq"].to_numpy()
        p = g["price"].to_numpy()

        if len(q) < 2:
            params[firm] = (np.mean(p), 0.0, 0.0)
            continue

        X = np.column_stack([np.ones_like(q), q, q*q])
        coef, *_ = np.linalg.lstsq(X, p, rcond=None)
        a, b, c = coef
        c = max(c, 0.0)
        params[firm] = (a, b, c)

    return params


def mc_from_quadratic(params, q):
    a, b, c = params
    return a + b*q + c*q*q


def fit_quadratic_demand(ts, P, Q, window=48):

    lo = ts - pd.Timedelta(hours=window)
    hi = ts + pd.Timedelta(hours=window)

    sub = pd.DataFrame({"p": P, "q": Q})
    sub = sub[(sub.index >= lo) & (sub.index <= hi)]

    P_vals = sub["p"].to_numpy()
    Q_vals = sub["q"].to_numpy()

    X = np.column_stack([np.ones_like(P_vals), -P_vals])
    coef, *_ = np.linalg.lstsq(X, Q_vals, rcond=None)

    A, B = coef
    B = abs(B) + 1e-6

    return A, B


# In[115]:


def compute_ne_hour(offers_hour, demand_params, strategic_firms):

    A, B = demand_params
    mc_params = fit_quadratic_mc_for_hour(offers_hour)

    def demand_Q(p):
        return max(A - B*p, 0)

    params = {f: (0.0, 0.01) for f in strategic_firms}

    for _ in range(20):
        for f in strategic_firms:

            def profit(x):
                alpha, beta = x

                def q_i(p):
                    return max((p-alpha)/beta, 0)

                def supply(p):
                    total = 0
                    for r in strategic_firms:
                        a, b = params[r]
                        if r == f:
                            total += q_i(p)
                        else:
                            total += max((p-a)/b, 0)
                    return total

                def excess(p):
                    return supply(p) - demand_Q(p)

                try:
                    p_eq = brentq(excess, -500, 5000)
                except:
                    return 1e6

                q = q_i(p_eq)
                mc = mc_from_quadratic(mc_params[f], q)

                return -(p_eq*q - mc*q)

            res = minimize(profit, [0.0, 0.01], bounds=[(-500,5000),(1e-4,100)])
            if res.success:
                params[f] = tuple(res.x)

    return params


# In[117]:


def compute_conduct(offers, prices, load, strategic_firms):

    results = []
    price_series = prices.set_index("ts")["price"]
    common_ts = sorted(price_series.index.intersection(load.index))
    print(f"Overlapping hours: {len(common_ts)}")

    P_series = price_series.reindex(common_ts)
    Q_series = load.reindex(common_ts)

    for ts in common_ts:

        offers_hour = offers[offers["ts"] == ts]
        if offers_hour.empty:
            continue

        p_eq = float(price_series.loc[ts])
        Q_eq = float(load.loc[ts])

        if Q_eq <= 0 or p_eq <= 0:
            continue

        # 1. Demand fit using zonal load directly
        try:
            A, B = fit_quadratic_demand(ts, P_series, Q_series, window=12)
        except ValueError:
            continue

        dQdP = -B
        if abs(dQdP) < 1e-8:
            continue

        # 2. MC from offer curves
        mc_params   = fit_quadratic_mc_for_hour(offers_hour)
        total_cap   = 0.0
        weighted_mc = 0.0

        for f in strategic_firms:
            if f not in mc_params:
                continue
            firm_offers = offers_hour[offers_hour["firm"] == f]
            dispatched  = firm_offers[firm_offers["price"] <= p_eq]["mw"].sum()
            if dispatched <= 0:
                continue
            mc_f = mc_from_quadratic(mc_params[f], dispatched)
            weighted_mc += mc_f * dispatched
            total_cap   += dispatched

        if total_cap <= 0:
            continue

        mc_avg = weighted_mc / total_cap   # <-- this line is missing in your code

        # 3. Back out theta
        theta = (p_eq - mc_avg) * dQdP / (-Q_eq)
        theta = float(np.clip(theta, 0, 1))

        results.append({
            "ts":     ts,
            "Q_eq":   Q_eq,
            "p_eq":   p_eq,
            "mc_avg": mc_avg,
            "theta":  theta
        })

    return pd.DataFrame(results)


# conduct function
def build_conduct_function(df):

    if df.empty:
        print("Conduct DF is empty.")
        return df

    df = df.copy()
    df["q_norm"] = df["Q_eq"] / df["Q_eq"].max()

    bins = np.linspace(0,1,15)
    df["bin"] = pd.cut(df["q_norm"], bins)

    g = df.groupby("bin")["theta"]
    out = g.agg(["mean","std","count"]).reset_index()

    out["se"] = out["std"] / np.sqrt(out["count"])
    out["q_center"] = out["bin"].apply(lambda x: (x.left+x.right)/2)

    return out


# In[118]:


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


# In[119]:


df_test = pd.read_csv("hrl_load_prelim.csv")
print(df_test.columns.tolist())


# In[120]:


# Define strategic firms on full offers data
strategic_firms = (
    offers.groupby("firm")["mw"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .index
    .tolist()
)
print("Strategic firms:", strategic_firms)

# Trim to overlapping window
price_series = prices.set_index("ts")["price"]
common_ts    = sorted(price_series.index.intersection(load.index))
load_trimmed    = load.reindex(common_ts).dropna()
offers_trimmed  = offers[offers["ts"].isin(common_ts)]

conduct_df = compute_conduct(offers_trimmed, prices, load_trimmed, strategic_firms)
phi_df     = build_conduct_function(conduct_df)
plot_conduct(phi_df)


# In[116]:


conduct_df = compute_conduct(offers_trimmed, prices, load_trimmed, strategic_firms)
print(conduct_df[["ts", "p_eq", "mc_avg", "theta"]].to_string())


# In[104]:


print(offers.columns.tolist())
print(offers.head(3))


# In[121]:


def compute_pjm_nash(offers, prices, load, strategic_firms):
    
    results = []
    price_series = prices.set_index("ts")["price"]
    common_ts = sorted(price_series.index.intersection(load.index))
    
    P_series = price_series.reindex(common_ts)
    Q_series = load.reindex(common_ts)
    
    for ts in common_ts:
        
        offers_hour = offers[offers["ts"] == ts]
        if offers_hour.empty:
            continue
        
        p_eq = float(price_series.loc[ts])
        Q_eq = float(load.loc[ts])
        
        if Q_eq <= 0 or p_eq <= 0:
            continue
        
        # Demand parameters for this hour
        try:
            A, B = fit_quadratic_demand(ts, P_series, Q_series, window=12)
        except ValueError:
            continue
        
        demand_params = (A, B)
        
        # Solve Nash equilibrium for this hour
        ne_params = compute_ne_hour(offers_hour, demand_params, strategic_firms)
        
        # For each strategic firm, compute Nash-implied price
        # using their equilibrium (alpha, beta) and clearing condition
        def demand_Q(p):
            return max(A - B*p, 0)
        
        def total_supply(p):
            total = 0
            for f in strategic_firms:
                alpha, beta = ne_params[f]
                total += max((p - alpha) / beta, 0)
            return total
        
        def excess(p):
            return total_supply(p) - demand_Q(p)
        
        try:
            p_nash = brentq(excess, -500, 3000)
            q_nash = demand_Q(p_nash)
        except Exception:
            continue
        
        results.append({
            "ts":       ts,
            "Q_eq":     Q_eq,
            "p_eq":     p_eq,       # observed DA price
            "p_nash":   p_nash,     # model-implied Nash price
            "q_nash":   q_nash,
            "ne_params": ne_params
        })
    
    return pd.DataFrame(results)


# In[126]:


def plot_pjm_offer_stack(offers_hour, p_obs, p_nash, ts, xlim=10000):
    
    stack = offers_hour.sort_values("price").copy()
    stack["cum_mw"] = stack["mw"].cumsum()
    
    # Filter to PSEG-relevant price range only
    stack = stack[stack["price"] <= 150]
    
    plt.figure(figsize=(8, 5))
    plt.step(stack["cum_mw"], stack["price"], where="post", 
             label="Observed offer curve", color="steelblue", linewidth=1.5)
    plt.axhline(p_obs,  color="blue",   linestyle="--", linewidth=1.5,
                label=f"Observed price = {p_obs:.2f}")
    plt.axhline(p_nash, color="orange", linestyle="--", linewidth=1.5,
                label=f"Nash p_eq = {p_nash:.2f}")
    
    plt.xlim(0, xlim)
    plt.ylim(-10, 150)   # tighter y range to see the curve shape
    plt.xlabel("Cumulative MW")
    plt.ylabel("Offer Price ($/MWh)")
    plt.title(f"PJM PSEG Offer Stack at {ts}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot a few representative hours
nash_df = compute_pjm_nash(offers_trimmed, prices, load_trimmed, strategic_firms)

for ts in nash_df["ts"].head(4):
    oh = offers_trimmed[offers_trimmed["ts"] == ts]
    row = nash_df[nash_df["ts"] == ts].iloc[0]
    plot_pjm_offer_stack(oh, row["p_eq"], row["p_nash"], ts)


# In[123]:


def plot_pjm_cleared_quantity(nash_df, offers, strategic_firms):
    
    # Inferred cleared Q from offer stacks at observed price
    inferred_q = []
    for _, row in nash_df.iterrows():
        oh = offers[offers["ts"] == row["ts"]]
        q_inferred = oh[oh["price"] <= row["p_eq"]]["mw"].sum()
        inferred_q.append(q_inferred)
    
    nash_df = nash_df.copy()
    nash_df["q_inferred"] = inferred_q
    
    plt.figure(figsize=(10, 4))
    plt.plot(nash_df["ts"], nash_df["q_inferred"], label="Inferred cleared Q (offers @ observed price)")
    plt.plot(nash_df["ts"], nash_df["q_nash"],     label="Model Nash Q (strategic firms only)")
    plt.xlabel("Hour")
    plt.ylabel("MW")
    plt.title("Cleared Quantity: Inferred vs Model — PJM PSEG")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_pjm_cleared_quantity(nash_df, offers_trimmed, strategic_firms)


# In[129]:


nash_df = compute_pjm_nash(offers_trimmed, prices, load_trimmed, strategic_firms)
print(nash_df[["ts", "p_eq", "p_nash"]].to_string())

# 2. Plot cleared quantity comparison
plot_pjm_cleared_quantity(nash_df, offers_trimmed, strategic_firms)

# 3. Plot offer stacks for representative hours
for ts in nash_df["ts"].head(24):
    oh = offers_trimmed[offers_trimmed["ts"] == ts]
    row = nash_df[nash_df["ts"] == ts].iloc[0]
    plot_pjm_offer_stack(oh, row["p_eq"], row["p_nash"], ts)


# In[128]:


# Also print a sanity check
print(f"Offers in 0-150 price range: {len(stack)} rows")
print(f"MW range: {stack['cum_mw'].min():.0f} – {stack['cum_mw'].max():.0f}")
print(f"Price range: {stack['price'].min():.2f} – {stack['price'].max():.2f}")


# In[ ]:




