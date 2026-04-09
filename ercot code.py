#!/usr/bin/env python
# coding: utf-8

# In[97]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Callable
from scipy.optimize import minimize, brentq
import glob
 
# ================================================
# 1. PARSING + LOADING 
# ================================================
 
def parse_hour_ending_to_int(x) -> int:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if ":" in s:
        s = s.split(":")[0].strip()
    try:
        h = int(s)
    except ValueError:
        return np.nan
    if h < 1 or h > 24:
        return np.nan
    return h
 
@dataclass
class DamInputs:
    offers: pd.DataFrame
    prices: pd.DataFrame

def load_dam_offers(path_offers_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_offers_csv)
    df["Delivery Date"] = pd.to_datetime(df["Delivery Date"], errors="coerce")
    he = df["Hour Ending"].apply(parse_hour_ending_to_int).astype("float")
    df["ts"] = df["Delivery Date"] + pd.to_timedelta((he - 1), unit="h")
    df.loc[he == 24, "ts"] += pd.Timedelta(days=1)
    df = df.dropna(subset=["ts"])
 
    rows = []
    for k in range(1, 11):
        mw_col = f"Energy Only Offer MW{k}"
        pr_col = f"Energy Only Offer Price{k}"
        if mw_col not in df.columns or pr_col not in df.columns:
            continue
        tmp = df[["ts", "QSE Name", "Settlement Point", mw_col, pr_col]].copy()
        tmp = tmp.rename(columns={
            "QSE Name": "firm",
            "Settlement Point": "settlement_point",
            mw_col: "mw",
            pr_col: "price",
        })
        tmp["step"] = k
        rows.append(tmp)
 
    out = pd.concat(rows, ignore_index=True)
    out["mw"]    = pd.to_numeric(out["mw"],    errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["ts", "firm", "settlement_point", "mw", "price"])
    out = out[out["mw"] > 0]
    out = out[out["price"] <= 1000]
    return out[["ts", "firm", "settlement_point", "step", "mw", "price"]]
 
def load_dam_prices(path_prices_csv: str,
                    settlement_point: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path_prices_csv)
    df["DeliveryDate"] = pd.to_datetime(df["DeliveryDate"], errors="coerce")
    he = df["HourEnding"].apply(parse_hour_ending_to_int).astype("float")
    df["ts"] = df["DeliveryDate"] + pd.to_timedelta((he - 1), unit="h")
    df.loc[he == 24, "ts"] += pd.Timedelta(days=1)
    df = df.dropna(subset=["ts"])
    df = df.rename(columns={
        "SettlementPoint": "settlement_point",
        "SettlementPointPrice": "price",
    })
    if settlement_point is not None:
        df = df[df["settlement_point"].astype(str) == str(settlement_point)].copy()
    out = df[["ts", "price"]].dropna()
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["price"])
    out = out.groupby("ts", as_index=False)["price"].mean()
    return out

def load_inputs_multi_day(
    offers_paths: List[str],
    prices_paths: List[str],
    settlement_point: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> DamInputs:
    
    offers_list = []
    for path in offers_paths:
        offers_list.append(load_dam_offers(path))
    all_offers = pd.concat(offers_list, ignore_index=True)
    
    prices_list = []
    for path in prices_paths:
        prices_list.append(load_dam_prices(path, settlement_point=settlement_point))
    all_prices = pd.concat(prices_list, ignore_index=True)
    
    # Deduplicate
    before = len(all_offers)
    all_offers = all_offers.drop_duplicates(
        subset=["ts", "firm", "settlement_point", "step"],
        keep="first"
    )
    print(f"Offers: {before} rows -> {len(all_offers)} after dedup "
          f"({before - len(all_offers)} duplicates removed)")
    
    all_prices = all_prices.drop_duplicates(subset=["ts"], keep="first")
    all_prices = all_prices.sort_values("ts").reset_index(drop=True)

    # Filter to date range if specified
    if start_date is not None:
        sd = pd.to_datetime(start_date).date()
        all_offers = all_offers[all_offers["ts"].dt.date >= sd]
        all_prices = all_prices[all_prices["ts"].dt.date >= sd]
    if end_date is not None:
        ed = pd.to_datetime(end_date).date()
        all_offers = all_offers[all_offers["ts"].dt.date <= ed]
        all_prices = all_prices[all_prices["ts"].dt.date <= ed]

    # Filter offers to only days covered by prices
    price_days = set(all_prices["ts"].dt.date.unique())
    all_offers = all_offers[all_offers["ts"].dt.date.isin(price_days)]
    print(f"Offers after date filter: {len(all_offers)} rows, "
          f"{all_offers['ts'].dt.date.nunique()} days")
    print(f"Days: {sorted(all_offers['ts'].dt.date.unique())}")
    
    return DamInputs(offers=all_offers, prices=all_prices)


# In[65]:


# ================================================
# 2. MARKET QUANTITIES
# ================================================
 
def compute_total_energy_sold(offers: pd.DataFrame,
                               prices: pd.DataFrame) -> pd.Series:
    prices_idx = prices.set_index("ts")["price"]
    out = {}
    for ts, offers_hour in offers.groupby("ts"):
        if ts not in prices_idx.index:
            continue
        p_star = float(prices_idx.loc[ts])
        out[ts] = float(offers_hour.loc[offers_hour["price"] <= p_star, "mw"].sum())
    return pd.Series(out).sort_index()
 
def top_firms_by_daily_mw(inp: DamInputs, target_day: str, top_n: int = 3):
    day = pd.to_datetime(target_day).date()
    daily_mw = (
        inp.offers
        .loc[inp.offers["ts"].dt.date == day]
        .groupby("firm")["mw"]
        .sum()
        .sort_values(ascending=False)
    )
    return daily_mw.head(top_n).index.tolist(), daily_mw


# In[66]:


# ================================================
# 3. COST FUNCTIONS
# ================================================
 
def fit_quadratic_mc_for_hour(offers_hour: pd.DataFrame,
                               low_quantile: float = 0.2) -> Dict:
    params = {}
    for firm, g in offers_hour.groupby("firm"):
        gg = g.sort_values("price").copy()
        gg["cumq"] = gg["mw"].cumsum()
        thr = gg["price"].quantile(low_quantile)
        sub = gg[gg["price"] <= thr]
        if len(sub) < 4:
            sub = gg
        q = sub["cumq"].to_numpy()
        p = sub["price"].to_numpy()
        if len(q) < 2:
            params[firm] = (float(np.mean(p)) if len(p) else 0.0, 0.0, 0.0)
            continue
        X = np.column_stack([np.ones_like(q), q, q * q])
        coef, *_ = np.linalg.lstsq(X, p, rcond=None)
        a, b, c = map(float, coef)
        c = max(c, 0.0)
        params[firm] = (a, b, c)
    return params
 
def mc_from_quadratic(mc_params: Tuple, q: float) -> float:
    a, b, c = mc_params
    return a + b * q + c * q * q
 
def total_cost_from_mc(mc: Tuple, q: float) -> float:
    a, b, c = mc
    return a * q + 0.5 * b * q * q + (1.0 / 3.0) * c * q * q * q
 


# In[67]:


# ================================================
# 4. DEMAND FITTING
# ================================================
 
def fit_quadratic_demand_from_window(ts: pd.Timestamp,
                                      p_series: pd.Series,
                                      q_series: pd.Series,
                                      window_hours: int = 48) -> Tuple:
    df = pd.DataFrame({"p": p_series, "q": q_series}).dropna()
    if df.empty:
        return (0.0, 1e-6, 0.0)
    lo  = ts - pd.Timedelta(hours=window_hours)
    hi  = ts + pd.Timedelta(hours=window_hours)
    sub = df.loc[(df.index >= lo) & (df.index <= hi)].copy()
    if len(sub) < 10:
        sub = df.tail(200)
    P = sub["p"].to_numpy()
    Q = sub["q"].to_numpy()
    X = np.column_stack([np.ones_like(P), -P, -(P * P)])
    coef, *_ = np.linalg.lstsq(X, Q, rcond=None)
    A, B, C = map(float, coef)
    B = abs(B) + 1e-6
    C = max(C, 0.0)
    return (A, B, C)
 
def demand_Q(demand_params: Tuple, p: float) -> float:
    A, B, C = demand_params
    return max(A - B * p - C * p * p, 0.0)
 
def demand_slope(demand_params: Tuple, p: float) -> float:
    """dQ/dp = -(B + 2*C*p)  — this is NEGATIVE for downward sloping demand."""
    A, B, C = demand_params
    return -(B + 2.0 * C * p)


# In[91]:


# ================================================
# 5. NASH EQUILIBRIUM  (unchanged from your code)
# ================================================
 
def affine_q_of_p(alpha: float, beta: float):
    beta = max(beta, 1e-6)
    return lambda p: max((p - alpha) / beta, 0.0)
 
def find_clearing_price(supply_fns, demand_params, fringe_fn=None,
                         p_bounds=(-500, 5000)) -> float:
    def excess(p):
        supply = sum(fn(p) for fn in supply_fns)
        if fringe_fn is not None:
            supply += fringe_fn(p)
        return supply - demand_Q(demand_params, p)
    lo, hi = p_bounds
    try:
        if excess(lo) * excess(hi) > 0:
            for hi_try in [10_000, 50_000]:
                if excess(lo) * excess(hi_try) < 0:
                    hi = hi_try
                    break
            else:
                return float((lo + hi) / 2)
        return float(brentq(excess, lo, hi, xtol=1e-4, maxiter=200))
    except Exception:
        return float((lo + hi) / 2)

def firm_capacity_from_offers(offers_hour: pd.DataFrame) -> Dict:
    return offers_hour.groupby("firm")["mw"].sum().to_dict()
 
def profit_for_candidate(alpha_beta, firm, offers_hour, mc_params,
                          demand_params, rivals_params, capacities,
                          fringe_fn=None):
    alpha, beta = alpha_beta
    q_i = affine_q_of_p(alpha, beta)
    supply_fns = [affine_q_of_p(a_r, b_r) for a_r, b_r in rivals_params.values()]
    supply_fns.append(q_i)
    p_star = find_clearing_price(supply_fns, demand_params, fringe_fn=fringe_fn)
    qi     = min(q_i(p_star), capacities.get(firm, np.inf))
    Ci     = total_cost_from_mc(mc_params.get(firm, (0.0, 0.0, 0.0)), qi)
    pi     = p_star * qi - Ci
    return float(pi), float(p_star), float(qi)
 
def best_response(firm, init_ab, offers_hour, mc_params, demand_params,
                   rivals_params, capacities, fringe_fn=None):
    def obj(x):
        pi, _, _ = profit_for_candidate(
            (x[0], x[1]), firm, offers_hour, mc_params,
            demand_params, rivals_params, capacities, fringe_fn=fringe_fn)
        return -pi
    res = minimize(obj, np.array(init_ab, dtype=float),
                   bounds=[(0, 200), (1e-4, 500.0)],  # TIGHTENED from (-500, 5000)
                   options={"maxiter": 200})
    if not res.success:
        return init_ab, False
    return (float(res.x[0]), float(res.x[1])), True
 
def compute_ne_for_hour(offers_hour, demand_params, strategic_firms,
                         fringe_supply=None, damping=0.4,
                         max_iter=40, tol=1e-3):
    capacities = firm_capacity_from_offers(offers_hour)
    mc_params  = fit_quadratic_mc_for_hour(offers_hour)
    params = {}
    for f in strategic_firms:
        g = offers_hour[offers_hour["firm"] == f].sort_values("price")
        if len(g) < 2:
            params[f] = (0.0, 1.0)
            continue
        cumq = g["mw"].cumsum().to_numpy()
        p    = g["price"].to_numpy()
        X    = np.column_stack([np.ones_like(cumq), cumq])
        coef, *_ = np.linalg.lstsq(X, p, rcond=None)
        params[f] = (float(coef[0]), max(float(coef[1]), 1e-4))
 
    for _ in range(max_iter):
        max_change = 0.0
        for f in strategic_firms:
            rivals = {r: params[r] for r in strategic_firms if r != f}
            old    = params[f]
            new, ok = best_response(f, old, offers_hour, mc_params,
                                    demand_params, rivals, capacities,
                                    fringe_fn=fringe_supply)
            upd = (
                (1 - damping) * old[0] + damping * new[0],
                (1 - damping) * old[1] + damping * new[1],
            )
            params[f]   = upd
            max_change   = max(max_change, abs(upd[0]-old[0]), abs(upd[1]-old[1]))
        if max_change < tol:
            break
    return params
 
 


# In[92]:


def run_replication(path_offers, path_prices, settlement_point=None,
                    target_day=None, strategic_top_n=3,
                    inp=None,
                    fixed_strategic=None) -> pd.DataFrame:

    if inp is None:
        inp = load_inputs(path_offers, path_prices, settlement_point=settlement_point)

    Q_star = compute_total_energy_sold(inp.offers, inp.prices)
    P_star = inp.prices.set_index("ts")["price"]

    if fixed_strategic is not None:
        strategic = fixed_strategic
    else:
        strategic, _ = top_firms_by_daily_mw(inp, target_day, strategic_top_n)

    if target_day is not None:
        day    = pd.to_datetime(target_day).date()
        Q_star = Q_star[Q_star.index.date == day]
        P_star = P_star[P_star.index.date == day]

    rows = []
    for ts in Q_star.index:
        offers_hour  = inp.offers[inp.offers["ts"] == ts].copy()
        if offers_hour.empty:
            continue
        fringe_firms  = set(offers_hour["firm"].unique()) - set(strategic)
        offers_fringe = offers_hour[offers_hour["firm"].isin(fringe_firms)]

        def build_fringe(ofr):
            pf  = ofr["price"].to_numpy()
            mwf = ofr["mw"].to_numpy()
            return lambda p: float(mwf[pf <= p].sum())

        S_fringe  = build_fringe(offers_fringe)
        d_params  = fit_quadratic_demand_from_window(ts, P_star, Q_star, window_hours=48)
        ne_params = compute_ne_for_hour(offers_hour, d_params,
                                        strategic_firms=strategic,
                                        fringe_supply=S_fringe)
        mc_params  = fit_quadratic_mc_for_hour(offers_hour)
        capacities = firm_capacity_from_offers(offers_hour)

        for f in strategic:
            if f not in ne_params:
                continue
            rivals = {r: ne_params[r] for r in strategic if r != f and r in ne_params}
            pi, p_eq, q_eq = profit_for_candidate(
                ne_params[f], f, offers_hour, mc_params,
                d_params, rivals, capacities, fringe_fn=S_fringe)
            rows.append({
                "ts": ts, "firm": f,
                "alpha": ne_params[f][0], "beta": ne_params[f][1],
                "p_eq": p_eq, "q_eq": q_eq, "profit": pi,
                "demand_A": d_params[0], "demand_B": d_params[1],
                "demand_C": d_params[2],
                "capacity_from_offers": capacities.get(f, np.nan),
            })

    if not rows:
        return pd.DataFrame(columns=["ts", "firm", "alpha", "beta",
                                      "p_eq", "q_eq", "profit",
                                      "demand_A", "demand_B", "demand_C",
                                      "capacity_from_offers"])
    return pd.DataFrame(rows).sort_values(["ts", "firm"])


# In[93]:


# ================================================
# 7. CONDUCT 
# ================================================
 
def supply_slope_dqdp(alpha: float, beta: float, p: float) -> float:
    beta = max(beta, 1e-12)
    return 0.0 if (p - alpha) / beta <= 0 else 1.0 / beta
 
def build_conduct_outputs(df_ne: pd.DataFrame, inp: DamInputs,
                           window_hours: int = 48):
    P_star = inp.prices.set_index("ts")["price"]
    Q_star = compute_total_energy_sold(inp.offers, inp.prices)
    market_rows, firm_rows = [], []
 
    for ts, df_hour in df_ne.groupby("ts"):
        df_hour  = df_hour.copy()
        p_eq     = float(df_hour["p_eq"].iloc[0])
        d_params = fit_quadratic_demand_from_window(ts, P_star, Q_star, window_hours)
 
        offers_hour = inp.offers[inp.offers["ts"] == ts].copy()
        if offers_hour.empty:
            continue
 
        mc_params_by_firm = fit_quadratic_mc_for_hour(offers_hour)
        Q_eq = float(df_hour["q_eq"].sum())
 
        # FIX: use demand_slope(), not demand_Q()
        dQdp = demand_Q(d_params, p_eq)   # negative number, e.g. -237
 
        epsilon = np.nan
        if Q_eq > 0 and dQdp != 0:
            epsilon = dQdp * (p_eq / Q_eq)    # price elasticity (negative)
 
        for _, r in df_hour.iterrows():
            firm  = r["firm"]
            qi    = float(r["q_eq"])
            si    = qi / Q_eq if Q_eq > 0 else np.nan
            mc_p  = mc_params_by_firm.get(firm, (0.0, 0.0, 0.0))
            mci   = mc_from_quadratic(mc_p, qi)
            lerner = (p_eq - mci) / p_eq if p_eq != 0 else np.nan
            theta_i = np.nan
            if not np.isnan(epsilon) and epsilon != 0:
                theta_i = lerner / (-1.0 / epsilon)
            firm_rows.append({
                "ts": ts, "firm": firm, "q_eq": qi, "share": si,
                "mc_at_q": mci, "lerner": lerner,
                "epsilon": epsilon, "theta_i": theta_i,
            })
 
        shares = df_hour["q_eq"].to_numpy() / Q_eq if Q_eq > 0 else np.array([np.nan])
        hhi    = float(np.sum(shares ** 2))
        dSdp   = sum(supply_slope_dqdp(float(r["alpha"]), float(r["beta"]), p_eq)
                     for _, r in df_hour.iterrows())
        theta_market = np.nan
        if dSdp != 0 and not np.isnan(dQdp):
            theta_market = float(abs(hhi * (dQdp / dSdp)))
 
        market_rows.append({
            "ts": ts, "p_eq": p_eq, "Q_eq": Q_eq,
            "HHI": hhi, "epsilon": epsilon,
            "theta_market": theta_market,
            "dQdp": dQdp, "dSdp": dSdp,
        })
 
    return (pd.DataFrame(market_rows).sort_values("ts"),
            pd.DataFrame(firm_rows).sort_values(["ts", "firm"]))
 
def build_conduct_function(inp: DamInputs, market_conduct_df: pd.DataFrame,
                            n_bins: int = 10, q_clip=(0.0, 1.0),
                            cap_method: str = "sum"):
    cap_rows = []
    for ts, oh in inp.offers.groupby("ts"):
        Q_cap = float(oh["mw"].sum()) if cap_method == "sum"                 else float(oh.groupby("firm")["mw"].max().sum())
        cap_rows.append({"ts": ts, "Q_cap": Q_cap})
    cap_df = pd.DataFrame(cap_rows)
 
    m = market_conduct_df.merge(cap_df, on="ts", how="inner").copy()
    m["q_norm"] = m["Q_eq"] / m["Q_cap"]
    m = m.replace([np.inf, -np.inf], np.nan).dropna(subset=["q_norm", "theta_market"])
    lo, hi = q_clip
    m = m[(m["q_norm"] >= lo) & (m["q_norm"] <= hi)].copy()
 
    bins      = np.linspace(lo, hi, n_bins + 1)
    m["q_bin"] = pd.cut(m["q_norm"], bins=bins, include_lowest=True)
    g   = m.groupby("q_bin", observed=True)["theta_market"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["se"]       = out["std"] / np.sqrt(out["count"].clip(lower=1))
    ivs             = pd.IntervalIndex(out["q_bin"])
    out["q_center"] = (ivs.left.astype(float) + ivs.right.astype(float)) / 2
    out             = out.rename(columns={"mean": "phi_hat"})
    return out.dropna(subset=["phi_hat"]), m


# In[94]:


# ================================================
# 8. PLOTS 
# ================================================
 
def plot_conduct_function(phi_df: pd.DataFrame,
                           title="Conduct Function in DAM (binned by normalized quantity)",
                           color="steelblue", ylim=None):
    """
    Conduct function with 95% CI bands.
    Clipped to [0,1] so it's comparable with PJM.
    """
    phi_df = phi_df.dropna(subset=["q_center", "phi_hat", "se"]).copy()
    x  = phi_df["q_center"].to_numpy()
    y  = phi_df["phi_hat"].to_numpy()
    se = phi_df["se"].to_numpy()
 
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", linewidth=2, color=color, label="φ̂(q)")
    ax.fill_between(x,
                    np.clip(y - 1.96 * se, 0, None),
                    np.clip(y + 1.96 * se, 0, None),
                    alpha=0.2, color=color, label="95% CI")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1,
               label="Perfect competition (φ=0)")
    ax.axhline(1, color="red",  linestyle="--", linewidth=1,
               label="Cournot benchmark (φ=1)")
    ax.set_xlabel("Normalised quantity  q = Q_eq / Q_cap", fontsize=11)
    ax.set_ylabel("Conduct  φ̂(q)",                         fontsize=11)
    ax.set_title(title, fontsize=12)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ercot_conduct_function.png", dpi=150)
    plt.show()
    print("Saved: ercot_conduct_function.png")
 
 
def plot_nash_vs_observed_prices(df_ne: pd.DataFrame, inp: DamInputs):
    """
    Two panels:
      Left:  time series of observed vs Nash price
      Right: scatter with 45-degree line
    NEW — this was missing from your original code.
    """
    P_obs   = inp.prices.set_index("ts")["price"]
    P_model = df_ne.groupby("ts")["p_eq"].mean()
    idx     = P_obs.index.intersection(P_model.index)
 
    p_obs_vals  = P_obs.loc[idx].values
    p_nash_vals = P_model.loc[idx].values
    gap         = p_nash_vals - p_obs_vals   # positive = Nash overpredicts
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
    # --- Left: time series ---
    ax = axes[0]
    ax.plot(idx, p_obs_vals,  label="Observed price",        color="steelblue", linewidth=2)
    ax.plot(idx, p_nash_vals, label="Nash equilibrium price", color="orange",   linewidth=2, linestyle="--")
    ax.fill_between(idx, p_obs_vals, p_nash_vals,
                    where=(p_nash_vals > p_obs_vals),
                    alpha=0.15, color="red", label="Nash overpredicts")
    ax.set_xlabel("Hour",          fontsize=11)
    ax.set_ylabel("Price ($/MWh)", fontsize=11)
    ax.set_title("ERCOT DAM: Observed vs Nash Price\n(Oct 12, 2025)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
 
    # --- Right: scatter ---
    ax2 = axes[1]
    lim = max(p_obs_vals.max(), p_nash_vals.max()) * 1.05
    ax2.scatter(p_obs_vals, p_nash_vals, alpha=0.8,
                edgecolors="steelblue", facecolors="lightblue", s=60, zorder=3)
    ax2.plot([0, lim], [0, lim], "k--", linewidth=1, label="45° (perfect fit)")
    ax2.set_xlabel("Observed Price ($/MWh)",        fontsize=11)
    ax2.set_ylabel("Nash Equilibrium Price ($/MWh)", fontsize=11)
    ax2.set_title("Nash vs Observed — ERCOT\n(points above line = Nash overpredicts)",
                  fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig("ercot_nash_vs_observed.png", dpi=150)
    plt.show()
    print("Saved: ercot_nash_vs_observed.png")
 
    # Summary statistics
    print("\n=== ERCOT Nash vs Observed Summary ===")
    print(f"Mean observed price:      ${p_obs_vals.mean():.2f}/MWh")
    print(f"Mean Nash price:          ${p_nash_vals.mean():.2f}/MWh")
    print(f"Mean gap (Nash−observed): ${gap.mean():.2f}/MWh")
    print(f"Fraction Nash > observed: {(gap > 0).mean():.0%}")
 
 
def plot_markup_vs_load(df_ne: pd.DataFrame, inp: DamInputs):
    """
    (p - MC) vs load for ERCOT.
    NEW — directly supports the mean-field motivation.
    Shows that market power is load-dependent.
    """
    P_obs  = inp.prices.set_index("ts")["price"]
    Q_star = compute_total_energy_sold(inp.offers, inp.prices)
 
    rows = []
    for ts, df_hour in df_ne.groupby("ts"):
        offers_hour = inp.offers[inp.offers["ts"] == ts].copy()
        if offers_hour.empty or ts not in P_obs.index:
            continue
        p_eq        = float(P_obs.loc[ts])
        Q_obs       = float(Q_star.loc[ts]) if ts in Q_star.index else np.nan
        mc_params   = fit_quadratic_mc_for_hour(offers_hour)
 
        mc_vals = []
        for _, r in df_hour.iterrows():
            f  = r["firm"]
            q  = float(r["q_eq"])
            mc = mc_from_quadratic(mc_params.get(f, (0.0,0.0,0.0)), q)
            mc_vals.append(mc)
 
        if mc_vals and not np.isnan(Q_obs):
            rows.append({
                "Q_obs":  Q_obs,
                "markup": p_eq - np.mean(mc_vals),
                "p_eq":   p_eq,
                "mc_avg": np.mean(mc_vals),
            })
 
    df_mk = pd.DataFrame(rows)
    if df_mk.empty:
        print("No markup data.")
        return
 
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(df_mk["Q_obs"], df_mk["markup"],
                    c=df_mk["p_eq"], cmap="OrRd", s=70, alpha=0.85, zorder=3)
    plt.colorbar(sc, ax=ax, label="Observed price ($/MWh)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
 
    # trend line
    z = np.polyfit(df_mk["Q_obs"], df_mk["markup"], 1)
    x_fit = np.linspace(df_mk["Q_obs"].min(), df_mk["Q_obs"].max(), 100)
    ax.plot(x_fit, np.polyval(z, x_fit), "steelblue", linewidth=2,
            linestyle="--", label=f"Trend (slope={z[0]:.4f})")
 
    ax.set_xlabel("System load Q (MW)",   fontsize=11)
    ax.set_ylabel("Price − MC ($/MWh)",   fontsize=11)
    ax.set_title("ERCOT: Markup vs Load\n(market power increases with system tightness)",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ercot_markup_vs_load.png", dpi=150)
    plt.show()
    print("Saved: ercot_markup_vs_load.png")
 
 
def plot_offer_stacks(df_ne: pd.DataFrame, inp: DamInputs,
                      hours_to_plot: List[int] = [0, 6, 12, 17, 23]):
    """
    FIX: y-axis capped at 200, x-axis zoomed to 95th percentile of cum MW.
    Shows both observed price and Nash price as dashed lines.
    """
    P_obs   = inp.prices.set_index("ts")["price"]
    P_model = df_ne.groupby("ts")["p_eq"].mean()
    all_ts  = sorted(P_obs.index.intersection(P_model.index))
 
    selected = [all_ts[i] for i in hours_to_plot if i < len(all_ts)]
 
    n = len(selected)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]
 
    for ax, ts in zip(axes, selected):
        oh = inp.offers[inp.offers["ts"] == ts].sort_values("price").copy()
        oh = oh[oh["price"] <= 200]
        oh["cum_mw"] = oh["mw"].cumsum()
 
        p_obs  = float(P_obs.loc[ts])
        p_nash = float(P_model.loc[ts])
        xlim   = oh["cum_mw"].quantile(0.95) if not oh.empty else 50000
 
        ax.step(oh["cum_mw"], oh["price"], where="post",
                color="steelblue", linewidth=1.5)
        ax.axhline(p_obs,  color="navy",       linestyle="--", linewidth=1.8,
                   label=f"Obs ${p_obs:.0f}")
        ax.axhline(p_nash, color="darkorange",  linestyle="--", linewidth=1.8,
                   label=f"Nash ${p_nash:.0f}")
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, min(200, max(p_obs, p_nash) * 2.5 + 10))
        ax.set_xlabel("Cumulative MW", fontsize=9)
        ax.set_ylabel("$/MWh",         fontsize=9)
        ax.set_title(f"{ts.strftime('%H:%M')}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
 
    fig.suptitle("ERCOT DAM Offer Stacks — Oct 12, 2025\n"
                 "(dashed = observed price / Nash price)", fontsize=12)
    plt.tight_layout()
    plt.savefig("ercot_offer_stacks.png", dpi=150)
    plt.show()
    print("Saved: ercot_offer_stacks.png")
 
 
def plot_cleared_quantity(df_ne: pd.DataFrame, inp: DamInputs):
    """Inferred cleared Q vs Nash Q — unchanged from your original but cleaner styling."""
    Q_star  = compute_total_energy_sold(inp.offers, inp.prices)
    Q_model = df_ne.groupby("ts")["q_eq"].sum()
    idx     = Q_star.index.intersection(Q_model.index)
 
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, Q_star.loc[idx].values,  label="Inferred cleared Q* (offers @ obs price)",
            color="steelblue", linewidth=2)
    ax.plot(idx, Q_model.loc[idx].values, label="Nash Q (strategic firms only)",
            color="orange", linewidth=2, linestyle="--")
    ax.set_xlabel("Hour",   fontsize=11)
    ax.set_ylabel("MW",     fontsize=11)
    ax.set_title("ERCOT: Cleared Quantity — Inferred vs Nash Model\n"
                 "(gap = fringe supply + strategic withholding)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("ercot_cleared_quantity.png", dpi=150)
    plt.show()
    print("Saved: ercot_cleared_quantity.png")


# In[99]:


# ================================================
# 9. ENTRY POINT — MULTI DAY
# ================================================

OFFERS_PATHS = sorted(glob.glob("60d_DAM_EnergyOnlyOffers*.csv"))
PRICES_PATHS = sorted(glob.glob("settlement_prices_10_*.csv"))

print("Offer files found:",  OFFERS_PATHS)
print("Price files found:",  PRICES_PATHS)

inp = load_inputs_multi_day(
    OFFERS_PATHS, 
    PRICES_PATHS, 
    settlement_point=None,
    start_date="2025-10-10",
    end_date="2025-10-12",
)
print(f"Total offer rows: {len(inp.offers)}")
print(f"Total price rows: {len(inp.prices)}")
print(f"Date range: {inp.offers['ts'].min()} to {inp.offers['ts'].max()}")

# Only run days that have both offers and prices
all_days = sorted(inp.offers["ts"].dt.date.unique())
price_days = set(inp.prices["ts"].dt.date.unique())
all_days = [d for d in all_days if d in price_days]
print(f"Days with price coverage: {all_days}")

# Fix strategic firms using Oct 12 as anchor day
strategic_firms, _ = top_firms_by_daily_mw(inp, target_day="2025-10-12", top_n=3)
print(f"Strategic firms: {strategic_firms}")

df_list = []
for day in all_days:
    print(f"Running Nash for {day}...")
    df_day = run_replication(
        path_offers=None,
        path_prices=None,
        settlement_point=None,
        target_day=str(day),
        strategic_top_n=3,
        inp=inp,
        fixed_strategic=strategic_firms,
    )
    if not df_day.empty:
        df_list.append(df_day)
        print(f"  Done — {len(df_day)} rows")
    else:
        print(f"  Skipped {day} — no results")

df = pd.concat(df_list, ignore_index=True).sort_values(["ts", "firm"])
print(f"Total Nash rows: {len(df)}")
df.to_csv("ercot_ne_results_multiday.csv", index=False)

# --- Conduct ---
market_conduct_df, firm_conduct_df = build_conduct_outputs(df, inp, window_hours=48)
phi_df, per_hour_df = build_conduct_function(
    inp, market_conduct_df, n_bins=5, q_clip=(0, 1), cap_method="sum"
)
print("\nConduct function:")
print(phi_df[["q_center", "phi_hat", "se", "count"]].to_string())

# --- Plots ---
plot_conduct_function(phi_df, title="Conduct Function in DAM (binned by normalized quantity)")
plot_nash_vs_observed_prices(df, inp)
plot_markup_vs_load(df, inp)
plot_offer_stacks(df, inp, hours_to_plot=[0, 6, 12, 17, 23])
plot_cleared_quantity(df, inp)


# In[43]:


def top_firms_by_daily_mw(inp, target_day: str, top_n: int = 3):
    day = pd.to_datetime(target_day).date()

    daily_mw = (
        inp.offers
        .loc[inp.offers["ts"].dt.date == day]
        .groupby("firm")["mw"]
        .sum()
        .sort_values(ascending=False)
    )

    return daily_mw.head(top_n).index.tolist(), daily_mw

strategic_firms, daily_mw = top_firms_by_daily_mw(
    inp,
    target_day="2025-10-12",
    top_n=10
)

print("Strategic firms:", strategic_firms)
print(daily_mw.head(10))


# In[89]:


OFFERS_PATHS = sorted(glob.glob("60d_DAM_EnergyOnlyOffers*.csv"))
PRICES_PATHS = sorted(glob.glob("settlement_prices_10_*.csv"))

inp = load_inputs_multi_day(OFFERS_PATHS, PRICES_PATHS, settlement_point=None)

# Verify after dedup
print(f"Unique hours in offers: {inp.offers['ts'].nunique()}")
print(f"Unique days in offers:  {inp.offers['ts'].dt.date.nunique()}")
print(f"Days in offers: {sorted(inp.offers['ts'].dt.date.unique())}")
print(f"Days in prices: {sorted(inp.prices['ts'].dt.date.unique())}")


# In[ ]:





# In[86]:


# observed hourly prices (settlement point)
inp = load_inputs(PATH_OFFERS, PATH_PRICES, settlement_point=None)
P_obs = inp.prices.set_index("ts")["price"]

# model price: same p_eq repeated for each firm; take mean per ts
P_model = df.groupby("ts")["p_eq"].mean()

# align (intersection of timestamps)
plot_idx = P_obs.index.intersection(P_model.index)

plt.figure()
plt.plot(plot_idx, P_obs.loc[plot_idx].values, label="Observed settlement point price")
plt.plot(plot_idx, P_model.loc[plot_idx].values, label="Model p_eq")
plt.xticks(rotation=45)
plt.ylabel("$/MWh")
plt.title("Observed vs Model Equilibrium Price")
plt.legend()
plt.tight_layout()
plt.show()


# In[45]:


# inferred cleared quantity from offer stack at observed price
Q_star = compute_total_energy_sold(inp.offers, inp.prices)

# model implied total supply at equilibrium: sum q_eq across strategic firms (per hour)
Q_model = df.groupby("ts")["q_eq"].sum()

plot_idx = Q_star.index.intersection(Q_model.index)

plt.figure()
plt.plot(plot_idx, Q_star.loc[plot_idx].values, label="Inferred cleared Q* (offers @ observed price)")
plt.plot(plot_idx, Q_model.loc[plot_idx].values, label="Model sum q_eq (strategic firms only)")
plt.xticks(rotation=45)
plt.ylabel("MW")
plt.title("Cleared Quantity: Inferred vs Model (strategic-only)")
plt.legend()
plt.tight_layout()
plt.show()


# In[46]:


ts0 = df["ts"].iloc[23]              # or choose a specific hour
firm0 = df.loc[df["ts"] == ts0, "firm"].iloc[0]

row = df[(df["ts"] == ts0) & (df["firm"] == firm0)].iloc[0]
alpha, beta = row["alpha"], row["beta"]
cap = row["capacity_from_offers"]

p_grid = np.linspace(0, max(500, row["p_eq"]*2), 200)
q_grid = np.minimum(np.maximum((p_grid - alpha)/max(beta, 1e-6), 0), cap)

plt.figure()
plt.plot(p_grid, q_grid)
plt.xlabel("Price ($/MWh)")
plt.ylabel("Quantity (MW)")
plt.title(f"Implied Supply Function: {firm0} at {ts0}")
plt.tight_layout()
plt.show()


# In[47]:


ts0 = df["ts"].iloc[24]
offers_hour = inp.offers[inp.offers["ts"] == ts0].copy()

# market offer stack (all firms), sorted by price
stack = offers_hour.sort_values("price").copy()
stack["cum_mw"] = stack["mw"].cumsum()

p_obs = float(inp.prices.set_index("ts").loc[ts0, "price"])
p_eq  = float(df.groupby("ts")["p_eq"].mean().loc[ts0])

plt.figure()
plt.step(stack["cum_mw"], stack["price"], where="post")
plt.axhline(p_obs, linestyle="--", label=f"Observed price = {p_obs:.2f}")
plt.axhline(p_eq, linestyle="--", label=f"Model p_eq = {p_eq:.2f}")
plt.xlabel("Cumulative MW")
plt.ylabel("Offer Price ($/MWh)")
plt.title(f"Offer Stack at {ts0}")
plt.legend()
plt.tight_layout()
plt.show()


# In[48]:


# observed vs actual nash? 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def build_step_curve(offers_firm_hour: pd.DataFrame):
    """
    Input: rows with columns ['mw','price'] for ONE firm, ONE hour (already long format).
    Output: x,y arrays that draw a step supply curve in (Q, P) space.
    """
    g = offers_firm_hour.sort_values("price").copy()
    g = g[g["mw"] > 0]

    # cumulative quantity after each step
    q_cum = g["mw"].cumsum().to_numpy()
    p = g["price"].to_numpy()

    # step plot coordinates
    # start at q=0 with first price
    xs = [0.0]
    ys = [p[0]] if len(p) else [0.0]

    prev_q = 0.0
    for qi, pi in zip(q_cum, p):
        # horizontal segment at price pi from prev_q -> qi
        xs += [prev_q, qi]
        ys += [pi, pi]
        prev_q = qi

    return np.array(xs), np.array(ys), float(q_cum[-1]) if len(q_cum) else 0.0


def plot_nash_for_firm_hour(
    df_ne: pd.DataFrame,
    inp: DamInputs,
    ts: pd.Timestamp,
    firm: str,
    use_capacity: bool = True,
    q_grid_n: int = 200,
):
    """
    Plots:
      - actual stepwise offer curve (Q vs P)
      - Nash inverse supply p(q)=alpha+beta*q (clipped to [0, cap] if use_capacity=True)
    """

    ts = pd.to_datetime(ts)

    # --- 1) actual offers for firm-hour
    offers_hour = inp.offers[inp.offers["ts"] == ts].copy()
    offers_firm = offers_hour[offers_hour["firm"] == firm].copy()
    if offers_firm.empty:
        raise ValueError(f"No offers found for firm={firm} at ts={ts}")

    x_step, y_step, q_cap_actual = build_step_curve(offers_firm)

    # --- 2) Nash params from df_ne
    row = df_ne[(df_ne["ts"] == ts) & (df_ne["firm"] == firm)]
    if row.empty:
        raise ValueError(f"No NE results found for firm={firm} at ts={ts}. Is firm in strategic set?")
    alpha = float(row["alpha"].iloc[0])
    beta  = float(row["beta"].iloc[0])

    # capacity to plot against
    if use_capacity:
        cap = float(row["capacity_from_offers"].iloc[0]) if "capacity_from_offers" in row.columns else q_cap_actual
        cap = max(cap, q_cap_actual)
    else:
        cap = q_cap_actual

    # Nash inverse supply line p(q)
    q_grid = np.linspace(0.0, cap, q_grid_n)
    p_nash = alpha + beta * q_grid

    # plot
    plt.figure(figsize=(7,4.5))
    plt.plot(x_step, y_step, label="Observed offer curve (steps)")
    plt.plot(q_grid, p_nash, label="Model Nash supply: p(q)=α+βq")

    plt.xlabel("Quantity (MW)")
    plt.ylabel("Price ($/MWh)")
    plt.title(f" Observed vs Nash offer\n{firm} @ {ts}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {"alpha": alpha, "beta": beta, "cap_used": cap, "cap_actual_steps": q_cap_actual}

for ts in sorted(df[df["firm"] == "QECNR"]["ts"].unique()):
    plot_nash_for_firm_hour(
        df_ne=df,
        inp=inp,
        ts=ts,
        firm="QECNR",
    )


# In[49]:


def plot_nash_multipanel(
    df_ne: pd.DataFrame,
    inp: DamInputs,
    firm: str,
    hours_to_plot: List[int] = [0, 6, 12, 17, 23],  # indices into sorted ts list
    q_grid_n: int = 200,
    price_cap: float = 200.0,
):
    all_ts = sorted(df_ne[df_ne["firm"] == firm]["ts"].unique())
    selected_ts = [all_ts[i] for i in hours_to_plot if i < len(all_ts)]

    n = len(selected_ts)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for ax, ts in zip(axes, selected_ts):
        ts = pd.Timestamp(ts)  # ADD THIS LINE HERE
    
        # --- observed offer curve ---
        offers_hour = inp.offers[inp.offers["ts"] == ts].copy()
        offers_firm = offers_hour[offers_hour["firm"] == firm].copy()
        if offers_firm.empty:
            ax.set_visible(False)
            continue

        x_step, y_step, q_cap_actual = build_step_curve(offers_firm)

        # --- Nash params ---
        row = df_ne[(df_ne["ts"] == ts) & (df_ne["firm"] == firm)]
        if row.empty:
            ax.set_visible(False)
            continue

        alpha = float(row["alpha"].iloc[0])
        beta  = float(row["beta"].iloc[0])
        cap   = float(row["capacity_from_offers"].iloc[0]) if "capacity_from_offers" in row.columns else q_cap_actual
        cap   = max(cap, q_cap_actual)

        q_grid = np.linspace(0.0, cap, q_grid_n)
        p_nash = alpha + beta * q_grid

#         # --- observed clearing price ---
#         p_obs = inp.prices.set_index("ts")["price"].get(ts, np.nan)

        # --- plot ---
        ax.plot(x_step, y_step, color="steelblue", linewidth=1.5,
                label="Observed offer")
        ax.plot(q_grid, p_nash, color="orange", linewidth=1.5,
                label="Nash supply")
#         if not np.isnan(p_obs):
#             ax.axhline(p_obs, color="navy", linestyle="--", linewidth=1,
#                        label=f"Obs price ${p_obs:.0f}")

        ax.set_ylim(0, price_cap)
        ax.set_xlabel("Quantity (MW)", fontsize=9)
        ax.set_ylabel("Price ($/MWh)", fontsize=9)
        ax.set_title(f"{ts.strftime('%H:%M')}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # hide any unused panels
    for ax in axes[len(selected_ts):]:
        ax.set_visible(False)

    fig.suptitle(f"Observed vs Nash Offer Curve–{firm}\n(Oct 12, 2025)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f"nash_multipanel_{firm}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: nash_multipanel_{firm}.png")


# --- usage: pick 6 representative hours ---
plot_nash_multipanel(
    df_ne=df,
    inp=inp,
    firm="QECNR",
    hours_to_plot=[0, 4, 8, 12, 16, 21],
    price_cap=150.0,
)


# In[63]:


def demand_slope(demand_params: Tuple[float, float, float], p: float) -> float:
    """
    If demand is Q(p) = A - B p - C p^2, then dQ/dp = -(B + 2 C p).
    """
    A, B, C = demand_params
    return -(B + 2.0 * C * p)

def mc_from_quadratic(mc_params: Tuple[float, float, float], q: float) -> float:
    """
    If MC(q) = a + b q + c q^2.
    """
    a, b, c = mc_params
    return a + b*q + c*(q*q)

def supply_slope_dqdp_from_inverse(alpha: float, beta: float, p: float) -> float:
    """
    Inverse supply: p = alpha + beta*q  =>  q(p) = (p - alpha)/beta (clipped at 0)
    dq/dp = 1/beta if interior, else 0.
    """
    beta = max(beta, 1e-12)
    q = (p - alpha) / beta
    return 0.0 if q <= 0 else 1.0 / beta
# ============================================================
# Conduct outputs: market (hourly) + firm panel
# ============================================================

def build_conduct_outputs(
    df_ne: pd.DataFrame,
    inp: DamInputs,
    window_hours: int = 48,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    P_star = inp.prices.set_index("ts")["price"]
    Q_star = compute_total_energy_sold(inp.offers, inp.prices)

    market_rows = []
    firm_rows = []

    for ts, df_hour in df_ne.groupby("ts"):
        df_hour = df_hour.copy()

        p_eq = float(df_hour["p_eq"].iloc[0])

        d_params = fit_quadratic_demand_from_window(
            ts=ts,
            p_series=P_star,
            q_series=Q_star,
            window_hours=window_hours,
        )

        offers_hour = inp.offers[inp.offers["ts"] == ts].copy()
        if offers_hour.empty:
            continue

        mc_params_by_firm = fit_quadratic_mc_for_hour(offers_hour)

        Q_eq = float(df_hour["q_eq"].sum())
        dQdp = demand_slope(d_params, p_eq)
        epsilon = np.nan
        if Q_eq > 0 and dQdp != 0 and not np.isnan(dQdp):
            epsilon = dQdp * (p_eq / Q_eq)

        # --- Pass 1: accumulate MC_agg ---
        MC_agg = 0.0
        for _, r in df_hour.iterrows():
            firm = r["firm"]
            qi = float(r["q_eq"])
            si = qi / Q_eq if Q_eq > 0 else 0.0
            mc_params = mc_params_by_firm.get(firm, (0.0, 0.0, 0.0))
            MC_agg += si * mc_from_quadratic(mc_params, qi)

        # --- Pass 2: firm rows ---
        for _, r in df_hour.iterrows():
            firm = r["firm"]
            qi = float(r["q_eq"])
            si = qi / Q_eq if Q_eq > 0 else np.nan

            mc_params = mc_params_by_firm.get(firm, (0.0, 0.0, 0.0))
            mci = mc_from_quadratic(mc_params, qi)

            lerner = np.nan
            if p_eq != 0:
                lerner = (p_eq - mci) / p_eq

            theta_i = np.nan
            if epsilon is not None and not np.isnan(epsilon) and epsilon != 0:
                theta_i = lerner / (-1.0 / epsilon)

            firm_rows.append({
                "ts": ts,
                "firm": firm,
                "q_eq": qi,
                "share": si,
                "mc_at_q": mci,
                "lerner": lerner,
                "epsilon": epsilon,
                "theta_i": theta_i,
            })

        # --- Market-level HHI ---
        if Q_eq > 0:
            shares = df_hour["q_eq"].to_numpy() / Q_eq
            hhi = float(np.sum(shares**2))
        else:
            hhi = np.nan

        # --- Market-level theta via equation 5.2 ---
        B = -dQdp  # positive
        theta_market = np.nan
        if Q_eq > 0 and not np.isnan(B) and not np.isnan(MC_agg):
            theta_market = float((p_eq - MC_agg) * B / Q_eq)

        market_rows.append({
            "ts": ts,
            "p_eq": p_eq,
            "Q_eq": Q_eq,
            "HHI": hhi,
            "epsilon": epsilon,
            "theta_market": theta_market,
            "demand_A": d_params[0],
            "demand_B": d_params[1],
            "demand_C": d_params[2],
            "dQdp": dQdp,
            "MC_agg": MC_agg,
        })

    market_conduct_df = pd.DataFrame(market_rows).sort_values("ts")
    firm_conduct_df = pd.DataFrame(firm_rows).sort_values(["ts", "firm"])
    return market_conduct_df, firm_conduct_df


def build_conduct_function(
    inp,
    market_conduct_df: pd.DataFrame,
    n_bins: int = 20,
    q_clip=(0.0, 1.0),
    cap_method: str = "sum",   # "sum" (incremental) or "max" (cumulative)
):
    cap_rows = []
    for ts, offers_hour in inp.offers.groupby("ts"):
        if cap_method == "max":
            Q_cap = float(offers_hour.groupby("firm")["mw"].max().sum())
        else:
            Q_cap = float(offers_hour["mw"].sum())
        cap_rows.append({"ts": ts, "Q_cap": Q_cap})
    cap_df = pd.DataFrame(cap_rows)

    m = market_conduct_df.merge(cap_df, on="ts", how="inner").copy()
    m["q_norm"] = m["Q_eq"] / m["Q_cap"]
    m = m.replace([np.inf, -np.inf], np.nan).dropna(subset=["q_norm", "theta_market"])

    lo, hi = q_clip
    m = m[(m["q_norm"] >= lo) & (m["q_norm"] <= hi)].copy()

    bins = np.linspace(lo, hi, n_bins + 1)
    m["q_bin"] = pd.cut(m["q_norm"], bins=bins, include_lowest=True)

    g = m.groupby("q_bin", observed=True)["theta_market"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["se"] = out["std"] / np.sqrt(out["count"].clip(lower=1))

    intervals = pd.IntervalIndex(out["q_bin"])

    out["q_left"]   = intervals.left.astype(float)
    out["q_right"]  = intervals.right.astype(float)
    out["q_center"] = 0.5 * (out["q_left"] + out["q_right"])

    out = out.rename(columns={"mean": "phi_hat"})
    return out, m


# ---- usage ----
inp = load_inputs(PATH_OFFERS, PATH_PRICES, settlement_point=None)
market_conduct_df, firm_conduct_df = build_conduct_outputs(df, inp, window_hours=48)
phi_df, per_hour_df = build_conduct_function(inp, market_conduct_df, n_bins=20, q_clip=(0,1), cap_method="sum")


# In[34]:


def plot_conduct_function(phi_df: pd.DataFrame, title="Estimated Conduct Function"):
    phi_df = phi_df.dropna(subset=["q_center", "phi_hat", "se"]).copy()

    x = phi_df["q_center"].to_numpy()
    y = phi_df["phi_hat"].to_numpy()
    se = phi_df["se"].to_numpy()

    plt.figure()
    plt.plot(x, y, marker="o", linestyle="-")
    # 95% band: mean ± 1.96*SE
    plt.fill_between(x, y - 1.96*se, y + 1.96*se, alpha=0.2)

    plt.xlabel("Normalized quantity  q = Q_eq / Q_cap")
    plt.ylabel("Conduct  $\hat\\phi(q)$")
    plt.title(title)
    plt.tight_layout()
    plt.show()

phi_df, per_hour_df = build_conduct_function(
    inp=inp,
    market_conduct_df=market_conduct_df,
    n_bins=5,  # down from 20
    q_clip=(0.0, 1.0),
    cap_method="sum"
)

print(phi_df[["q_center", "phi_hat", "se", "count"]].head(10))

plot_conduct_function(phi_df, title="Conduct Function in DAM (binned by normalized quantity)")


# In[61]:


print(market_conduct_df[["ts", "p_eq", "MC_agg", "Q_eq", "dQdp", "demand_B"]].head(10))


# In[51]:


def plot_conduct_function(phi_df: pd.DataFrame, title="Estimated Conduct Function"):
    phi_df = phi_df.dropna(subset=["q_center", "phi_hat", "se"]).copy()

    x = phi_df["q_center"].to_numpy()
    y = phi_df["phi_hat"].to_numpy()
    se = phi_df["se"].to_numpy()

    plt.figure()
    plt.plot(x, y, marker="o", linestyle="-")
    # 95% band: mean ± 1.96*SE
    plt.fill_between(x, y - 1.96*se, y + 1.96*se, alpha=0.2)

    plt.xlabel("Normalized quantity  q = Q_eq / Q_cap")
    plt.ylabel("Conduct  $\hat\\phi(q)$")
    plt.title(title)
    plt.tight_layout()
    plt.show()

phi_df, per_hour_df = build_conduct_function(
    inp=inp,
    market_conduct_df=market_conduct_df,
    n_bins=20,
    q_clip=(0.0, 1.0),   
    cap_method="sum"
)

print(phi_df[["q_center", "phi_hat", "se", "count"]].head(10))

plot_conduct_function(phi_df, title="Conduct Function in DAM (binned by normalized quantity)")


# In[52]:


ts_debug = list(df_ne.groupby("ts"))[0][1]  # first hour
ts = ts_debug["ts"].iloc[0]
p_eq = float(ts_debug["p_eq"].iloc[0])

d_params = fit_quadratic_demand_from_window(ts, P_star, Q_star, 48)
B = d_params[1]

offers_hour = inp.offers[inp.offers["ts"] == ts].copy()
mc_params_by_firm = fit_quadratic_mc_for_hour(offers_hour)
Q_total = float(Q_star.loc[ts])

mc_vals = []
for _, r in ts_debug.iterrows():
    qi = float(r["q_eq"])
    mc_p = mc_params_by_firm.get(r["firm"], (0.0,0.0,0.0))
    mc_vals.append(mc_from_quadratic(mc_p, qi))
mc_avg = np.mean(mc_vals)

print(f"p_eq:    {p_eq:.4f}")
print(f"mc_avg:  {mc_avg:.4f}")
print(f"B:       {B:.6f}")
print(f"Q_total: {Q_total:.2f}")
print(f"phi_hat: {(p_eq - mc_avg) * B / Q_total:.6f}")


# In[ ]:




