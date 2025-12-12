#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Callable, Set, Tuple, List, Optional
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

# target date 10/12/2025


# In[2]:


# calculate energy bought and sold on 10/12/2025 (ERCOT doesn't provide this data)
offers = pd.read_csv("60d_DAM_EnergyOnlyOffers-11-DEC-25.csv")
prices = pd.read_csv("settlement_pt_price.csv")
TARGET_DATE = "10/12/2025"  # MM/DD/YYYY
offers = offers[offers["Delivery Date"] == TARGET_DATE].copy()
prices = prices[prices["DeliveryDate"] == TARGET_DATE].copy()
offers["HE"] = offers["Hour Ending"].astype(str).str.extract(r"(\d+)").astype(int)
prices["HE"] = prices["HourEnding"].str.extract(r"(\d+)").astype(int)
price_table = (
    prices[["SettlementPoint", "HE", "SettlementPointPrice"]]
    .rename(columns={"SettlementPointPrice": "p_star"})
)
merged = offers.merge(
    price_table,
    left_on=["Settlement Point", "HE"],
    right_on=["SettlementPoint", "HE"],
    how="inner"   # inner is fine now — coverage will be high
)

def accepted_quantity(row):
    mws, prices = [], []

    for k in range(1, 11):
        mw = row.get(f"Energy Only Offer MW{k}")
        pr = row.get(f"Energy Only Offer Price{k}")

        if pd.notna(mw) and pd.notna(pr):
            mws.append(float(mw))
            prices.append(float(pr))

    if not prices:
        return 0.0

    prices = np.array(prices)
    mws = np.array(mws)

    mask = prices <= row["p_star"]
    return float(mws[mask].sum()) if mask.any() else 0.0

merged["q_accept"] = merged.apply(accepted_quantity, axis=1)

dam_energy_sold = (
    merged
    .groupby("HE")["q_accept"]
    .sum()
    .sort_index()
)

dam_energy_sold


# In[3]:


def parse_hour_ending_to_int(x) -> int:
    """
    Returns hour-ending as an int in 1..24 from values like:
    1, "1", "01", "01:00", "1:00", "24:00", 24.
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()

    # If format like "01:00" or "24:00"
    if ":" in s:
        s = s.split(":")[0].strip()
    try:
        h = int(s)
    except ValueError:
        return np.nan

    if h < 1 or h > 24:
        return np.nan
    return h


# In[25]:


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

strategic_firms, daily_mw = top_firms_by_daily_mw(
    inp,
    target_day="2025-10-12",
    top_n=3
)

print("Strategic firms:", strategic_firms)
print(daily_mw.head(10))


# In[26]:


# 0) Utilities / robust column picking
def pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    """
    Pick the first column in df whose lowercase name contains any candidate substring.
    """
    cols = list(df.columns)
    lower = {c: c.lower() for c in cols}
    for cand in candidates:
        cand = cand.lower()
        for c in cols:
            if cand in lower[c]:
                return c
    raise KeyError(f"Could not find any of {candidates} in columns: {cols[:30]}...")

def to_datetime_safe(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)

def parse_hour_ending_to_int(x):
    """
    Accepts: 1, "1", "01", "01:00", "24", "24:00"
    Returns: int in [1,24] or np.nan
    """
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

# 1) Load DAM CSVs

@dataclass
class DamInputs:
    offers: pd.DataFrame
    prices: pd.DataFrame

def load_dam_offers(path_offers_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_offers_csv)

    # build timestamp from Delivery Date + Hour Ending (HE24 -> next day)
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
    out["mw"] = pd.to_numeric(out["mw"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["ts", "firm", "settlement_point", "mw", "price"])
    out = out[out["mw"] > 0]
    PRICE_CAP = 1000
    out = out[out["price"] <= PRICE_CAP]
    return out[["ts","firm","settlement_point","step","mw","price"]]

def load_dam_prices(path_prices_csv: str,
                    settlement_point: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path_prices_csv)

    # robust timestamp build: DeliveryDate + HourEnding (handles "01:00")
    df["DeliveryDate"] = pd.to_datetime(df["DeliveryDate"], errors="coerce")
    he = df["HourEnding"].apply(parse_hour_ending_to_int).astype("float")
    df["ts"] = df["DeliveryDate"] + pd.to_timedelta((he - 1), unit="h")
    df.loc[he == 24, "ts"] += pd.Timedelta(days=1)
    df = df.dropna(subset=["ts"])

    df = df.rename(columns={
        "SettlementPoint": "settlement_point",
        "SettlementPointPrice": "price",
    })

    # keep one settlement point if provided
    if settlement_point is not None:
        df = df[df["settlement_point"].astype(str) == str(settlement_point)].copy()

    out = df[["ts", "price"]].dropna()
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["price"])
    out = out.groupby("ts", as_index=False)["price"].mean()
    return out

def load_inputs(path_offers: str, path_prices: str, settlement_point: Optional[str]=None) -> DamInputs:
    offers = load_dam_offers(path_offers)
    prices = load_dam_prices(path_prices, settlement_point=settlement_point)
    return DamInputs(offers=offers, prices=prices)

# 2) Compute DAM total energy sold per hour from offers + price
#    (stack all offers and clear at observed p*)

def clear_quantity_at_price(offers_hour: pd.DataFrame, p_star: float) -> float:
    # simple step-stack approximation: all MW offered at price <= p_star clears
    return float(offers_hour.loc[offers_hour["price"] <= p_star, "mw"].sum())

def compute_total_energy_sold(offers: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """
    offers: ts, settlement_point, price, mw
    prices: ts, price (already filtered to ONE settlement point if you passed one)
    """
    prices_idx = prices.set_index("ts")["price"]

    out = {}
    for ts, offers_hour in offers.groupby("ts"):
        if ts not in prices_idx.index:
            continue
        p_star = float(prices_idx.loc[ts])
        out[ts] = float(offers_hour.loc[offers_hour["price"] <= p_star, "mw"].sum())
    return pd.Series(out).sort_index()

# 3) Quadratic marginal cost fit (least squares, per firm per hour)
#    MC_i(q) = a + b q + c q^2

def fit_quadratic_mc_for_hour(offers_hour: pd.DataFrame,
                              low_quantile: float = 0.2) -> Dict[str, Tuple[float,float,float]]:
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

        X = np.column_stack([np.ones_like(q), q, q*q])
        coef, *_ = np.linalg.lstsq(X, p, rcond=None)
        a,b,c = map(float, coef)
        if c < 0: c = 0.0
        params[firm] = (a,b,c)
    return params

def total_cost_from_mc(mc: Tuple[float,float,float], q: float) -> float:
    a,b,c = mc
    return a*q + 0.5*b*q*q + (1.0/3.0)*c*q*q*q

# 4) Demand curve fit (least squares)

def fit_quadratic_demand_from_window(ts: pd.Timestamp,
                                     p_series: pd.Series,
                                     q_series: pd.Series,
                                     window_hours: int = 48) -> Tuple[float,float,float]:
    df = pd.DataFrame({"p": p_series, "q": q_series}).dropna()
    if df.empty:
        return (0.0, 1e-6, 0.0)

    lo = ts - pd.Timedelta(hours=window_hours)
    hi = ts + pd.Timedelta(hours=window_hours)
    sub = df.loc[(df.index >= lo) & (df.index <= hi)].copy()
    if len(sub) < 10:
        sub = df.tail(200)

    P = sub["p"].to_numpy()
    Q = sub["q"].to_numpy()

    X = np.column_stack([np.ones_like(P), -P, -(P*P)])
    coef, *_ = np.linalg.lstsq(X, Q, rcond=None)
    A,B,C = map(float, coef)
    if B <= 0: B = abs(B) + 1e-6
    if C < 0:  C = 0.0
    return (A,B,C)

def demand_Q(demand_params: Tuple[float,float,float], p: float) -> float:
    A,B,C = demand_params
    return max(A - B*p - C*(p*p), 0.0)

# 5) Supply-function NE approximation (iterative best response, affine supply)

def affine_q_of_p(alpha: float, beta: float):
    beta = max(beta, 1e-6)
    return lambda p: max((p - alpha)/beta, 0.0)

def find_clearing_price(supply_q_fns: List,
                        demand_params: Tuple[float,float,float],
                        p_bounds=(-500, 5000)) -> float:
    def f(p):
        s = sum(fn(p) for fn in supply_q_fns)
        d = demand_Q(demand_params, p)
        return s - d

    xs = np.linspace(p_bounds[0], p_bounds[1], 400)
    fs = np.array([f(x) for x in xs])
    for i in range(len(xs)-1):
        if fs[i] == 0:
            return float(xs[i])
        if fs[i]*fs[i+1] < 0:
            return float(brentq(f, xs[i], xs[i+1]))
    return float(xs[np.argmin(np.abs(fs))])

def firm_capacity_from_offers(offers_hour: pd.DataFrame) -> Dict[str, float]:
    return offers_hour.groupby("firm")["mw"].sum().to_dict()

def profit_for_candidate(alpha_beta: Tuple[float,float],
                         firm: str,
                         offers_hour: pd.DataFrame,
                         mc_params: Dict[str, Tuple[float,float,float]],
                         demand_params: Tuple[float,float,float],
                         rivals_params: Dict[str, Tuple[float,float]],
                         capacities: Dict[str, float]) -> Tuple[float,float,float]:
    alpha, beta = alpha_beta
    q_i = affine_q_of_p(alpha, beta)

    supply_fns = []
    for r, (a_r, b_r) in rivals_params.items():
        supply_fns.append(affine_q_of_p(a_r, b_r))
    supply_fns.append(q_i)

    p_star = find_clearing_price(supply_fns, demand_params)

    qi = min(q_i(p_star), capacities.get(firm, np.inf))
    Ci = total_cost_from_mc(mc_params.get(firm, (0.0,0.0,0.0)), qi)
    pi = p_star * qi - Ci
    return float(pi), float(p_star), float(qi)

def best_response(firm: str,
                  init_ab: Tuple[float,float],
                  offers_hour: pd.DataFrame,
                  mc_params: Dict[str, Tuple[float,float,float]],
                  demand_params: Tuple[float,float,float],
                  rivals_params: Dict[str, Tuple[float,float]],
                  capacities: Dict[str, float]) -> Tuple[Tuple[float,float], bool]:
    def obj(x):
        pi, _, _ = profit_for_candidate((x[0], x[1]), firm, offers_hour, mc_params, demand_params, rivals_params, capacities)
        return -pi

    bounds = [(-500, 5000), (1e-4, 2000.0)]
    res = minimize(obj, np.array(init_ab, dtype=float), bounds=bounds, options={"maxiter": 200})
    if not res.success:
        return init_ab, False
    return (float(res.x[0]), float(res.x[1])), True

def compute_ne_for_hour(offers_hour: pd.DataFrame,
                        demand_params: Tuple[float,float,float],
                        strategic_firms: List[str],
                        damping: float = 0.4,
                        max_iter: int = 40,
                        tol: float = 1e-3) -> Dict[str, Tuple[float,float]]:
    capacities = firm_capacity_from_offers(offers_hour)
    mc_params = fit_quadratic_mc_for_hour(offers_hour)

    params = {}
    for f in strategic_firms:
        g = offers_hour[offers_hour["firm"] == f].sort_values("price")
        if len(g) < 2:
            params[f] = (0.0, 1.0)
            continue
        cumq = g["mw"].cumsum().to_numpy()
        p = g["price"].to_numpy()
        X = np.column_stack([np.ones_like(cumq), cumq])
        coef, *_ = np.linalg.lstsq(X, p, rcond=None)
        params[f] = (float(coef[0]), max(float(coef[1]), 1e-4))

    for _ in range(max_iter):
        max_change = 0.0
        for f in strategic_firms:
            rivals = {r: params[r] for r in strategic_firms if r != f}
            old = params[f]
            new, ok = best_response(f, old, offers_hour, mc_params, demand_params, rivals, capacities)
            upd = ((1-damping)*old[0] + damping*new[0],
                   (1-damping)*old[1] + damping*new[1])
            params[f] = upd
            max_change = max(max_change, abs(upd[0]-old[0]), abs(upd[1]-old[1]))
        if max_change < tol:
            break
    return params

def run_replication(path_offers: str,
                    path_prices: str,
                    settlement_point: Optional[str] = None,
                    target_day: Optional[str] = None,
                    strategic_top_n: int = 3) -> pd.DataFrame:
    inp = load_inputs(path_offers, path_prices, settlement_point=settlement_point)

    # hourly inferred cleared quantity
    Q_star = compute_total_energy_sold(inp.offers, inp.prices)
    P_star = inp.prices.set_index("ts")["price"]
    strategic, _ = top_firms_by_daily_mw(inp, target_day, strategic_top_n)

    if target_day is not None:
        day = pd.to_datetime(target_day).date()
        Q_star = Q_star[Q_star.index.date == day]
        P_star = P_star[P_star.index.date == day]

    rows = []
    for ts in Q_star.index:
        offers_hour = inp.offers[inp.offers["ts"] == ts].copy()
        if offers_hour.empty:
            continue

        cap = firm_capacity_from_offers(offers_hour)
        strategic_firms = strategic
        #strategic = [k for k,_ in sorted(cap.items(), key=lambda kv: kv[1], reverse=True)[:strategic_top_n]]

        d_params = fit_quadratic_demand_from_window(ts, P_star, Q_star, window_hours=48)
        ne_params = compute_ne_for_hour(offers_hour, d_params, strategic_firms=strategic)

        for f in strategic:
            rivals = {r: ne_params[r] for r in strategic if r != f}
            mc_params = fit_quadratic_mc_for_hour(offers_hour)
            capacities = firm_capacity_from_offers(offers_hour)
            pi, p_eq, q_eq = profit_for_candidate(ne_params[f], f, offers_hour, mc_params, d_params, rivals, capacities)
            rows.append({
                "ts": ts,
                "firm": f,
                "alpha": ne_params[f][0],
                "beta": ne_params[f][1],
                "p_eq": p_eq,
                "q_eq": q_eq,
                "profit": pi,
                "demand_A": d_params[0],
                "demand_B": d_params[1],
                "demand_C": d_params[2],
                "capacity_from_offers": capacities.get(f, np.nan),
            })

    return pd.DataFrame(rows).sort_values(["ts","firm"])

if __name__ == "__main__":
    PATH_OFFERS = "60d_DAM_EnergyOnlyOffers-11-DEC-25.csv"
    PATH_PRICES = "settlement_pt_price.csv"  
    TARGET_DAY  = "2025-10-12"

    df = run_replication(PATH_OFFERS, PATH_PRICES, settlement_point=None, target_day=TARGET_DAY, strategic_top_n=3)
    print(df.head(20))
    df.to_csv("dam_ne_results.csv", index=False)
    print("Wrote dam_ne_results.csv")


# In[27]:


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


# In[28]:


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


# In[29]:


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


# In[31]:


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


# In[32]:


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


# In[33]:


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
    """
    df_ne: Nash equilibrium results from run_replication()
          must include columns: ts, firm, alpha, beta, p_eq, q_eq
    inp:   DamInputs from load_inputs()
    window_hours: rolling window for demand fit

    Returns:
      market_conduct_df: columns include ts, p_eq, Q_eq, HHI, epsilon, theta_market, demand params
      firm_conduct_df:   columns include ts, firm, q_eq, share, mc_at_q, lerner, epsilon, theta_i
    """
    # Observed prices used for demand fit window (same as your pipeline)
    P_star = inp.prices.set_index("ts")["price"]
    Q_star = compute_total_energy_sold(inp.offers, inp.prices)

    market_rows = []
    firm_rows = []

    for ts, df_hour in df_ne.groupby("ts"):
        df_hour = df_hour.copy()

        # Equilibrium price for this hour (same across firms)
        p_eq = float(df_hour["p_eq"].iloc[0])

        # Demand params from rolling window around ts
        d_params = fit_quadratic_demand_from_window(
            ts=ts,
            p_series=P_star,
            q_series=Q_star,
            window_hours=window_hours,
        )

        # Offers for this hour (used to fit MC)
        offers_hour = inp.offers[inp.offers["ts"] == ts].copy()
        if offers_hour.empty:
            continue

        # --- Firm-level: MC(q), Lerner, theta_i ---
        mc_params_by_firm = fit_quadratic_mc_for_hour(offers_hour)

        Q_eq = float(df_hour["q_eq"].sum())
        dQdp = demand_Q(d_params, p_eq)  # negative
        epsilon = np.nan
        if Q_eq > 0 and dQdp != 0 and not np.isnan(dQdp):
            epsilon = dQdp * (p_eq / Q_eq)  # should be negative

        # firm rows
        for _, r in df_hour.iterrows():
            firm = r["firm"]
            qi = float(r["q_eq"])
            si = qi / Q_eq if Q_eq > 0 else np.nan

            mc_params = mc_params_by_firm.get(firm, (0.0, 0.0, 0.0))
            mci = mc_from_quadratic(mc_params, qi)

            lerner = np.nan
            if p_eq != 0:
                lerner = (p_eq - mci) / p_eq

            # conduct index proxy: Lerner_i ≈ theta_i * (-1/epsilon)
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

        # --- Market-level: HHI + SFE-style conduct proxy ---
        if Q_eq > 0:
            shares = df_hour["q_eq"].to_numpy() / Q_eq
            hhi = float(np.sum(shares**2))
        else:
            hhi = np.nan

        # dS/dp from q_i(p) slopes implied by inverse supply p = alpha + beta q
        dSdp = 0.0
        for _, r in df_hour.iterrows():
            dSdp += supply_slope_dqdp_from_inverse(
                alpha=float(r["alpha"]),
                beta=float(r["beta"]),
                p=p_eq
            )

        theta_market = np.nan
        if dSdp != 0 and not np.isnan(dSdp) and not np.isnan(dQdp):
            # proxy (sign-adjusted): theta = |HHI * (dQ/dp) / (dS/dp)|
            theta_market = float(abs(hhi * (dQdp / dSdp)))

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
            "dSdp": dSdp,
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
    n_bins=20,
    q_clip=(0.0, 1.0),   
    cap_method="sum"
)

print(phi_df[["q_center", "phi_hat", "se", "count"]].head(10))

plot_conduct_function(phi_df, title="Conduct Function in DAM (binned by normalized quantity)")


# In[ ]:




