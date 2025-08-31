import numpy as np
import matplotlib.pyplot as plt
import random
from math import ceil, sqrt
from collections import deque
# If you have scipy: from scipy.stats import norm
# We'll hardcode z ~ 1.645 for 95% service to keep it dependency-free.

# =================== CONFIG ===================

np.random.seed(42)
random.seed(42)

SIM_DAYS = 300

# ----- Demand with smooth trends + noise -----
BASE_DEMAND = 18                 # baseline level
SEASONAL_AMPLITUDE = 2           # weekly seasonality amplitude (0 to disable)
TREND_MIN_LEN = 100               # min length of a trend regime (days)
TREND_MAX_LEN = 200               # max length of a trend regime (days)
UPTREND_DRIFT_RANGE = (0.04, 0.10)   # daily drift added during uptrends
DOWNTREND_DRIFT_RANGE = (-0.03, -0.01) # daily drift during downtrends (smaller magnitude)
UPTREND_PROB = 0.65              # bias: more/longer uptrends than downtrends
DEMAND_NOISE_STD = 1.0           # Gaussian noise on top of trend
POISSON_OBS = True               # if True, sample Poisson around positive mean

# ----- Lead times -----
LT_WAREHOUSE_TO_FACTORY = 2
LT_SUPPLIER_RANGE = (3, 6)
SUPPLIER_DELAY_PROB = 0.05
SUPPLIER_EXTRA_DELAY = 3
SUPPLIER_PARTIAL_FILL_PROB = 0.05
SUPPLIER_PARTIAL_MIN_FRAC = 0.6

# ----- Backorders vs Lost Sales -----
ALLOW_BACKORDERS = True

# ----- Inventory / Supplier capacity -----
initial_inventory = {"supplier": 100, "warehouse": 70, "factory": 30}
SUPPLIER_INFINITE_CAPACITY = True
SUPPLIER_MONTHLY_REPLENISH = 500

# ----- Periodic Review Policy (forecast-driven) -----
REVIEW_PERIOD_R = 7        # review every R days (weekly)
SERVICE_LEVEL = 0.95       # target cycle service (z≈1.645)
Z = 1.645                  # approx for 95%
MOQ = 20                   # minimum order quantity from supplier
LOT_SIZE = 10              # order in multiples of LOT_SIZE

# SES Forecast parameters
SES_ALPHA = 0.3            # smoothing for level
ERR_ALPHA = 0.3            # smoothing for forecast error variance (EWMA of squared error)

# ----- Costs -----
UNIT_COST = 100
HOLDING_COST_PER_UNIT = 3
BACKORDER_COST_PER_UNIT = 25
EXPEDITE = False           # optional: could add expedite logic if backorders high

# =================== DEMAND GENERATOR ===================

def generate_trended_demand_series(n_days):
    """Create a smooth demand mean series with biased trend regimes + weekly seasonality + noise."""
    mu = BASE_DEMAND
    series = []
    day = 0
    while day < n_days:
        # choose regime type & length
        up = random.random() < UPTREND_PROB
        length = random.randint(TREND_MIN_LEN, TREND_MAX_LEN)
        drift = random.uniform(*UPTREND_DRIFT_RANGE) if up else random.uniform(*DOWNTREND_DRIFT_RANGE)

        for _ in range(length):
            if day >= n_days: break
            # seasonal component (weekly)
            season = SEASONAL_AMPLITUDE * np.sin(2*np.pi * (day % 7) / 7.0) if SEASONAL_AMPLITUDE else 0.0
            mu = max(0.1, mu + drift)  # evolve mean with drift, keep positive
            mean_today = max(0.1, mu + season)

            # add smooth noise on mean (then actual observation sampled later)
            noisy_mean = max(0.1, mean_today + np.random.normal(0, DEMAND_NOISE_STD))
            series.append(noisy_mean)
            day += 1

    # Convert mean series to integer demand
    if POISSON_OBS:
        demand = [int(np.random.poisson(max(0.1, m))) for m in series]
    else:
        demand = [max(0, int(round(m + np.random.normal(0, DEMAND_NOISE_STD)))) for m in series]
    return np.array(demand), np.array(series)

# =================== HELPERS ===================

def supplier_lead_time():
    lt = random.randint(*LT_SUPPLIER_RANGE)
    if random.random() < SUPPLIER_DELAY_PROB:
        lt += SUPPLIER_EXTRA_DELAY
    return lt

def maybe_partial_fill(q):
    if random.random() < SUPPLIER_PARTIAL_FILL_PROB:
        return max(0, int(round(q * random.uniform(SUPPLIER_PARTIAL_MIN_FRAC, 1.0))))
    return q

def compute_working_capital(inv, pipe_sup, pipe_wh):
    on_hand = inv["warehouse"] + inv["factory"] + (0 if SUPPLIER_INFINITE_CAPACITY else inv["supplier"])
    in_transit = sum(q for _, q in pipe_sup) + sum(q for _, q in pipe_wh)
    return UNIT_COST * (on_hand + in_transit)

def round_moq_lot(q):
    if q <= 0: return 0
    q = max(q, MOQ)
    # round up to lot size
    return int(ceil(q / LOT_SIZE) * LOT_SIZE)

# =================== SIM ===================

def run_sim():
    demand, demand_mean = generate_trended_demand_series(SIM_DAYS)

    inv = initial_inventory.copy()
    pipeline_supplier = []   # (arrival_day, qty)
    pipeline_warehouse = []  # (arrival_day, qty)
    backorders = 0

    # SES state & forecast error variance (EWMA)
    # initialize with first observed demand
    level = max(1.0, demand[0])
    err_var = 5.0 ** 2

    hist = {k: [] for k in [
        "day","demand","forecast","served","factory_inv","warehouse_inv","supplier_inv",
        "backorders","holding_cost","backorder_cost","working_capital","order_placed"
    ]}

    for day in range(SIM_DAYS):

        # Supplier finite capacity monthly replenish (if used)
        if (not SUPPLIER_INFINITE_CAPACITY) and (day % 30 == 0) and day > 0:
            inv["supplier"] += SUPPLIER_MONTHLY_REPLENISH

        # Receive arrivals
        arr = [q for (t,q) in pipeline_supplier if t == day]
        if arr: inv["warehouse"] += sum(arr)
        pipeline_supplier = [(t,q) for (t,q) in pipeline_supplier if t > day]

        arr = [q for (t,q) in pipeline_warehouse if t == day]
        if arr: inv["factory"] += sum(arr)
        pipeline_warehouse = [(t,q) for (t,q) in pipeline_warehouse if t > day]

        # Demand today
        d = int(demand[day])

        # Serve backorders first (FIFO lumped)
        served = 0
        if ALLOW_BACKORDERS and backorders > 0:
            serve = min(backorders, inv["factory"])
            inv["factory"] -= serve
            backorders -= serve
            served += serve

        # Serve today's demand
        serve_today = min(inv["factory"], d)
        inv["factory"] -= serve_today
        served += serve_today

        shortage_today = d - serve_today
        if ALLOW_BACKORDERS:
            backorders += shortage_today

        # ---- Forecast update (SES) ----
        # Forecast used for next period (t+1)
        forecast = level
        error = d - forecast
        level = SES_ALPHA * d + (1 - SES_ALPHA) * level
        err_var = ERR_ALPHA * (error**2) + (1 - ERR_ALPHA) * err_var
        err_std = sqrt(max(1e-6, err_var))

        # ---- Periodic review: place order every R days ----
        order_qty = 0
        if day % REVIEW_PERIOD_R == 0:
            # Lead time to cover = supplier LT + warehouse→factory LT + review period (protection window)
            # Use expected supplier LT mid-point for target window
            exp_lt = (LT_SUPPLIER_RANGE[0] + LT_SUPPLIER_RANGE[1]) / 2
            protection = exp_lt + LT_WAREHOUSE_TO_FACTORY + REVIEW_PERIOD_R

            # Order-up-to level = forecast * protection + safety stock
            # Safety stock from forecast error (very rough approximation)
            safety = Z * err_std * sqrt(max(1.0, protection))
            target = forecast * protection + safety

            # Inventory position at warehouse echelon (we order at warehouse from supplier)
            invpos_wh = inv["warehouse"] + sum(q for _, q in pipeline_supplier) - backorders
            raw_order = max(0.0, target - invpos_wh)
            order_qty = round_moq_lot(raw_order)

            # Place order (infinite supplier or finite with partial)
            if order_qty > 0:
                if SUPPLIER_INFINITE_CAPACITY:
                    ship = maybe_partial_fill(order_qty)
                else:
                    ship = min(order_qty, inv["supplier"])
                    ship = maybe_partial_fill(ship)
                    inv["supplier"] -= ship

                if ship > 0:
                    lt = supplier_lead_time()
                    pipeline_supplier.append((day + lt, ship))

        # ---- Warehouse -> Factory "replenishment pull" daily (partial allowed) ----
        # Simple pull: try to keep factory above a small buffer using forecast for next R days
        factory_target = min(80, max(20, int(round(forecast * (LT_WAREHOUSE_TO_FACTORY + 2)))))
        needed = max(0, factory_target - (inv["factory"] + sum(q for _, q in pipeline_warehouse) - backorders))
        ship_wf = min(needed, inv["warehouse"])
        if ship_wf > 0:
            inv["warehouse"] -= ship_wf
            pipeline_warehouse.append((day + LT_WAREHOUSE_TO_FACTORY, ship_wf))

        # ---- Costs & WC ----
        holding_cost = HOLDING_COST_PER_UNIT * (inv["warehouse"] + inv["factory"])
        backorder_cost = BACKORDER_COST_PER_UNIT * (backorders if ALLOW_BACKORDERS else 0)
        wc = compute_working_capital(inv, pipeline_supplier, pipeline_warehouse)

        # ---- Log ----
        hist["day"].append(day)
        hist["demand"].append(d)
        hist["forecast"].append(forecast)
        hist["served"].append(serve_today)
        hist["factory_inv"].append(inv["factory"])
        hist["warehouse_inv"].append(inv["warehouse"])
        hist["supplier_inv"].append(inv["supplier"])
        hist["backorders"].append(backorders if ALLOW_BACKORDERS else 0)
        hist["holding_cost"].append(holding_cost)
        hist["backorder_cost"].append(backorder_cost)
        hist["working_capital"].append(wc)
        hist["order_placed"].append(order_qty)

    return hist, demand_mean

# =================== RUN & PLOT ===================

if __name__ == "__main__":
    hist, mean_series = run_sim()

    days = hist["day"]
    plt.figure(figsize=(15,12))

    # 1. Demand vs Forecast
    plt.subplot(3,2,1)
    plt.plot(days, hist["demand"], label="Demand (obs)")
    plt.plot(days, hist["forecast"], label="SES Forecast", alpha=0.9)
    plt.plot(days, mean_series, label="Underlying Mean (trend)", alpha=0.6)
    plt.title("Demand with Smooth Trend + Forecast")
    plt.ylabel("Units")
    plt.legend(loc="upper left")

    # 2. Inventory
    plt.subplot(3,2,2)
    plt.plot(days, hist["factory_inv"], label="Factory Inv")
    plt.plot(days, hist["warehouse_inv"], label="Warehouse Inv")
    plt.title("Inventory by Echelon")
    plt.ylabel("Units")
    plt.legend(loc="upper left")

    # 3. Backorders + Orders placed
    plt.subplot(3,2,3)
    plt.plot(days, hist["backorders"], label="Open Backorders")
    plt.bar(days, hist["order_placed"], alpha=0.3, label="Supplier Orders (qty)")
    plt.title("Backorders and Supplier Orders")
    plt.ylabel("Units")
    plt.legend(loc="upper left")

    # 4. Working Capital
    plt.subplot(3,2,4)
    plt.plot(days, hist["working_capital"], label="Working Capital", color='blue')
    plt.title("Working Capital")
    plt.xlabel("Day")
    plt.ylabel("$")
    plt.legend(loc="upper left")

    # 5. Costs: Holding 
    plt.subplot(3,2,5)
    plt.plot(days, hist["holding_cost"], label="Holding Cost/day", color='orange')
    plt.title("Holding Costs")
    plt.xlabel("Day")
    plt.ylabel("$")
    plt.legend(loc="upper left")

     # 6. Costs: Backorder
    plt.subplot(3,2,6) 
    plt.plot(days, hist["backorder_cost"], label="", color='red')
    plt.title("Backorder Costs")
    plt.xlabel("Day")
    plt.ylabel("$")
    plt.legend(loc="upper left")

    plt.tight_layout()
    plt.show()