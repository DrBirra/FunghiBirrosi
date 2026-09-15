"""Modello dell'indice di fruttificazione, condiviso da update.py e backtest.py.

I parametri meteo di default vengono sostituiti da quelli di data/model.json,
stimati da backtest.py confrontando il modello con i ritrovamenti degli anni passati.
"""
import numpy as np

LAPSE = 0.0065  # °C per metro
DEFAULTS = {"peak": 8, "rain_scale": 25, "dry_k": 25, "tshift": 0.0}


def params_for(g, model):
    p = dict(DEFAULTS)
    p.update({k: v for k, v in (model.get("global") or {}).items() if k in DEFAULTS})
    p.update({k: v for k, v in (model.get("groups", {}).get(g["id"]) or {}).items() if k in DEFAULTS})
    return p


def trap(x, lo, blo, bhi, hi):
    x = np.asarray(x, float)
    up = np.clip((x - lo) / max(blo - lo, 1e-9), 0, 1)
    down = np.clip((hi - x) / max(hi - bhi, 1e-9), 0, 1)
    return np.nan_to_num(np.minimum(up, down))


def kernel(peak, max_lag=20):
    """Peso della pioggia caduta l giorni fa: nullo a 0, massimo al picco, coda lunga."""
    l = np.arange(max_lag + 1, dtype=float)
    k = (l / peak) ** 2 * np.exp(2 * (1 - l / peak))
    return k / k.max()


def eff_rain(P, t, peak):
    """P: (punti, giorni). t: indice del giorno (scalare o array per punto)."""
    K = kernel(peak)
    t = np.broadcast_to(np.asarray(t), (P.shape[0],))
    idx = np.clip(t[:, None] - np.arange(len(K))[None, :], 0, P.shape[1] - 1)
    return np.take_along_axis(P, idx, 1) @ K


def window_mean(A, t, back):
    t = np.broadcast_to(np.asarray(t), (A.shape[0],))
    idx = np.clip(t[:, None] - np.arange(back)[None, :], 0, A.shape[1] - 1)
    return np.nanmean(np.take_along_axis(A, idx, 1), axis=1)


def window_sum(A, t, back):
    return window_mean(np.nan_to_num(A), t, back) * back


def window_min(A, t, back):
    t = np.broadcast_to(np.asarray(t), (A.shape[0],))
    idx = np.clip(t[:, None] - np.arange(back)[None, :], 0, A.shape[1] - 1)
    return np.nanmin(np.take_along_axis(A, idx, 1), axis=1)


def water_factor(rain, soil, deficit, p):
    s_rain = 1 - np.exp(-np.asarray(rain) / p["rain_scale"])
    s_dry = 1.0 if p["dry_k"] == 0 else np.clip(1 - (np.asarray(deficit) - 10) / p["dry_k"], 0.2, 1)
    if soil is None:  # verifica storica: l'umidità del suolo non è nell'archivio giornaliero
        return s_rain * s_dry
    s_soil = trap(soil, 0.10, 0.22, 0.40, 0.55)
    return (0.55 * s_rain + 0.45 * s_soil) * s_dry


def temp_factor(tmin, tmax, frost, g, p):
    lo, blo, bhi, hi = np.array(g["tmin"], float) + p["tshift"]
    heat = np.clip(1 - (np.asarray(tmax) - g["tmax_max"] - p["tshift"]) / 6, 0, 1)
    return trap(tmin, lo, blo, bhi, hi) * heat * np.where(np.asarray(frost) < -2, 0.4, 1.0)


def month_weight(g, cal, day):
    c = cal.get(g["id"], {})
    if "month_w" in c:
        return c["month_w"][day.month - 1]
    m, months = day.month, g["months"]
    if m in months:
        return 1.0
    return 0.4 if (m % 12) + 1 in months or ((m - 2) % 12) + 1 in months else 0.0


def elev_factor(g, cal, elev):
    lo, hi = cal.get(g["id"], {}).get("elev", g["elev"])
    return trap(elev, lo - 250, lo, hi, hi + 250)


def habitat_factor(g, st):
    ft, fo = st["f_tree"], st["f_open"]
    if g["cover"] == "open":
        h = trap(fo, 0.1, 0.45, 1, 1.01) * (1 - 0.5 * trap(ft, 0.5, 0.9, 1, 1.01))
    elif g["cover"] == "edge":
        h = np.clip(4 * ft * fo, 0, 1) * 0.8 + 0.2 * trap(fo, 0.2, 0.6, 1, 1.01)
    else:
        h = trap(ft, 0.1, 0.5, 1, 1.01)
    leaf = g.get("leaf", "any")
    if leaf != "any" and "f_broad" in st:
        known = (st["f_broad"] + st["f_conif"]) > 0
        share = st["f_broad"] if leaf == "broad" else st["f_conif"]
        h = h * np.where(known, 0.25 + 0.75 * share, 0.8)
    return h * (1 - trap(st["f_built"], 0.1, 0.4, 1, 1.01))


def aspect_factor(st, tmin, tmax, g):
    north = np.cos(np.radians(st["aspect"])) * np.clip(st["slope"] / 15, 0, 1)
    hot = tmax > g["tmax_max"] - 3
    cold = tmin < g["tmin"][1]
    return 1 + 0.15 * north * np.where(hot, 1, np.where(cold, -1, 0))


def auc(pos, neg):
    """Probabilità che un giorno con ritrovamento abbia punteggio più alto di un giorno senza."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(allv.size)
    sv = allv[order]
    i = 0
    while i < sv.size:  # ranghi medi per i pari merito
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))
