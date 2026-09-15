#!/usr/bin/env python3
"""Verifica del modello sul passato e taratura dei parametri meteo.

Idea (caso-controllo): per ogni ritrovamento confermato su iNaturalist negli ultimi anni
confronto il meteo dei giorni precedenti con quello dello stesso punto, nello stesso
periodo dell'anno ma in anni diversi. Luogo e stagione sono uguali, quindi conta solo
il meteo. L'AUC misura quanto spesso il giorno del ritrovamento ha un punteggio più alto
del giorno di controllo (0,5 = a caso, 1 = perfetto).

Il meteo storico (Open-Meteo Archive) è pesante per i limiti gratuiti, quindi lo scarico
a rate su più esecuzioni e lo tengo in data/history.npz. Quando è completo stimo
i parametri e scrivo data/model.json.
"""
import itertools, json, os, time, urllib.parse
from datetime import date, datetime, timedelta

import numpy as np

import model as M
from common import DATA, area_filter, get_json, group_of, load_config, load_groups, log, resolve_taxa, warn

HIST = os.path.join(DATA, "history.npz")
MODEL = os.path.join(DATA, "model.json")
VARS = "precipitation_sum,temperature_2m_min,temperature_2m_max,et0_fao_evapotranspiration"
CALLS_PER_MINUTE = 450
REBUILD_AFTER_DAYS = 150


# ---------------- dati ----------------
def fetch_cases(cfg, groups, start, end, in_area):
    t2g = resolve_taxa(groups)
    b = cfg["bbox"]
    out = []
    for g in groups:
        ids = [str(t) for t, gid in t2g.items() if gid == g["id"]]
        if not ids:
            continue
        for page in range(1, cfg["backtest"]["max_pages_per_species"] + 1):
            q = urllib.parse.urlencode({
                "taxon_id": ",".join(ids), "quality_grade": "research", "geo": "true",
                "swlat": b["south"], "swlng": b["west"], "nelat": b["north"], "nelng": b["east"],
                "d1": start.isoformat(), "d2": end.isoformat(),
                "per_page": 200, "page": page, "order": "desc", "order_by": "id",
            })
            data = get_json("https://api.inaturalist.org/v1/observations?" + q)
            for o in data.get("results", []):
                if not o.get("geojson") or not o.get("observed_on"):
                    continue
                if group_of(o.get("taxon") or {}, t2g) != g["id"]:
                    continue
                lon, lat = o["geojson"]["coordinates"]
                if in_area(lat, lon):
                    out.append((g["id"], lat, lon, o["observed_on"]))
            time.sleep(1.2)
            if page * 200 >= data.get("total_results", 0):
                break
        log(f"  {g['id']}: {sum(1 for c in out if c[0] == g['id'])} ritrovamenti")
    return out


def choose_points(cases, cfg):
    bt, b = cfg["backtest"], cfg["bbox"]
    step = bt["point_step_deg"]
    counts = {}
    for _, lat, lon, _ in cases:
        key = (int((lat - b["south"]) // step), int((lon - b["west"]) // step))
        counts[key] = counts.get(key, 0) + 1
    top = sorted(counts, key=counts.get, reverse=True)[: bt["max_points"]]
    return np.array([[b["south"] + (r + 0.5) * step, b["west"] + (c + 0.5) * step] for r, c in top])


def load_history(cfg, cases):
    today = date.today()
    if os.path.isfile(HIST):
        h = dict(np.load(HIST, allow_pickle=False))
        created = date.fromisoformat(str(h["created"]))
        if (today - created).days < REBUILD_AFTER_DAYS:
            return h
        log("Archivio meteo vecchio: lo ricostruisco")
    end = today - timedelta(days=7)
    start = date(end.year - cfg["backtest"]["years"], 1, 1)
    pts = choose_points([c for c in cases if start.isoformat() <= c[3] <= end.isoformat()], cfg)
    D = (end - start).days + 1
    n = len(pts)
    log(f"Nuovo archivio: {n} punti, {start} → {end}")
    return {"created": np.array(today.isoformat()), "start": np.array(start.isoformat()), "end": np.array(end.isoformat()),
            "points": pts, "done": np.zeros(n, bool), "EL": np.zeros(n),
            "P": np.zeros((n, D), "float32"), "TN": np.zeros((n, D), "float32"),
            "TX": np.zeros((n, D), "float32"), "ET": np.zeros((n, D), "float32")}


def save_history(h):
    np.savez_compressed(HIST, **h)


def fetch_history(h, cfg):
    start, end = str(h["start"]), str(h["end"])
    D = h["P"].shape[1]
    cost = max(1.0, D / 14)                                   # conteggio frazionario di Open-Meteo
    per_batch = max(1, int(CALLS_PER_MINUTE // cost))
    budget = cfg["backtest"]["calls_per_run"]
    todo = np.flatnonzero(~h["done"])
    spent = 0.0
    for b in range(0, len(todo), per_batch):
        chunk = todo[b: b + per_batch]
        if spent + cost * len(chunk) > budget:
            break
        q = urllib.parse.urlencode({
            "latitude": ",".join(f"{h['points'][i, 0]:.3f}" for i in chunk),
            "longitude": ",".join(f"{h['points'][i, 1]:.3f}" for i in chunk),
            "start_date": start, "end_date": end, "daily": VARS, "timezone": cfg["timezone"],
        })
        res = get_json("https://archive-api.open-meteo.com/v1/archive?" + q)
        res = res if isinstance(res, list) else [res]
        for i, r in zip(chunk, res):
            d = r["daily"]
            arr = lambda k: np.array([np.nan if v is None else v for v in d[k]], "float32")[:D]
            h["P"][i], h["TN"][i], h["TX"][i], h["ET"][i] = (
                np.nan_to_num(arr("precipitation_sum")), arr("temperature_2m_min"),
                arr("temperature_2m_max"), np.nan_to_num(arr("et0_fao_evapotranspiration")))
            h["EL"][i] = r.get("elevation") or 0
            h["done"][i] = True
        spent += cost * len(chunk)
        log(f"  archivio: {int(h['done'].sum())}/{len(h['done'])} punti (≈{spent:.0f} chiamate)")
        save_history(h)
        time.sleep(65)
    return bool(h["done"].all())


# ---------------- casi e controlli ----------------
def build_samples(h, cases, cfg, groups, static):
    start = date.fromisoformat(str(h["start"]))
    D = h["P"].shape[1]
    pts, step, b = h["points"], cfg["backtest"]["point_step_deg"], cfg["bbox"]
    key_to_p = {(int((la - b["south"]) // step), int((lo - b["west"]) // step)): i for i, (la, lo) in enumerate(pts)}
    elev_of = cell_elevation(static)

    case_set = {}
    for gid, lat, lon, day in cases:
        p = key_to_p.get((int((lat - b["south"]) // step), int((lon - b["west"]) // step)))
        t = (date.fromisoformat(day) - start).days
        if p is None or t < 21 or t >= D:
            continue
        e = elev_of(lat, lon)
        dz = 0.0 if e is None else e - h["EL"][p]
        case_set.setdefault((gid, p, t), dz)

    by_gp = {}
    for gid, p, t in case_set:
        by_gp.setdefault((gid, p), []).append(t)

    samples = {g["id"]: {"p": [], "t": [], "dz": [], "y": []} for g in groups}
    for (gid, p, t), dz in case_set.items():
        s = samples[gid]
        s["p"].append(p); s["t"].append(t); s["dz"].append(dz); s["y"].append(1)
        d0 = start + timedelta(days=t)
        for dy in range(-cfg["backtest"]["years"] - 1, cfg["backtest"]["years"] + 2):
            if dy == 0:
                continue
            try:
                base = d0.replace(year=d0.year + dy)
            except ValueError:
                continue
            for off in (-15, 0, 15):
                tc = (base - start).days + off
                if tc < 21 or tc >= D:
                    continue
                if any(abs(tc - tt) <= 10 for tt in by_gp[(gid, p)]):
                    continue
                s["p"].append(p); s["t"].append(tc); s["dz"].append(dz); s["y"].append(0)
    return {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in samples.items() if np.sum(v["y"]) > 0}


def cell_elevation(static):
    if static is None:
        return lambda lat, lon: None
    south, west, step, rows, cols = static["grid"]
    pos = {int(i): j for j, i in enumerate(static["idx"])}
    def f(lat, lon):
        r, c = int((lat - south) // step), int((lon - west) // step)
        j = pos.get(r * int(cols) + c)
        return None if j is None else float(static["elev"][j])
    return f


def gather(A, p, t, back):
    idx = np.clip(t[:, None] - np.arange(back)[None, :], 0, A.shape[1] - 1)
    return A[p[:, None], idx]


def sample_features(h, s, peaks):
    p, t, dz = s["p"], s["t"], s["dz"]
    P = gather(h["P"], p, t, 21)
    f = {
        "deficit": gather(h["ET"], p, t, 7).sum(1) - P[:, :7].sum(1),
        "tmin": np.nanmean(gather(h["TN"], p, t, 7), 1) - M.LAPSE * dz,
        "tmax": np.nanmean(gather(h["TX"], p, t, 7), 1) - M.LAPSE * dz,
        "frost": np.nanmin(gather(h["TN"], p, t, 3), 1) - M.LAPSE * dz,
    }
    f["rain"] = {pk: P @ M.kernel(pk) for pk in peaks}
    return f


def weather_score(f, g, prm):
    return (M.temp_factor(f["tmin"], f["tmax"], f["frost"], g, prm)
            * M.water_factor(f["rain"][prm["peak"]], None, f["deficit"], prm))


# ---------------- taratura ----------------
def fit(h, samples, groups):
    peaks, scales, dries, shifts = [5, 6, 7, 8, 10, 12, 14], [10, 15, 25, 40, 60], [0, 15, 25, 40], [-3, -1.5, 0, 1.5, 3]
    gmap = {g["id"]: g for g in groups}
    feats = {gid: sample_features(h, s, peaks) for gid, s in samples.items()}
    n_cases = {gid: int(s["y"].sum()) for gid, s in samples.items()}
    usable = [gid for gid, n in n_cases.items() if n >= 15]
    if not usable:
        return None

    def group_auc(gid, prm):
        y = samples[gid]["y"]
        sc = weather_score(feats[gid], gmap[gid], prm)
        return M.auc(sc[y == 1], sc[y == 0])

    def pooled(prm):
        w = np.array([n_cases[g] for g in usable], float)
        a = np.array([group_auc(g, prm) for g in usable])
        ok = np.isfinite(a)
        return float(np.average(a[ok], weights=w[ok])) if ok.any() else float("nan")

    base = dict(M.DEFAULTS)
    auc_default = pooled(base)
    best, best_auc = base, auc_default
    for pk, sc, dk in itertools.product(peaks, scales, dries):
        prm = {**base, "peak": pk, "rain_scale": sc, "dry_k": dk}
        a = pooled(prm)
        if a > best_auc + 1e-9:
            best, best_auc = prm, a
    if best_auc - auc_default < 0.01:          # miglioramento trascurabile: tengo i default
        best, best_auc = base, auc_default
    log(f"Globale: AUC {auc_default:.3f} → {best_auc:.3f} con {best}")

    out = {"global": {"peak": best["peak"], "rain_scale": best["rain_scale"], "dry_k": best["dry_k"],
                      "auc": round(best_auc, 3), "auc_default": round(auc_default, 3),
                      "n": int(sum(n_cases[g] for g in usable))},
           "groups": {}}
    for gid in samples:
        entry = {"n": n_cases[gid]}
        if gid in usable:
            a0 = group_auc(gid, best)
            gbest, ga = {}, a0
            if n_cases[gid] >= 40:
                for pk, ts in itertools.product(peaks, shifts):
                    a = group_auc(gid, {**best, "peak": pk, "tshift": ts})
                    if a > ga + 1e-9:
                        gbest, ga = {"peak": pk, "tshift": ts}, a
                if ga - a0 < 0.02:
                    gbest, ga = {}, a0
            entry.update(gbest)
            entry["auc"] = round(ga, 3)
            log(f"  {gid}: n={n_cases[gid]} AUC {a0:.3f} → {ga:.3f} {gbest or ''}")
        out["groups"][gid] = entry
    return out


def main():
    cfg, groups = load_config(), load_groups()
    static_path = os.path.join(DATA, "static.npz")
    static = dict(np.load(static_path)) if os.path.isfile(static_path) else None
    today = date.today()
    start_guess = date(today.year - cfg["backtest"]["years"], 1, 1)

    log("Ritrovamenti storici…")
    if static is None:
        raise SystemExit("Manca data/static.npz: lancia prima «Preparazione dati statici».")
    cases = fetch_cases(cfg, groups, start_guess, today, area_filter(static))
    h = load_history(cfg, cases)
    if len(h["points"]) == 0:
        warn("nessun ritrovamento utilizzabile in regione")
        return
    if not h["done"].all():
        log("Scarico meteo storico…")
        complete = fetch_history(h, cfg)
        save_history(h)
        if not complete:
            log(f"Archivio incompleto ({int(h['done'].sum())}/{len(h['done'])}): riprendo alla prossima esecuzione.")
            return

    if os.path.isfile(MODEL):
        old = json.load(open(MODEL))
        if old.get("_history") == str(h["created"]) and old.get("_cases") == len(cases):
            log("Modello già aggiornato.")
            return

    samples = build_samples(h, cases, cfg, groups, static)
    log(f"Campioni: {sum(int(s['y'].sum()) for s in samples.values())} casi, "
        f"{sum(int((s['y'] == 0).sum()) for s in samples.values())} controlli")
    result = fit(h, samples, groups)
    if result is None:
        warn("troppo pochi ritrovamenti per tarare il modello")
        return
    result["_generated"] = today.isoformat()
    result["_history"] = str(h["created"])
    result["_cases"] = len(cases)
    with open(MODEL, "w") as f:
        json.dump(result, f, indent=1)
    log("Modello salvato.")


if __name__ == "__main__":
    main()
