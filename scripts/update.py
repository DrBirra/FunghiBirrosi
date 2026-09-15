#!/usr/bin/env python3
"""Aggiornamento giornaliero con previsione a 7 giorni.

Meteo Open-Meteo (21 giorni passati + 7 di previsione) sulle celle da ~5 km, riportato
sulle celle da ~1 km con la quota. Per ogni specie e ogni giorno da oggi a +6:
  stagione × quota × habitat (bosco, latifoglie/conifere) × temperature × acqua × esposizione
con i parametri meteo tarati da backtest.py (data/model.json) quando disponibili.
"""
import json, os, time, urllib.parse, warnings
from datetime import date, datetime, timedelta, timezone

import numpy as np

import model as M
from common import DATA, area_filter, get_json, group_of, load_config, load_groups, log, resolve_taxa, warn

BATCH = 50            # punti per richiesta Open-Meteo
PAUSE = 12            # secondi tra richieste (limite ~600 chiamate/minuto)


def fetch_weather(st, cfg):
    lats, lons = st["w_lat"], st["w_lon"]
    n, past, fut = len(lats), cfg["past_days"], cfg["forecast_days"]
    days = past + fut
    P = np.zeros((n, days)); TN = np.full((n, days), np.nan); TX = np.full((n, days), np.nan)
    ET = np.zeros((n, days)); SM = np.full((n, days), np.nan); EL = np.zeros(n)
    for b in range(0, n, BATCH):
        sl = slice(b, min(n, b + BATCH))
        q = urllib.parse.urlencode({
            "latitude": ",".join(f"{x:.4f}" for x in lats[sl]),
            "longitude": ",".join(f"{x:.4f}" for x in lons[sl]),
            "daily": "precipitation_sum,temperature_2m_min,temperature_2m_max,et0_fao_evapotranspiration",
            "hourly": "soil_moisture_9_to_27cm",
            "past_days": past, "forecast_days": fut, "timezone": cfg["timezone"],
        })
        log(f"Open-Meteo {b // BATCH + 1}/{-(-n // BATCH)}")
        res = get_json("https://api.open-meteo.com/v1/forecast?" + q)
        res = res if isinstance(res, list) else [res]
        for j, r in enumerate(res):
            i = b + j
            d = r["daily"]
            f = lambda k: np.array([np.nan if v is None else v for v in d[k]], float)[:days]
            P[i], TN[i], TX[i], ET[i] = f("precipitation_sum"), f("temperature_2m_min"), f("temperature_2m_max"), f("et0_fao_evapotranspiration")
            h = np.array([np.nan if v is None else v for v in r["hourly"]["soil_moisture_9_to_27cm"]], float)[: days * 24]
            h = np.pad(h, (0, days * 24 - h.size), constant_values=np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # giorni senza dati orari
                SM[i] = np.nanmean(h.reshape(days, 24), axis=1)
            EL[i] = r.get("elevation") or 0
        if b + BATCH < n:
            time.sleep(PAUSE)
    # l'umidità del suolo prevista a volte manca negli ultimi giorni: porto avanti l'ultimo valore
    for i in range(n):
        last = np.nan
        for t in range(days):
            if np.isnan(SM[i, t]):
                SM[i, t] = last
            else:
                last = SM[i, t]
    return {"P": np.nan_to_num(P), "TN": TN, "TX": TX, "ET": np.nan_to_num(ET), "SM": SM, "EL": EL}


def features(w, t, peak):
    return {
        "rain": M.eff_rain(w["P"], t, peak),
        "soil": M.window_mean(w["SM"], t, 5),
        "deficit": M.window_sum(w["ET"], t, 7) - M.window_sum(w["P"], t, 7),
        "tmin": M.window_mean(w["TN"], t, 7),
        "tmax": M.window_mean(w["TX"], t, 7),
        "frost": M.window_min(w["TN"], t, 3),
    }


def reliability(day_offset, peak=8):
    """Quanto l'indice dipende da dati previsti invece che misurati (0 = solo misurati)."""
    K = M.kernel(peak)
    rain_future = K[:day_offset].sum() / K.sum() if day_offset > 0 else 0.0
    temp_future = min(day_offset, 7) / 7
    r = 1 - (0.6 * rain_future + 0.4 * temp_future) * 1.6
    return max(0.0, min(1.0, r))


def score_day(g, st, w, t, day, cal, p, feat_cache):
    key = (p["peak"], t)
    if key not in feat_cache:
        feat_cache[key] = features(w, t, p["peak"])
    f = feat_cache[key]
    k = st["k"]
    dz = st["elev"] - st["w_elev"][k]
    tmin = f["tmin"][k] - M.LAPSE * dz
    tmax = f["tmax"][k] - M.LAPSE * dz
    frost = f["frost"][k] - M.LAPSE * dz
    s = (M.month_weight(g, cal, day) * M.elev_factor(g, cal, st) * M.habitat_factor(g, st)
         * M.temp_factor(tmin, tmax, frost, g, p)
         * M.water_factor(f["rain"][k], f["soil"][k], f["deficit"][k], p)
         * M.aspect_factor(st, tmin, tmax, g))
    return np.clip(100 * s, 0, 100)


# ---------------- zone ----------------
MAX_ZONE_KM2 = 25


def components(cells, cols, rows):
    """Gruppi di celle adiacenti (8 vicini). cells: set di indici piatti."""
    seen, out = set(), []
    for start in cells:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            cur = stack.pop(); comp.append(cur)
            r, c = divmod(cur, cols)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    nxt = nr * cols + nc
                    if 0 <= nr < rows and 0 <= nc < cols and nxt in cells and nxt not in seen:
                        seen.add(nxt); stack.append(nxt)
        out.append(comp)
    return out


def hotspots(st, grid, sc, nxt, max_zones=8):
    """Zone compatte: se un gruppo è troppo esteso alzo la soglia al suo interno
    finché si divide in nuclei più piccoli."""
    south, west, step, rows, cols = grid[0], grid[1], grid[2], int(grid[3]), int(grid[4])
    if not np.any(sc > 0):
        return []
    pos = {int(i): j for j, i in enumerate(st["idx"])}
    cell_km2 = (step * 111.32) ** 2 * np.cos(np.radians(south + rows * step / 2))
    thr0 = max(50.0, float(np.percentile(sc[sc > 0], 90)))
    queue = [({i for i, j in pos.items() if sc[j] >= thr0}, thr0)]
    zones = []
    while queue:
        cells, thr = queue.pop()
        for comp in components(cells, cols, rows):
            j = np.array([pos[i] for i in comp])
            if j.size < 2:
                continue
            if j.size * cell_km2 > MAX_ZONE_KM2:
                if thr + 4 < sc[j].max():      # prima provo a separare i nuclei migliori
                    queue.append(({i for i in comp if sc[pos[i]] >= thr + 4}, thr + 4))
                    continue
                # punteggi uniformi: divido in blocchi da ~5x5 celle
                blocks = {}
                for i in comp:
                    r, c = divmod(i, cols)
                    blocks.setdefault((r // 5, c // 5), set()).add(i)
                if len(blocks) > 1:
                    for bl in blocks.values():
                        queue.append((bl, 101))    # soglia 101: niente ulteriori divisioni
                    continue
            r, c = np.divmod(st["idx"][j], cols)
            lat = south + (r + 0.5) * step; lon = west + (c + 0.5) * step
            zones.append({
                "lat": round(float(np.average(lat, weights=sc[j])), 4),
                "lon": round(float(np.average(lon, weights=sc[j])), 4),
                "bounds": [round(float(lat.min() - step / 2), 4), round(float(lon.min() - step / 2), 4),
                           round(float(lat.max() + step / 2), 4), round(float(lon.max() + step / 2), 4)],
                "km2": round(float(j.size * cell_km2), 1),
                "max": int(round(sc[j].max())), "mean": int(round(sc[j].mean())),
                "elev": [int(st["elev"][j].min()), int(st["elev"][j].max())],
                "trend": int(np.round(nxt[j].mean() - sc[j].mean())),
            })
    zones.sort(key=lambda z: z["mean"] + 3 * np.log1p(z["km2"]), reverse=True)
    return zones[:max_zones]


# ---------------- osservazioni recenti ----------------
def fetch_observations(cfg, groups, in_area):
    b = cfg["bbox"]
    t2g = resolve_taxa(groups)
    if not t2g:
        return [], {}
    d1 = (date.today() - timedelta(days=cfg["inat_days_back"])).isoformat()
    obs, counts = [], {g["id"]: 0 for g in groups}
    for page in range(1, cfg["inat_max_pages"] + 1):
        q = urllib.parse.urlencode({
            "taxon_id": ",".join(str(t) for t in t2g), "swlat": b["south"], "swlng": b["west"],
            "nelat": b["north"], "nelng": b["east"], "d1": d1, "geo": "true",
            "per_page": 200, "page": page, "order_by": "observed_on", "locale": "it",
        })
        data = get_json("https://api.inaturalist.org/v1/observations?" + q)
        for o in data.get("results", []):
            t = o.get("taxon") or {}
            gid = group_of(t, t2g)
            if not o.get("geojson") or not gid or not in_area(o["geojson"]["coordinates"][1], o["geojson"]["coordinates"][0]):
                continue
            obs.append({
                "lat": o["geojson"]["coordinates"][1], "lon": o["geojson"]["coordinates"][0],
                "date": o.get("observed_on"), "group": gid, "name": t.get("name"),
                "common": t.get("preferred_common_name"), "confirmed": o.get("quality_grade") == "research",
                "photo": ((o.get("photos") or [{}])[0].get("url") or "").replace("square", "small"),
                "url": o.get("uri"),
            })
            counts[gid] += 1
        if page * 200 >= data.get("total_results", 0):
            break
        time.sleep(1.2)
    return obs, counts


def main():
    cfg, groups = load_config(), load_groups()
    static_path = os.path.join(DATA, "static.npz")
    if not os.path.isfile(static_path):
        raise SystemExit("Manca data/static.npz: lancia prima il workflow «Preparazione dati statici».")
    st = dict(np.load(static_path))
    load = lambda name: json.load(open(os.path.join(DATA, name))) if os.path.isfile(os.path.join(DATA, name)) else {}
    cal, model = load("calibration.json"), load("model.json")
    grid = st["grid"]
    ndays = cfg["forecast_days"]
    log(f"{st['idx'].size} celle, {st['w_lat'].size} punti meteo, {ndays} giorni")

    w = fetch_weather(st, cfg)
    st["w_elev"] = w["EL"]
    M.neighbourhood(st)
    t0 = cfg["past_days"]
    today = date.today()
    cache = {}

    os.makedirs(os.path.join(DATA, "layers"), exist_ok=True)
    zones = {}
    for g in groups:
        p = M.params_for(g, model)
        days = [score_day(g, st, w, t0 + d, today + timedelta(days=d), cal, p, cache) for d in range(ndays)]
        S = np.stack(days)                                  # (giorni, celle)
        keep = S.max(0) >= 5
        with open(os.path.join(DATA, "layers", f"{g['id']}.json"), "w") as f:
            json.dump({"i": st["idx"][keep].tolist(), "s": np.round(S[:, keep]).astype(int).tolist()}, f, separators=(",", ":"))
        zones[g["id"]] = [hotspots(st, grid, S[d], S[min(d + 1, ndays - 1)]) for d in range(ndays)]
        log(f"  {g['id']}: oggi max {S[0].max():.0f}, fra 6 giorni max {S[-1].max():.0f}")

    try:
        obs, counts = fetch_observations(cfg, groups, area_filter(st))
    except Exception as e:
        warn(f"iNaturalist non disponibile: {e}")
        obs, counts = [], {}

    r1 = lambda a, n=1: np.round(np.nan_to_num(a), n).tolist()
    weather_days = []
    for d in range(ndays):
        f = features(w, t0 + d, M.DEFAULTS["peak"])
        weather_days.append({
            "date": (today + timedelta(days=d)).isoformat(),
            "reliability": round(reliability(d), 2),
            "rain": r1(f["rain"]), "soil": r1(f["soil"], 3), "tmin": r1(f["tmin"]), "tmax": r1(f["tmax"]),
            "day_rain": r1(w["P"][:, t0 + d]),
        })
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "region": cfg["region_name"], "bbox": cfg["bbox"],
        "grid": {"south": float(grid[0]), "west": float(grid[1]), "step": float(grid[2]), "rows": int(grid[3]), "cols": int(grid[4])},
        "w_elev": r1(w["EL"], 0), "days": weather_days,
        "zones": zones, "counts": counts, "observations": obs,
        "model": {"generated": model.get("_generated"), "global": model.get("global"),
                  "groups": {k: {kk: v[kk] for kk in ("auc", "n") if kk in v} for k, v in model.get("groups", {}).items()}},
    }
    with open(os.path.join(DATA, "latest.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    log("Fatto.")


if __name__ == "__main__":
    main()
