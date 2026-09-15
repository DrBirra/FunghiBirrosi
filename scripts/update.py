#!/usr/bin/env python3
"""Aggiornamento giornaliero: meteo (Open-Meteo) + osservazioni (iNaturalist) -> data/latest.json
Solo libreria standard, nessuna dipendenza."""
import json, os, sys, time, urllib.parse, urllib.request
from datetime import date, datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = "funghi-map/1.0 (github actions)"


def get_json(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))


# ---------- griglia ----------
def point_in_polygon(lon, lat, ring):
    inside, j = False, len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def load_polygons(path):
    if not path or not os.path.isfile(path):
        return None
    gj = json.load(open(path))
    feats = gj.get("features", [gj])
    polys = []
    for f in feats:
        g = f.get("geometry", f)
        if g["type"] == "Polygon":
            polys.append(g["coordinates"][0])
        elif g["type"] == "MultiPolygon":
            polys += [p[0] for p in g["coordinates"]]
    return polys


def build_grid(cfg):
    b, step = cfg["bbox"], cfg["grid_step_deg"]
    polys = load_polygons(os.path.join(ROOT, cfg.get("boundary_geojson", "")))
    pts, lat = [], b["south"] + step / 2
    while lat < b["north"]:
        lon = b["west"] + step / 2
        while lon < b["east"]:
            if polys is None or any(point_in_polygon(lon, lat, r) for r in polys):
                pts.append((round(lat, 4), round(lon, 4)))
            lon += step
        lat += step
    return pts


# ---------- indice di fruttificazione (euristico) ----------
def trap(x, lo, best_lo, best_hi, hi):
    """Trapezio: 0 fuori da [lo,hi], 1 dentro [best_lo,best_hi]."""
    if x is None or x <= lo or x >= hi:
        return 0.0
    if best_lo <= x <= best_hi:
        return 1.0
    return (x - lo) / (best_lo - lo) if x < best_lo else (hi - x) / (hi - best_hi)


def avg(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def window(arr, today_idx, days_ago_from, days_ago_to):
    """Valori da `days_ago_to` a `days_ago_from` giorni fa (inclusi)."""
    return arr[max(0, today_idx - days_ago_to): today_idx - days_ago_from + 1]


def month_factor(months, today):
    """1 in stagione, 0.4 nel mese prima/dopo, 0 fuori stagione."""
    m = today.month
    if m in months:
        return 1.0
    if (m % 12) + 1 in months or ((m - 2) % 12) + 1 in months:
        return 0.4
    return 0.0


def elev_factor(elev, rng):
    lo, hi = rng
    if lo <= elev <= hi:
        return 1.0
    dist = lo - elev if elev < lo else elev - hi
    return max(0.0, 1 - dist / 300)


def weather_summary(daily, soil_daily, today_idx):
    p, tmin, tmax = daily["precipitation_sum"], daily["temperature_2m_min"], daily["temperature_2m_max"]
    return {
        "rain_5_14": sum(x or 0 for x in window(p, today_idx, 5, 14)),
        "rain_0_4": sum(x or 0 for x in window(p, today_idx, 0, 4)),
        "tmin7": avg(window(tmin, today_idx, 0, 6)),
        "tmax7": avg(window(tmax, today_idx, 0, 6)),
        "soil7": avg(window(soil_daily, today_idx, 0, 6)),
    }


def group_score(w, elev, g, today):
    """La fruttificazione segue di ~5-14 giorni una pioggia significativa.
    Temperature, quota e stagione dipendono dalla specie (species.json)."""
    s_rain = min(w["rain_5_14"] / 40.0, 1.0)       # ~40 mm = ottimo
    s_recent = min(w["rain_0_4"] / 15.0, 1.0)
    s_soil = trap(w["soil7"], 0.08, 0.25, 0.45, 0.6)
    tmax7 = w["tmax7"] or 0
    heat = 1.0 if tmax7 < g["tmax_max"] else max(0.0, 1 - (tmax7 - g["tmax_max"]) / 6)
    s_temp = trap(w["tmin7"], *g["tmin"]) * heat
    gate = month_factor(g["months"], today) * elev_factor(elev, g["elev"])
    return round(100 * gate * s_temp * (0.5 * s_rain + 0.3 * s_soil + 0.2 * s_recent))


def fetch_weather(points, cfg, groups):
    today = date.today()
    cells, batch = [], 80
    for i in range(0, len(points), batch):
        chunk = points[i:i + batch]
        q = urllib.parse.urlencode({
            "latitude": ",".join(str(p[0]) for p in chunk),
            "longitude": ",".join(str(p[1]) for p in chunk),
            "daily": "precipitation_sum,temperature_2m_min,temperature_2m_max",
            "hourly": "soil_moisture_3_to_9cm",
            "past_days": cfg["past_days"], "forecast_days": 1,
            "timezone": cfg["timezone"],
        })
        print(f"Open-Meteo batch {i // batch + 1} ({len(chunk)} punti)")
        res = get_json("https://api.open-meteo.com/v1/forecast?" + q)
        res = res if isinstance(res, list) else [res]
        for (lat, lon), r in zip(chunk, res):
            elev = r.get("elevation") or 0
            if elev < cfg.get("min_elevation_m", 1):
                continue  # mare / lagune
            d = r["daily"]
            hs = r["hourly"]["soil_moisture_3_to_9cm"]
            soil_daily = [avg(hs[k:k + 24]) for k in range(0, len(hs), 24)]
            w = weather_summary(d, soil_daily, len(d["time"]) - 1)
            cell = {"lat": lat, "lon": lon, "elev": round(elev),
                    **{k: (None if v is None else round(v, 3 if k == "soil7" else 1)) for k, v in w.items()},
                    "s": {g["id"]: group_score(w, elev, g, today) for g in groups}}
            cells.append(cell)
        time.sleep(1.5)
    return cells


def resolve_taxa(groups):
    """Nome scientifico -> ID iNaturalist, con cache in data/taxa.json."""
    cache_path = os.path.join(ROOT, "data", "taxa.json")
    cache = json.load(open(cache_path)) if os.path.isfile(cache_path) else {}
    for g in groups:
        for name in g["taxa"]:
            if name in cache:
                continue
            q = urllib.parse.urlencode({"q": name, "is_active": "true", "per_page": 10})
            res = get_json("https://api.inaturalist.org/v1/taxa?" + q).get("results", [])
            match = next((t for t in res if t.get("name", "").lower() == name.lower()), None)
            if match:
                cache[name] = match["id"]
                print(f"  taxon {name} -> {match['id']}")
            else:
                print(f"  ATTENZIONE: taxon non trovato: {name}", file=sys.stderr)
            time.sleep(1.2)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=1, ensure_ascii=False)
    return {tid: g["id"] for g in groups for n in g["taxa"] if (tid := cache.get(n))}


def fetch_observations(cfg, groups):
    b = cfg["bbox"]
    taxon_to_group = resolve_taxa(groups)
    if not taxon_to_group:
        return [], []
    d1 = (date.today() - timedelta(days=cfg["inat_days_back"])).isoformat()
    obs, counts = [], {g["id"]: 0 for g in groups}
    for page in range(1, cfg["inat_max_pages"] + 1):
        q = urllib.parse.urlencode({
            "taxon_id": ",".join(str(t) for t in taxon_to_group), "swlat": b["south"], "swlng": b["west"],
            "nelat": b["north"], "nelng": b["east"], "d1": d1,
            "geo": "true", "per_page": 200, "page": page,
            "order_by": "observed_on", "locale": "it",
        })
        print(f"iNaturalist pagina {page}")
        data = get_json("https://api.inaturalist.org/v1/observations?" + q)
        for o in data.get("results", []):
            gj, t = o.get("geojson"), o.get("taxon") or {}
            if not gj:
                continue
            lineage = [t.get("id")] + list(t.get("ancestor_ids") or [])
            group = next((taxon_to_group[i] for i in lineage if i in taxon_to_group), None)
            if not group:
                continue
            photo = ((o.get("photos") or [{}])[0].get("url") or "").replace("square", "small")
            obs.append({
                "lat": gj["coordinates"][1], "lon": gj["coordinates"][0],
                "date": o.get("observed_on"), "group": group,
                "name": t.get("name"), "common": t.get("preferred_common_name"),
                "confirmed": o.get("quality_grade") == "research",
                "photo": photo, "url": o.get("uri"),
            })
            counts[group] += 1
        if page * 200 >= data.get("total_results", 0):
            break
        time.sleep(1.2)  # iNaturalist chiede ~1 richiesta/secondo
    return obs, counts


def main():
    cfg = json.load(open(os.path.join(ROOT, "config.json")))
    groups = json.load(open(os.path.join(ROOT, "species.json")))["groups"]
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    points = build_grid(cfg)
    print(f"{len(points)} punti griglia")
    cells = fetch_weather(points, cfg, groups)
    try:
        obs, counts = fetch_observations(cfg, groups)
    except Exception as e:  # se iNaturalist è giù, pubblichiamo comunque il meteo
        print(f"iNaturalist non disponibile: {e}", file=sys.stderr)
        obs, counts = [], {}
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "region": cfg["region_name"], "bbox": cfg["bbox"], "grid_step": cfg["grid_step_deg"],
        "cells": cells, "observations": obs, "counts": counts,
    }
    path = os.path.join(ROOT, "data", "latest.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Scritto {path}: {len(cells)} celle, {len(obs)} osservazioni")


if __name__ == "__main__":
    main()
