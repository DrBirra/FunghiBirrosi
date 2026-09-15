#!/usr/bin/env python3
"""Preparazione dei dati statici (da lanciare una volta e poi ogni tanto).

Per ogni cella da ~1 km della regione calcola:
  - quota media, pendenza ed esposizione (Copernicus DEM GLO-90)
  - frazione di bosco, prati/arbusteti, coltivi, urbano, acqua (ESA WorldCover 10 m)
e scarta le celle senza habitat utile. Poi calibra quote e stagionalità
di ogni specie sulle osservazioni storiche confermate di iNaturalist.

Output: data/static.npz (per lo script giornaliero), data/cells.json (per la mappa),
        data/calibration.json
"""
import json, math, os, time, urllib.parse
from datetime import date

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.windows import from_bounds
from rasterio.errors import RasterioIOError
from shapely import contains_xy
from shapely.geometry import shape
from shapely.ops import unary_union

from common import DATA, ROOT, area_name, get_bytes, get_json, group_of, load_config, load_groups, log, resolve_taxa, warn

M_PER_DEG = 111_320


class Grid:
    def __init__(self, bbox, step):
        self.south, self.west, self.step = bbox["south"], bbox["west"], step
        self.rows = math.ceil(round((bbox["north"] - bbox["south"]) / step, 6))
        self.cols = math.ceil(round((bbox["east"] - bbox["west"]) / step, 6))
        self.n = self.rows * self.cols
        self.bbox = bbox

    def flat_index(self, lat, lon):
        r = np.floor((lat - self.south) / self.step).astype(np.int64)
        c = np.floor((lon - self.west) / self.step).astype(np.int64)
        ok = (r >= 0) & (r < self.rows) & (c >= 0) & (c < self.cols)
        return np.where(ok, r * self.cols + c, -1)

    def centers(self):
        r, c = np.divmod(np.arange(self.n), self.cols)
        return self.south + (r + 0.5) * self.step, self.west + (c + 0.5) * self.step


def pixel_centers(transform, shape_):
    h, w = shape_
    cols = transform.c + (np.arange(w) + 0.5) * transform.a
    rows = transform.f + (np.arange(h) + 0.5) * transform.e
    return np.meshgrid(rows, cols, indexing="ij")  # lat, lon


def read_window(url, bbox, decimate=1):
    """Legge solo la parte del tile dentro il bbox, opzionalmente sottocampionata
    (i COG hanno overview, quindi si scarica poco)."""
    with rasterio.open(url) as src:
        win = from_bounds(bbox["west"], bbox["south"], bbox["east"], bbox["north"], src.transform)
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        win = win.round_offsets().round_lengths()
        if win.width < 1 or win.height < 1:
            return None, None, None
        out_shape = (max(1, int(win.height // decimate)), max(1, int(win.width // decimate)))
        arr = src.read(1, window=win, out_shape=out_shape)
        t = src.window_transform(win) * rasterio.Affine.scale(win.width / out_shape[1], win.height / out_shape[0])
        return arr, t, src.nodata


def dem_tiles(bbox):
    for la in range(math.floor(bbox["south"]), math.ceil(bbox["north"])):
        for lo in range(math.floor(bbox["west"]), math.ceil(bbox["east"])):
            yield (f"Copernicus_DSM_COG_30_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_"
                   f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM")


def landcover_tiles(bbox):
    for la in range(math.floor(bbox["south"] / 3) * 3, math.ceil(bbox["north"]), 3):
        for lo in range(math.floor(bbox["west"] / 3) * 3, math.ceil(bbox["east"]), 3):
            yield f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"


def terrain(grid, cfg):
    """Quota media e vettore gradiente medio per cella."""
    s_elev = np.zeros(grid.n); s_gx = np.zeros(grid.n); s_gy = np.zeros(grid.n)
    s_slope = np.zeros(grid.n); cnt = np.zeros(grid.n)
    for name in dem_tiles(grid.bbox):
        url = cfg["sources"]["dem"].format(name=name)
        try:
            arr, t, nodata = read_window(url, grid.bbox)
        except RasterioIOError:
            log(f"  DEM {name}: assente (mare)")
            continue
        if arr is None:
            continue
        log(f"  DEM {name}: {arr.shape}")
        z = arr.astype("float64")
        if nodata is not None:
            z[z == nodata] = np.nan
        lat, lon = pixel_centers(t, z.shape)
        dy = abs(t.e) * M_PER_DEG
        dx = t.a * M_PER_DEG * np.cos(np.radians(lat))
        gy_rows, gx_cols = np.gradient(z)
        gx = gx_cols / dx                      # dz verso est
        gy = -gy_rows / dy                     # dz verso nord (le righe scendono verso sud)
        idx = grid.flat_index(lat, lon).ravel()
        ok = (idx >= 0) & np.isfinite(z.ravel()) & np.isfinite(gx.ravel()) & np.isfinite(gy.ravel())
        i = idx[ok]
        s_elev += np.bincount(i, z.ravel()[ok], grid.n)
        s_gx += np.bincount(i, gx.ravel()[ok], grid.n)
        s_gy += np.bincount(i, gy.ravel()[ok], grid.n)
        s_slope += np.bincount(i, np.hypot(gx, gy).ravel()[ok], grid.n)
        cnt += np.bincount(i, minlength=grid.n)
    with np.errstate(invalid="ignore", divide="ignore"):
        elev = s_elev / cnt
        gx, gy = s_gx / cnt, s_gy / cnt
        slope = np.degrees(np.arctan(s_slope / cnt))
    # esposizione = direzione verso cui scende il versante (0 = nord, 90 = est)
    aspect = (np.degrees(np.arctan2(-gx, -gy)) + 360) % 360
    return elev, slope, aspect, cnt > 0


LC_CLASSES = {"tree": [10], "open": [20, 30], "crop": [40], "built": [50], "water": [80, 90]}


def landcover(grid, cfg):
    counts = {k: np.zeros(grid.n) for k in LC_CLASSES}
    total = np.zeros(grid.n)
    for tile in landcover_tiles(grid.bbox):
        url = cfg["sources"]["landcover"].format(tile=tile)
        try:
            arr, t, _ = read_window(url, grid.bbox, decimate=8)   # ~80 m
        except RasterioIOError:
            warn(f"WorldCover {tile} non disponibile")
            continue
        if arr is None:
            continue
        log(f"  WorldCover {tile}: {arr.shape}")
        lat, lon = pixel_centers(t, arr.shape)
        idx = grid.flat_index(lat, lon).ravel()
        v = arr.ravel()
        ok = (idx >= 0) & (v > 0)
        total += np.bincount(idx[ok], minlength=grid.n)
        for k, cls in LC_CLASSES.items():
            m = ok & np.isin(v, cls)
            counts[k] += np.bincount(idx[m], minlength=grid.n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {k: np.nan_to_num(c / total) for k, c in counts.items()}


def leaf_type(grid, cfg):
    """Frazione di latifoglie e conifere (Copernicus HRL Dominant Leaf Type 2018),
    scaricata a blocchi da 1° a ~100 m. Se il servizio non risponde il modello
    tratta il tipo di bosco come sconosciuto."""
    broad = np.zeros(grid.n); conif = np.zeros(grid.n)
    b = grid.bbox
    url_t = cfg["sources"].get("leaftype")
    if not url_t:
        return broad, conif, False
    ok_any = False
    for la in range(math.floor(b["south"]), math.ceil(b["north"])):
        for lo in range(math.floor(b["west"]), math.ceil(b["east"])):
            s, n = max(la, b["south"]), min(la + 1, b["north"])
            w, e = max(lo, b["west"]), min(lo + 1, b["east"])
            W, H = max(1, round((e - w) / 0.001)), max(1, round((n - s) / 0.001))
            url = url_t.format(west=w, south=s, east=e, north=n, w=W, h=H)
            try:
                raw = get_bytes(url)
                with MemoryFile(raw) as mf, mf.open() as src:
                    arr = src.read(1)
            except Exception as ex:
                warn(f"tipo di bosco {la}N {lo}E non disponibile: {ex}")
                continue
            t = transform_from_bounds(w, s, e, n, arr.shape[1], arr.shape[0])
            lat, lon = pixel_centers(t, arr.shape)
            idx = grid.flat_index(lat, lon).ravel()
            v = arr.ravel()
            m = idx >= 0
            broad += np.bincount(idx[m & (v == 1)], minlength=grid.n)
            conif += np.bincount(idx[m & (v == 2)], minlength=grid.n)
            ok_any = True
            log(f"  tipo di bosco {la}N {lo}E: {arr.shape}")
    tot = broad + conif
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(broad / tot), np.nan_to_num(conif / tot), ok_any


def build_area(cfg):
    """Scarica i confini ISTAT/openpolis, tiene le regioni scelte e ricava il riquadro."""
    path = os.path.join(DATA, "region.geojson")
    if cfg.get("regions"):
        gj = json.loads(get_bytes(cfg["sources"]["regions"]))
        wanted = {r.lower() for r in cfg["regions"]}
        feats = [f for f in gj["features"] if f["properties"].get("reg_name", "").lower() in wanted]
        found = {f["properties"]["reg_name"] for f in feats}
        missing = [r for r in cfg["regions"] if r.lower() not in {x.lower() for x in found}]
        if missing:
            names = sorted(f["properties"]["reg_name"] for f in gj["features"])
            raise SystemExit(f"Regioni non trovate: {missing}. Nomi validi: {names}")
        with open(path, "w") as fp:
            json.dump({"type": "FeatureCollection", "features": feats}, fp, separators=(",", ":"))
        name = area_name(cfg["regions"])
    elif os.path.isfile(path) and "bbox" in cfg:
        feats, name = json.load(open(path))["features"], cfg.get("region_name", "Area")
    else:
        raise SystemExit("Indica le regioni in config.json (\"regions\")")
    geom = unary_union([shape(f["geometry"]) for f in feats])
    w, s_, e, n = geom.bounds
    bbox = {"south": math.floor(s_ * 100) / 100, "west": math.floor(w * 100) / 100,
            "north": math.ceil(n * 100) / 100, "east": math.ceil(e * 100) / 100}
    with open(os.path.join(DATA, "area.json"), "w") as fp:
        json.dump({"regions": cfg.get("regions"), "name": name, "bbox": bbox}, fp, ensure_ascii=False)
    cfg["bbox"], cfg["region_name"] = bbox, name
    log(f"Area: {name}, {bbox}")
    return geom


def region_mask(grid, geom):
    lat, lon = grid.centers()
    return contains_xy(geom, lon, lat)


def calibrate(grid, elev_all, inside, cfg, groups):
    """Quote (5°-95° percentile) e stagionalità dalle osservazioni confermate in regione."""
    t2g = resolve_taxa(groups)
    b = grid.bbox
    elevs = {g["id"]: [] for g in groups}
    months = {g["id"]: np.zeros(12) for g in groups}
    for g in groups:
        ids = [str(t) for t, gid in t2g.items() if gid == g["id"]]
        if not ids:
            continue
        for page in range(1, cfg["calibration_max_pages"] + 1):
            q = urllib.parse.urlencode({
                "taxon_id": ",".join(ids), "quality_grade": "research", "geo": "true",
                "swlat": b["south"], "swlng": b["west"], "nelat": b["north"], "nelng": b["east"],
                "per_page": 200, "page": page, "order": "desc", "order_by": "id",
            })
            data = get_json("https://api.inaturalist.org/v1/observations?" + q)
            for o in data.get("results", []):
                if not o.get("geojson") or not o.get("observed_on") or group_of(o.get("taxon") or {}, t2g) != g["id"]:
                    continue
                if (o.get("positional_accuracy") or 0) > 2000 or o.get("obscured"):
                    continue  # coordinate troppo imprecise per stimare la quota
                lon, lat = o["geojson"]["coordinates"]
                i = grid.flat_index(np.array([lat]), np.array([lon]))[0]
                if i < 0 or not inside[i]:
                    continue
                if np.isfinite(elev_all[i]):
                    elevs[g["id"]].append(float(elev_all[i]))
                months[g["id"]][int(o["observed_on"][5:7]) - 1] += 1
            time.sleep(1.2)
            if page * 200 >= data.get("total_results", 0):
                break
    out = {}
    for g in groups:
        e, m = np.array(elevs[g["id"]]), months[g["id"]]
        entry = {"n_elev": int(e.size), "n_month": int(m.sum())}
        if e.size >= 30:
            entry["elev"] = [int(max(0, np.percentile(e, 5) - 100)), int(np.percentile(e, 95) + 100)]
        if m.sum() >= 40:
            sm = np.convolve(np.r_[m[-1], m, m[0]], [0.25, 0.5, 0.25], "valid")  # mesi circolari
            w = sm / sm.max()
            entry["month_w"] = [round(float(x), 2) if x >= 0.08 else 0 for x in w]
        out[g["id"]] = entry
        log(f"  {g['id']}: {entry}")
    return out


def main():
    cfg, groups = load_config(), load_groups()
    os.makedirs(DATA, exist_ok=True)
    geom = build_area(cfg)
    grid = Grid(cfg["bbox"], cfg["fine_step_deg"])
    log(f"Griglia {grid.rows}x{grid.cols} celle da {cfg['fine_step_deg']}°")

    log("Terreno…")
    elev, slope, aspect, has_dem = terrain(grid, cfg)
    log("Copertura del suolo…")
    lc = landcover(grid, cfg)
    log("Tipo di bosco…")
    f_broad, f_conif, has_leaf = leaf_type(grid, cfg)
    inside = region_mask(grid, geom)

    habitat = lc["tree"] + lc["open"]
    keep = inside & has_dem & (habitat >= 0.2) & (lc["built"] < 0.5) & (lc["water"] < 0.6)
    idx = np.flatnonzero(keep)
    log(f"Celle con habitat utile: {idx.size} su {int(inside.sum())} in regione")

    # cella meteo (più grossa) di appartenenza
    lat, lon = grid.centers()
    ws = cfg["weather_step_deg"]
    cr = np.floor((lat[idx] - grid.south) / ws).astype(int)
    cc = np.floor((lon[idx] - grid.west) / ws).astype(int)
    keys, k = np.unique(np.stack([cr, cc], 1), axis=0, return_inverse=True)
    w_lat = grid.south + (keys[:, 0] + 0.5) * ws
    w_lon = grid.west + (keys[:, 1] + 0.5) * ws
    days = cfg["past_days"] + cfg["forecast_days"]
    calls = len(keys) * max(1, days / 14)
    log(f"Punti meteo necessari: {len(keys)} (≈{calls:.0f} chiamate Open-Meteo al giorno)")
    if calls > 4500:
        warn("troppe chiamate meteo per il limite orario gratuito: aumenta weather_step_deg in config.json")

    os.makedirs(DATA, exist_ok=True)
    np.savez_compressed(
        os.path.join(DATA, "static.npz"),
        idx=idx, inside=np.flatnonzero(inside), elev=elev[idx], slope=slope[idx], aspect=aspect[idx],
        f_tree=lc["tree"][idx], f_open=lc["open"][idx], f_built=lc["built"][idx],
        f_broad=f_broad[idx], f_conif=f_conif[idx],
        k=k.ravel(), w_lat=w_lat, w_lon=w_lon,
        grid=np.array([grid.south, grid.west, grid.step, grid.rows, grid.cols]),
    )
    cells = {
        "grid": {"south": grid.south, "west": grid.west, "step": grid.step, "rows": grid.rows, "cols": grid.cols},
        "i": idx.tolist(), "e": np.round(elev[idx]).astype(int).tolist(),
        "sl": np.round(slope[idx]).astype(int).tolist(), "a": np.round(aspect[idx]).astype(int).tolist(),
        "ft": np.round(lc["tree"][idx] * 100).astype(int).tolist(),
        "fo": np.round(lc["open"][idx] * 100).astype(int).tolist(),
        "fb": np.round(f_broad[idx] * 100).astype(int).tolist(),
        "fc": np.round(f_conif[idx] * 100).astype(int).tolist(),
        "k": k.ravel().tolist(),
    }
    with open(os.path.join(DATA, "cells.json"), "w") as f:
        json.dump(cells, f, separators=(",", ":"))

    log("Calibrazione sulle osservazioni storiche…")
    try:
        cal = calibrate(grid, elev, inside, cfg, groups)
        cal["_generated"] = date.today().isoformat()
        with open(os.path.join(DATA, "calibration.json"), "w") as f:
            json.dump(cal, f, indent=1)
    except Exception as e:
        warn(f"calibrazione saltata: {e}")
    log("Fatto.")


if __name__ == "__main__":
    main()
