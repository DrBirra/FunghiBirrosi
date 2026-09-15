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
import json, math, os, shutil, struct, time, urllib.parse, zlib
from datetime import date

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.windows import from_bounds
from rasterio.errors import RasterioIOError
from shapely import contains_xy, intersects, prepare as shapely_prepare
from shapely.geometry import box
from shapely.geometry import shape
from shapely.ops import unary_union

import model as M
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


def raster_index(grid, t, shape_):
    """Indice piatto della cella di griglia per ogni pixel del raster (-1 = fuori)."""
    h, w = shape_
    lat = t.f + (np.arange(h) + 0.5) * t.e
    lon = t.c + (np.arange(w) + 0.5) * t.a
    r = np.floor((lat - grid.south) / grid.step).astype(np.int64)
    c = np.floor((lon - grid.west) / grid.step).astype(np.int64)
    okr = (r >= 0) & (r < grid.rows); okc = (c >= 0) & (c < grid.cols)
    idx = r[:, None] * grid.cols + c[None, :]
    return np.where(okr[:, None] & okc[None, :], idx, -1), lat


def read_window(url, bbox, decimate=1, target_res=None):
    """Legge solo la parte del tile dentro il bbox, opzionalmente sottocampionata
    (i COG hanno overview, quindi si scarica poco)."""
    with rasterio.open(url) as src:
        win = from_bounds(bbox["west"], bbox["south"], bbox["east"], bbox["north"], src.transform)
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        win = win.round_offsets().round_lengths()
        if win.width < 1 or win.height < 1:
            return None, None, None
        if target_res:   # sottocampiono fino a circa la risoluzione richiesta
            decimate = max(1, int(target_res / abs(src.transform.a)))
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


def terrain(grid, cfg, quiet=False):
    """Quota media e vettore gradiente medio per cella."""
    s_elev = np.zeros(grid.n); s_gx = np.zeros(grid.n); s_gy = np.zeros(grid.n)
    s_slope = np.zeros(grid.n); cnt = np.zeros(grid.n)
    for name in dem_tiles(grid.bbox):
        url = cfg["sources"]["dem"].format(name=name)
        try:
            arr, t, nodata = read_window(url, grid.bbox)
        except RasterioIOError:
            quiet or log(f"  DEM {name}: assente (mare)")
            continue
        if arr is None:
            continue
        quiet or log(f"  DEM {name}: {arr.shape}")
        z = arr.astype("float64")
        if nodata is not None:
            z[z == nodata] = np.nan
        idx2, lat_rows = raster_index(grid, t, z.shape)
        dy = abs(t.e) * M_PER_DEG
        dx = t.a * M_PER_DEG * np.cos(np.radians(lat_rows))[:, None]
        gy_rows, gx_cols = np.gradient(z)
        gx = gx_cols / dx                      # dz verso est
        gy = -gy_rows / dy                     # dz verso nord (le righe scendono verso sud)
        idx = idx2.ravel()
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


def landcover(grid, cfg, decimate=8, quiet=False, target_res=None):
    counts = {k: np.zeros(grid.n) for k in LC_CLASSES}
    total = np.zeros(grid.n)
    for tile in landcover_tiles(grid.bbox):
        url = cfg["sources"]["landcover"].format(tile=tile)
        try:
            arr, t, _ = read_window(url, grid.bbox, decimate=decimate, target_res=target_res)
        except RasterioIOError:
            quiet or warn(f"WorldCover {tile} non disponibile")
            continue
        if arr is None:
            continue
        quiet or log(f"  WorldCover {tile}: {arr.shape}")
        idx = raster_index(grid, t, arr.shape)[0].ravel()
        v = arr.ravel()
        ok = (idx >= 0) & (v > 0)
        total += np.bincount(idx[ok], minlength=grid.n)
        for k, cls in LC_CLASSES.items():
            m = ok & np.isin(v, cls)
            counts[k] += np.bincount(idx[m], minlength=grid.n)
        del idx, v, arr
    with np.errstate(invalid="ignore", divide="ignore"):
        return {k: np.nan_to_num(c / total) for k, c in counts.items()}


def leaf_type(grid, cfg, quiet=False):
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
            idx = raster_index(grid, t, arr.shape)[0].ravel()
            v = arr.ravel()
            m = idx >= 0
            broad += np.bincount(idx[m & (v == 1)], minlength=grid.n)
            conif += np.bincount(idx[m & (v == 2)], minlength=grid.n)
            ok_any = True
            quiet or log(f"  tipo di bosco {la}N {lo}E: {arr.shape}")
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
    parts = [(f["properties"].get("reg_name", name), shape(f["geometry"])) for f in feats]
    return geom, parts


def neighbourhood_full(grid, lc, radius=2):
    """Media nel raggio di ~2 km di bosco, ambiente naturale e urbano, su tutte le celle
    (anche quelle scartate), così un paese o una distesa di campi pesano davvero."""
    out = {}
    for name, arr in (("tree", lc["tree"]), ("nat", np.clip(lc["tree"] + lc["open"], 0, 1)), ("built", lc["built"])):
        a = arr.reshape(grid.rows, grid.cols)
        cs = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        r = np.arange(grid.rows); c = np.arange(grid.cols)
        r0, r1 = np.clip(r - radius, 0, grid.rows), np.clip(r + radius + 1, 0, grid.rows)
        c0, c1 = np.clip(c - radius, 0, grid.cols), np.clip(c + radius + 1, 0, grid.cols)
        tot = cs[np.ix_(r1, c1)] - cs[np.ix_(r0, c1)] - cs[np.ix_(r1, c0)] + cs[np.ix_(r0, c0)]
        out[name] = (tot / np.outer(r1 - r0, c1 - c0)).ravel()
    return out


def region_mask(grid, geom):
    lat, lon = grid.centers()
    return contains_xy(geom, lon, lat)


ELEV_BANDS = [0, 200, 400, 600, 800, 1000, 1300, 1600, 2000, 5000]


def fetch_fungi_background(cfg, t2g):
    """Tutte le osservazioni confermate di funghi nell'area, paginando per id.
    Servono come "sfondo": dicono dove e quando la gente fotografa funghi in generale."""
    b = cfg["bbox"]
    out, id_above = [], 0
    for page in range(cfg["background_max_pages"]):
        q = urllib.parse.urlencode({
            "taxon_id": 47170, "quality_grade": "research", "geo": "true",
            "swlat": b["south"], "swlng": b["west"], "nelat": b["north"], "nelng": b["east"],
            "per_page": 200, "order_by": "id", "order": "asc", "id_above": id_above,
        })
        res = get_json("https://api.inaturalist.org/v1/observations?" + q).get("results", [])
        if not res:
            break
        for o in res:
            if not o.get("geojson") or not o.get("observed_on"):
                continue
            lon, lat = o["geojson"]["coordinates"]
            precise = not o.get("obscured") and (o.get("positional_accuracy") or 0) <= 1000
            out.append((lat, lon, int(o["observed_on"][5:7]), group_of(o.get("taxon") or {}, t2g), precise))
        id_above = res[-1]["id"]
        if page % 25 == 0:
            log(f"  sfondo: {len(out)} osservazioni")
        time.sleep(1.1)
    else:
        warn(f"sfondo troncato a {len(out)} osservazioni: aumenta background_max_pages")
    log(f"  sfondo: {len(out)} osservazioni di funghi confermate")
    return out


def band_share(sp, al, prior, k):
    """Quota della specie tra i funghi osservati in ogni fascia, tirata verso `prior`
    quando i dati sono pochi (stima bayesiana con k osservazioni fittizie)."""
    return (sp + k * prior) / (al + k)


def to_weights(share):
    w = np.convolve(np.r_[share[0], share, share[-1]], [0.15, 0.7, 0.15], "valid")
    w = w / w.max()
    w[w < 0.15] = 0
    return w


def calibrate(grid, elev_all, region_of, region_names, cfg, groups):
    t2g = resolve_taxa(groups)
    bg = fetch_fungi_background(cfg, t2g)
    nb, nr = len(ELEV_BANDS) - 1, len(region_names)
    all_c = np.zeros((nr, nb)); sp_c = {g["id"]: np.zeros((nr, nb)) for g in groups}
    months = {g["id"]: np.zeros(12) for g in groups}
    for lat, lon, month, gid, precise in bg:
        i = grid.flat_index(np.array([lat]), np.array([lon]))[0]
        if i < 0 or region_of[i] < 0:
            continue
        if gid:
            months[gid][month - 1] += 1
        if not precise or not np.isfinite(elev_all[i]):
            continue
        band = min(nb - 1, max(0, np.searchsorted(ELEV_BANDS, elev_all[i], side="right") - 1))
        all_c[region_of[i], band] += 1
        if gid:
            sp_c[gid][region_of[i], band] += 1

    out = {"_elev_bands": ELEV_BANDS, "_regions": region_names,
           "_background": {r: int(all_c[j].sum()) for j, r in enumerate(region_names)}}
    for g in groups:
        gid, m = g["id"], months[g["id"]]
        entry = {"n": int(sp_c[gid].sum()), "n_month": int(m.sum())}
        n_sp = sp_c[gid].sum()
        if n_sp >= 25:
            # Correzione per lo sforzo di osservazione: conta la quota della specie tra TUTTI i funghi
            # fotografati in quella fascia, non il numero assoluto di foto (che è alto vicino alle città).
            p_all = n_sp / max(1, all_c.sum())
            pooled_share = band_share(sp_c[gid].sum(0), all_c.sum(0), p_all, k=30)
            pooled = to_weights(pooled_share)
            per_region = {}
            for j, r in enumerate(region_names):
                # ogni regione parte dalla stima complessiva e se ne discosta solo dove ha dati
                share = band_share(sp_c[gid][j], all_c[j], pooled_share, k=60)
                per_region[r] = [round(float(x), 2) for x in to_weights(share)]
            entry["elev_w"] = per_region
            good = np.flatnonzero(pooled >= 0.3)
            entry["elev"] = [ELEV_BANDS[good.min()], ELEV_BANDS[good.max() + 1]]
        if m.sum() >= 40:
            sm = np.convolve(np.r_[m[-1], m, m[0]], [0.25, 0.5, 0.25], "valid")
            w = sm / sm.max()
            entry["month_w"] = [round(float(x), 2) if x >= 0.08 else 0 for x in w]
        out[gid] = entry
        log(f"  {gid}: n={entry['n']} quota {entry.get('elev', 'generica')} "
            + " ".join(f"{r[:3]}={v}" for r, v in entry.get("elev_w", {}).items()))
    return out


FINE_STEP = 0.001          # ~100 m
FINE_BUFFER = 25           # celle di margine per il vicinato (~2,5 km)


def write_png_gray(path, img):
    """PNG in scala di grigi a 8 bit, riga 0 = nord. Solo libreria standard."""
    h, w = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))
    chunk = lambda tag, data: struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def fine_habitat(cfg, geom, parts, groups, cal):
    """Idoneità statica (habitat × quota × tipo di bosco × urbano) a ~100 m, per specie,
    in tile da 1°×1°. La mappa la moltiplica per l'indice meteo della cella da 1 km."""
    out_dir = os.path.join(DATA, "fine")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    for g in groups:
        os.makedirs(os.path.join(out_dir, g["id"]))
    shapely_prepare(geom)
    b = cfg["bbox"]
    buf = FINE_BUFFER * FINE_STEP
    tiles = []
    todo = [(la, lo) for la in range(math.floor(b["south"]), math.ceil(b["north"]))
            for lo in range(math.floor(b["west"]), math.ceil(b["east"]))
            if intersects(geom, box(lo, la, lo + 1, la + 1))]
    for n_tile, (la, lo) in enumerate(todo, 1):
        t0 = time.time()
        tb = {"south": la - buf, "west": lo - buf, "north": la + 1 + buf, "east": lo + 1 + buf}
        grid = Grid(tb, FINE_STEP)
        elev, slope, aspect, has_dem = terrain(grid, cfg, quiet=True)
        lc = landcover(grid, cfg, quiet=True, target_res=FINE_STEP / 3)   # ~30-40 m, almeno 6 pixel per cella
        f_broad, f_conif, _ = leaf_type(grid, cfg, quiet=True)
        nb = neighbourhood_full(grid, lc, radius=FINE_BUFFER)
        lat_c, lon_c = grid.centers()
        region = np.full(grid.n, -1, np.int8)
        for j, (_, poly) in enumerate(parts):
            shapely_prepare(poly)
            region[contains_xy(poly, lon_c, lat_c)] = j
        del lat_c, lon_c

        inner = np.zeros((grid.rows, grid.cols), bool)
        inner[FINE_BUFFER:FINE_BUFFER + 1000, FINE_BUFFER:FINE_BUFFER + 1000] = True
        sel = inner.ravel() & (region >= 0) & has_dem
        if not sel.any():
            continue
        st = {"elev": np.nan_to_num(elev[sel]), "region": region[sel],
              "f_tree": lc["tree"][sel], "f_open": lc["open"][sel], "f_built": lc["built"][sel],
              "f_broad": f_broad[sel], "f_conif": f_conif[sel],
              "nb_tree": nb["tree"][sel], "nb_nat": nb["nat"][sel], "nb_built": nb["built"][sel]}
        rr, cc = np.divmod(np.flatnonzero(sel), grid.cols)
        rr, cc = rr - FINE_BUFFER, cc - FINE_BUFFER
        for g in groups:
            v = M.habitat_factor(g, st) * M.elev_factor(g, cal, st)
            img = np.zeros((1000, 1000), np.uint8)
            img[999 - rr, cc] = np.clip(np.round(v * 255), 0, 255).astype(np.uint8)   # riga 0 = nord
            write_png_gray(os.path.join(out_dir, g["id"], f"{la}_{lo}.png"), img)
        tiles.append([la, lo])
        log(f"  dettaglio 100 m: tile {la}N {lo}E ({n_tile}/{len(todo)}) in {time.time() - t0:.0f}s")
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump({"step": FINE_STEP, "size": 1000, "tiles": tiles}, f)
    return len(tiles)


def main():
    cfg, groups = load_config(), load_groups()
    os.makedirs(DATA, exist_ok=True)
    geom, parts = build_area(cfg)
    grid = Grid(cfg["bbox"], cfg["fine_step_deg"])
    log(f"Griglia {grid.rows}x{grid.cols} celle da {cfg['fine_step_deg']}°")

    log("Terreno…")
    elev, slope, aspect, has_dem = terrain(grid, cfg)
    log("Copertura del suolo…")
    lc = landcover(grid, cfg)
    log("Tipo di bosco…")
    f_broad, f_conif, has_leaf = leaf_type(grid, cfg)
    inside = region_mask(grid, geom)
    lat_c, lon_c = grid.centers()
    region_names = [name for name, _ in parts]
    region_of = np.full(grid.n, -1, np.int8)
    for j, (_, poly) in enumerate(parts):
        region_of[contains_xy(poly, lon_c, lat_c)] = j
    nb = neighbourhood_full(grid, lc)

    habitat = lc["tree"] + lc["open"]
    keep = inside & has_dem & (habitat >= 0.03)
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
        f_broad=f_broad[idx], f_conif=f_conif[idx], region=region_of[idx],
        nb_tree=nb["tree"][idx], nb_nat=nb["nat"][idx], nb_built=nb["built"][idx],
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
        "r": region_of[idx].tolist(), "regions": region_names,
    }
    with open(os.path.join(DATA, "cells.json"), "w") as f:
        json.dump(cells, f, separators=(",", ":"))

    log("Calibrazione sulle osservazioni storiche…")
    try:
        cal = calibrate(grid, elev, region_of, region_names, cfg, groups)
        cal["_generated"] = date.today().isoformat()
        with open(os.path.join(DATA, "calibration.json"), "w") as f:
            json.dump(cal, f, indent=1)
    except Exception as e:
        warn(f"calibrazione saltata: {e}")
        cal = {}

    if cfg.get("fine_detail", True):
        log("Dettaglio a 100 m…")
        n = fine_habitat(cfg, geom, parts, groups, cal)
        log(f"  {n} tile scritti in data/fine")
    log("Fatto.")


if __name__ == "__main__":
    main()
