import json, os, sys, time, urllib.error, urllib.parse, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
UA = "funghi-map/2.0 (github actions)"


def log(*a):
    print(*a, flush=True)


def warn(*a):
    print("ATTENZIONE:", *a, file=sys.stderr, flush=True)


def get_json(url, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if attempt == retries - 1:
                raise
            wait = 70 if e.code == 429 else 5 * (attempt + 1)   # 429 = limite al minuto
            warn(f"HTTP {e.code}, riprovo tra {wait}s")
            time.sleep(wait)
        except Exception as e:
            if attempt == retries - 1:
                raise
            warn(f"{e}, riprovo")
            time.sleep(5 * (attempt + 1))


def get_bytes(url, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except Exception as e:
            if attempt == retries - 1:
                raise
            warn(f"{e}, riprovo")
            time.sleep(10 * (attempt + 1))


def load_config():
    """config.json + area calcolata dalla preparazione (nome, riquadro)."""
    cfg = json.load(open(os.path.join(ROOT, "config.json")))
    area = os.path.join(DATA, "area.json")
    if os.path.isfile(area) and "bbox" not in cfg:
        a = json.load(open(area))
        if a.get("regions") == cfg.get("regions"):
            cfg["bbox"], cfg["region_name"] = a["bbox"], a["name"]
        else:
            warn("regioni cambiate in config.json: rilancia «Preparazione dati statici»")
    return cfg


def area_name(regions):
    return regions[0] if len(regions) == 1 else ", ".join(regions[:-1]) + " e " + regions[-1]


def area_filter(static):
    """Funzione (lat, lon) -> True se il punto cade dentro le regioni scelte."""
    if static is None or "inside" not in static:
        return lambda lat, lon: True
    south, west, step, rows, cols = static["grid"]
    rows, cols = int(rows), int(cols)
    inside = set(static["inside"].tolist())
    def f(lat, lon):
        r, c = int((lat - south) // step), int((lon - west) // step)
        return 0 <= r < rows and 0 <= c < cols and (r * cols + c) in inside
    return f


def load_groups():
    return json.load(open(os.path.join(ROOT, "species.json")))["groups"]


def resolve_taxa(groups):
    """Nome scientifico -> ID iNaturalist, con cache in data/taxa.json.
    Ritorna {taxon_id: group_id}."""
    path = os.path.join(DATA, "taxa.json")
    cache = json.load(open(path)) if os.path.isfile(path) else {}
    for g in groups:
        for name in g["taxa"]:
            if name in cache:
                continue
            q = urllib.parse.urlencode({"q": name, "is_active": "true", "per_page": 10})
            res = get_json("https://api.inaturalist.org/v1/taxa?" + q).get("results", [])
            match = next((t for t in res if t.get("name", "").lower() == name.lower()), None)
            if match:
                cache[name] = match["id"]
                log(f"  taxon {name} -> {match['id']}")
            else:
                warn(f"taxon non trovato: {name}")
            time.sleep(1.2)
    os.makedirs(DATA, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=1, ensure_ascii=False)
    return {tid: g["id"] for g in groups for n in g["taxa"] if (tid := cache.get(n))}


def group_of(taxon, taxon_to_group):
    lineage = [taxon.get("id")] + list(taxon.get("ancestor_ids") or [])
    return next((taxon_to_group[i] for i in lineage if i in taxon_to_group), None)
