#!/usr/bin/env python3
"""Su un push di codice recupera i dati del giorno dal sito già pubblicato,
così non serve rifare le chiamate meteo. Esce con errore se non ci riesce."""
import json, os, sys, urllib.request

from common import DATA, UA, load_groups

owner, repo = os.environ["GITHUB_REPOSITORY"].split("/")
base = f"https://{owner.lower()}.github.io/" + ("" if repo.lower() == f"{owner.lower()}.github.io" else f"{repo}/")


def get(path):
    req = urllib.request.Request(base + path, headers={"User-Agent": UA, "Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


try:
    latest = get("data/latest.json")
    if "days" not in json.loads(latest):
        raise ValueError("dati pubblicati in un formato vecchio")
    files = {"data/latest.json": latest}
    for g in load_groups():
        files[f"data/layers/{g['id']}.json"] = get(f"data/layers/{g['id']}.json")
except Exception as e:
    print(f"Dati pubblicati non recuperabili ({e}): serve un aggiornamento completo", file=sys.stderr)
    sys.exit(1)

for path, raw in files.items():
    full = os.path.join(os.path.dirname(DATA), path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    open(full, "wb").write(raw)
print(f"Recuperati {len(files)} file da {base}")
