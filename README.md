# Mappa funghi

Webapp statica (PWA) con un indice di crescita per ciascun fungo commestibile comune e i relativi avvistamenti recenti, aggiornata ogni mattina da GitHub Actions.

## Setup
1. Modifica `config.json` (regione e `bbox`) e aggiungi `data/region.geojson`.
2. Crea un repo pubblico su GitHub e fai push con git (non con l'upload web, che salta la cartella `.github`).
3. Settings → Pages → Source: "GitHub Actions".
4. Actions → "Aggiornamento dati funghi" → Run workflow.
5. Apri https://TUONOME.github.io/NOMEREPO/ dal telefono e aggiungila alla schermata Home.

Test locale: `python scripts/update.py && python -m http.server`

## Note
- Cron `15 3 * * *` (UTC). GitHub può ritardare i job schedulati di alcuni minuti, e li sospende dopo 60 giorni senza attività sul repo: i commit giornalieri del bot bastano a tenerlo attivo.
- Griglia 0,1° (~10 km). A 0,05° le celle quadruplicano: verifica i limiti gratuiti di Open-Meteo (uso non commerciale).
- Specie, stagioni, soglie di temperatura, quote, habitat e sosia sono in `species.json`: aggiungere un fungo significa aggiungere un oggetto lì. I nomi scientifici vengono convertiti in ID iNaturalist al primo run e salvati in `data/taxa.json`; se un nome non viene trovato lo script lo segnala nel log.
- L'indice in `group_score()` è euristico: tara soglie e pesi confrontandolo con gli avvistamenti della tua zona.
- Le coordinate iNaturalist di alcune osservazioni sono volutamente offuscate.
