# Mappa funghi

PWA statica che mostra, per ogni fungo commestibile comune, le zone da circa 1 km più favorevoli oggi, con tendenza a 3 giorni e avvistamenti recenti. Tutto gira su GitHub Actions + Pages.

## Come funziona
**Preparazione (`prepare_static.py`, ogni 3 mesi o a mano)**
- Confini delle regioni scelte in `config.json` scaricati da openpolis/geojson-italy (dati ISTAT); riquadro calcolato in automatico e salvato in `data/area.json`.
- Griglia da 0,01° (circa 1 km) ritagliata sui confini.
- Quota media, pendenza ed esposizione da Copernicus DEM GLO-90.
- Frazione di bosco, prati/arbusteti, coltivi, urbano e acqua da ESA WorldCover 10 m; le celle senza habitat utile vengono scartate.
- Calibrazione sulle osservazioni confermate iNaturalist, corretta per lo sforzo di osservazione: scarica tutte le osservazioni di funghi dell'area come sfondo e, per ogni fascia di quota e regione, usa la quota della specie tra i funghi fotografati invece del numero assoluto di foto (che è alto vicino alle città). Le stime per regione partono da quella complessiva e se ne discostano solo dove ci sono dati.
- Vicinato di ~2 km (bosco, ambiente naturale, urbano) calcolato sulla griglia completa: filari, golene e paesi vengono penalizzati.

**Aggiornamento (`update.py`, ogni mattina)**
- Meteo Open-Meteo su celle da 0,07° (circa 6–8 km): 21 giorni passati e 7 di previsione. Per Emilia-Romagna, Toscana e Liguria sono circa 1.000–1.300 punti, ~2.500 chiamate al giorno.
- Temperature riportate alla quota di ogni cella da 1 km (0,65 °C ogni 100 m).
- Indice per specie e per giorno, da oggi a +6 = stagione × quota × habitat (bosco/prati, latifoglie/conifere) × temperature × acqua × esposizione.
- Affidabilità per giorno: quanto l'indice dipende da pioggia e temperature previste invece che misurate.
- Zone: gruppi di celle adiacenti sopra soglia, divisi finché non superano ~25 km².
- `latest.json` e `data/layers/` non vengono salvati nel repo: vanno direttamente su Pages. Sui push di codice lo script `fetch_live.py` li recupera dal sito pubblicato.

**Verifica sul passato (`backtest.py`, ogni lunedì)**
- Ritrovamenti confermati iNaturalist degli ultimi 3 anni in regione.
- Per ognuno confronta il meteo del ritrovamento con lo stesso punto e periodo dell'anno in anni diversi (caso-controllo: luogo e stagione uguali, conta solo il meteo).
- Il meteo storico (Open-Meteo Archive) pesa molto sui limiti gratuiti: viene scaricato a rate, circa 3.000 chiamate per esecuzione, e salvato in `data/history.npz`.
- A archivio completo stima ritardo della pioggia, scala della pioggia, penalità per la siccità e, per le specie con abbastanza dati, lo spostamento delle temperature ideali. Accetta i nuovi parametri solo se migliorano l'AUC in modo apprezzabile. Risultati in `data/model.json`, mostrati nell'app.
- L'archivio si ricostruisce ogni ~5 mesi.

## Setup
1. `config.json`: elenco `regions` con i nomi ISTAT (es. "Emilia-Romagna", "Toscana", "Liguria"). Se cambi regioni rilancia la preparazione.
2. Settings → Pages → Source: GitHub Actions.
3. Actions → «Preparazione dati statici» → Run workflow. Al termine avvia da solo «Aggiornamento dati funghi».
4. Almeno un'ora dopo: Actions → «Verifica sul passato» → Run workflow. Poi prosegue da sola ogni lunedì finché l'archivio è completo.

## Note
- `species.json` contiene i parametri di partenza: tarali confrontando la mappa con i tuoi ritrovamenti.
- Open-Meteo gratuito solo per uso non commerciale; circa 1.000 punti meteo al giorno restano sotto i limiti.
- Limiti Open-Meteo gratuiti: 600 chiamate/minuto, 5.000/ora, 10.000/giorno. Aggiornamento e verifica girano a orari diversi per restarci dentro.
- Attribuzioni: © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium; Copernicus HRL Dominant Leaf Type 2018 © European Union, Copernicus Land Monitoring Service, EEA; Copernicus DEM © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA.
