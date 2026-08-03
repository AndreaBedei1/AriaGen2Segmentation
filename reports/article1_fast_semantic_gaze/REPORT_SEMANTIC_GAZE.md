# Semantic gaze full recording

La segmentazione è indipendente dal gaze. Ogni campione gaze reale proiettato a circa 30 Hz viene associato per timestamp al frame segmentato reale più vicino; l'associazione è accettata entro 120 ms. Nessun frame rate entra nelle feature e nessuna maschera è interpolata.

| Metrica | Auto | Moto |
|---|---:|---:|
| campioni gaze | 11,283 | 30,138 |
| campioni semanticamente validi | 10,514 | 27,018 |
| copertura | **93.18%** | **89.65%** |
| tempo gaze semanticamente valido | 350.46 s | 900.59 s |
| fissazioni I-VT | 410 | 1,270 |
| quota top-1 road-relevant | 31.79% | 30.45% |
| massa foveale road-relevant | 34.84% | 33.19% |
| candidati mirror deboli | 0 | 6 |

La massa foveale usa una finestra gaussiana da 1.5° (raggio operativo 29 px) ed è il complemento più robusto alla classe top-1: vicino al punto di fuga, la strada può occupare meno della finestra foveale rispetto alla scena circostante. Le metriche includono dwell in secondi e percentuali, fissazioni, transizioni, tempo alla prima osservazione e run off-road; non riportano conteggi per frame come unità inferenziale.

Le quote più grandi top-1 sono vegetazione (46.48% auto; 45.34% moto), veicolo (27.42%; 20.23%), built environment (13.54%; 8.28%), cielo (5.76%; 14.36%) e road surface (3.39%; 8.26%). Questi valori descrivono questa singola acquisizione e sono sensibili a percorso, geometria dell'abitacolo e qualità del modello: non supportano conclusioni generalizzabili o mediche.

Il proxy mirror è deliberatamente separato: `included_in_primary_metrics=false` e `included_in_primary_masks=false`.
