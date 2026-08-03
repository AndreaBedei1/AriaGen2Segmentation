# Semantic gaze nativo

La segmentazione densa è calcolata indipendentemente dal gaze. Per ogni campione valido, il semantic gaze usa una finestra foveale gaussiana da 1,5° (raggio 29 px) sulla maschera del frame reale temporalmente più vicino.

## Copertura e dinamica

| Metrica | Auto | Moto |
|---|---:|---:|
| campioni validi | 10.515 / 11.283 | 27.023 / 30.138 |
| copertura | 93,193% | 89,664% |
| tempo valido | 350,496 s | 900,758 s |
| fissazioni | 410 | 1.270 |
| durata fissazione media | 0,616 s | 0,349 s |
| durata fissazione mediana | 0,433 s | 0,233 s |
| transizioni semantiche/s | 5,837 | 7,158 |
| entropy semantica | 0,525 | 0,555 |
| scanpath | 41,553 deg/s | 53,501 deg/s |

Le classi dominanti per massa foveale sono `vegetation` (40,8% auto; 40,3% moto), `vehicle` (27,7%; 19,8%), `built_environment` (9,6%; 9,0%), `sky` (8,0%; 16,0%) e `road_surface` (5,1%; 9,9%). I confronti auto-moto sono descrittivi e mantengono lo stato `exploratory_pilot`.

## Tassonomia e affidabilità

La tassonomia conserva 16 ID, inclusi `unknown`, `mirror` e il residuo `other_environment`. Le 65 classi Mapillary sono completamente mappate; non ci sono etichette native non mappate. Il residuo resta sotto lo 0,12% medio dei pixel.

`interior_cockpit` è una singola macroclasse. Nel video auto domina geometricamente il frame perché tetto, montanti e cruscotto occupano gran parte della camera; non viene suddivisa in volante, display o mani. Il QA mostra che alcune mani possono essere classificate come `pedestrian`, perciò le classi rare richiedono revisione visiva.

Il mirror non è una classe scientifica attiva: 0 candidati auto e 6 moto sono conservati come proxy debole. La revisione manuale non ne conferma nessuno.

## Figure

Le nove figure principali native e la figura 5 Hz/native sono state ispezionate dopo il rendering. Nei confronti auto-moto usano ordine classi, palette, binning, unità, normalizzazione e limiti condivisi. Le metriche comportamentali sono espresse in secondi, percentuali, eventi o bin spaziali, mai come conteggi di frame indipendenti.

I dati completi sono in `output/article1/fast_semantic_gaze_native/`; le figure PNG/PDF sono in `reports/article1_fast_semantic_gaze_native/figures/`.
