# Profiling della pipeline fast semantic gaze

Stato del risultato: `exploratory_pilot`. Le misure sono state raccolte sulla RTX 6000 Ada disponibile (46 GB nominali), con 40 CPU logiche, 88.22 GB di RAM totali (circa 82.35 GB disponibili al rilevamento) e circa 350 GB liberi su disco. Il modello è il checkpoint locale Mask2Former/Mapillary; Grounding DINO e SAM2 non sono eseguiti.

## Ablation del batch GPU

| Batch | ms/frame profilati | frame/s GPU | picco VRAM |
|---:|---:|---:|---:|
| 1 | 76.22 | 13.12 | 1,115 MB |
| **4** | **32.20** | **31.06** | **1,818 MB** |
| 8 | 33.67 | 29.70 | 2,755 MB |
| 16 | 34.12 | 29.31 | 4,629 MB |

Batch 4 è il massimo misurato e lascia molto margine VRAM. Batch più grandi riducono il costo del forward, ma aumentano preprocessing e latenza senza migliorare il throughput totale.

## Ablation temporale

L'accordo è misurato contro inferenza nativa su finestre reali di 4 s. Le frequenze inferiori riutilizzano esclusivamente la maschera del frame reale selezionato più vicino: nessun frame o mask è sintetizzato.

| Dominio | Frequenza | Frame selezionati full run | Riduzione | Accordo pixel vs nativa | dt gaze-mask mediano | dt p95 |
|---|---:|---:|---:|---:|---:|---:|
| auto | nativa (~10 Hz) | 3,762 | 0.0% | 100.0% | 23.8 ms | 47.7 ms |
| auto | **5 Hz** | **1,881** | **50.0%** | **91.87%** | **50.0 ms** | **93.3 ms** |
| auto | 2.5 Hz | 941 | 75.0% | 86.67% | 99.3 ms | 189.2 ms |
| moto | nativa (~15 Hz) | 15,069 | 0.0% | 100.0% | 16.7 ms | 31.4 ms |
| moto | **5 Hz** | **5,023** | **66.7%** | **92.06%** | **50.0 ms** | **94.1 ms** |
| moto | 2.5 Hz | 2,512 | 83.3% | 86.27% | 100.0 ms | 189.3 ms |

La scelta finale è 5 Hz: conserva circa il 92% dell'accordo pixel temporale e dimezza/riduce di due terzi le inferenze; 2.5 Hz perde altri 5–6 punti di accordo e porta il p95 vicino a 190 ms.

## Misure full run

| Stadio | Auto | Moto | Nota |
|---|---:|---:|---|
| decode + rectification + JPEG | 74.50 s | 233.23 s | 8 writer CPU asincroni |
| segmentazione finale | 98.61 s (19.08 fps) | 197.50 s (25.43 fps) | una sola coda GPU, batch 4 |
| semantic gaze | 25.85 s | 65.00 s | tutti i campioni gaze reali |
| export video completo | 51.4 s | 129.8 s | H.264, 1008×756, 5 fps |
| export shared-route | \- | 39.1 s | 1920×720, 716 coppie |

Il picco di produzione è 1,561 MB VRAM e circa 2.0 GB RAM nel processo di segmentazione. Il totale esatto dei wall time dei sei stadi core, se eseguiti serialmente, è circa 694.7 s (11 min 35.7 s). Con le estrazioni CPU concorrenti, una sola inferenza GPU alla volta e il gaze sovrapposto quando possibile, il critical path ricostruito è circa 495.7 s (8 min 15.7 s); gli export video concorrenti aggiungono circa 2 min 10 s.

I dati grezzi sono in `output/article1/fast_semantic_gaze/profiling/` (`benchmark.json`, `batch_profiles.csv`, `temporal_quality.csv`, `frequency_coverage.csv`).
