# Full run auto e moto

La pipeline è stata costruita dalla baseline stabile `feature/article1-multimodal-behavior-analysis` sul branch `feature/article1-fast-full-semantic-gaze`. Un guard test verifica la discendenza dalla baseline (quando il ref locale è disponibile), che il branch attivo non sia full-FOV e che il percorso fast non contenga artefatti full-FOV tracciati.

## Provenienza e copertura

| Dominio | SHA-256 VRS | Frame sorgente | Durata reale | Frame segmentati a 5 Hz | Risoluzione |
|---|---|---:|---:|---:|---:|
| auto | `2e84f0c3e245a7afc353e7c981ed44916df5d8c723e61aab1dd7735903490bf9` | 3,762 | 376.071 s | 1,881 | 1008×756 |
| moto | `5ab8604a14dfafecb15fc58a86c8d99ac59e4c83b1cb6025e43d22efe264f6d7` | 15,069 | 1004.516 s | 5,023 | 1008×756 |

I timestamp provengono dai VRS. La selezione segue una griglia temporale e sceglie frame osservati; `resampled=false`, `interpolated=false`, `synthetic_frames=0`. La rectification usa la geometria pinhole della baseline stabile prima del resize compatto.

## Output strutturati

Per ciascun dominio in `output/article1/fast_semantic_gaze/{car,motorcycle}/` sono presenti:

- `frames/frames.parquet` e `frames/extraction_summary.json`;
- `segmentation/segmentation_index.parquet`, mask ID PNG, confidence ed entropy normalizzata;
- `gaze/projected_gaze.parquet`, `gaze/semantic_gaze.parquet`, `gaze/fixations.parquet` e relativi summary.

Gli output condivisi includono `paired_route_bins.parquet` (1,176 righe), `event_metrics.parquet` (320 righe) e `full_run_summary.json`. Il confronto usa secondi, percentuali, bin spaziali ed eventi; i frame non sono trattati come campioni indipendenti e il frame rate non è una feature.

## Video validati

| Video | Frame | Durata | Codec/geometria |
|---|---:|---:|---|
| `car_semantic_camera_full.mp4` | 1,881 | 376.20 s | H.264 High, yuv420p, 1008×756, 5 fps |
| `motorcycle_semantic_camera_full.mp4` | 5,023 | 1004.60 s | H.264 High, yuv420p, 1008×756, 5 fps |
| `paired_shared_route_comparison.mp4` | 716 coppie | 143.20 s | H.264 High, yuv420p, 1920×720, 5 fps |

Le preview auto e moto hanno entrambe 151 frame (30.20 s). Tutti e cinque i file sono stati decodificati integralmente dopo l'export.

La verifica finale indipendente ha inoltre ricontrollato monotonicità e unicità dei timestamp, corrispondenza fra indici RGB/segmentazione, somma a uno delle frazioni di classe, assenza degli ID `unknown` e `mirror` nelle mask primarie campionate e vincolo gaze-mask entro 120 ms per tutti i campioni semanticamente validi.
