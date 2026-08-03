# Full run della semantic camera a frequenza nativa

La pipeline ha elaborato ogni frame RGB reale disponibile, senza subsampling, duplicazione, interpolazione o frame sintetici. Le frequenze sono misurate dai timestamp, non hardcodate.

| Misura | Auto | Moto |
|---|---:|---:|
| frequenza RGB misurata | 10,000763 Hz | 15,000256 Hz |
| durata timestamp sorgente | 376,071 s | 1.004,516 s |
| frame sorgente/processati | 3.762 / 3.762 | 15.069 / 15.069 |
| frame falliti | 0 | 0 |
| batch GPU | 8 | 8 |
| throughput segmentazione | 18,058 frame/s | 19,511 frame/s |
| tempo segmentazione end-to-end | 208,329 s | 772,349 s |
| tempo forward GPU | 39,173 s | 156,590 s |
| picco VRAM | 2.245 MB | 2.245 MB |
| picco RAM | 2.057 MB | 2.176 MB |
| RGB rettificati | 320.795.384 byte | 1.358.369.266 byte |
| output segmentazione | 1.012.931.246 byte | 2.804.354.004 byte |

L'associazione gaze-maschera è `nearest_real_segmentation_timestamp`. Non è mai usato l'indice del frame come sostituto del tempo.

| Gaze | Auto | Moto |
|---|---:|---:|
| campioni totali | 11.283 | 30.138 |
| associati | 10.515 | 27.023 |
| esclusi | 768 | 3.115 |
| copertura | 93,1933% | 89,6642% |
| tempo gaze valido | 350,496 s | 900,758 s |
| errore assoluto mediano | 23,926 ms | 16,650 ms |
| errore assoluto p95 | 47,775 ms | 31,372 ms |
| errore assoluto massimo | 49,996 ms | 60,266 ms |

## Contratti scientifici

- Le maschere raw native sono gli unici input di semantic gaze e statistiche.
- Non è applicata stabilizzazione temporale scientifica.
- Grounding DINO e SAM2 non sono usati.
- Tutte le 65 classi native Mapillary risultano mappate.
- `unknown` è 0% in entrambi i domini.
- `other_environment` occupa in media lo 0,0529% dei pixel auto e lo 0,1168% dei pixel moto.
- `mirror` non entra nelle maschere primarie; il proxy spaziale è esclusivamente review-only.

Gli output strutturati sono disponibili in Parquet/CSV sotto `car/gaze/`, `motorcycle/gaze/`, `paired_route_bins.*`, `event_metrics.*`, `comparison/` e `profiling/` del nuovo output root. Il run 5 Hz non è stato usato come destinazione di scrittura.
