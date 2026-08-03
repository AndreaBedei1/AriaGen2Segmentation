# Profiling del full run a frequenza RGB nativa

Stato del risultato: `exploratory_pilot`. Il benchmark e il full run usano il checkpoint locale Mask2Former/Mapillary, mixed precision, maschere 1008×756, una sola coda GPU e writer CPU asincroni. Grounding DINO e SAM2 non sono stati caricati.

La macchina rilevata dispone di una RTX 6000 Ada (46.068 MiB), 40 CPU logiche, 88,22 GB di RAM e circa 348 GB liberi all'avvio.

## Benchmark del batch

Sono stati usati 300 frame auto e 450 frame moto, cioè circa 30 s reali per dominio, con cinque ripetizioni per configurazione.

| Batch | ms/frame profilati | Throughput | Latenza batch | Picco VRAM | Picco RAM | Finito |
|---:|---:|---:|---:|---:|---:|:---:|
| 4 | 69,03 | 14,49 frame/s | 276,1 ms | 1.819 MB | 1.991 MB | sì |
| 6 | 60,47 | 16,54 frame/s | 362,8 ms | 2.287 MB | 2.047 MB | sì |
| **8** | **56,83** | **17,60 frame/s** | 454,6 ms | 2.759 MB | 2.112 MB | sì |

È stato scelto batch 8: aumenta il throughput profilato del 21,5% rispetto a batch 4, usa solo il 5,8% della VRAM disponibile e non mostra valori non finiti. L'aumento della latenza batch non compromette il full run offline.

## Ablation temporale iniziale

Le frequenze inferiori associano esclusivamente frame reali; non sono state create maschere sintetiche. L'accordo pixel è calcolato su finestre reali di circa 30 s rispetto all'inferenza di ogni frame nativo.

| Dominio | Frequenza | Frame full run | Riduzione | Accordo pixel vs nativa | dt gaze-mask mediano | dt p95 |
|---|---:|---:|---:|---:|---:|---:|
| auto | **nativa (10,0008 Hz)** | **3.762** | 0% | 100% | 23,81 ms | 47,69 ms |
| auto | 5 Hz | 1.881 | 50,0% | 94,05% | 50,00 ms | 93,34 ms |
| auto | 2,5 Hz | 941 | 75,0% | 89,61% | 99,29 ms | 189,17 ms |
| moto | **nativa (15,0003 Hz)** | **15.069** | 0% | 100% | 16,67 ms | 31,38 ms |
| moto | 5 Hz | 5.023 | 66,7% | 92,36% | 50,00 ms | 94,05 ms |
| moto | 2,5 Hz | 2.512 | 83,3% | 86,90% | 100,00 ms | 189,26 ms |

## Profiling del full run

| Stadio | Auto | Moto | Nota |
|---|---:|---:|---|
| estrazione/rectification/JPEG | 146,55 s | 590,84 s | eseguiti in parallelo su CPU |
| segmentazione | 208,33 s, 18,06 frame/s | 772,35 s, 19,51 frame/s | seriale sulla singola GPU |
| solo forward GPU | 39,17 s | 156,59 s | batch 8, mixed precision |
| semantic gaze | 47,09 s | 168,51 s | timestamp reali, tutti i campioni gaze |

Il picco di produzione è 2.245 MB VRAM e 2.176 MB RAM. Il throughput combinato della segmentazione è 19,20 frame/s su 18.831 frame. I principali costi medi per frame sono preprocessing (31,65–31,71 ms), scrittura asincrona (25,23–27,17 ms) e forward (10,39–10,41 ms).

La somma seriale di estrazione, segmentazione e semantic gaze dei due domini è 1.933,67 s (32 min 13,7 s). Il critical path realmente adottato — estrazioni CPU concorrenti, poi auto e moto seriali sulla GPU — è 1.787,12 s (29 min 47,1 s). Questa misura non include benchmark, figure, QA o codifica video.

I dati completi sono in `output/article1/fast_semantic_gaze_native/profiling/`.
