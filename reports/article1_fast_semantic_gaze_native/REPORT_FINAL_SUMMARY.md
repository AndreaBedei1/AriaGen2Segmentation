# Verdetto finale — semantic gaze a frequenza RGB nativa

Il secondo full run è completo per entrambe le registrazioni, separato dalla baseline 5 Hz e mantenuto nello stato `exploratory_pilot`.

1. **Configurazione eseguita.** Mask2Former/Mapillary locale, 1008×756, mixed precision, batch 8, 8 writer CPU, ogni frame RGB reale, frequenza misurata dai timestamp, associazione gaze-maschera al timestamp reale più vicino, maschere raw senza stabilizzazione. Grounding DINO e SAM2 non sono usati.
2. **Tempo totale reale.** Il core scientifico sul critical path è 1.787,12 s (29 min 47,1 s): estrazioni auto/moto concorrenti e una sola inferenza GPU seriale. Gli export video concorrenti più lo shared-route richiedono circa 442 s; il percorso di produzione core+video è quindi circa 2.229 s (37 min 9 s). Benchmark, figure, confronto, QA e test sono attività aggiuntive riportate separatamente, non incluse in un cronometro monolitico.
3. **Throughput e batch.** Batch 8 raggiunge 17,60 frame/s nel benchmark e 19,20 frame/s combinati nel full run; il picco di produzione è 2.245 MB VRAM e 2.176 MB RAM. Sono stati processati 18.831/18.831 frame, con zero fallimenti.
4. **Copertura auto.** 10.515/11.283 campioni gaze, 93,1933%, 350,496 s validi.
5. **Copertura moto.** 27.023/30.138 campioni gaze, 89,6642%, 900,758 s validi.
6. **Errore temporale gaze-maschera.** Auto: mediana 49,962→23,926 ms e p95 93,300→47,775 ms. Moto: mediana 49,895→16,650 ms e p95 93,953→31,372 ms. Questo è il beneficio netto e riproducibile del sampling nativo.
7. **Differenze semantic gaze.** Fissazioni e loro durate sono identiche; entropy e scanpath cambiano meno dello 0,6% e dello 0,02%. La massima variazione di massa foveale per classe è 0,194 punti percentuali auto e 0,183 moto. Nei 42 bin shared-route il cambiamento assoluto mediano è 0,023 pp.
8. **Eventi brevi.** Sono segnalati 749 micro-run candidati, 8,47 s complessivi, con durata p95 circa 67 ms. Nel QA mirato, 8/10 contatti brevi con cartelli sono plausibili e 2 incerti. Il nativo aiuta questi casi, ma il guadagno non modifica materialmente le distribuzioni aggregate.
9. **Flicker.** Le transizioni semantiche cambiano +4,11% auto e −1,96% moto; i run brevi +5,66% e −2,21%. L'accordo sui medesimi frame reali è 99,512%/99,917%. Non emerge un peggioramento sistematico e non è stata prodotta una variante stabilizzata.
10. **Mirror.** I 6 candidati moto sono tutti falsi o indeterminati alla revisione; 0 auto. Il proxy non è utile per metriche primarie e resta review-only.
11. **Grafici.** Le nove figure native e la figura 5 Hz/native sono leggibili e usano scale, ordine classi, palette, binning, unità e normalizzazioni coerenti. Sono state ispezionate visivamente dopo la generazione.
12. **Integrità.** Suite finale: 505 test passati, 14 saltati. Tutti i cinque MP4 sono stati decodificati integralmente con FFmpeg. La baseline 5 Hz è invariata byte-per-byte: 27.661 file, 2.387.020.397 byte, digest del manifest SHA-256 `b099465e5889e78b39959a529ac3e31d061a3079dbf8104be65af8124727a031`.

## Frequenza raccomandata per l'articolo

**C. La frequenza nativa non modifica materialmente i risultati e 5 Hz è sufficiente.**

Raccomandazione: mantenere 5 Hz come analisi principale efficiente e usare il run nativo come ablation/robustness check, oppure per analisi mirate di oggetti piccoli e contatti inferiori a 200 ms. Il nativo dimezza o riduce di due terzi l'errore temporale, ma costa 2× i frame auto e 3× i frame moto e non cambia le conclusioni descrittive aggregate.

## Video principali da guardare

- **Auto:** `output/article1/fast_semantic_gaze_native/car_semantic_camera_full_native.mp4` — 3.762 frame, stream 10 Hz.
- **Moto:** `output/article1/fast_semantic_gaze_native/motorcycle_semantic_camera_full_native.mp4` — 15.069 frame, stream 15 Hz.
- **Confronto shared-route:** `output/article1/fast_semantic_gaze_native/paired_shared_route_comparison_native.mp4` — 1.674 coppie/pannelli, 42 bin, timebase auto per il solo MP4 di presentazione.

Per una verifica rapida: `previews/car_preview_native_30s.mp4` e `previews/motorcycle_preview_native_30s.mp4`. I dati scientifici auto e moto mantengono sempre le rispettive frequenze e i rispettivi timestamp; il video shared-route non ricampiona le maschere usate nelle statistiche.

**Verdetto obbligatorio finale: C. La frequenza nativa non modifica materialmente i risultati e 5 Hz è sufficiente.**
