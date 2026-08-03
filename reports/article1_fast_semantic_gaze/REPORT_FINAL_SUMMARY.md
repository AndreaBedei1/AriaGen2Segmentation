# Verdetto finale — fast full semantic gaze

La pipeline `semantic_gaze_fast_external` è completa per entrambe le registrazioni e conserva lo stato `exploratory_pilot`.

1. **Configurazione finale.** Mask2Former/Mapillary locale, tassonomia a 16 ID, output 1008×756, segmentazione a 5 Hz, batch GPU 4, mixed precision, 8 writer CPU asincroni, mask/confidence/entropy compatti, nessuna stabilizzazione temporale senza compensazione del moto. Grounding DINO e SAM2 non sono usati nel full run.
2. **Tempo stimato e reale.** Il benchmark indicava circa 8–9 minuti per il core con sovrapposizione sicura. I wall time misurati sommano 694.7 s se tutti i sei stadi core sono seriali; il critical path con estrazione CPU concorrente e una sola inferenza GPU è circa 495.7 s (8 min 15.7 s). Gli export video concorrenti aggiungono circa 2 min 10 s. Sono misure per stadio, non un singolo cronometro monolitico.
3. **Copertura semantic gaze.** Auto: 10,514/11,283 campioni, 93.18%, 350.46 s. Moto: 27,018/30,138, 89.65%, 900.59 s. La differenza residua deriva dal vincolo timestamp massimo di 120 ms, non da frame inventati.
4. **Affidabilità tassonomia.** Tutte le 65 classi Mapillary sono mappate; `unknown` è 0% e `other_environment` resta 0.052% auto / 0.122% moto. L'interno è una macroclasse con proxy di bordo conservativi. La qualità è buona per studio esplorativo e QA visivo, ma non è accuracy-validata da ground truth completa.
5. **Proxy mirror.** Non è abbastanza affidabile per l'uso primario. Rimane review-only, fuori da mask e metriche: 0 candidati auto, 6 moto.
6. **Qualità grafici.** Le nove figure sono leggibili, ridotte e con scale/unità/ordine/normalizzazione condivisi; il confronto principale usa bin da 50 m, finestre ed eventi, non frame indipendenti.
7. **Test.** Suite finale nell'ambiente integrato: 499 passati, 14 saltati dopo l'aggiunta del guard di discendenza/non-full-FOV (l'ambiente VRS senza Torch produce 498 passati e 15 saltati). I test nuovi coprono tassonomia, mapping completo, frequenze temporali, assenza dei modelli lenti, same-scale plotting, proxy cockpit, lineage stabile e video puliti. I cinque MP4 sono stati anche decodificati integralmente con FFmpeg senza errori.

## Video principali da guardare

- **Auto:** `output/article1/fast_semantic_gaze/car_semantic_camera_full.mp4`
- **Moto:** `output/article1/fast_semantic_gaze/motorcycle_semantic_camera_full.mp4`
- **Confronto shared-route:** `output/article1/fast_semantic_gaze/paired_shared_route_comparison.mp4`

Per una valutazione rapida usare prima `previews/car_preview_30s.mp4` e `previews/motorcycle_preview_30s.mp4`; per verificare direttamente l'allineamento auto/moto usare il video shared-route.
