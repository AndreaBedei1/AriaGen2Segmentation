# Confronto esatto tra 5 Hz e frequenza RGB nativa

Il confronto usa gli stessi timestamp gaze, blocchi temporali da 10 s, eventi e 42 bin spaziali. I frame non sono trattati come campioni comportamentali indipendenti. Sono disponibili 162 metriche aggregate, 7.506 righe di differenze a blocchi, 320 coppie evento e 1.176 righe shared-route.

## Risultati globali

| Dominio | Metrica | 5 Hz | Nativa | Differenza |
|---|---|---:|---:|---:|
| auto | copertura | 93,1844% | 93,1933% | +0,0089 pp |
| moto | copertura | 89,6476% | 89,6642% | +0,0166 pp |
| auto | dt mediano | 49,962 ms | 23,926 ms | −26,036 ms |
| moto | dt mediano | 49,895 ms | 16,650 ms | −33,245 ms |
| auto | dt p95 | 93,300 ms | 47,775 ms | −45,524 ms |
| moto | dt p95 | 93,953 ms | 31,372 ms | −62,581 ms |
| auto | fissazioni | 410 | 410 | 0 |
| moto | fissazioni | 1.270 | 1.270 | 0 |
| auto | transizioni semantiche/s | 5,607 | 5,837 | +4,11% |
| moto | transizioni semantiche/s | 7,302 | 7,158 | −1,96% |
| auto | entropy semantica | 0,5239 | 0,5247 | +0,0008 |
| moto | entropy semantica | 0,5587 | 0,5555 | −0,0032 |
| auto | scanpath deg/s | 41,550 | 41,553 | +0,007% |
| moto | scanpath deg/s | 53,492 | 53,501 | +0,018% |
| auto | off-road time | 68,223% | 68,274% | +0,051 pp |
| moto | off-road time | 69,550% | 69,789% | +0,239 pp |

Numero e durata media/mediana delle fissazioni sono identici perché le fissazioni derivano dallo stesso gaze temporale, non dalla frequenza di segmentazione.

## Classi ed eventi

La massima variazione assoluta della massa foveale è 0,194 punti percentuali per l'auto (`interior_cockpit`) e 0,183 punti per la moto (`vehicle`). Le variazioni top-1 più grandi sono +0,491 pp per `vegetation` e −0,436 pp per `sky` sulla moto. `unknown` resta zero; `other_environment` varia di +0,008 pp auto e −0,015 pp moto.

Il dwell cambia poco in termini assoluti: l'escursione maggiore è +4,50 s per `vegetation` moto e −3,90 s per `sky` moto su oltre 900 s di gaze valido. Auto: massimo +1,23 s per `built_environment` e −0,80 s per `interior_cockpit`.

Negli eventi, le differenze medie della massa road-relevant restano generalmente sotto 1 punto percentuale. Le fissazioni event-related non cambiano materialmente. Nei 42 bin shared-route, la differenza assoluta mediana della massa foveale è 0,023 pp e il p95 è 1,53 pp.

## Micro-eventi recuperati e QA

Il rilevatore segnala 228 micro-run auto (3,83 s complessivi) e 521 moto (4,63 s) presenti nel nativo ma assorbiti dalla classe dominante del frame 5 Hz associato. Sono candidati temporali, non 749 eventi comportamentali indipendenti: la durata mediana è un singolo campione e il p95 è circa 67 ms.

Nel QA mirato dei cartelli, i dieci casi selezionati cambiano tutti classe rispetto al 5 Hz; otto mostrano un cartello plausibile nel punto foveale, due restano incerti per dimensione/occlusione. Il beneficio esiste per contatti molto brevi, ma non altera le distribuzioni aggregate.

La revisione completa è descritta in `QA_TEMPORAL_MANUAL_REVIEW.md`.

## Flicker

L'accordo delle maschere sui medesimi frame reali è 99,512% auto e 99,917% moto. I run semantici brevi aumentano da 253,38 a 267,72/min nell'auto (+5,66%) e diminuiscono da 339,17 a 331,67/min nella moto (−2,21%). Il cambio pixel consecutivo per secondo cresce perché vengono osservati due o tre volte più frame e include moto della camera/scena; non è interpretato da solo come flicker del modello.

Non emerge un peggioramento sistematico della stabilità semantica. La versione raw resta il risultato scientifico; non è stato necessario produrre una variante stabilizzata.

La figura `figures/native_vs_5hz_semantic_gaze.png` usa una scala comune per i valori e una scala comune per le differenze dei due domini.
