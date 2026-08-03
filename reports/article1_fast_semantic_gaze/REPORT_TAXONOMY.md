# Tassonomia semplificata

La tassonomia finale contiene 16 ID stabili: `unknown`, le 12 macroclassi esterne richieste, `interior_cockpit`, `mirror` e `other_environment`. L'ordine, i colori e i gruppi sono definiti in `configs/article1/classes_fast_semantic_gaze.yaml` e sono condivisi da mask, metriche, legenda, grafici e video.

## Mapping Mapillary

Tutte le 65 classi native del checkpoint sono mappate esplicitamente. Vegetazione, ambiente costruito e cielo restano separati. Solo sei classi native rare e non riconducibili alle macroclassi richieste confluiscono nel residuo: `Bird`, `Ground Animal`, `Sand`, `Snow`, `Water`, `Boat`.

`other_environment` occupa in media lo 0.052% dei pixel auto e lo 0.122% dei pixel moto: è quindi un residuo controllato, non una classe dominante. `unknown` è 0% nel full run.

## Interno del mezzo

`interior_cockpit` è una sola macroclasse. La sorgente combina:

- classi native `Ego Vehicle` e `Car Mount`;
- componenti `vehicle`/`two_wheeler` nella parte bassa che toccano realmente il bordo inferiore, utili per serbatoio/carena propri;
- solo per l'auto, componenti `built_environment` che toccano il bordo superiore e hanno almeno il 55% dell'area nella fascia superiore (42% dell'immagine), utili per cielo e montanti dell'abitacolo.

La regola superiore riclassifica l'intero componente qualificato, evitando tagli orizzontali artificiali. È geometrica, deterministica, indipendente dal gaze e disabilitata per la moto. L'occupazione media finale di `interior_cockpit` è 73.07% in auto e 1.95% in moto, coerente con le due geometrie di ripresa ma non utilizzabile come misura comportamentale tra domini.

## Specchietto

`mirror` resta nella tassonomia ma è escluso dalla mask primaria (`eval: false`). Non è stato trovato un detector abbastanza affidabile da giustificare Grounding DINO/SAM2 sull'intera registrazione. Il sistema conserva solo `possible_mirror_gaze_candidate`, basato su zone spaziali deboli e marcato `review_only`: 0 candidati auto e 6 moto. Nessun candidato entra nelle metriche primarie.

## Affidabilità

Il mapping è completo e internamente consistente, e il residuo è minimo. Il contact sheet `semantic_camera_qa_contact_sheet.jpg` mostra copertura densa e leggibile. Non esiste tuttavia ground truth pixel-wise revisionata sull'intero percorso: l'affidabilità è adeguata a un'analisi esplorativa e a QA umano, non a dichiarazioni di accuratezza o sicurezza.
