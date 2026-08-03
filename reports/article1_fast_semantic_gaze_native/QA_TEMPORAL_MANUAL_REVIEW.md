# Revisione manuale del QA temporale

Sono stati esaminati tutti gli otto contact sheet generati, per un totale di 76 casi: 10 per ciascuna delle sette categorie obbligatorie e tutti i 6 candidati mirror. Ogni caso mostra il frame 5 Hz e il frame nativo realmente associato, la classe top-1, il gaze e l'errore temporale assoluto.

| Categoria | Casi | Esito della revisione |
|---|---:|---|
| sguardi brevi su cartelli | 10 | 8 plausibili, 2 incerti; il nativo riduce il dt e recupera contatti piccoli/brevi |
| transizioni strada-veicolo | 10 | classe centrale visivamente plausibile in 10/10; nessun artefatto sistematico evidente |
| pedoni | 10 | 4 supportati, 2 incerti, 4 non supportati; mani/guanti dell'ego possono essere confusi con `pedestrian` |
| veicoli in movimento | 10 | veicolo visibile in 10/10; il moto non è certificabile dal singolo contact sheet |
| gaze vicino a lane marking | 10 | marking visibile e coerente in 10/10; un caso perso a 5 Hz è recuperato dal nativo |
| eventi in rotonda | 10 | appartenenza alla finestra evento verificata; classi locali coerenti, due cambi 5 Hz/native |
| eventi presso incroci | 10 | appartenenza alla finestra evento verificata; classi locali coerenti |
| possible mirror | 6 | 0 confermati; tutti falsi positivi o indeterminati |

## Interpretazione

La frequenza nativa riduce chiaramente il disallineamento gaze-maschera e può recuperare contatti inferiori a 200 ms. Non produce però un miglioramento uniforme delle classi rare: la qualità locale resta limitata dalla tassonomia nativa e dalla dimensione degli oggetti, non soltanto dal sampling temporale.

Il proxy mirror non è utile in questa forma. I sei casi moto non mostrano uno sguardo inequivocabile allo specchietto; il segnale resta fuori dalle maschere e dalle metriche primarie.

La classe `pedestrian` richiede cautela: il modello segmenta talvolta mani o guanti dell'ego come persona. Questo limite è già visibile a entrambe le frequenze e non è causato dal run nativo.

Contact sheet: `output/article1/fast_semantic_gaze_native/qa/contact_sheets/`. La selezione riproducibile è in `temporal_qa_cases.csv`; nessun caso è stato rimosso dopo l'ispezione.
