# Figure principali semplificate

Le nove figure principali sono in `reports/article1_fast_semantic_gaze/figures/` in PNG 300 dpi e PDF vettoriale. Il manifest dichiara `same_scale_car_motorcycle=true` per ciascuna figura comparativa.

1. `01_route_map_shared`: percorso e tratto condiviso.
2. `02_semantic_gaze_distribution`: quota temporale per classe, stesso ordine.
3. `03_dwell_fixation_by_class`: dwell e fissazioni con scale comuni.
4. `04_semantic_gaze_shared_route`: metriche su bin spaziali condivisi.
5. `05_event_related_gaze`: rotonde, incroci e curve, unità percentuali comuni.
6. `06_speed_head_motion`: velocità e movimento testa comparabili.
7. `07_ppg_hr_event_related`: variazione HR rispetto alla baseline locale.
8. `08_solid_line_candidates`: soli candidati, con stato revisionabile.
9. `09_paired_summary_difference`: differenza moto meno auto separata dai valori assoluti.

I confronti condividono limiti, unità, binning, palette e ordine delle classi. Il tratto comune usa bin da 50 m e traversal order entro bin; sono disponibili 1,176 righe paired e 320 righe evento. Le linee continue restano `candidate_compliant`, `candidate_noncompliant`, `uncertain` o `not_evaluable`; non sono etichette definitive senza revisione umana.

La nuova serie è molto meno densa della precedente: ogni figura risponde a una sola domanda principale, mentre confidence, entropy, profiling e QA restano nei report/output supplementari. Il contact sheet semantic-camera è stato mantenuto come JPEG leggero (circa 148 KiB) al posto del PNG diagnostico da 4.2 MiB.
