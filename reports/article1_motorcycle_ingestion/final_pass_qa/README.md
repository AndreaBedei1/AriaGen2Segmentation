# Article 1 semantic-camera final QA

Preview leggere della sola modalità finale offline/presentation. Il risultato precedente è la presentazione stabilizzata del branch di partenza; gli output scientifici raw sono rimasti invariati.

| Sequenza | Frame | Timestamp clip | Motivo | Contact sheet |
|---|---:|---:|---|---|
| [Linee stradali che scompaiono e ricompaiono](01_lane_dropout/README.md) | 13944–13953 | 929.933 s | La finestra flow-aligned ±10 recupera evidenza lane nei frame intermedi e il filtro geometrico la limita al supporto stradale. | [JPG](01_lane_dropout/contact_sheet.jpg) |
| [Linee spezzate con gap compatibili](02_broken_lane_link/README.md) | 14177–14186 | 945.465 s | Closing multi-orientamento e collegamento controllato degli estremi riducono piccoli gap senza attraversare barriere semantiche. | [JPG](02_broken_lane_link/contact_sheet.jpg) |
| [Cordolo o bordo strada intermittente](03_intermittent_boundary/README.md) | 14300–14309 | 953.664 s | Persistenza temporale, prossimità al bordo strada e pulizia delle componenti rendono più continuo road_boundary_or_obstacle. | [JPG](03_intermittent_boundary/contact_sheet.jpg) |
| [Cockpit, mani, specchi o display con flicker](04_internal_flicker/README.md) | 13923–13932 | 928.533 s | Persistenza lunga e isteresi per le classi interne recuperano i dropout; le mani restano nella classe control_and_ego_vehicle. | [JPG](04_internal_flicker/contact_sheet.jpg) |
| [Conflitto tra interno ed esterno](05_internal_external_conflict/README.md) | 13969–13978 | 931.599 s | La fusione conserva le regioni esterne forti e vieta ai collegamenti di attraversare cockpit, veicoli o pedoni. | [JPG](05_internal_external_conflict/contact_sheet.jpg) |

## Indice completo delle immagini

### Linee stradali che scompaiono e ricompaiono

- [Contact sheet](01_lane_dropout/contact_sheet.jpg)
- Frame 13944: [RGB](01_lane_dropout/frame_013944_rgb.jpg), [maschera PNG](01_lane_dropout/frame_013944_mask.png), [overlay](01_lane_dropout/frame_013944_overlay.jpg)
- Frame 13949: [RGB](01_lane_dropout/frame_013949_rgb.jpg), [maschera PNG](01_lane_dropout/frame_013949_mask.png), [overlay](01_lane_dropout/frame_013949_overlay.jpg)
- Frame 13953: [RGB](01_lane_dropout/frame_013953_rgb.jpg), [maschera PNG](01_lane_dropout/frame_013953_mask.png), [overlay](01_lane_dropout/frame_013953_overlay.jpg)

### Linee spezzate con gap compatibili

- [Contact sheet](02_broken_lane_link/contact_sheet.jpg)
- Frame 14177: [RGB](02_broken_lane_link/frame_014177_rgb.jpg), [maschera PNG](02_broken_lane_link/frame_014177_mask.png), [overlay](02_broken_lane_link/frame_014177_overlay.jpg)
- Frame 14182: [RGB](02_broken_lane_link/frame_014182_rgb.jpg), [maschera PNG](02_broken_lane_link/frame_014182_mask.png), [overlay](02_broken_lane_link/frame_014182_overlay.jpg)
- Frame 14186: [RGB](02_broken_lane_link/frame_014186_rgb.jpg), [maschera PNG](02_broken_lane_link/frame_014186_mask.png), [overlay](02_broken_lane_link/frame_014186_overlay.jpg)

### Cordolo o bordo strada intermittente

- [Contact sheet](03_intermittent_boundary/contact_sheet.jpg)
- Frame 14300: [RGB](03_intermittent_boundary/frame_014300_rgb.jpg), [maschera PNG](03_intermittent_boundary/frame_014300_mask.png), [overlay](03_intermittent_boundary/frame_014300_overlay.jpg)
- Frame 14305: [RGB](03_intermittent_boundary/frame_014305_rgb.jpg), [maschera PNG](03_intermittent_boundary/frame_014305_mask.png), [overlay](03_intermittent_boundary/frame_014305_overlay.jpg)
- Frame 14309: [RGB](03_intermittent_boundary/frame_014309_rgb.jpg), [maschera PNG](03_intermittent_boundary/frame_014309_mask.png), [overlay](03_intermittent_boundary/frame_014309_overlay.jpg)

### Cockpit, mani, specchi o display con flicker

- [Contact sheet](04_internal_flicker/contact_sheet.jpg)
- Frame 13923: [RGB](04_internal_flicker/frame_013923_rgb.jpg), [maschera PNG](04_internal_flicker/frame_013923_mask.png), [overlay](04_internal_flicker/frame_013923_overlay.jpg)
- Frame 13928: [RGB](04_internal_flicker/frame_013928_rgb.jpg), [maschera PNG](04_internal_flicker/frame_013928_mask.png), [overlay](04_internal_flicker/frame_013928_overlay.jpg)
- Frame 13932: [RGB](04_internal_flicker/frame_013932_rgb.jpg), [maschera PNG](04_internal_flicker/frame_013932_mask.png), [overlay](04_internal_flicker/frame_013932_overlay.jpg)

### Conflitto tra interno ed esterno

- [Contact sheet](05_internal_external_conflict/contact_sheet.jpg)
- Frame 13969: [RGB](05_internal_external_conflict/frame_013969_rgb.jpg), [maschera PNG](05_internal_external_conflict/frame_013969_mask.png), [overlay](05_internal_external_conflict/frame_013969_overlay.jpg)
- Frame 13974: [RGB](05_internal_external_conflict/frame_013974_rgb.jpg), [maschera PNG](05_internal_external_conflict/frame_013974_mask.png), [overlay](05_internal_external_conflict/frame_013974_overlay.jpg)
- Frame 13978: [RGB](05_internal_external_conflict/frame_013978_rgb.jpg), [maschera PNG](05_internal_external_conflict/frame_013978_mask.png), [overlay](05_internal_external_conflict/frame_013978_overlay.jpg)


Le maschere sono PNG; RGB, overlay e contact sheet sono JPG. Le preview individuali sono 960×720 e non includono l’intero output.

La selezione è diagnostica e non implica un miglioramento di accuratezza in assenza di ground truth.

Codici provenance del final pass: `1` current_model, `2` temporal_evidence, `3` morphological_link, `4` final_fill
