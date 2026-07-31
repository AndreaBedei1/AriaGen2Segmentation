# Article 1 semantic-camera final QA

Preview leggere della sola modalità finale offline/presentation. Il risultato precedente è la presentazione stabilizzata del branch di partenza; gli output scientifici raw sono rimasti invariati.

| Sequenza | Frame | Timestamp clip | Motivo | Contact sheet |
|---|---:|---:|---|---|
| [Linee stradali che scompaiono e ricompaiono](01_lane_dropout/README.md) | 1911–1920 | 191.499 s | La finestra flow-aligned ±10 recupera evidenza lane nei frame intermedi e il filtro geometrico la limita al supporto stradale. | [JPG](01_lane_dropout/contact_sheet.jpg) |
| [Linee spezzate con gap compatibili](02_broken_lane_link/README.md) | 2082–2091 | 208.598 s | Closing multi-orientamento e collegamento controllato degli estremi riducono piccoli gap senza attraversare barriere semantiche. | [JPG](02_broken_lane_link/contact_sheet.jpg) |
| [Cordolo o bordo strada intermittente](03_intermittent_boundary/README.md) | 1809–1818 | 181.300 s | Persistenza temporale, prossimità al bordo strada e pulizia delle componenti rendono più continuo road_boundary_or_obstacle. | [JPG](03_intermittent_boundary/contact_sheet.jpg) |
| [Cockpit, mani, specchi o display con flicker](04_internal_flicker/README.md) | 1804–1813 | 180.800 s | Persistenza lunga e isteresi per le classi interne recuperano i dropout; le mani restano nella classe control_and_ego_vehicle. | [JPG](04_internal_flicker/contact_sheet.jpg) |
| [Conflitto tra interno ed esterno](05_internal_external_conflict/README.md) | 1878–1887 | 188.200 s | La fusione conserva le regioni esterne forti e vieta ai collegamenti di attraversare cockpit, veicoli o pedoni. | [JPG](05_internal_external_conflict/contact_sheet.jpg) |

## Indice completo delle immagini

### Linee stradali che scompaiono e ricompaiono

- [Contact sheet](01_lane_dropout/contact_sheet.jpg)
- Frame 1911: [RGB](01_lane_dropout/frame_001911_rgb.jpg), [maschera PNG](01_lane_dropout/frame_001911_mask.png), [overlay](01_lane_dropout/frame_001911_overlay.jpg)
- Frame 1916: [RGB](01_lane_dropout/frame_001916_rgb.jpg), [maschera PNG](01_lane_dropout/frame_001916_mask.png), [overlay](01_lane_dropout/frame_001916_overlay.jpg)
- Frame 1920: [RGB](01_lane_dropout/frame_001920_rgb.jpg), [maschera PNG](01_lane_dropout/frame_001920_mask.png), [overlay](01_lane_dropout/frame_001920_overlay.jpg)

### Linee spezzate con gap compatibili

- [Contact sheet](02_broken_lane_link/contact_sheet.jpg)
- Frame 2082: [RGB](02_broken_lane_link/frame_002082_rgb.jpg), [maschera PNG](02_broken_lane_link/frame_002082_mask.png), [overlay](02_broken_lane_link/frame_002082_overlay.jpg)
- Frame 2087: [RGB](02_broken_lane_link/frame_002087_rgb.jpg), [maschera PNG](02_broken_lane_link/frame_002087_mask.png), [overlay](02_broken_lane_link/frame_002087_overlay.jpg)
- Frame 2091: [RGB](02_broken_lane_link/frame_002091_rgb.jpg), [maschera PNG](02_broken_lane_link/frame_002091_mask.png), [overlay](02_broken_lane_link/frame_002091_overlay.jpg)

### Cordolo o bordo strada intermittente

- [Contact sheet](03_intermittent_boundary/contact_sheet.jpg)
- Frame 1809: [RGB](03_intermittent_boundary/frame_001809_rgb.jpg), [maschera PNG](03_intermittent_boundary/frame_001809_mask.png), [overlay](03_intermittent_boundary/frame_001809_overlay.jpg)
- Frame 1814: [RGB](03_intermittent_boundary/frame_001814_rgb.jpg), [maschera PNG](03_intermittent_boundary/frame_001814_mask.png), [overlay](03_intermittent_boundary/frame_001814_overlay.jpg)
- Frame 1818: [RGB](03_intermittent_boundary/frame_001818_rgb.jpg), [maschera PNG](03_intermittent_boundary/frame_001818_mask.png), [overlay](03_intermittent_boundary/frame_001818_overlay.jpg)

### Cockpit, mani, specchi o display con flicker

- [Contact sheet](04_internal_flicker/contact_sheet.jpg)
- Frame 1804: [RGB](04_internal_flicker/frame_001804_rgb.jpg), [maschera PNG](04_internal_flicker/frame_001804_mask.png), [overlay](04_internal_flicker/frame_001804_overlay.jpg)
- Frame 1809: [RGB](04_internal_flicker/frame_001809_rgb.jpg), [maschera PNG](04_internal_flicker/frame_001809_mask.png), [overlay](04_internal_flicker/frame_001809_overlay.jpg)
- Frame 1813: [RGB](04_internal_flicker/frame_001813_rgb.jpg), [maschera PNG](04_internal_flicker/frame_001813_mask.png), [overlay](04_internal_flicker/frame_001813_overlay.jpg)

### Conflitto tra interno ed esterno

- [Contact sheet](05_internal_external_conflict/contact_sheet.jpg)
- Frame 1878: [RGB](05_internal_external_conflict/frame_001878_rgb.jpg), [maschera PNG](05_internal_external_conflict/frame_001878_mask.png), [overlay](05_internal_external_conflict/frame_001878_overlay.jpg)
- Frame 1883: [RGB](05_internal_external_conflict/frame_001883_rgb.jpg), [maschera PNG](05_internal_external_conflict/frame_001883_mask.png), [overlay](05_internal_external_conflict/frame_001883_overlay.jpg)
- Frame 1887: [RGB](05_internal_external_conflict/frame_001887_rgb.jpg), [maschera PNG](05_internal_external_conflict/frame_001887_mask.png), [overlay](05_internal_external_conflict/frame_001887_overlay.jpg)


Le maschere sono PNG; RGB, overlay e contact sheet sono JPG. Le preview individuali sono 960×720 e non includono l’intero output.

La selezione è diagnostica e non implica un miglioramento di accuratezza in assenza di ground truth.

Codici provenance del final pass: `1` current_model, `2` temporal_evidence, `3` morphological_link, `4` final_fill
