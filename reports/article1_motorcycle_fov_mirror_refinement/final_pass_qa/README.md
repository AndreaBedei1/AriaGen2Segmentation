# Article 1 semantic-camera final QA

Preview leggere della sola modalità finale offline/presentation. Il risultato precedente è la presentazione stabilizzata del branch di partenza; gli output scientifici raw sono rimasti invariati.

| Sequenza | Frame | Timestamp clip | Motivo | Contact sheet |
|---|---:|---:|---|---|
| [Linee stradali che scompaiono e ricompaiono](01_lane_dropout/README.md) | 13984–13993 | 932.599 s | La finestra flow-aligned ±10 recupera evidenza lane nei frame intermedi e il filtro geometrico la limita al supporto stradale. | [JPG](01_lane_dropout/contact_sheet.jpg) |
| [Linee spezzate con gap compatibili](02_broken_lane_link/README.md) | 14263–14272 | 951.198 s | Closing multi-orientamento e collegamento controllato degli estremi riducono piccoli gap senza attraversare barriere semantiche. | [JPG](02_broken_lane_link/contact_sheet.jpg) |
| [Cordolo o bordo strada intermittente](03_intermittent_boundary/README.md) | 14240–14249 | 949.665 s | Persistenza temporale, prossimità al bordo strada e pulizia delle componenti rendono più continuo road_boundary_or_obstacle. | [JPG](03_intermittent_boundary/contact_sheet.jpg) |
| [Cockpit, mani, specchi o display con flicker](04_internal_flicker/README.md) | 14040–14049 | 936.332 s | Persistenza lunga e isteresi per le classi interne recuperano i dropout; le mani restano nella classe control_and_ego_vehicle. | [JPG](04_internal_flicker/contact_sheet.jpg) |
| [Conflitto tra interno ed esterno](05_internal_external_conflict/README.md) | 14045–14054 | 936.666 s | La fusione conserva le regioni esterne forti e vieta ai collegamenti di attraversare cockpit, veicoli o pedoni. | [JPG](05_internal_external_conflict/contact_sheet.jpg) |

## Indice completo delle immagini

### Linee stradali che scompaiono e ricompaiono

- [Contact sheet](01_lane_dropout/contact_sheet.jpg)
- Frame 13984: [RGB](01_lane_dropout/frame_013984_rgb.jpg), [maschera PNG](01_lane_dropout/frame_013984_mask.png), [overlay](01_lane_dropout/frame_013984_overlay.jpg)
- Frame 13989: [RGB](01_lane_dropout/frame_013989_rgb.jpg), [maschera PNG](01_lane_dropout/frame_013989_mask.png), [overlay](01_lane_dropout/frame_013989_overlay.jpg)
- Frame 13993: [RGB](01_lane_dropout/frame_013993_rgb.jpg), [maschera PNG](01_lane_dropout/frame_013993_mask.png), [overlay](01_lane_dropout/frame_013993_overlay.jpg)

### Linee spezzate con gap compatibili

- [Contact sheet](02_broken_lane_link/contact_sheet.jpg)
- Frame 14263: [RGB](02_broken_lane_link/frame_014263_rgb.jpg), [maschera PNG](02_broken_lane_link/frame_014263_mask.png), [overlay](02_broken_lane_link/frame_014263_overlay.jpg)
- Frame 14268: [RGB](02_broken_lane_link/frame_014268_rgb.jpg), [maschera PNG](02_broken_lane_link/frame_014268_mask.png), [overlay](02_broken_lane_link/frame_014268_overlay.jpg)
- Frame 14272: [RGB](02_broken_lane_link/frame_014272_rgb.jpg), [maschera PNG](02_broken_lane_link/frame_014272_mask.png), [overlay](02_broken_lane_link/frame_014272_overlay.jpg)

### Cordolo o bordo strada intermittente

- [Contact sheet](03_intermittent_boundary/contact_sheet.jpg)
- Frame 14240: [RGB](03_intermittent_boundary/frame_014240_rgb.jpg), [maschera PNG](03_intermittent_boundary/frame_014240_mask.png), [overlay](03_intermittent_boundary/frame_014240_overlay.jpg)
- Frame 14245: [RGB](03_intermittent_boundary/frame_014245_rgb.jpg), [maschera PNG](03_intermittent_boundary/frame_014245_mask.png), [overlay](03_intermittent_boundary/frame_014245_overlay.jpg)
- Frame 14249: [RGB](03_intermittent_boundary/frame_014249_rgb.jpg), [maschera PNG](03_intermittent_boundary/frame_014249_mask.png), [overlay](03_intermittent_boundary/frame_014249_overlay.jpg)

### Cockpit, mani, specchi o display con flicker

- [Contact sheet](04_internal_flicker/contact_sheet.jpg)
- Frame 14040: [RGB](04_internal_flicker/frame_014040_rgb.jpg), [maschera PNG](04_internal_flicker/frame_014040_mask.png), [overlay](04_internal_flicker/frame_014040_overlay.jpg)
- Frame 14045: [RGB](04_internal_flicker/frame_014045_rgb.jpg), [maschera PNG](04_internal_flicker/frame_014045_mask.png), [overlay](04_internal_flicker/frame_014045_overlay.jpg)
- Frame 14049: [RGB](04_internal_flicker/frame_014049_rgb.jpg), [maschera PNG](04_internal_flicker/frame_014049_mask.png), [overlay](04_internal_flicker/frame_014049_overlay.jpg)

### Conflitto tra interno ed esterno

- [Contact sheet](05_internal_external_conflict/contact_sheet.jpg)
- Frame 14045: [RGB](05_internal_external_conflict/frame_014045_rgb.jpg), [maschera PNG](05_internal_external_conflict/frame_014045_mask.png), [overlay](05_internal_external_conflict/frame_014045_overlay.jpg)
- Frame 14050: [RGB](05_internal_external_conflict/frame_014050_rgb.jpg), [maschera PNG](05_internal_external_conflict/frame_014050_mask.png), [overlay](05_internal_external_conflict/frame_014050_overlay.jpg)
- Frame 14054: [RGB](05_internal_external_conflict/frame_014054_rgb.jpg), [maschera PNG](05_internal_external_conflict/frame_014054_mask.png), [overlay](05_internal_external_conflict/frame_014054_overlay.jpg)


Le maschere sono PNG; RGB, overlay e contact sheet sono JPG. Le preview individuali sono 960×720 e non includono l’intero output.

La selezione è diagnostica e non implica un miglioramento di accuratezza in assenza di ground truth.

Codici provenance del final pass: `1` current_model, `2` temporal_evidence, `3` morphological_link, `4` final_fill
