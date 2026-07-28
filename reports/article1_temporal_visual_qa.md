# Article 1 temporal checkpoint 1 — visual QA

Date: 2026-07-28  
Clip: frames 1801–2100, nominal 10 fps, VRS interval 180–210 s  
Views: `output/article1/temporal_videos_30s/qa_contact_categories.jpg`,
`contact_sheet.jpg`, the four `preview_*.png` images, and videos 03–06.

## Status and method

This is a visual, non-GT review. “Apparently correct/incorrect” means only that the
overlay appears consistent/inconsistent with the RGB image at contact-sheet resolution.
It is not an accuracy annotation. Candidate selection was reproducible and
metric-driven; the visual judgments below were made afterwards. `valid` reports the
flow-valid fraction. A reset means that no prior temporal evidence was used.

## Stable lane-marking candidates

| frame | valid | apparent review |
|---:|---:|---|
| 1847 | 0.514 | Raw and temporal lane/road boundary are visually stable; temporal adds only a small edge fragment. No obvious false persistence. |
| 1839 | 0.390, reset | Current-only result; raw and temporal are visually equivalent apart from grid resampling. No recovery claim. |
| 1838 | 0.390, reset | Current-only result under head motion; no propagated line is visible. |
| 1826 | 0.391, reset | Current-only result; thin fragments remain fragmented, as expected after reset. |
| 1845 | 0.475 | Small temporal extension along the right road edge appears plausible, but cannot be certified without GT. |
| 1925 | 0.547 | Main lane region remains stable; temporal fills a small boundary gap. Useful-looking recovery, low visible persistence. |
| 2060 | 0.561 | Dense traffic; static and temporal road/lane geometry remain aligned. |
| 1859 | 0.603 | Highest-flow-quality sample in this group; narrow lane structure is retained with little change. |
| 1926 | 0.530 | Minor edge recovery; no obvious override of vehicles. |
| 1865 | 0.470 | Raw and temporal are close; the temporal right-edge extension is plausible but also a reflection-risk area. |

## One-frame lane-dropout candidates

| frame | valid | apparent review |
|---:|---:|---|
| 1938 | 0.528 | Clear additional white thin-line continuity in temporal; useful-looking recovery, with some local widening. |
| 1916 | 0.532 | Temporal reconnects small missing lane fragments; no obvious vehicle overlap. |
| 1887 | 0.578 | Modest recovery along the left/centre thin marking; apparently useful. |
| 1883 | 0.503 | Several thin fragments are retained; some fragmentation remains and correctness is uncertain. |
| 1900 | 0.498 | Temporal fills a lane gap, but the broader white region could be over-persistence. |
| 1910 | 0.512 | Small, spatially aligned lane recovery; apparently useful. |
| 2022 | 0.617 | Recovery occurs in dense traffic; boundaries stay mostly off vehicles, but the crowded scene raises false-persistence risk. |
| 1878 | 0.531 | Minor line recovery with limited visual change. |
| 2047 | 0.485 | Thin recovery is visible between vehicles; potential occlusion-boundary leakage remains. |
| 1921 | 0.524 | Short missing segment is retained; no conspicuous displaced trail. |

## Regulatory-road-marking candidates

| frame | valid | apparent review |
|---:|---:|---|
| 1956 | 0.450 | Large painted-road region is retained, but fragmented vehicle/road boundaries make the temporal gain ambiguous. |
| 2036 | 0.575 | Marking continuity improves in dense traffic; small fragments near vehicles may be false persistence. |
| 2033 | 0.520 | Raw and temporal are broadly consistent; temporal fills isolated gaps. |
| 2032 | 0.529 | Strongest visible modification in the group; useful-looking road-paint fill with non-negligible overfill risk. |
| 2031 | 0.624 | Good flow support; marking persists, though static errors near the cockpit remain present. |

## Vehicle candidates

| frame | valid | apparent review |
|---:|---:|---|
| 1805 | 0.422 | Dominant “vehicle” area is actually cockpit/specular structure in the RGB. This is a static false positive and temporal preserves parts of it. |
| 1817 | 0.490 | Same cockpit false-positive failure mode; temporal does not repair semantic taxonomy errors. |
| 1806 | 0.485 | Cockpit/specchietto dominates the candidate count; false-positive persistence is visible but short-lived. |
| 1807 | 0.482 | Cockpit is again mislabeled as vehicle; temporal differences are small relative to the static error. |
| 1810 | 0.479 | Large static cockpit error is carried into temporal output; this is the clearest vehicle-class failure in the clip. |
| 2053 | 0.448 | Real vehicles in dense traffic remain spatially aligned; conservative TTL avoids an obvious trail. |
| 2052 | 0.494 | Vehicle masks are stable; small boundary differences are visible without clear ghosting. |
| 2027 | 0.622 | Good flow support and aligned vehicle boundaries; apparently conservative propagation. |
| 2054 | 0.373, reset | Current-only due insufficient valid flow; no vehicle propagation through the motion event. |
| 2045 | 0.436 | Multiple vehicles remain aligned; minor edge retention is visible, with no long trail. |

## Pedestrian/two-wheeler candidates

| frame | valid | apparent review |
|---:|---:|---|
| 1823 | 0.568 | Candidate pixels largely correspond to the driver's hands/arms, not a clearly visible external pedestrian/two-wheeler. Static false positive persists. |
| 1822 | 0.431 | Same hand/arm failure mode; temporal does not materially enlarge it. |
| 1825 | 0.428 | Driver hand dominates the candidate; no external target can be confidently reviewed. |
| 1820 | 0.486 | Large hand/arm region is apparently mislabeled; temporal preservation is undesirable but conservative in duration. |
| 1819 | 0.491 | No clearly reviewable external pedestrian/two-wheeler; candidate is mainly ego-body/cockpit confusion. |

No clearly visible external pedestrian or two-wheeler was found among these five
metric-selected frames. They are retained in the QA precisely because they expose a
static-domain failure rather than being silently replaced with easier examples.

## Traffic-sign and traffic-light candidates

| class | frame | valid | apparent review |
|---|---:|---:|---|
| sign | 1966 | 0.530 | Yellow sign regions align with visible roadside signs; temporal change is small. |
| sign | 2036 | 0.575 | Several small signs in dense traffic remain stable; boundary correctness is uncertain at this scale. |
| sign | 2044 | 0.476 | Main sign remains present with no obvious trail. |
| sign | 1954 | 0.512 | Sign persistence appears spatially aligned; nearby road-marking change is larger than sign change. |
| sign | 2037 | 0.535 | Small sign candidates remain stable; no obvious false extension. |
| light | 1951 | 0.472 | Only a very small traffic-light prediction is present; temporal is visually similar and too small for an accuracy claim. |
| light | 2065 | 0.562 | Candidate is tiny/ambiguous in the RGB; no visible ghosting. |
| light | 2073 | 0.538 | Candidate remains too small for confident semantic review; temporal change is limited. |
| light | 2075 | 0.475 | Tiny candidate, raw and temporal close. |
| light | 1950 | 0.562 | Tiny candidate, no obvious persistence trail; correctness unresolved. |

## Strong motion / low-flow validity

| frame | valid | apparent review |
|---:|---:|---|
| 2054 | 0.373, reset | Correct fail-closed behavior; current-only output under strong motion. |
| 1838 | 0.390, reset | Correct reset; no previous mask trail. |
| 1839 | 0.390, reset | Consecutive reset because flow remains below threshold. |
| 1826 | 0.391, reset | Correct reset; fragmentation is preferred to unsupported propagation. |
| 1804 | 0.396, reset | Correct reset close to the 0.40 validity gate. |
| 1836 | 0.399, reset | Correct reset at the threshold boundary. |
| 1802 | 0.408 | Flow just passes; visible temporal modification is larger (1.34%) but remains spatially local. |
| 1955 | 0.413 | No reset; dense-scene boundary changes merit caution. |
| 1841 | 0.413 | No obvious displaced trail, but right-edge thin persistence is uncertain. |
| 1837 | 0.416 | Borderline valid flow; changes remain small before the following resets. |

## Reflection candidates

| frame | valid | apparent review |
|---:|---:|---|
| 1840 | 0.474 | Temporal extends thin/road pixels near a reflective boundary; plausible recovery and false-persistence risk coexist. |
| 1865 | 0.470 | Small, stable change; no conspicuous reflection trail. |
| 1880 | 0.552 | Temporal boundary remains aligned, with isolated uncertain fragments. |
| 1905 | 0.562 | Hand/reflection region remains a static semantic failure; lane recovery is modest. |
| 1920 | 0.459 | Broader road/lane change near reflective cockpit geometry; potential over-persistence. |

## Reset audit

All resets were inspected: 1801 (`initial_frame`) and 1804, 1826, 1836, 1838,
1839, 2054 (`insufficient_valid_flow`). Provenance is `reset_current_only`; prior
temporal evidence is not used. Small raw/temporal differences (0.16–0.42%) come from
the half-resolution temporal probability grid and resampling, not propagation.

## Overall visual conclusion

- Useful-looking recoveries are concentrated in short lane/regulatory gaps, notably
  1938, 1916, 1887, 1910 and 1921.
- No long displaced dynamic-object trail was apparent; TTL=1 and current-support gates
  look conservative on the visible vehicles.
- The most important failure mode is inherited from the static model: cockpit,
  mirrors and hands can be assigned to vehicle/pedestrian-like classes. Temporal
  stabilization cannot correct this semantic error and can briefly preserve it.
- Border/reflection areas can receive plausible but uncertain thin-line extensions.
- The review does not establish that T4 is more accurate than T0. GT review is required.
