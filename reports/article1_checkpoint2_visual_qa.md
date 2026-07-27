# Article 1 checkpoint 2 — visual QA

The review used the same 180–210 s car clip and a 30-frame contact sheet sampled uniformly
from the 60 rendered frames. This is visual QA, not an accuracy evaluation.

## Explicit frame checks

- Lane marking (10): 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910.
- Regulatory marking (5): 1830, 1835, 1940, 1955, 2035.
- Reflections/high-contrast rejection (5): 1840, 1865, 1880, 1905, 1920.
- Cockpit/native ego region visible (5): 1805, 1845, 1885, 1955, 2035.
- Vehicles ahead/adjacent (5): 1875, 1895, 1955, 1985, 2045.

## Findings

- The scientific `control_and_ego_vehicle` output is gone from the external model; the
  native ego region remains visible through its flag/provenance.
- Road support removes many thin responses on cockpit edges and reflections.
- Raw thin candidates remain available and visibly denser than filtered output.
- Lane continuity remains acceptable in many road frames, but some true thin fragments are
  removed; GT is needed to quantify the precision/recall trade-off.
- Regulatory markings remain the most uncertain class: a few wide white regions and
  crosswalk-like responses still require manual calibration.
- Unknown reason visualization is sparse but nonzero and follows confidence/margin/entropy
  boundaries rather than a target percentage.

Across all 60 frames, mean raw thin pixels fell from 26,112 to 6,040 after filtering.
This reduction is not itself evidence of improved accuracy.
