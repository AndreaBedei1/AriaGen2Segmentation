# Article 1 temporal class policies

TTL and previous-evidence weights are initial, pre-GT values:

| class | TTL | previous weight | current support |
|---|---:|---:|---|
| road_surface | 3 | 0.55 | no |
| lane_marking | 3 | 0.75 | road support |
| regulatory_road_marking | 2 | 0.65 | road support |
| vehicle | 1 | 0.35 | required |
| two_wheeler | 1 | 0.25 | required |
| pedestrian | 1 | 0.20 | required |
| traffic_light / traffic_sign | 2 | 0.45 | conservative |
| road_boundary_or_obstacle | 2 | 0.45 | conservative |
| other_environment / future cockpit classes | 1 | 0.15 | conservative |
| unknown | 0 | 0 | never propagated |

Dynamic classes require current top-2, unknown, or minimum same-class probability support.
A confident new current class wins immediately. Unknown recovery requires valid flow,
eligible previous confidence and unexpired TTL.

Thin markings use their static filtered layer, current road support and a separate decaying
state. Lane TTL is three frames and regulatory TTL two. Propagation cannot cross current
non-road exclusions or invalid flow.
