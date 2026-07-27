# Article 1 taxonomy

The authoritative file is `configs/article1/classes_article1.yaml`.

IDs 0–13 are: unknown, road_surface, lane_marking, regulatory_road_marking, vehicle,
two_wheeler, pedestrian, traffic_light, traffic_sign, road_boundary_or_obstacle, mirror,
instrument_display, control_and_ego_vehicle and other_environment.

`unknown` means unsupported or insufficiently confident. `other_environment` means a known,
non-central semantic region. Thin marking classes are stored separately and may overwrite
road_surface in the composited view without deleting the base mask.

Unknown is decided in macro-probability space from configurable probability, top-1/top-2
margin, normalized entropy and unsupported-native evidence. Its reason map is a bitmask
(1 low probability, 2 low margin, 4 high entropy, 8 unsupported native). It is not tuned to
a target percentage. Mapillary ego pixels remain traceable via `mapillary_ego_region` and
provenance, but map to `other_environment`.
