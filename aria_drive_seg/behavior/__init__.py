"""Article 1 multimodal driver-behaviour analysis.

Everything in this package is **exploratory pilot** work: one participant, one
session per vehicle, a car recorded at ~10 fps and a motorcycle at ~15 fps. No
result produced here generalises to a population and none of it is a medical
measurement.

The package sits on top of the frozen Article 1 semantic-camera baseline
(`feature/article1-motorcycle-ingestion`, commit `a9ee61b`). It reads that
baseline's outputs; it never re-tunes the segmentation and never feeds gaze back
into it.
"""
from __future__ import annotations

#: Marker attached to every artefact this package writes.
EXPLORATORY_MARKER = "exploratory_pilot"

#: The semantic-camera baseline these analyses are allowed to consume.
BASELINE_BRANCH = "feature/article1-motorcycle-ingestion"
BASELINE_COMMIT = "a9ee61bc77cdf61f364eb0eec98c2975fd933829"

#: External classes treated as reliable enough to carry a primary metric.
RELIABLE_EXTERNAL_CLASSES = (
    "road_surface",
    "lane_marking",
    "regulatory_road_marking",
    "vehicle",
    "two_wheeler",
    "pedestrian",
    "traffic_light",
    "traffic_sign",
    "road_boundary_or_obstacle",
)

#: Classes that stay exploratory proxies and need manual review before use.
PROXY_CLASSES = (
    "mirror",
    "instrument_display",
    "control_and_ego_vehicle",
    "hands",
)

#: The `temperature` stream is a device thermal sensor, never a body temperature.
TEMPERATURE_SEMANTICS = "device_or_environment_temperature_context"
