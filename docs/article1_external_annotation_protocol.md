# Article 1 external annotation protocol

Annotate only visible external semantics: road_surface, lane_marking,
regulatory_road_marking, vehicle, two_wheeler, pedestrian, traffic_light, traffic_sign,
road_boundary_or_obstacle and other_environment. Use unknown/void for genuinely ambiguous,
occluded or unsupported pixels. Do not annotate mirror, displays or ego controls here.

Lane and regulatory markings are allowed to overlap the conceptual road surface, but the
exported composited mask gives the marking priority. Keep thin boundaries precise; do not
merge parallel lines. Regulatory attributes may record crosswalk, stop_line, arrow or other.

Model seeds are editable proposals with provenance, never GT. At least part of the
development set should remain unseeded. A reviewed export needs a reviewer marker before the
conversion validator accepts it. Resolve disagreements by consensus and document void areas.
