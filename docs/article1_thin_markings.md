# Thin road-marking filtering

Raw lane/regulatory probabilities and masks are never overwritten. A filtered component
must meet its probability and margin thresholds, lie within a dilation of independently
predicted road_surface, avoid conflicting non-road classes and pass component-area limits.
Conservative opening/closing parameters are configurable and default to no expansion.

Reason codes are: none, low_probability, low_margin, outside_road_support,
conflicts_with_nonroad, component_too_small, component_too_large and accepted. The final
composite applies accepted lane marking, then regulatory marking, over the base mask.
Precision, recall, boundary F1 and temporal continuity remain blocked on reviewed GT.
