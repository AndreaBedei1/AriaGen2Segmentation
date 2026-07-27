# Article 1 cockpit annotation protocol

Primary classes are mirror, instrument_display, control_and_ego_vehicle,
background_internal and ignore. Use one symmetric definition for car and motorcycle.

Attributes:

- mirror_side: left, right, central, unknown;
- control_type: steering_wheel, handlebar, hand, console, ego_body, other;
- display_type: instrument_cluster, navigation, central_display, motorcycle_display, other.

Attributes are not training classes. Include partial occlusions, hands on/off controls,
display on/off states and visible motorcycle body consistently. Grounded-SAM2 B/C seeds may
be reviewed; D is excluded. Training is forbidden until both domains are represented and
review status is recorded.
