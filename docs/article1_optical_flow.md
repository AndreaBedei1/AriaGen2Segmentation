# Article 1 optical flow

The mandatory backend is local OpenCV DIS. It operates on rectified RGB-derived grayscale
images at the temporal-grid resolution. No model is downloaded. An unconfigured backend,
including RAFT without verified local weights, fails closed.

Both previous→current and current→previous flows are computed. Validity requires:

- current coordinates map inside the previous frame;
- finite vectors;
- forward/backward residual below the configured threshold;
- photometric residual below its threshold.

Invalid pixels form the occlusion map. Vector magnitudes are scaled whenever flow geometry
changes. A frame resets to current-only when valid fraction, median motion or photometric
error violates policy. Flow arrays are stored only for selected diagnostic frames.
