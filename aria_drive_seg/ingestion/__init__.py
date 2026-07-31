"""Article 1 motorcycle ingestion: inventory, stream QA, timestamp-driven extraction.

Everything in this package is frame-rate agnostic. Temporal parameters are always
expressed in seconds and converted to a frame count from the *measured* timestamps
of the recording being processed (never from a nominal profile name, never from a
hard-coded 10 or 15 fps constant).
"""
