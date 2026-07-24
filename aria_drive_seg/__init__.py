"""aria_drive_seg — fully-local dual-method segmentation + eye-gaze analysis
for Meta Project Aria Gen2 driving recordings.

The top-level package intentionally imports nothing heavy (no torch, no
projectaria_tools) so it loads in every environment. Heavy dependencies are
imported lazily inside the submodule that needs them.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
