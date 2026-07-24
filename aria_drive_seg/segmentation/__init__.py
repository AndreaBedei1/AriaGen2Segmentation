"""Segmentation subpackage (Method 1: Grounded-SAM2, Method 2: OneFormer).

Heavy deps (torch, transformers, sam2, detectron2) are imported lazily inside the
method modules so the shared schema/overlap utilities import anywhere.
"""
