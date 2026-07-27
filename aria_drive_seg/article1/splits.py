"""Leakage-safe scientific splits for Article 1."""
from __future__ import annotations

from typing import Iterable

import pandas as pd


ALLOWED_UNITS = ("participant_id", "session_id", "matched_pair_id", "route_segment_id")


def grouped_split(df: pd.DataFrame, unit: str, validation_groups: Iterable[str]):
    if unit not in ALLOWED_UNITS:
        raise ValueError(f"split unit must be one of {ALLOWED_UNITS}")
    if unit not in df:
        raise ValueError(f"missing split column {unit}")
    groups = {str(x) for x in validation_groups}
    valid = df[df[unit].astype(str).isin(groups)].copy()
    train = df[~df[unit].astype(str).isin(groups)].copy()
    assert_no_group_leakage(train, valid, unit)
    return train, valid


def assert_no_group_leakage(train: pd.DataFrame, valid: pd.DataFrame, unit: str):
    overlap = set(train[unit].dropna().astype(str)) & set(valid[unit].dropna().astype(str))
    if overlap:
        raise ValueError(f"leakage in {unit}: {sorted(overlap)}")


def assert_no_near_frame_leakage(train: pd.DataFrame, valid: pd.DataFrame,
                                 min_frame_distance: int = 10):
    for session in set(train.session_id) & set(valid.session_id):
        a = train.loc[train.session_id == session, "frame_index"].to_numpy()
        b = valid.loc[valid.session_id == session, "frame_index"].to_numpy()
        if a.size and b.size and abs(a[:, None] - b[None, :]).min() < min_frame_distance:
            raise ValueError(f"near-frame leakage in session {session}")
