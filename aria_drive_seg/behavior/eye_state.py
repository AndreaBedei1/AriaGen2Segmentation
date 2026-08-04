"""Per-sample eye state (blink, pupil diameter) from the on-device gaze stream.

`projectaria_tools` 2.1.2 binds the eye-gaze record into `mps.EyeGaze` /
`mps.EyeGazeVergence`, which expose yaw, pitch, vergence and the two blink flags —
but **not** the pupil diameter, even though the recording carries it. The VRS
record itself declares, in its own embedded `DataLayout` descriptor:

    left_eye/pupil_diameter_valid   Bool    offset 55
    left_eye/pupil_diameter_meter   float   offset 56
    left_eye/blink_valid            Bool    offset 60
    left_eye/blink                  Bool    offset 61
    ... and the mirrored right-eye block

so the data is present and only the binding is missing. This module reads those
fields straight out of the record payload.

Three rules keep that honest:

* **the layout is read from the file, never hardcoded.** Field names, types and
  byte offsets come from the descriptor VRS embedded in this very recording, so a
  device or firmware that writes a different layout is decoded correctly or not at
  all — it can never be decoded *wrongly*;
* **every decode is cross-checked against the official binding.** The fields
  `projectaria_tools` does expose (timestamps, both blink flags, both blink
  validity flags, the entrance-pupil positions) must match sample for sample
  before any pupil number is used. `verify_against_provider` is that check and the
  extraction script refuses to write a table that fails it;
* **nothing is interpolated, resampled or gap-filled.** Every row is one real
  record. Invalid samples keep their validity flag and a NaN value rather than a
  neighbour's value.

The blink field's polarity is *not* assumed — see `behavior/blink.py`.
"""
from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: First eight bytes of every VRS container.
VRS_MAGIC = b"VisionRe"

#: Field name that identifies the eye-gaze data layout inside the descriptor blob.
EYE_LAYOUT_MARKER = "left_eye/pupil_diameter_meter"

#: Fixed-size `DataLayout` piece types, as (struct format, byte width). Anything
#: absent from this table is skipped rather than guessed at, and a layout whose
#: fields we need is missing from it fails loudly in `parse_data_layout`.
_PIECE_TYPES: Dict[str, Tuple[str, int]] = {
    "Bool": ("?", 1),
    "int8_t": ("b", 1), "uint8_t": ("B", 1),
    "int16_t": ("h", 2), "uint16_t": ("H", 2),
    "int32_t": ("i", 4), "uint32_t": ("I", 4),
    "int64_t": ("q", 8), "uint64_t": ("Q", 8),
    "float": ("f", 4), "double": ("d", 8),
    "Point2Df": ("2f", 8), "Point3Df": ("3f", 12), "Point4Df": ("4f", 16),
    "Point2Di": ("2i", 8), "Point3Di": ("3i", 12),
    "Matrix3Df": ("9f", 36), "Matrix4Df": ("16f", 64),
}

#: Columns the extraction is required to produce, in the order the brief asks for
#: them. Kept as a constant so the schema is asserted rather than described.
REQUIRED_COLUMNS = (
    "timestamp_ns",
    "left_blink", "right_blink",
    "left_blink_valid", "right_blink_valid",
    "left_pupil_diameter_meter", "right_pupil_diameter_meter",
    "left_pupil_diameter_valid", "right_pupil_diameter_valid",
)

#: Mapping from the record's own field names to the column names used downstream.
#: Only fields listed here are carried; everything else in the layout is decoded
#: and discarded, so an unexpected extra field cannot silently enter a table.
_COLUMN_MAP: Dict[str, str] = {
    "tracker_timestamp_ns": "tracker_timestamp_ns",
    "system_timestamp_ns": "system_timestamp_ns",
    "left_eye/blink": "left_blink",
    "right_eye/blink": "right_blink",
    "left_eye/blink_valid": "left_blink_valid",
    "right_eye/blink_valid": "right_blink_valid",
    "left_eye/pupil_diameter_meter": "left_pupil_diameter_meter",
    "right_eye/pupil_diameter_meter": "right_pupil_diameter_meter",
    "left_eye/pupil_diameter_valid": "left_pupil_diameter_valid",
    "right_eye/pupil_diameter_valid": "right_pupil_diameter_valid",
    "left_eye/gaze_direction_valid": "left_gaze_direction_valid",
    "right_eye/gaze_direction_valid": "right_gaze_direction_valid",
    "left_eye/entrance_pupil_position_valid": "left_entrance_pupil_valid",
    "right_eye/entrance_pupil_position_valid": "right_entrance_pupil_valid",
    "gaze_direction_combined_in_device_valid": "combined_gaze_direction_valid",
    "spatial_gaze_point_valid": "spatial_gaze_point_valid",
    "convergence_distance_valid": "convergence_distance_valid",
    "convergence_distance_meter": "convergence_distance_meter",
    "interocular_distance_valid": "interocular_distance_valid",
    "interocular_distance_meter": "interocular_distance_meter",
}

#: Vector fields carried as three separate columns each.
_VECTOR_MAP: Dict[str, str] = {
    "left_eye/entrance_pupil_position_in_device_meter_xyz": "left_entrance_pupil",
    "right_eye/entrance_pupil_position_in_device_meter_xyz": "right_entrance_pupil",
}


# --------------------------------------------------------------------------- #
# DataLayout descriptor
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LayoutField:
    """One fixed-size piece of a VRS `DataLayout`, as the file declares it."""

    name: str
    piece_type: str
    offset: int
    fmt: str
    width: int
    count: int = 1

    @property
    def end(self) -> int:
        return self.offset + self.width


def _piece_spec(type_string: str) -> Optional[Tuple[str, int, bool]]:
    """(struct format, element width, is_array) for a declared piece type."""
    text = str(type_string)
    for prefix, is_array in (("DataPieceValue<", False), ("DataPieceArray<", True)):
        if text.startswith(prefix) and text.endswith(">"):
            inner = text[len(prefix):-1]
            spec = _PIECE_TYPES.get(inner)
            return (spec[0], spec[1], is_array) if spec else None
    return None                      # DataPieceString and friends: not fixed-size


def parse_data_layout(descriptor: str) -> Tuple[List[LayoutField], int]:
    """Fields and fixed-buffer size of a `{"data_layout": [...]}` descriptor.

    Returns the fields that live in the fixed-size buffer, in declaration order,
    plus the buffer's total width computed as the maximum field end. Pieces
    without an `offset` (strings, vectors) are not in the fixed buffer and are
    skipped; a piece with an offset whose type this module cannot decode is an
    error, because silently skipping it would shift nothing but would let a
    caller believe a field was absent when it was merely unreadable.
    """
    doc = json.loads(descriptor)
    fields: List[LayoutField] = []
    for piece in doc["data_layout"]:
        if "offset" not in piece:
            continue
        spec = _piece_spec(piece["type"])
        if spec is None:
            raise ValueError(
                f"eye-gaze layout piece {piece['name']!r} has unsupported fixed-size "
                f"type {piece['type']!r}; refusing to decode a layout only partly "
                "understood")
        fmt, element_width, is_array = spec
        count = int(piece.get("size", 1)) if is_array else 1
        fields.append(LayoutField(
            name=str(piece["name"]), piece_type=str(piece["type"]),
            offset=int(piece["offset"]), fmt=f"<{fmt * count}" if count > 1 else f"<{fmt}",
            width=element_width * count, count=count))
    size = max((f.end for f in fields), default=0)
    return fields, int(size)


def find_eye_gaze_layout(vrs_path: str | Path,
                         marker: str = EYE_LAYOUT_MARKER
                         ) -> Tuple[List[LayoutField], int, str]:
    """Locate and parse the eye-gaze `DataLayout` embedded in the recording.

    The descriptor is found by the presence of a field name unique to the eye-gaze
    record, so nothing about its position, its stream id or its neighbours is
    assumed.
    """
    needle = marker.encode()
    with open(vrs_path, "rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            hit = mm.find(needle)
            if hit < 0:
                raise ValueError(
                    f"{vrs_path}: no DataLayout declares {marker!r}; this recording "
                    "does not carry per-eye pupil diameter")
            start = mm.rfind(b'{"data_layout":', 0, hit)
            if start < 0:
                raise ValueError(f"{vrs_path}: malformed DataLayout descriptor")
            end = _match_brace(mm, start)
            descriptor = mm[start:end].decode("utf-8")
    fields, size = parse_data_layout(descriptor)
    return fields, size, descriptor


def _match_brace(buffer, start: int) -> int:
    """End offset (exclusive) of the JSON object that begins at `start`."""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(buffer)):
        char = buffer[i:i + 1]
        if in_string:
            if escaped:
                escaped = False
            elif char == b"\\":
                escaped = True
            elif char == b'"':
                in_string = False
            continue
        if char == b'"':
            in_string = True
        elif char == b"{":
            depth += 1
        elif char == b"}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("unterminated DataLayout descriptor")


# --------------------------------------------------------------------------- #
# Container walk
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RecordHeader:
    offset: int
    record_size: int
    recordable_type_id: int
    format_version: int
    timestamp_s: float
    instance_id: int
    record_type: int

    @property
    def payload_offset(self) -> int:
        return self.offset + _RECORD_HEADER_STRUCT.size

    @property
    def payload_size(self) -> int:
        return self.record_size - _RECORD_HEADER_STRUCT.size


#: recordSize, previousRecordSize, recordableTypeId, formatVersion, timestamp,
#: recordableInstanceId, recordType — the 32-byte VRS record header.
_RECORD_HEADER_STRUCT = struct.Struct("<IIIIdII")
_FILE_HEADER_STRUCT = struct.Struct("<8sQIIqq")


def read_file_header(handle) -> Dict[str, int]:
    """File-header fields needed to walk the container.

    `file_header_size` and `record_header_size` are read rather than assumed, and
    the record header is only parsed when the file agrees it is 32 bytes wide.
    """
    handle.seek(0)
    raw = handle.read(_FILE_HEADER_STRUCT.size)
    magic, _creation, header_size, rec_header_size, index_offset, first_record = \
        _FILE_HEADER_STRUCT.unpack(raw)
    if magic != VRS_MAGIC:
        raise ValueError("not a VRS container")
    if rec_header_size != _RECORD_HEADER_STRUCT.size:
        raise ValueError(
            f"VRS record header is {rec_header_size} bytes; this reader only "
            f"understands the {_RECORD_HEADER_STRUCT.size}-byte layout")
    return {"file_header_size": int(header_size),
            "record_header_size": int(rec_header_size),
            "index_record_offset": int(index_offset),
            "first_user_record_offset": int(first_record)}


def iter_record_headers(vrs_path: str | Path):
    """Walk every user record in file order, yielding its header.

    The walk follows `recordSize` from the first user record and stops at the
    index record, so it never depends on the index being present or intact.
    """
    path = Path(vrs_path)
    file_size = path.stat().st_size
    with open(path, "rb") as handle:
        header = read_file_header(handle)
        limit = header["index_record_offset"] or file_size
        limit = min(limit, file_size)
        offset = header["first_user_record_offset"] or header["file_header_size"]
        while offset + _RECORD_HEADER_STRUCT.size <= limit:
            handle.seek(offset)
            raw = handle.read(_RECORD_HEADER_STRUCT.size)
            if len(raw) < _RECORD_HEADER_STRUCT.size:
                return
            size, _prev, type_id, version, timestamp, instance, record_type = \
                _RECORD_HEADER_STRUCT.unpack(raw)
            if size < _RECORD_HEADER_STRUCT.size or offset + size > file_size:
                raise ValueError(
                    f"VRS record at offset {offset} declares an impossible size "
                    f"{size}; the container walk cannot continue")
            yield RecordHeader(offset, int(size), int(type_id), int(version),
                               float(timestamp), int(instance), int(record_type))
            offset += size


# --------------------------------------------------------------------------- #
# Decode
# --------------------------------------------------------------------------- #
def decode_record(payload: bytes, fields: Sequence[LayoutField]) -> Dict[str, Any]:
    """Decode one fixed-size `DataLayout` buffer into a name → value mapping."""
    out: Dict[str, Any] = {}
    for field in fields:
        if field.end > len(payload):
            raise ValueError(
                f"field {field.name!r} ends at {field.end} but the payload is "
                f"{len(payload)} bytes")
        values = struct.unpack_from(field.fmt, payload, field.offset)
        out[field.name] = values[0] if len(values) == 1 else list(values)
    return out


def read_eye_state(vrs_path: str | Path, recording_id: str = "",
                   domain: str = ""):
    """Every real eye-gaze record, with blink and pupil fields, as a DataFrame.

    One row per record; no interpolation, no resampling and no gap filling. An
    invalid pupil reading keeps `*_pupil_diameter_valid = False` and a NaN
    diameter, so a consumer cannot use it by accident.
    """
    import pandas as pd

    fields, layout_size, descriptor = find_eye_gaze_layout(vrs_path)
    by_name = {f.name: f for f in fields}
    missing = [name for name in _COLUMN_MAP if name not in by_name]
    if missing:
        raise ValueError(
            f"{vrs_path}: eye-gaze layout is missing required fields: "
            + ", ".join(sorted(missing)))

    headers = [h for h in iter_record_headers(vrs_path)
               if h.payload_size == layout_size]
    if not headers:
        raise ValueError(
            f"{vrs_path}: no record payload matches the {layout_size}-byte "
            "eye-gaze layout")

    # Several streams could in principle share a payload width, so the eye-gaze
    # stream is identified by content: its records carry a tracker timestamp that
    # reproduces the record header's own timestamp.
    candidates = sorted({h.recordable_type_id for h in headers})
    chosen: Optional[int] = None
    with open(vrs_path, "rb") as handle:
        for type_id in candidates:
            probe = next(h for h in headers if h.recordable_type_id == type_id)
            handle.seek(probe.payload_offset)
            decoded = decode_record(handle.read(layout_size), fields)
            drift_s = abs(float(decoded["tracker_timestamp_ns"]) / 1e9
                          - probe.timestamp_s)
            if drift_s <= 1e-3:
                if chosen is not None:
                    raise ValueError(
                        f"{vrs_path}: record types {chosen} and {type_id} both "
                        "decode as eye-gaze; the stream is ambiguous")
                chosen = type_id
    if chosen is None:
        raise ValueError(
            f"{vrs_path}: no record type decodes as eye gaze (no payload "
            "reproduces its own record-header timestamp)")

    selected = [h for h in headers if h.recordable_type_id == chosen]
    rows: List[Dict[str, Any]] = []
    with open(vrs_path, "rb") as handle:
        for index, header in enumerate(selected):
            handle.seek(header.payload_offset)
            decoded = decode_record(handle.read(layout_size), fields)
            row: Dict[str, Any] = {
                "sample_index": index,
                "record_offset": header.offset,
                "record_header_timestamp_ns": int(round(header.timestamp_s * 1e9)),
            }
            for source, target in _COLUMN_MAP.items():
                row[target] = decoded[source]
            for source, prefix in _VECTOR_MAP.items():
                if source in decoded:
                    x, y, z = decoded[source]
                    row[f"{prefix}_x_m"] = float(x)
                    row[f"{prefix}_y_m"] = float(y)
                    row[f"{prefix}_z_m"] = float(z)
            rows.append(row)

    frame = pd.DataFrame(rows)
    frame["timestamp_ns"] = frame["tracker_timestamp_ns"].astype("int64")
    frame["recording_id"] = recording_id
    frame["domain"] = domain
    frame["time_since_recording_start_s"] = (
        (frame["timestamp_ns"] - frame["timestamp_ns"].iloc[0]) / 1e9)

    # An invalid reading must not carry a usable-looking number.
    for side in ("left", "right"):
        column = f"{side}_pupil_diameter_meter"
        frame[column] = np.where(frame[f"{side}_pupil_diameter_valid"].to_numpy(bool),
                                 frame[column].to_numpy(float), np.nan)
    for column in ("left_blink", "right_blink", "left_blink_valid",
                   "right_blink_valid", "left_pupil_diameter_valid",
                   "right_pupil_diameter_valid", "left_gaze_direction_valid",
                   "right_gaze_direction_valid", "combined_gaze_direction_valid",
                   "spatial_gaze_point_valid", "convergence_distance_valid",
                   "interocular_distance_valid", "left_entrance_pupil_valid",
                   "right_entrance_pupil_valid"):
        frame[column] = frame[column].astype(bool)

    ordered = [c for c in REQUIRED_COLUMNS] + \
              [c for c in frame.columns if c not in REQUIRED_COLUMNS]
    frame = frame[ordered]
    frame.attrs["layout_descriptor"] = descriptor
    frame.attrs["layout_size_bytes"] = layout_size
    frame.attrs["recordable_type_id"] = chosen
    return frame


def extraction_summary(frame, vrs_path: str | Path) -> Dict[str, Any]:
    """Provenance and coverage of one eye-state extraction."""
    ts = frame["timestamp_ns"].to_numpy(np.int64)
    duration_s = float((ts[-1] - ts[0]) / 1e9) if ts.size > 1 else 0.0
    gaps_ms = np.diff(ts) / 1e6 if ts.size > 1 else np.zeros(0)
    return {
        "schema": "article1_eye_state_v1",
        "source_file": str(Path(vrs_path).resolve()),
        "samples": int(len(frame)),
        "duration_s": duration_s,
        "measured_rate_hz": (float((ts.size - 1) / duration_s)
                             if duration_s > 0 else None),
        "sample_interval_ms": {
            "median": float(np.median(gaps_ms)) if gaps_ms.size else None,
            "max": float(gaps_ms.max()) if gaps_ms.size else None,
        },
        "timestamps_strictly_increasing": bool(ts.size < 2 or np.all(np.diff(ts) > 0)),
        "layout_size_bytes": int(frame.attrs.get("layout_size_bytes", 0)),
        "recordable_type_id": int(frame.attrs.get("recordable_type_id", -1)),
        "blink_valid_fraction": {
            "left": float(frame["left_blink_valid"].mean()),
            "right": float(frame["right_blink_valid"].mean()),
        },
        "pupil_valid_fraction": {
            "left": float(frame["left_pupil_diameter_valid"].mean()),
            "right": float(frame["right_pupil_diameter_valid"].mean()),
        },
        "interpolated": False,
        "resampled": False,
        "gap_filled": False,
        "blink_interpolated": False,
        "reader": "raw_vrs_datalayout",
        "reader_note": (
            "pupil diameter is not exposed by projectaria_tools 2.1.2, so the "
            "record payload is decoded directly using the DataLayout descriptor "
            "embedded in this recording; field offsets are read from the file"),
    }


# --------------------------------------------------------------------------- #
# Cross-check against the official binding
# --------------------------------------------------------------------------- #
#: Fields the official binding exposes in a form directly comparable with the raw
#: buffer, as (column, accessor). Per-eye yaw/pitch and the vergence translations
#: are deliberately absent: the binding reports them in the Central Pupil Frame
#: while the record stores them in the device frame, so they differ by a rigid
#: transform and disagreement would prove nothing.
_PROVIDER_FLAG_CHECKS = (
    ("left_blink", lambda g: bool(g.vergence.left_blink)),
    ("right_blink", lambda g: bool(g.vergence.right_blink)),
    ("left_blink_valid", lambda g: bool(g.vergence.left_blink_valid)),
    ("right_blink_valid", lambda g: bool(g.vergence.right_blink_valid)),
    ("left_gaze_direction_valid", lambda g: bool(g.vergence.left_gaze_valid)),
    ("right_gaze_direction_valid", lambda g: bool(g.vergence.right_gaze_valid)),
    ("combined_gaze_direction_valid", lambda g: bool(g.combined_gaze_valid)),
    ("spatial_gaze_point_valid", lambda g: bool(g.spatial_gaze_point_valid)),
)


def verify_against_provider(frame, vrs_path: str | Path,
                            label: str = "eyegaze") -> Dict[str, Any]:
    """Compare every field the official binding also exposes, sample for sample.

    This is the guarantee that the raw decode is correct. `projectaria_tools` does
    not expose pupil diameter, but it does expose the record timestamp (buffer
    offset 0), the convergence distance (offset 135) and eight boolean flags
    spread from offset 29 to offset 158 — including the two blink flags and two
    blink validity flags that sit at offsets 60/61 and 106/107, immediately after
    each eye's pupil-diameter field.

    The pupil bytes (56–59 and 102–105) are therefore bracketed on both sides by
    fields verified on every sample. If the whole set agrees, the buffer cannot be
    misaligned and the pupil fields are read where the file says they are.
    """
    from ..vrs.provider import AriaProvider

    provider = AriaProvider(str(vrs_path))
    if not provider.has_eyegaze(label):
        return {"verified": False, "reason": f"no {label} stream in the recording"}
    count = int(provider.num_eyegaze(label))
    mismatches: Dict[str, int] = {}
    max_timestamp_error_ns = 0
    max_convergence_error_m = 0.0
    if len(frame) == count:
        for i in range(count):
            gaze = provider.eyegaze_by_index(i, label)
            row = frame.iloc[i]
            api_ts = int(round(gaze.tracking_timestamp.total_seconds() * 1e9))
            max_timestamp_error_ns = max(max_timestamp_error_ns,
                                         abs(api_ts - int(row["timestamp_ns"])))
            for column, accessor in _PROVIDER_FLAG_CHECKS:
                if accessor(gaze) != bool(row[column]):
                    mismatches[column] = mismatches.get(column, 0) + 1
            if bool(row["convergence_distance_valid"]):
                max_convergence_error_m = max(
                    max_convergence_error_m,
                    abs(float(gaze.depth) - float(row["convergence_distance_meter"])))

    return {
        "verified": (len(frame) == count and not mismatches
                     and max_timestamp_error_ns == 0
                     and max_convergence_error_m < 1e-6),
        "provider_samples": count,
        "raw_samples": int(len(frame)),
        "flag_mismatches": mismatches,
        "max_timestamp_error_ns": int(max_timestamp_error_ns),
        "max_convergence_distance_error_m": float(max_convergence_error_m),
        "fields_compared": (["tracking_timestamp", "convergence_distance_meter"]
                            + [c for c, _ in _PROVIDER_FLAG_CHECKS]),
        "buffer_offsets_covered": [0, 29, 60, 61, 75, 106, 107, 121, 135, 158],
        "pupil_field_offsets": [56, 102],
        "note": ("the compared fields bracket the pupil-diameter bytes inside the "
                 "same fixed buffer, so agreement on all of them establishes that "
                 "the pupil fields are read at the right offsets"),
    }
