"""
Glitch frame filtering for 3D pose sequences.

Public API:
    filter_glitch_frames(pickle_abs_path: str,
                         padding: int = 30,
                         threshold: float = 4.0,
                         min_abnormal_bones: int = 1) -> list | np.ndarray

Given an absolute path to a 3D poses pickle file, detects frames with abnormal
bone lengths (glitches) using robust statistics, removes those frames and
`padding` frames on both sides, and returns the filtered poses in the SAME format
as the input (list vs ndarray; and per-segment shape: (T,17,3) or (1,T,17,3)).
"""

from __future__ import annotations

import os
import pickle
from typing import List, Sequence, Tuple, Union

import numpy as np


# Bone connections (start_index, end_index) – Human3.6M-style 17 joints
H36M_CONNECTIONS: Sequence[Tuple[int, int]] = (
    (0, 1),   # Hip -> Right Hip
    (1, 2),   # Right Hip -> Right Knee
    (2, 3),   # Right Knee -> Right Ankle
    (0, 4),   # Hip -> Left Hip
    (4, 5),   # Left Hip -> Left Knee
    (5, 6),   # Left Knee -> Left Ankle
    (0, 7),   # Hip -> Spine
    (7, 8),   # Spine -> Thorax
    (8, 9),   # Thorax -> Neck
    (9, 10),  # Neck -> Head
    (8, 11),  # Thorax -> Left Shoulder
    (11, 12), # Left Shoulder -> Left Elbow
    (12, 13), # Left Elbow -> Left Hand
    (8, 14),  # Thorax -> Right Shoulder
    (14, 15), # Right Shoulder -> Right Elbow
    (15, 16), # Right Elbow -> Right Hand
)


def _normalize_segment(segment: np.ndarray) -> Tuple[np.ndarray, str]:
    """Normalize a segment to (T,17,3) and return (normalized, original_kind).

    original_kind is either:
      - 'T'  for (T,17,3)
      - '1T' for (1,T,17,3)
    """
    if not isinstance(segment, np.ndarray):
        raise ValueError("Pose segment is not a numpy array")

    if segment.ndim == 3 and segment.shape[1:] == (17, 3):
        return segment, 'T'

    if segment.ndim == 4 and segment.shape[-2:] == (17, 3):
        if segment.shape[0] == 1:
            return segment[0], '1T'
        # If multiple sequences along axis 0, concatenate along time
        return np.concatenate([segment[i] for i in range(segment.shape[0])], axis=0), 'T'

    raise ValueError(f"Unsupported segment shape: {getattr(segment, 'shape', None)}")


def _load_segments_with_meta(pickle_abs_path: str):
    """Load poses from pickle, returning (segments_T173, original_container, per_segment_kind).

    - segments_T173: List[np.ndarray] each (T,17,3)
    - original_container: 'list' or 'ndarray' or 'dict'
    - per_segment_kind: List[str] with values 'T' or '1T' (for reshape-back)
    """
    if not os.path.isabs(pickle_abs_path):
        raise ValueError("Expected an absolute path to the pickle file")
    if not os.path.exists(pickle_abs_path):
        raise FileNotFoundError(f"Pickle file not found: {pickle_abs_path}")

    with open(pickle_abs_path, 'rb') as f:
        obj = pickle.load(f)

    segments_T173: List[np.ndarray] = []
    kinds: List[str] = []
    original_container: str = type(obj).__name__

    def add_seg(x):
        norm, kind = _normalize_segment(np.asarray(x))
        segments_T173.append(norm)
        kinds.append(kind)

    if isinstance(obj, list):
        for s in obj:
            try:
                add_seg(s)
            except Exception:
                continue
        original_container = 'list'
    elif isinstance(obj, np.ndarray):
        add_seg(obj)
        original_container = 'ndarray'
    elif isinstance(obj, dict):
        for key in ("poses", "data", "segment", "segments"):
            if key in obj:
                add_seg(obj[key])
                break
        original_container = 'dict'
    else:
        raise ValueError("Unsupported pickle structure. Expected list/ndarray/dict.")

    # Drop empties
    keep = [(s, k) for (s, k) in zip(segments_T173, kinds) if isinstance(s, np.ndarray) and s.size > 0]
    if not keep:
        raise ValueError("No valid pose segments found in the pickle file")
    segments_T173, kinds = zip(*keep)
    return list(segments_T173), original_container, list(kinds)


def _compute_robust_bone_stats(segments_T173: List[np.ndarray]):
    """Return dict conn->(median, robust_scale) computed across all segments/frames."""
    bone_lengths: dict = {c: [] for c in H36M_CONNECTIONS}
    for seg in segments_T173:
        if seg.shape[1:] != (17, 3):
            continue
        for (s, e) in H36M_CONNECTIONS:
            p1 = seg[:, s, :]
            p2 = seg[:, e, :]
            valid = (np.linalg.norm(p1, axis=1) > 1e-6) & (np.linalg.norm(p2, axis=1) > 1e-6)
            if not np.any(valid):
                continue
            d = p1[valid] - p2[valid]
            lens = np.linalg.norm(d, axis=1)
            bone_lengths[(s, e)].extend(lens.tolist())

    stats = {}
    for conn, vals in bone_lengths.items():
        if not vals:
            stats[conn] = (float('nan'), 0.0)
            continue
        arr = np.asarray(vals, dtype=np.float64)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        stats[conn] = (med, 1.4826 * mad)
    return stats


def filter_glitch_frames(
    pickle_abs_path: str,
    padding: int = 30,
    threshold: float = 4.0,
    min_abnormal_bones: int = 1,
) -> Union[List[np.ndarray], np.ndarray]:
    """Detect glitch frames and remove them with padding on both sides.

    Parameters
    ----------
    pickle_abs_path: str
        Absolute path to a 3D poses pickle file.
    padding: int
        Number of frames to remove on each side of a glitched frame.
    threshold: float
        Robust z-score threshold (median/MAD) to flag a bone length as abnormal.
    min_abnormal_bones: int
        Minimum number of abnormal bones required to flag a frame as glitched.

    Returns
    -------
    List[np.ndarray] or np.ndarray
        Filtered poses in the SAME format as the input:
        - If input was a list of segments, returns a list of segments, each shaped
          like the original per-segment format ((T,17,3) or (1,T,17,3)). Empty
          segments are removed.
        - If input was a single ndarray, returns a single ndarray in the original
          format. If all frames are removed, returns an empty array with the same
          rank and trailing dimensions.
    """
    segments, container, kinds = _load_segments_with_meta(pickle_abs_path)

    # Compute robust per-bone stats
    stats = _compute_robust_bone_stats(segments)

    filtered_segments: List[np.ndarray] = []
    for seg_idx, seg in enumerate(segments):
        T = seg.shape[0]
        if T == 0:
            continue
        keep_mask = np.ones(T, dtype=bool)

        # Detect glitched frames in this segment
        for t in range(T):
            pose = seg[t]
            abnormal = 0
            for (s, e) in H36M_CONNECTIONS:
                median, scale = stats[(s, e)]
                if not np.isfinite(median) and scale == 0.0:
                    continue
                p1 = pose[s]
                p2 = pose[e]
                if np.linalg.norm(p1) <= 1e-6 or np.linalg.norm(p2) <= 1e-6:
                    continue
                length = float(np.linalg.norm(p1 - p2))
                denom = scale if scale > 1e-12 else 1e-12
                z = abs(length - median) / denom
                if z > threshold:
                    abnormal += 1
            if abnormal >= max(1, int(min_abnormal_bones)):
                # Invalidate [t - padding, t + padding]
                start = max(0, t - padding)
                end = min(T - 1, t + padding)
                keep_mask[start:end + 1] = False

        kept = seg[keep_mask]
        if kept.shape[0] == 0:
            # Drop empty segments
            continue
        # Restore original per-segment format
        if kinds[seg_idx] == '1T':
            kept = kept[np.newaxis, ...]  # (1, T, 17, 3)
        filtered_segments.append(kept)

    # Restore original container format
    if container == 'ndarray' or (container == 'dict'):
        # Single segment originally
        if not filtered_segments:
            # Return empty with original-like shape
            if kinds and kinds[0] == '1T':
                return np.zeros((1, 0, 17, 3), dtype=np.float32)
            return np.zeros((0, 17, 3), dtype=np.float32)
        return filtered_segments[0]

    # List case
    return filtered_segments

