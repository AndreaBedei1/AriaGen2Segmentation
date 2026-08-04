"""Descriptive statistics for a one-session-per-vehicle pilot.

There is one participant and one session per vehicle. Nothing here can support an
inference about drivers in general, and the module has no function that would let
it try: no t-test against a population, no p-value from treating frames as
independent draws.

What it does provide:

* effect sizes with **block** bootstrap confidence intervals, because consecutive
  samples of a drive are strongly autocorrelated and an i.i.d. bootstrap would
  report an interval several times too narrow;
* permutation tests that permute whole blocks, for the same reason;
* Benjamini-Hochberg FDR control, since a comparison run across many metrics will
  produce small p-values by construction.

The unit of analysis is a route bin, an event or a time window — never a frame.
Every function takes the unit count and reports it, so a confidence interval
built on nine segments is never mistaken for one built on nine hundred.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class EffectSize:
    name: str
    n_units: int
    estimate: float
    ci_low: Optional[float]
    ci_high: Optional[float]
    method: str
    caveat: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "n_units": self.n_units,
                "estimate": self.estimate, "ci_low": self.ci_low,
                "ci_high": self.ci_high, "method": self.method,
                "caveat": self.caveat}


#: Below this many units, an interval is reported but flagged: a bootstrap
#: resamples what it was given, and eight segments cannot describe a route.
MIN_UNITS_FOR_INTERVAL = 8


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Non-parametric effect size in [-1, 1]: P(a > b) - P(a < b).

    Preferred over a standardised mean difference here because these
    distributions are small, skewed and bounded, and none of that troubles a
    rank-based statistic.
    """
    x = np.asarray(a, float)
    y = np.asarray(b, float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    greater = float(np.sum(x[:, None] > y[None, :]))
    less = float(np.sum(x[:, None] < y[None, :]))
    return (greater - less) / (x.size * y.size)


def paired_median_difference(a: Sequence[float], b: Sequence[float]) -> float:
    """Median of the per-unit differences. Pairs are never broken apart."""
    x = np.asarray(a, float)
    y = np.asarray(b, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.median(x[ok] - y[ok])) if ok.any() else float("nan")


def _blocks(n: int, block_size: int) -> List[np.ndarray]:
    block_size = max(1, int(block_size))
    return [np.arange(i, min(i + block_size, n))
            for i in range(0, n, block_size)]


def block_bootstrap_ci(values: Sequence[float],
                       statistic: Callable[[np.ndarray], float],
                       block_size: int = 4,
                       iterations: int = 2000,
                       alpha: float = 0.05,
                       rng: Optional[np.random.Generator] = None
                       ) -> Tuple[Optional[float], Optional[float]]:
    """Percentile interval from resampling contiguous blocks, not single values.

    Resampling individual samples would assume they are independent. They are
    not: speed, heart rate and gaze all persist for seconds, so an i.i.d.
    bootstrap understates the spread badly. Blocks preserve the local dependence.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return (None, None)
    rng = rng or np.random.default_rng(0)
    blocks = _blocks(v.size, block_size)
    k = len(blocks)
    stats = np.empty(iterations)
    for i in range(iterations):
        pick = rng.integers(0, k, size=k)
        sample = np.concatenate([blocks[j] for j in pick])[:v.size]
        stats[i] = statistic(v[sample])
    finite = stats[np.isfinite(stats)]
    if finite.size < iterations * 0.5:
        return (None, None)
    return (float(np.percentile(finite, 100 * alpha / 2)),
            float(np.percentile(finite, 100 * (1 - alpha / 2))))


def paired_block_bootstrap(a: Sequence[float], b: Sequence[float],
                           block_size: int = 4, iterations: int = 2000,
                           alpha: float = 0.05,
                           rng: Optional[np.random.Generator] = None
                           ) -> EffectSize:
    """Paired median difference with a block-bootstrap interval."""
    x = np.asarray(a, float)
    y = np.asarray(b, float)
    ok = np.isfinite(x) & np.isfinite(y)
    d = x[ok] - y[ok]
    est = float(np.median(d)) if d.size else float("nan")
    lo, hi = block_bootstrap_ci(d, np.median, block_size, iterations, alpha, rng)
    caveat = None
    if d.size < MIN_UNITS_FOR_INTERVAL:
        caveat = (f"{d.size} paired units: the interval is reported for "
                  "completeness but a bootstrap cannot manufacture information "
                  "this few units do not contain")
    return EffectSize("paired_median_difference", int(d.size), est, lo, hi,
                      f"block bootstrap, block={block_size}, "
                      f"{iterations} iterations", caveat)


def two_sample_block_bootstrap(a: Sequence[float], b: Sequence[float],
                               block_size: int = 4, iterations: int = 2000,
                               alpha: float = 0.05,
                               rng: Optional[np.random.Generator] = None
                               ) -> EffectSize:
    """Difference of medians with a block-bootstrap interval, for unpaired units.

    Used where the two vehicles have no common unit to pair on — separate events,
    separate time blocks — so each group is resampled in contiguous blocks
    independently and the difference is recomputed. It is not a paired test and
    does not pretend to be one: the interval is wider than the paired version for
    the same data, which is correct, because the pairing information is genuinely
    absent.
    """
    x = np.asarray(a, float)
    y = np.asarray(b, float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 3 or y.size < 3:
        return EffectSize("median_difference", int(min(x.size, y.size)),
                          float("nan"), None, None, "insufficient units",
                          f"{x.size} and {y.size} units: too few to resample")
    rng = rng or np.random.default_rng(0)
    est = float(np.median(x) - np.median(y))
    blocks_x = _blocks(x.size, block_size)
    blocks_y = _blocks(y.size, block_size)
    stats = np.empty(iterations)
    for i in range(iterations):
        px = rng.integers(0, len(blocks_x), size=len(blocks_x))
        py = rng.integers(0, len(blocks_y), size=len(blocks_y))
        sx = np.concatenate([blocks_x[j] for j in px])[:x.size]
        sy = np.concatenate([blocks_y[j] for j in py])[:y.size]
        stats[i] = np.median(x[sx]) - np.median(y[sy])
    finite = stats[np.isfinite(stats)]
    if finite.size < iterations * 0.5:
        return EffectSize("median_difference", int(min(x.size, y.size)), est,
                          None, None, "block bootstrap failed to converge")
    caveat = None
    if min(x.size, y.size) < MIN_UNITS_FOR_INTERVAL:
        caveat = (f"{x.size} and {y.size} units: the interval is reported for "
                  "completeness but a bootstrap cannot manufacture information "
                  "this few units do not contain")
    return EffectSize(
        "median_difference", int(min(x.size, y.size)), est,
        float(np.percentile(finite, 100 * alpha / 2)),
        float(np.percentile(finite, 100 * (1 - alpha / 2))),
        f"two-sample block bootstrap, block={block_size}, "
        f"{iterations} iterations", caveat)


def effective_sample_size(n_units: int, block_size: int) -> int:
    """Independent blocks behind a comparison, not its row count.

    Consecutive units of one drive are strongly autocorrelated, so the number of
    rows overstates the information available by whatever factor the block length
    represents. This is the number every interval actually rests on.
    """
    return int(np.ceil(int(n_units) / max(1, int(block_size)))) if n_units else 0


def block_permutation_test(a: Sequence[float], b: Sequence[float],
                           statistic: Callable[[np.ndarray, np.ndarray], float],
                           block_size: int = 4, iterations: int = 5000,
                           rng: Optional[np.random.Generator] = None
                           ) -> Dict[str, Any]:
    """Two-sided permutation test that permutes contiguous blocks.

    The exchangeable unit is a block, not a sample. Permuting samples would break
    the autocorrelation that the null has to preserve and would return a p-value
    far smaller than the data supports.
    """
    x = np.asarray(a, float)
    y = np.asarray(b, float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 3 or y.size < 3:
        return {"p_value": None, "observed": None, "iterations": 0,
                "reason": f"{x.size} and {y.size} units: too few to permute"}
    rng = rng or np.random.default_rng(0)
    observed = float(statistic(x, y))
    pooled = np.concatenate([x, y])
    blocks = _blocks(pooled.size, block_size)
    count = 0
    done = 0
    for _ in range(iterations):
        order = rng.permutation(len(blocks))
        shuffled = np.concatenate([blocks[j] for j in order])
        s = pooled[shuffled]
        stat = float(statistic(s[:x.size], s[x.size:]))
        if not np.isfinite(stat):
            continue
        done += 1
        if abs(stat) >= abs(observed):
            count += 1
    if done == 0:
        return {"p_value": None, "observed": observed, "iterations": 0,
                "reason": "statistic was not finite under permutation"}
    # (count + 1) / (done + 1): a permutation p-value is never exactly zero.
    return {"p_value": float((count + 1) / (done + 1)), "observed": observed,
            "iterations": done, "block_size": block_size,
            "exchangeable_unit": "contiguous block", "reason": None}


def benjamini_hochberg(p_values: Sequence[Optional[float]],
                       alpha: float = 0.05) -> Dict[str, Any]:
    """FDR control across a family of comparisons.

    A comparison sweep over many metrics produces small p-values by construction;
    without this the smallest one is not evidence of anything.
    """
    idx = [i for i, p in enumerate(p_values) if p is not None and np.isfinite(p)]
    if not idx:
        return {"alpha": alpha, "tested": 0, "rejected": 0,
                "adjusted": list(p_values), "threshold": None}
    p = np.array([p_values[i] for i in idx], float)
    order = np.argsort(p)
    m = p.size
    ranked = p[order]
    adj = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    adj = np.clip(adj, 0, 1)

    out: List[Optional[float]] = list(p_values)
    for pos, orig in enumerate(order):
        out[idx[orig]] = float(adj[pos])
    below = np.flatnonzero(ranked <= alpha * np.arange(1, m + 1) / m)
    threshold = float(ranked[below.max()]) if below.size else None
    return {"alpha": alpha, "tested": int(m),
            "rejected": int(np.sum(adj <= alpha)),
            "adjusted": out, "threshold": threshold,
            "method": "Benjamini-Hochberg"}


def describe(values: Sequence[float], unit: str = "unit") -> Dict[str, Any]:
    """Descriptive summary with the unit count front and centre."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n_units": 0, "unit": unit, "median": None, "iqr": None,
                "p05": None, "p95": None, "mean": None, "std": None}
    return {
        "n_units": int(v.size), "unit": unit,
        "median": float(np.median(v)),
        "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
        "p05": float(np.percentile(v, 5)), "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)), "p95": float(np.percentile(v, 95)),
        "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if v.size > 1 else None,
        "min": float(np.min(v)), "max": float(np.max(v)),
    }


def blocks_for_duration(unit_duration_s: float, block_length_s: float) -> int:
    """Bootstrap block length in units, from a block length in seconds."""
    if unit_duration_s <= 0:
        return 1
    return max(1, int(round(float(block_length_s) / float(unit_duration_s))))


PILOT_CAVEAT = (
    "One participant, one session per vehicle. These are descriptive statistics "
    "of two specific drives. They do not estimate a population parameter, they "
    "cannot be generalised, and no confidence interval here is a statement about "
    "drivers in general."
)
