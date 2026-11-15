"""Polar concentricity indices for a nested (parent, child) mask pair: the tree v2 rule's offset_idx / shape_idx.

COPIED WORD FOR WORD (lines 27-85: NB, polar_profile, harmonics, strip_low, metrics) from the ISEVxTech 2026 poster's
scripts/nesting_polar_probe.py (sha256 7141fcef72ef7220421121379d17f53bc8e4392e09538353cccf0ac93bcd8671, 2026-11-14).
Only the I/O is left out (its CLI, file reading and selftest; the selftest cases are unit tests in
results/tests/test_nesting_tree.py). Do not edit the copied block: the tree v2 thresholds (offset 0.65, shape 0.35)
were set on these exact numbers, and its sampling differs from nesting's gap_stats on purpose.

Both outlines are unwrapped around the CHILD centroid into 72 angle bins (5 deg): r(t) = child radius (max per bin),
R(t) = parent radius (median per bin), gap g(t) = R - r. Fourier series of g: a0 mean gap; offset_idx = a1 / a0
(0 = concentric, ~1.1 = touching on one side); shape_idx = a2 / a0 (oval mismatch); rough_idx = harmonics >= 3.
metrics() returns None when the child has no pixels or either outline covers fewer than NB // 3 = 24 bins.
"""
import math

import cv2
import numpy as np

NB = 72


def polar_profile(mask, cx, cy, reduce):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.vstack([c[:, 0, :] for c in cs]).astype(float) + 0.5
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    th = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)
    rho = np.hypot(dx, dy)
    b = np.minimum((th / (2 * np.pi) * NB).astype(int), NB - 1)
    out = np.full(NB, np.nan)
    for k in range(NB):
        m = b == k
        if m.any():
            out[k] = reduce(rho[m])
    ok = ~np.isnan(out)
    if ok.sum() < NB // 3:
        return None
    idx = np.arange(NB)
    out[~ok] = np.interp(idx[~ok], idx[ok], out[ok], period=NB)
    return out


def harmonics(x):
    F = np.fft.rfft(x) / len(x)
    amp = 2 * np.abs(F)
    amp[0] = np.abs(F[0])
    return F, amp


def strip_low(x, kmax=1):
    F = np.fft.rfft(x)
    F[:kmax + 1] = 0
    return np.fft.irfft(F, len(x))


def metrics(P, Cm, nm=1.0):
    """P, Cm: same-shape binary masks (parent, child). Returns dict or None."""
    M = cv2.moments(Cm.astype(np.uint8), True)
    if M["m00"] == 0:
        return None
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    r = polar_profile(Cm, cx, cy, np.max)
    R = polar_profile(P, cx, cy, np.median)
    if r is None or R is None:
        return None
    g = R - r
    _, amp = harmonics(g)
    a0 = max(amp[0], 1e-6)
    hi = math.sqrt(float(np.sum(amp[3:] ** 2) / 2))
    rd, Rd = strip_low(r), strip_low(R)
    coup = float(np.corrcoef(rd, Rd)[0, 1]) if rd.std() > 1e-9 and Rd.std() > 1e-9 else float("nan")
    med = np.median(g)
    even = float(np.mean(np.abs(g - med) <= max(2.0, 0.25 * abs(med))))
    Mp = cv2.moments(P.astype(np.uint8), True)
    d_cent = math.hypot(Mp["m10"] / Mp["m00"] - cx, Mp["m01"] / Mp["m00"] - cy)
    return {"a0_nm": round(a0 * nm, 2), "offset_idx": round(amp[1] / a0, 3), "shape_idx": round(amp[2] / a0, 3),
            "rough_idx": round(hi / a0, 3), "coupling": round(coup, 3), "deform_nm": round(float(rd.std()) * nm, 2),
            "even_frac": round(even, 3), "centroid_offset_norm": round(d_cent / a0, 3)}
