"""Pure-numpy pitch metrics used during mel->f0 training (logged to TensorBoard).

`decompose` gives a frame-pooled error decomposition + headline accuracies (RPA/RCA/OA/VR/VFA,
octave/gross/voicing-miss rates) and `committed_accuracy` — the co-voiced (ref-voiced AND
pred-voiced) within-tolerance fraction, which removes the voicing operating-point so train-vs-val
and cross-model comparisons reflect *pitch* quality, not voicing recall. `rpa_at_tolerances` gives
RPA at several cent tolerances (low RPA@10 but high RPA@50 => precision-limited, e.g. coarse mel).

Convention (matches the repo): cent = 1200*log2(f0/10); reference unvoiced frames decode to
cent == 0. The model's pitch value is threshold-independent; voicing is `max_prob > voicing_th`.

Plain numpy + python types so they can be unit-tested without torch. Consumed by training/fcpe.py.
"""
from __future__ import annotations

import numpy as np


def _octave_distance(d):
    """Cent distance to the nearest +-1/+-2 octave multiple (0 if d is a perfect octave)."""
    ad = np.abs(d)
    return np.minimum.reduce([np.abs(ad - 1200.0), np.abs(ad - 2400.0)])


def _chroma_distance(d):
    """Cent distance ignoring octave (folds onto [0,600])."""
    dd = np.mod(np.abs(d), 1200.0)
    return np.minimum(dd, 1200.0 - dd)


def decompose(ref_cent, pred_cent, pred_maxprob, voicing_th,
              tol=50.0, octave_tol=50.0):
    """Frame-pooled error decomposition + headline accuracies.

    Returns a dict of counts, rates, and the standard RPA/RCA/OA/VR/VFA (frame-pooled).
    """
    ref_cent = np.asarray(ref_cent, dtype=np.float64)
    pred_cent = np.asarray(pred_cent, dtype=np.float64)
    pred_maxprob = np.asarray(pred_maxprob, dtype=np.float64)

    ref_v = ref_cent > 0
    pred_v = pred_maxprob > voicing_th
    n = ref_cent.shape[0]
    n_ref_v = int(ref_v.sum())
    n_ref_u = int((~ref_v).sum())

    d = pred_cent - ref_cent
    ad = np.abs(d)
    is_fine = ad <= tol
    is_oct = (~is_fine) & (_octave_distance(d) <= octave_tol)
    is_chroma = _chroma_distance(d) <= tol  # pitch class correct (any octave)

    rv = ref_v
    correct = int(np.sum(rv & pred_v & is_fine))
    octave = int(np.sum(rv & pred_v & is_oct))
    gross = int(np.sum(rv & pred_v & ~is_fine & ~is_oct))
    voicing_miss = int(np.sum(rv & ~pred_v))
    false_alarm = int(np.sum((~rv) & pred_v))
    chroma_correct = int(np.sum(rv & pred_v & is_chroma))

    safe_rv = max(n_ref_v, 1)
    safe_ru = max(n_ref_u, 1)

    return {
        'voicing_th': float(voicing_th),
        'cent_tolerance': float(tol),
        'n_frames': int(n),
        'n_ref_voiced': n_ref_v,
        'n_ref_unvoiced': n_ref_u,
        # raw counts over ref-voiced frames (these four sum to n_ref_voiced)
        'correct': correct,
        'octave_error': octave,
        'gross_error': gross,
        'voicing_miss': voicing_miss,
        'false_alarm': false_alarm,  # over ref-unvoiced frames
        # rates
        'RPA': correct / safe_rv,
        # committed/co-voiced accuracy: among frames the model commits to (ref voiced AND pred voiced),
        # fraction within tolerance. This removes the voicing operating-point so train-vs-val and
        # cross-model comparisons reflect *pitch* quality, not voicing recall.
        'committed_accuracy': correct / max(correct + octave + gross, 1),
        'co_voiced_frames': int(correct + octave + gross),
        'RCA': chroma_correct / safe_rv,
        'octave_error_rate': octave / safe_rv,
        'gross_error_rate': gross / safe_rv,
        'voicing_miss_rate': voicing_miss / safe_rv,
        'VR': int(np.sum(rv & pred_v)) / safe_rv,
        'VFA': false_alarm / safe_ru,
        'OA': (int(np.sum(rv & pred_v & is_fine)) + int(np.sum((~rv) & (~pred_v)))) / max(n, 1),
    }


def rpa_at_tolerances(ref_cent, pred_cent, pred_maxprob, voicing_th, tols=(10.0, 25.0, 50.0)):
    """RPA at several cent tolerances. Low RPA@10 but high RPA@50 => precision-limited (e.g. coarse mel)."""
    ref_cent = np.asarray(ref_cent, dtype=np.float64)
    pred_cent = np.asarray(pred_cent, dtype=np.float64)
    pred_v = np.asarray(pred_maxprob, dtype=np.float64) > voicing_th
    ref_v = ref_cent > 0
    ad = np.abs(pred_cent - ref_cent)
    safe_rv = max(int(ref_v.sum()), 1)
    return {f'RPA@{int(t)}': float(np.sum(ref_v & pred_v & (ad <= t)) / safe_rv) for t in tols}
