"""Pure-numpy diagnostics for mel->f0 pitch models.

Given per-frame decoded cents (model + reference) and the model's per-frame max bin
probability, decompose the error so you can tell *where* a model is bad:

  - voicing errors (false alarm / miss) vs pitch errors
  - octave errors (chroma right, register wrong) vs other gross errors vs fine errors
  - accuracy as a function of the reference f0 register  (the key test for whether a
    coarse / high-fmax mel is the bottleneck: if low registers are much worse, the input is)
  - cent-error bias & spread (systematic offset vs noise)
  - per-file ranking, and a voicing-threshold sweep

Convention (matches the repo): cent = 1200*log2(f0/10); reference unvoiced frames decode to
cent == 0. The model's pitch value is threshold-independent; voicing is `max_prob > voicing_th`.

All functions take/return plain numpy + python types so they can be unit-tested without torch.
"""
from __future__ import annotations

import numpy as np

DEFAULT_REGISTER_EDGES_HZ = [0.0, 100.0, 200.0, 400.0, 800.0, 1e9]
DEFAULT_SWEEP_THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]


def cents_to_hz(cents):
    cents = np.asarray(cents, dtype=np.float64)
    return np.where(cents > 0, 10.0 * 2.0 ** (cents / 1200.0), 0.0)


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


def accuracy_by_register(ref_cent, pred_cent, pred_maxprob, voicing_th,
                         edges_hz=None, tol=50.0, octave_tol=50.0):
    """Per-register-band accuracy. If the low bands are much worse, the input representation is the bottleneck."""
    if edges_hz is None:
        edges_hz = DEFAULT_REGISTER_EDGES_HZ
    ref_cent = np.asarray(ref_cent, dtype=np.float64)
    pred_cent = np.asarray(pred_cent, dtype=np.float64)
    pred_v = np.asarray(pred_maxprob, dtype=np.float64) > voicing_th
    ref_v = ref_cent > 0
    ref_hz = cents_to_hz(ref_cent)
    d = pred_cent - ref_cent
    ad = np.abs(d)
    is_fine = ad <= tol
    is_oct = (~is_fine) & (_octave_distance(d) <= octave_tol)

    rows = []
    for lo, hi in zip(edges_hz[:-1], edges_hz[1:]):
        in_band = ref_v & (ref_hz >= lo) & (ref_hz < hi)
        cnt = int(in_band.sum())
        if cnt == 0:
            rows.append({'band_hz': [float(lo), float(hi)], 'n': 0,
                         'RPA': float('nan'), 'octave_rate': float('nan'), 'median_abs_cent': float('nan')})
            continue
        band_fine = int(np.sum(in_band & pred_v & is_fine))
        band_oct = int(np.sum(in_band & pred_v & is_oct))
        # precision proxy: median |cent err| among non-octave, non-gross frames in band
        prec_mask = in_band & pred_v & (ad <= 200.0)
        med = float(np.median(ad[prec_mask])) if int(prec_mask.sum()) > 0 else float('nan')
        rows.append({'band_hz': [float(lo), float(hi)], 'n': cnt,
                     'RPA': band_fine / cnt, 'octave_rate': band_oct / cnt, 'median_abs_cent': med})
    return rows


def cent_error_stats(ref_cent, pred_cent, pred_maxprob, voicing_th, precision_clip=200.0):
    """Bias/spread of the cent error. Large |bias| => systematic offset (alignment / cents-mapping)."""
    ref_cent = np.asarray(ref_cent, dtype=np.float64)
    pred_cent = np.asarray(pred_cent, dtype=np.float64)
    pred_v = np.asarray(pred_maxprob, dtype=np.float64) > voicing_th
    ref_v = ref_cent > 0
    both = ref_v & pred_v
    d_all = (pred_cent - ref_cent)[both]
    if d_all.size == 0:
        return {'n': 0}
    d_prec = d_all[np.abs(d_all) <= precision_clip]
    return {
        'n': int(d_all.size),
        'bias_cent_all': float(np.mean(d_all)),
        'std_cent_all': float(np.std(d_all)),
        'n_precision': int(d_prec.size),
        'bias_cent_precision': float(np.mean(d_prec)) if d_prec.size else float('nan'),
        'std_cent_precision': float(np.std(d_prec)) if d_prec.size else float('nan'),
        'mae_cent_precision': float(np.mean(np.abs(d_prec))) if d_prec.size else float('nan'),
    }


def threshold_sweep(ref_cent, pred_cent, pred_maxprob, thresholds=None, tol=50.0):
    """RPA / VR / VFA as a function of the voicing threshold."""
    if thresholds is None:
        thresholds = DEFAULT_SWEEP_THRESHOLDS
    ref_cent = np.asarray(ref_cent, dtype=np.float64)
    pred_cent = np.asarray(pred_cent, dtype=np.float64)
    pred_maxprob = np.asarray(pred_maxprob, dtype=np.float64)
    ref_v = ref_cent > 0
    ad = np.abs(pred_cent - ref_cent)
    safe_rv = max(int(ref_v.sum()), 1)
    safe_ru = max(int((~ref_v).sum()), 1)
    out = []
    for th in thresholds:
        pred_v = pred_maxprob > th
        out.append({
            'th': float(th),
            'RPA': float(np.sum(ref_v & pred_v & (ad <= tol)) / safe_rv),
            'VR': float(np.sum(ref_v & pred_v) / safe_rv),
            'VFA': float(np.sum((~ref_v) & pred_v) / safe_ru),
        })
    return out


def per_file_rpa(ref_cent, pred_cent, pred_maxprob, file_ids, voicing_th, tol=50.0):
    """RPA@tol per file. Returns list of (file_id, n_ref_voiced, rpa) sorted ascending by rpa."""
    ref_cent = np.asarray(ref_cent, dtype=np.float64)
    pred_cent = np.asarray(pred_cent, dtype=np.float64)
    pred_v = np.asarray(pred_maxprob, dtype=np.float64) > voicing_th
    ref_v = ref_cent > 0
    ad = np.abs(pred_cent - ref_cent)
    file_ids = np.asarray(file_ids)
    rows = []
    for fid in dict.fromkeys(file_ids.tolist()):
        m = file_ids == fid
        nrv = int(np.sum(ref_v & m))
        if nrv == 0:
            rows.append((fid, 0, float('nan')))
            continue
        rows.append((fid, nrv, float(np.sum(m & ref_v & pred_v & (ad <= tol)) / nrv)))
    rows.sort(key=lambda r: (np.inf if np.isnan(r[2]) else r[2]))
    return rows


def build_report(ref_cent, pred_cent, pred_maxprob, file_ids, files, voicing_th,
                 register_edges_hz=None, sweep_thresholds=None):
    """Assemble the full diagnostic dict from per-frame arrays."""
    summary = decompose(ref_cent, pred_cent, pred_maxprob, voicing_th)
    summary.update(rpa_at_tolerances(ref_cent, pred_cent, pred_maxprob, voicing_th))
    pf = per_file_rpa(ref_cent, pred_cent, pred_maxprob, file_ids, voicing_th)
    id2name = {i: files[i] for i in range(len(files))}
    return {
        'summary': summary,
        'cent_error': cent_error_stats(ref_cent, pred_cent, pred_maxprob, voicing_th),
        'by_register': accuracy_by_register(ref_cent, pred_cent, pred_maxprob, voicing_th, register_edges_hz),
        'threshold_sweep': threshold_sweep(ref_cent, pred_cent, pred_maxprob, sweep_thresholds),
        'per_file': [{'file': id2name.get(fid, str(fid)), 'n_ref_voiced': nrv, 'RPA': rpa} for fid, nrv, rpa in pf],
    }


def _fmt(x, p=3):
    if isinstance(x, float):
        return 'nan' if np.isnan(x) else f'{x:.{p}f}'
    return str(x)


def render_markdown(report, title='Pitch model diagnostic report', meta=None):
    """Render the report dict to a markdown string with an interpretation legend."""
    s = report['summary']
    lines = [f'# {title}', '']
    if meta:
        for k, v in meta.items():
            lines.append(f'- **{k}**: {v}')
        lines.append('')

    lines += [
        '## How to read this',
        '- **RPA@10 low but RPA@50 high** → precision-limited (likely the coarse / high-fmax mel). Compare with the fmax8000 diagnostic run.',
        '- **RCA ≫ RPA** or high octave-error rate → octave errors dominate.',
        '- **High VFA / voicing-miss** → voicing threshold or voicing modelling issue (try the threshold sweep below).',
        '- **|bias| large (cent error)** → systematic offset (mel/f0 alignment or cents-mapping), not noise.',
        '- **Low registers much worse than high** (by-register table) → the input representation is the bottleneck.',
        '- **A few files dominate the errors** (per-file table) → data/teacher problem on those clips, not the model.',
        '',
        '## Headline (frame-pooled)',
        '| metric | value |',
        '|---|---|',
        f"| RPA@50 | {_fmt(s['RPA'])} |",
        f"| RPA@25 | {_fmt(s.get('RPA@25'))} |",
        f"| RPA@10 | {_fmt(s.get('RPA@10'))} |",
        f"| RCA (chroma) | {_fmt(s['RCA'])} |",
        f"| Overall Acc | {_fmt(s['OA'])} |",
        f"| Voicing Recall | {_fmt(s['VR'])} |",
        f"| Voicing False Alarm | {_fmt(s['VFA'])} |",
        f"| voicing threshold | {_fmt(s['voicing_th'])} |",
        '',
        '## Error decomposition (over reference-voiced frames)',
        f"reference-voiced frames: **{s['n_ref_voiced']}**, reference-unvoiced: **{s['n_ref_unvoiced']}**",
        '',
        '| outcome | count | share of ref-voiced |',
        '|---|---|---|',
        f"| correct (≤50¢) | {s['correct']} | {_fmt(s['RPA'])} |",
        f"| octave error | {s['octave_error']} | {_fmt(s['octave_error_rate'])} |",
        f"| other gross error | {s['gross_error']} | {_fmt(s['gross_error_rate'])} |",
        f"| voicing miss (said unvoiced) | {s['voicing_miss']} | {_fmt(s['voicing_miss_rate'])} |",
        f"| false alarm (over ref-unvoiced) | {s['false_alarm']} | {_fmt(s['VFA'])} |",
        '',
    ]

    ce = report['cent_error']
    lines += ['## Cent-error bias / spread (correct-ish frames, |err|≤200¢)']
    if ce.get('n', 0) == 0:
        lines += ['(no voiced-voiced frames)', '']
    else:
        lines += [
            f"- bias: **{_fmt(ce['bias_cent_precision'])}** ¢, std: **{_fmt(ce['std_cent_precision'])}** ¢, MAE: **{_fmt(ce['mae_cent_precision'])}** ¢ (n={ce['n_precision']})",
            f"- including octave/gross: bias {_fmt(ce['bias_cent_all'])} ¢, std {_fmt(ce['std_cent_all'])} ¢",
            '',
        ]

    lines += ['## Accuracy by reference register', '| band (Hz) | frames | RPA@50 | octave rate | median |err| ¢ |', '|---|---|---|---|---|']
    for r in report['by_register']:
        lo, hi = r['band_hz']
        hi_s = '∞' if hi >= 1e8 else f'{hi:.0f}'
        lines.append(f"| {lo:.0f}–{hi_s} | {r['n']} | {_fmt(r['RPA'])} | {_fmt(r['octave_rate'])} | {_fmt(r['median_abs_cent'],1)} |")
    lines.append('')

    lines += ['## Voicing-threshold sweep', '| th | RPA@50 | VR | VFA |', '|---|---|---|---|']
    for r in report['threshold_sweep']:
        lines.append(f"| {_fmt(r['th'])} | {_fmt(r['RPA'])} | {_fmt(r['VR'])} | {_fmt(r['VFA'])} |")
    lines.append('')

    worst = report['per_file'][:10]
    lines += ['## Worst files (lowest RPA@50)', '| file | ref-voiced frames | RPA@50 |', '|---|---|---|']
    for r in worst:
        lines.append(f"| {r['file']} | {r['n_ref_voiced']} | {_fmt(r['RPA'])} |")
    lines.append('')

    return '\n'.join(lines)
