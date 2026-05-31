"""Post-training diagnostic report for mel->f0 models (FCPE or RMVPE).

Runs full-sequence inference over the validation set, decodes model f0 and the reference
(teacher) f0 from the dataset labels, and writes a markdown report + metrics.json + figures
that localise *where* the model is bad (octave vs voicing vs gross vs precision; by register;
per file). Works for any task whose `generator(mel)->(hidden, pred)` outputs a (B,T,n_class)
sigmoid and that exposes `to_local_average_cents` (RMVPETask / FCPETask both do).

NOTE on the reference: it is decoded from the validation-set labels, i.e. the teacher
(harvest/parselmouth). The report measures agreement with that teacher; if the teacher itself
is wrong on some frames, that shows up as model "error". Use the by-register breakdown and the
fmax8000 diagnostic run to separate input-resolution limits from model limits.

Usage (run from the SingingVocoders dir):
    python eval_pitch_report.py --config configs/fcpe.yaml --exp_name fcpe_fmax16000
    python eval_pitch_report.py --config configs/fcpe.yaml --ckpt path/to/model.ckpt --out reports/fcpe
"""
import importlib
import json
import os
import pathlib

import click
import numpy as np
import torch
import torch.nn.functional as F

from utils.config_utils import read_full_config
from utils.pitch_report import build_report, render_markdown, cents_to_hz

try:
    from utils.training_utils import get_latest_checkpoint_path
except Exception:
    get_latest_checkpoint_path = None


def _build_task(config):
    pkg = ".".join(config["task_cls"].split(".")[:-1])
    cls_name = config["task_cls"].split(".")[-1]
    task_cls = getattr(importlib.import_module(pkg), cls_name)
    return task_cls(config=config)


def _resolve_ckpt(ckpt, work_dir, exp_name):
    if ckpt:
        return pathlib.Path(ckpt)
    if exp_name is None:
        raise click.ClickException("Provide --ckpt or --exp_name to locate a checkpoint.")
    wd = pathlib.Path(work_dir) if work_dir else (pathlib.Path(__file__).parent / 'experiments')
    exp_dir = wd / exp_name
    if get_latest_checkpoint_path is not None:
        p = get_latest_checkpoint_path(exp_dir)
        if p:
            return pathlib.Path(p)
    cks = sorted(exp_dir.glob('*.ckpt'))
    if not cks:
        raise click.ClickException(f"No checkpoint found in {exp_dir}")
    return cks[-1]


def _pad_to_multiple(mel, m=32):
    """Pad time (last) dim up to a multiple of m so UNet-style models accept it; returns (mel, orig_T)."""
    T = mel.shape[-1]
    pad = (-T) % m
    if pad:
        mel = F.pad(mel, (0, pad))
    return mel, T


@torch.no_grad()
def _run_inference(task, dataset, device, pad_multiple=32):
    """Return per-frame arrays concatenated over the valid set + per-file contour data."""
    decode = task.to_local_average_cents
    ref_cent_all, pred_cent_all, pred_max_all, file_id_all = [], [], [], []
    files, contours = [], []

    for fid, item in enumerate(dataset.data):
        mel = item['mel']                       # (n_mels, T)
        label = item['pitch'].cpu().numpy()     # (T, n_class)
        mel_in = mel.unsqueeze(0).to(device)    # (1, n_mels, T)
        mel_in, T = _pad_to_multiple(mel_in, pad_multiple)

        _, pred = task.generator(mel_in)        # (1, Tpad, n_class)
        pred = pred[0, :T].float().cpu().numpy()  # (T, n_class)

        ref_cent = decode(label, None, 0.0)            # 0 where unvoiced
        pred_cent = decode(pred, None, 0.0)            # always a value (sigmoid > 0)
        pred_max = pred.max(axis=-1)

        ref_cent_all.append(ref_cent)
        pred_cent_all.append(pred_cent)
        pred_max_all.append(pred_max)
        file_id_all.append(np.full(T, fid, dtype=np.int64))
        files.append(os.path.basename(str(item['file'])))
        contours.append({'ref_cent': ref_cent, 'pred_cent': pred_cent, 'pred_max': pred_max})

    return (np.concatenate(ref_cent_all), np.concatenate(pred_cent_all),
            np.concatenate(pred_max_all), np.concatenate(file_id_all), files, contours)


def _make_figures(report, ref_cent, pred_cent, pred_max, voicing_th, files, contours, out_dir, max_contour_files):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        return [f"(matplotlib unavailable, figures skipped: {e})"]

    figs = []
    ref_v = ref_cent > 0
    pred_v = pred_max > voicing_th
    both = ref_v & pred_v
    d = (pred_cent - ref_cent)[both]

    # 1) cent-error histogram
    if d.size:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(np.clip(d, -1500, 1500), bins=120)
        ax.axvline(0, color='k', lw=0.8)
        ax.set_title('Cent error (pred - ref), voiced-voiced frames (clipped ±1500¢)')
        ax.set_xlabel('cent error'); ax.set_ylabel('frames')
        p = out_dir / 'cent_error_hist.png'; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
        figs.append(p.name)

    # 2) RPA by register
    rows = [r for r in report['by_register'] if r['n'] > 0]
    if rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = []
        for r in rows:
            lo, hi = r['band_hz']
            hi_s = '∞' if hi >= 1e8 else '{:.0f}'.format(hi)
            labels.append('{:.0f}-{}'.format(lo, hi_s))
        ax.bar(labels, [r['RPA'] for r in rows])
        ax.set_ylim(0, 1); ax.set_title('RPA@50 by reference register (Hz)')
        ax.set_ylabel('RPA@50'); ax.set_xlabel('register (Hz)')
        p = out_dir / 'rpa_by_register.png'; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
        figs.append(p.name)

    # 3) threshold sweep
    sw = report['threshold_sweep']
    if sw:
        fig, ax = plt.subplots(figsize=(8, 4))
        th = [r['th'] for r in sw]
        ax.plot(th, [r['RPA'] for r in sw], '-o', label='RPA@50')
        ax.plot(th, [r['VR'] for r in sw], '-s', label='VR')
        ax.plot(th, [r['VFA'] for r in sw], '-^', label='VFA')
        ax.set_xlabel('voicing threshold'); ax.set_ylim(0, 1); ax.legend(); ax.set_title('Voicing-threshold sweep')
        p = out_dir / 'threshold_sweep.png'; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
        figs.append(p.name)

    # 4) worst-file f0 contour overlays
    name2id = {f: i for i, f in enumerate(files)}
    worst = [r['file'] for r in report['per_file'] if not np.isnan(r['RPA'])][:max_contour_files]
    if worst:
        k = len(worst)
        fig, axes = plt.subplots(k, 1, figsize=(10, 2.2 * k), squeeze=False)
        for ax, fname in zip(axes[:, 0], worst):
            c = contours[name2id[fname]]
            rc, pc, pm = c['ref_cent'], c['pred_cent'], c['pred_max']
            ref_hz = cents_to_hz(rc); ref_hz[ref_hz == 0] = np.nan
            pred_hz = cents_to_hz(pc); pred_hz[pm <= voicing_th] = np.nan
            ax.plot(ref_hz, label='ref (teacher)', lw=1.2)
            ax.plot(pred_hz, label='pred', lw=1.0, alpha=0.8)
            ax.set_title(fname, fontsize=8); ax.set_ylabel('Hz')
        axes[0, 0].legend(loc='upper right', fontsize=7)
        p = out_dir / 'worst_files_contours.png'; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
        figs.append(p.name)

    return figs


@click.command(help='Generate a mel->f0 diagnostic report from a trained checkpoint.')
@click.option('--config', required=True, metavar='FILE', help='Config yaml (e.g. configs/fcpe.yaml).')
@click.option('--ckpt', required=False, metavar='FILE', help='Checkpoint path. If omitted, located via --exp_name.')
@click.option('--exp_name', required=False, metavar='EXP', help='Experiment name (to find latest ckpt + default out dir).')
@click.option('--work_dir', required=False, metavar='DIR', help='Experiments root (default: ./experiments).')
@click.option('--out', required=False, metavar='DIR', help='Output dir for the report (default: <exp_dir>/pitch_report).')
@click.option('--device', required=False, metavar='DEV', help="'cuda' / 'cpu' (default: auto).")
@click.option('--voicing_th', required=False, type=float, help='Override voicing threshold (default: config pitch_th or 0.03).')
@click.option('--max_contour_files', default=6, show_default=True, help='How many worst files to plot.')
def main(config, ckpt, exp_name, work_dir, out, device, voicing_th, max_contour_files):
    cfg_path = pathlib.Path(config)
    cfg = read_full_config(cfg_path)

    device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = _resolve_ckpt(ckpt, work_dir, exp_name)
    click.echo(f"[report] config={cfg_path}  ckpt={ckpt_path}  device={device}")

    task = _build_task(cfg)
    state = torch.load(str(ckpt_path), map_location='cpu')
    sd = state.get('state_dict', state) if isinstance(state, dict) else state
    missing, unexpected = task.load_state_dict(sd, strict=False)
    if missing:
        click.echo(f"[report] missing keys: {len(missing)} (first few: {list(missing)[:4]})")
    if unexpected:
        click.echo(f"[report] unexpected keys: {len(unexpected)} (first few: {list(unexpected)[:4]})")
    task.eval().to(device)

    # Build validation dataset (test=True -> full-sequence items)
    from training.rmvpe import RMVPE_dataset
    val_path = os.path.join(cfg['DataIndexPath'], cfg['valid_set_name'])
    click.echo(f"[report] loading valid set from {val_path} ...")
    dataset = RMVPE_dataset(config=cfg, path=val_path, test=True)
    if len(dataset.data) == 0:
        raise click.ClickException(f"No data found at {val_path}")

    pad_multiple = 2 ** int(cfg.get('en_de_layers', 5))  # RMVPE needs T % 32; harmless for FCPE
    vth = float(voicing_th) if voicing_th is not None else float(cfg.get('pitch_th', 0.03))

    ref_cent, pred_cent, pred_max, file_ids, files, contours = _run_inference(task, dataset, device, pad_multiple)

    report = build_report(ref_cent, pred_cent, pred_max, file_ids, files, vth)

    out_dir = pathlib.Path(out) if out else (ckpt_path.parent / 'pitch_report')
    out_dir.mkdir(parents=True, exist_ok=True)

    figs = _make_figures(report, ref_cent, pred_cent, pred_max, vth, files, contours, out_dir, max_contour_files)

    meta = {
        'config': str(cfg_path),
        'checkpoint': str(ckpt_path),
        'task_cls': cfg['task_cls'],
        'pitch_extractor (teacher)': cfg.get('pitch_extractor'),
        'mel': f"n_mels={cfg['audio_num_mel_bins']}, hop={cfg['hop_size']}, sr={cfg['audio_sample_rate']}, "
               f"fmin={cfg['fmin']}, fmax={cfg['fmax']}",
        'valid files': len(files),
        'voicing threshold': vth,
    }
    md = render_markdown(report, title='mel→f0 diagnostic report', meta=meta)
    if figs:
        md += '\n## Figures\n' + '\n'.join(f'![{n}]({n})' for n in figs if n.endswith('.png')) + '\n'

    (out_dir / 'report.md').write_text(md, encoding='utf8')
    (out_dir / 'metrics.json').write_text(json.dumps({'meta': meta, **report}, indent=2, ensure_ascii=False), encoding='utf8')

    s = report['summary']
    click.echo(f"[report] RPA@50={s['RPA']:.3f} RCA={s['RCA']:.3f} octave_rate={s['octave_error_rate']:.3f} "
               f"VFA={s['VFA']:.3f} VR={s['VR']:.3f}")
    click.echo(f"[report] written to {out_dir / 'report.md'}")


if __name__ == '__main__':
    main()
