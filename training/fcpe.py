import importlib
import os

import numpy as np
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from modules.loss.rmvpe_loss import FL, bce
from training.rmvpe import RMVPETask, build_dataset  # reuse task base / dataset_cls switch / decoder / metrics
from utils.pitch_metrics import decompose, rpa_at_tolerances


class FCPETask(RMVPETask):
    """
    Benchmark task for the FCPE backbone, instrumented for train-vs-val self-diagnosis.

    - Backbone is FCPE's CFNaiveMelPE instead of the 2D DeepUNet; everything else (CachedF0Dataset,
      one-hot labels, FL loss, to_local_average_cents decode) is identical => isolates the backbone.
    - Validation runs full-sequence (no chunking).
    - A shared `_eval_pitch` computes the SAME rich metric set (committed/co-voiced accuracy,
      RPA@10/25/50, octave/gross/voicing-miss rates, RCA/OA/VR/VFA) for BOTH the val set and a
      fixed train-probe subset, logged as `val/*` and `train/*`. The train-vs-val gap on
      committed_accuracy tells under/overfit:
        train >> val  -> overfit (add diverse data, not capacity)
        train ~= val (both low) -> label/input/optimization ceiling (clean labels / Gaussian / train longer)
    """

    def __init__(self, config: dict):
        # Bypass RMVPETask.__init__ (which builds the DeepUNet E2E); set up the same attributes with FCPE.
        pl.LightningModule.__init__(self)
        self.config = config
        self.save_hyperparameters(config)

        # Backbone is configurable so the SAME instrumented task (eval / committed_accuracy /
        # train-probe / early stopping) can drive different architectures for a fair A/B.
        # Default = FCPE; set generator_cls: models.rmvpe.model.E2E to train RMVPE on identical data.
        gen_cls_path = config.get('generator_cls', 'models.fcpe.FCPE_E2E')
        gmod, gname = gen_cls_path.rsplit('.', 1)
        GenCls = getattr(importlib.import_module(gmod), gname)
        self.generator = GenCls(config=config)

        self.alpha = config['alpha']
        self.gamma = config['gamma']
        self.learning_rate = float(config['learning_rate'])
        self.clip_grad_norm = config.get('clip_grad_norm', None)
        self.pitch_th = float(config.get('pitch_th', 0.03))
        self.train_probe_num = int(config.get('train_probe_num', 0))
        self.pad_multiple = 2 ** int(config.get('en_de_layers', 5))  # harmless for FCPE; needed if backbone downsamples

        # validation outputs grouped by dataloader_idx (0=val, 1=train-probe)
        self.validation_step_outputs = {}
        self.skip_immediate_ckpt_save = False  # Required by DsModelCheckpoint

    # ----- shared evaluation path (val and train-probe use the EXACT same code -> comparable) -----
    @torch.no_grad()
    def _eval_pitch(self, mel, pitch_label):
        """mel: (1, n_mels, T); pitch_label: (1, T, n_class). Returns (metrics, (ref_cent, pred_cent, pred_max))."""
        T = mel.shape[-1]
        pad = (-T) % self.pad_multiple
        mel_in = F.pad(mel, (0, pad)) if pad else mel
        _, pred = self.generator(mel_in)        # (1, T(+pad), n_class)
        pred = pred[:, :T]

        loss = bce(pred, pitch_label)
        pred_np = pred.squeeze(0).float().cpu().numpy()        # (T, n_class)
        label_np = pitch_label.squeeze(0).float().cpu().numpy()

        pred_cent = self.to_local_average_cents(pred_np, None, 0.0)    # always a value (sigmoid > 0)
        ref_cent = self.to_local_average_cents(label_np, None, 0.0)    # 0 where unvoiced
        pred_max = pred_np.max(axis=-1)

        metrics = decompose(ref_cent, pred_cent, pred_max, self.pitch_th)
        metrics.update(rpa_at_tolerances(ref_cent, pred_cent, pred_max, self.pitch_th))
        metrics['loss'] = float(loss.item())
        return metrics, (ref_cent, pred_cent, pred_max)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        mel = batch['mel']            # (1, n_mels, T)
        pitch_label = batch['pitch']  # (1, T, n_class)
        metrics, arrays = self._eval_pitch(mel, pitch_label)
        self.validation_step_outputs.setdefault(dataloader_idx, []).append(metrics)

        if dataloader_idx == 0 and batch_idx == 0:
            ref_cent, pred_cent, pred_max = arrays
            freq = np.array([10 * (2 ** (c / 1200)) if c > 0 else 0 for c in ref_cent])
            freq_pred = np.array([10 * (2 ** (c / 1200)) if (c > 0 and m > self.pitch_th) else 0
                                  for c, m in zip(pred_cent, pred_max)])
            self.log_f0_comparison(freq, freq_pred)
        return metrics

    _LOG_KEYS = ['committed_accuracy', 'RPA', 'RPA@25', 'RPA@10', 'RCA', 'OA', 'VR', 'VFA',
                 'octave_error_rate', 'gross_error_rate', 'voicing_miss_rate', 'loss']

    def on_validation_epoch_end(self):
        names = {0: 'val', 1: 'train'}
        for didx, outs in self.validation_step_outputs.items():
            if not outs:
                continue
            prefix = names.get(didx, f'val{didx}')
            for k in self._LOG_KEYS:
                vals = [o[k] for o in outs if k in o]
                if vals:
                    prog = (prefix == 'val' and k in ('RPA', 'committed_accuracy'))
                    self.log(f'{prefix}/{k}', float(np.mean(vals)), prog_bar=prog)
        self.validation_step_outputs = {}

    def val_dataloader(self):
        val_ds = build_dataset(
            config=self.config,
            path=os.path.join(self.config['DataIndexPath'], self.config['valid_set_name']),
            test=True,
        )
        loaders = [DataLoader(val_ds, batch_size=1, shuffle=False, drop_last=False,
                              collate_fn=val_ds.collate_fn, num_workers=self.config.get('ds_workers', 4))]
        if self.train_probe_num > 0:
            probe_ds = build_dataset(
                config=self.config,
                path=os.path.join(self.config['DataIndexPath'], self.config['train_set_name']),
                test=True,
                max_files=self.train_probe_num,
            )
            loaders.append(DataLoader(probe_ds, batch_size=1, shuffle=False, drop_last=False,
                                      collate_fn=probe_ds.collate_fn, num_workers=self.config.get('ds_workers', 4)))
        return loaders
