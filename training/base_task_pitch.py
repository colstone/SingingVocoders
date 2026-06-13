import functools

import numpy as np
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

from modules.loss.rmvpe_loss import FL, bce
from utils.pitch_metrics import decompose, rpa_at_tolerances


@functools.lru_cache(maxsize=16)
def _load_npz(path):
    """Decompress one process.py clip into memory (small LRU cache for repeated segment access).

    process.py saves mel as (T, n_mels) and f0 interpolated through unvoiced regions (interp_uv=True),
    with a separate `uv` mask. We transpose mel to (n_mels, T) and re-zero f0 at unvoiced frames so the
    `f0 > 0` voicing convention in _f0_to_labels holds.
    """
    with np.load(path) as z:
        mel = np.ascontiguousarray(np.asarray(z['mel'], dtype=np.float32).T)   # (T, n_mels) -> (n_mels, T)
        f0 = np.asarray(z['f0'], dtype=np.float32)
        uv = np.asarray(z['uv']).astype(bool)
        f0 = np.where(uv, np.float32(0.0), f0).astype(np.float32)              # undo interp at unvoiced
    return {'mel': mel, 'f0': f0}


class CachedF0Dataset(Dataset):
    """
    Lazy dataset over process.py's PRECOMPUTED per-clip features (the standard vocoder preprocessing).
    Shared by both pitch backbones (cf. BaseDataset living in base_task_gan.py).

    process.py writes one `.npz` per clip holding: mel (T, n_mels) log-compressed (clip_val=1e-9),
    f0 (T,) Hz (interpolated through unvoiced, interp_uv=True), uv (T,), audio, pe; plus a train/valid
    INDEX TEXT FILE (one .npz path per line) at DataIndexPath/{train,valid}_set_name. This dataset
    reads that index file (like nsf_HiFigan_dataset). _load_npz transposes mel to (n_mels, T) and
    re-zeros f0 at unvoiced frames (via uv); one-hot pitch labels are built per __getitem__.

    `time_multiple`: the backbone's required input time-divisibility (set by the task's build_dataset;
    FCPE=1, RMVPE=2**en_de_layers). __getitem__ returns each clip at its REAL length; collate_fn pads a
    batch's time axis up to max(real lengths) rounded up to time_multiple, recording the real `length`
    of each item so the task can crop predictions back. Training segments are equal-length -> no padding.

    Built to scale to hundreds of hours: __init__ only indexes segments (reads f0 length per clip);
    mel is loaded/sliced lazily in __getitem__ with a small LRU cache.

    Segmentation: frame stride = seq_l // hop, window = stride + 1, plus a tail segment; clips
    shorter than one window are skipped (training). test=True -> one full-clip item.
    """

    def __init__(self, config: dict, path, test=False, time_multiple=1, max_files=None):
        super().__init__()
        self.config = config
        self.path = path
        self.hop_length = config['hop_size']
        self.num_class = config['n_class']
        self.CONST = config['const']
        self.time_multiple = int(time_multiple)
        self.seq_len = config['seq_l'] if not test else None
        if self.seq_len is not None:
            assert self.seq_len % self.hop_length == 0
            # training segments must be divisible by the backbone's time multiple (checked at build time)
            assert ((self.seq_len // self.hop_length) + 1) % self.time_multiple == 0

        # process.py writes an index text file (one .npz path per line) at `path`.
        with open(self.path, 'r', encoding='utf8') as f:
            files = [ln.strip() for ln in f if ln.strip()]
        if max_files is not None:
            files = files[:max_files]               # deterministic fixed subset (e.g. train-probe)
        if not files:
            import logging
            logging.warning(f"No .npz paths found in index file {self.path}")

        # index entries: (npz_path, begin_step, end_step)
        self.index = []
        if self.seq_len is None:
            for p in files:
                T = self._frames(p)
                self.index.append((p, 0, T))
        else:
            stride = self.seq_len // self.hop_length          # 255
            n_steps = stride + 1                              # 256
            for p in files:
                T = self._frames(p)
                if T < n_steps:
                    continue  # too short -> skip
                start = 0
                while start + n_steps <= T:
                    self.index.append((p, start, start + n_steps))
                    start += stride
                tail = T - n_steps
                if tail > 0 and (not self.index or self.index[-1][0] != p or self.index[-1][1] != tail):
                    self.index.append((p, tail, tail + n_steps))

    @staticmethod
    def _frames(path):
        # read only f0 length (== mel frames) to size the index without decompressing mel
        with np.load(path) as z:
            return int(z['f0'].shape[0])

    def _f0_to_labels(self, f0):
        """Vectorized per-frame one-hot label build (cent = 1200*log2(f0/10), 20¢/bin, same CONST)."""
        n = len(f0)
        pitch = torch.zeros(n, self.num_class, dtype=torch.float)
        voice = torch.zeros(n, dtype=torch.float)
        f0 = np.asarray(f0, dtype=np.float64)
        voiced = f0 > 0
        cents = np.zeros(n, dtype=np.float64)
        cents[voiced] = 1200.0 * np.log2(f0[voiced] / 10.0)
        idx = np.round((cents - self.CONST) / 20.0).astype(np.int64)
        valid = voiced & (idx >= 0) & (idx < self.num_class)
        rows = np.nonzero(valid)[0]
        if rows.size:
            pitch[torch.from_numpy(rows), torch.from_numpy(idx[rows])] = 1.0
            voice[torch.from_numpy(rows)] = 1.0
        return pitch, voice

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        path, b, e = self.index[i]
        data = _load_npz(path)
        mel = torch.from_numpy(data['mel'][:, b:e]).float()      # (n_mels, T) at REAL length
        f0 = data['f0'][b:e]
        pitch, voice = self._f0_to_labels(f0)                    # (T, n_class), (T,)
        return dict(mel=mel, pitch=pitch, voice=voice, file=path)

    def collate_fn(self, batch):
        """Pad the batch's time axis to max(real lengths) rounded up to time_multiple; record each real
        `length` so the task can crop predictions back. Training segments are equal-length -> no padding."""
        files = [item['file'] for item in batch]
        lengths = [item['mel'].shape[1] for item in batch]        # real (unpadded) time lengths
        T = max(lengths)
        T = T + (-T % self.time_multiple)                         # round up to the backbone's time multiple
        mel = torch.stack([F.pad(it['mel'], (0, T - it['mel'].shape[1])) for it in batch])           # (B, n_mels, T)
        pitch = torch.stack([F.pad(it['pitch'], (0, 0, 0, T - it['pitch'].shape[0])) for it in batch])  # (B, T, n_class)
        voice = torch.stack([F.pad(it['voice'], (0, T - it['voice'].shape[0])) for it in batch])     # (B, T)
        return {'mel': mel, 'pitch': pitch, 'voice': voice, 'length': lengths, 'file': files}


class PitchBaseTask(pl.LightningModule):
    """
    Backbone-agnostic mel->f0 training task (instrumented for train-vs-val self-diagnosis).

    Subclasses (training/fcpe_task.py, training/rmvpe_task.py) implement the two per-backbone hooks —
    `build_model()` (construct self.generator) and `build_dataset()` (construct the datasets with the
    backbone's required input time-divisibility `time_multiple`) — same hook style as the vocoder
    GanBaseTask. EVERYTHING else is shared so the backbones train identically and comparably:
      - one-hot labels, FL loss, Adam+StepLR, to_local_average_cents decode;
      - a single per-clip metric path computing the SAME rich set (committed/co-voiced accuracy,
        RPA@10/25/50, octave/gross/voicing-miss, RCA/OA/VR/VFA) for BOTH the val set and a fixed
        train-probe subset, logged to TensorBoard as `val/*` and `train/*`. The train-vs-val gap on
        committed_accuracy tells under/overfit:
          train >> val            -> overfit (add diverse data, not capacity)
          train ~= val (both low)  -> label/input/optimization ceiling
    Validation supports batch_size>1: the dataset's collate_fn zero-pads each batch to a common length
    (a multiple of the backbone's time_multiple) and records each clip's real length; here we crop every
    prediction back to its real length (`_crop_to_length`) before scoring. All monitoring is TensorBoard
    scalars (no matplotlib).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)

        # per-backbone hook: the subclass's build_model() sets self.generator (cf. GanBaseTask.build_model)
        self.build_model()

        self.alpha = config['alpha']
        self.gamma = config['gamma']
        self.learning_rate = float(config['learning_rate'])
        self.clip_grad_norm = config.get('clip_grad_norm', None)
        self.pitch_th = float(config.get('pitch_th', 0.03))
        self.train_probe_num = int(config.get('train_probe_num', 0))

        self.train_dataset = None
        self.valid_dataset = None
        self.train_probe_dataset = None
        # validation outputs grouped by dataloader_idx (0=val, 1=train-probe)
        self.validation_step_outputs = {}
        self.skip_immediate_ckpt_save = False  # Required by DsModelCheckpoint

    # ----- per-backbone hooks (implemented by FCPETask / RMVPETask) -----
    def build_model(self):
        """Set self.generator to the backbone (B, n_mels, T) -> (hidden, pred[B, T, n_class]). Subclass-only."""
        raise NotImplementedError("PitchBaseTask is abstract; use FCPETask (fcpe_task.py) / RMVPETask (rmvpe_task.py).")

    def build_dataset(self):
        """Set self.train_dataset / valid_dataset / train_probe_dataset (with the backbone's time_multiple). Subclass-only."""
        raise NotImplementedError("PitchBaseTask is abstract; use FCPETask (fcpe_task.py) / RMVPETask (rmvpe_task.py).")

    def setup(self, stage=None):
        # Lightning calls this automatically per-process at the start of fit/validate; build datasets here
        # (same place as the vocoder GanBaseTask.setup -> build_dataset).
        self.build_dataset()

    def forward(self, mel):
        return self.generator(mel)

    def training_step(self, batch, batch_idx):
        hidden_vec, pitch_pred = self.generator(batch['mel'])
        loss = FL(pitch_pred, batch['pitch'], self.alpha, self.gamma)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        scheduler = StepLR(
            optimizer,
            step_size=self.config.get('learning_rate_decay_steps', 1000),
            gamma=self.config.get('learning_rate_decay_rate', 0.98),
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.config['batch_size'], shuffle=True, drop_last=True,
            collate_fn=self.train_dataset.collate_fn, num_workers=self.config.get('ds_workers', 4),
            persistent_workers=True,
        )

    def val_dataloader(self):
        bs = self.config.get('val_batch_size', 1)   # variable-length clips are padded to batch-max -> raise with care
        loaders = [DataLoader(self.valid_dataset, batch_size=bs, shuffle=False, drop_last=False,
                              collate_fn=self.valid_dataset.collate_fn, num_workers=self.config.get('ds_workers', 4))]
        if self.train_probe_dataset is not None:
            loaders.append(DataLoader(self.train_probe_dataset, batch_size=bs, shuffle=False, drop_last=False,
                                      collate_fn=self.train_probe_dataset.collate_fn,
                                      num_workers=self.config.get('ds_workers', 4)))
        return loaders

    # ----- shared evaluation path (val and train-probe use the EXACT same code -> comparable) -----
    @staticmethod
    def _crop_to_length(x, length):
        """Crop a padded (B, T, ...) tensor's time axis back to a clip's real length. Shared by both
        backbones, since batched validation pads variable-length clips to a common (padded) length."""
        return x[:, :length]

    def _eval_metrics(self, pred, pitch_label):
        """pred, pitch_label: (1, L, n_class), already cropped to the clip's real length L -> metric dict."""
        loss = bce(pred, pitch_label)
        pred_np = pred.squeeze(0).float().cpu().numpy()        # (L, n_class)
        label_np = pitch_label.squeeze(0).float().cpu().numpy()

        pred_cent = self.to_local_average_cents(pred_np, None, 0.0)    # always a value (sigmoid > 0)
        ref_cent = self.to_local_average_cents(label_np, None, 0.0)    # 0 where unvoiced
        pred_max = pred_np.max(axis=-1)

        metrics = decompose(ref_cent, pred_cent, pred_max, self.pitch_th)
        metrics.update(rpa_at_tolerances(ref_cent, pred_cent, pred_max, self.pitch_th))
        metrics['loss'] = float(loss.item())
        return metrics

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        _, pred = self.generator(batch['mel'])           # (B, T_pad, n_class)
        pitch_label = batch['pitch']                     # (B, T_pad, n_class)
        for b, L in enumerate(batch['length']):          # crop each clip back to its real length, then score
            L = int(L)
            metrics = self._eval_metrics(self._crop_to_length(pred[b:b + 1], L),
                                         self._crop_to_length(pitch_label[b:b + 1], L))
            self.validation_step_outputs.setdefault(dataloader_idx, []).append(metrics)

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

    @staticmethod
    def to_local_average_cents(salience, center=None, thred=0.0):
        """Weighted average of cents near the argmax bin (9-bin window). 2D -> per-frame."""
        if not hasattr(PitchBaseTask.to_local_average_cents, 'cents_mapping'):
            PitchBaseTask.to_local_average_cents.cents_mapping = (
                    np.linspace(0, 7180, 360) + 1997.3794084376191)
        mapping = PitchBaseTask.to_local_average_cents.cents_mapping

        if salience.ndim == 1:
            if center is None:
                center = int(np.argmax(salience))
            start = max(0, center - 4)
            end = min(len(salience), center + 5)
            salience = salience[start:end]
            if len(salience) == 0:
                return 0
            product_sum = np.sum(salience * mapping[start:end])
            weight_sum = np.sum(salience)
            return product_sum / weight_sum if np.max(salience) > thred else 0
        if salience.ndim == 2:
            return np.array([PitchBaseTask.to_local_average_cents(salience[i, :], None, thred)
                             for i in range(salience.shape[0])])
        raise Exception("label should be either 1d or 2d ndarray")
