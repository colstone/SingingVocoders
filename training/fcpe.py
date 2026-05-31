import numpy as np
import torch
import lightning.pytorch as pl

from mir_eval.melody import (
    raw_pitch_accuracy, to_cent_voicing, raw_chroma_accuracy, overall_accuracy,
    voicing_recall, voicing_false_alarm,
)

from models.fcpe import FCPE_E2E
from modules.loss.rmvpe_loss import FL, bce
from training.rmvpe import RMVPETask  # reuse RMVPE_dataset / dataloaders / optimizer / decoder / metrics


class FCPETask(RMVPETask):
    """
    Benchmark task for the FCPE backbone.

    Identical to the RMVPE training setup in EVERY respect (same RMVPE_dataset, same unchanged mel,
    same one-hot labels, same FL loss, same `to_local_average_cents` decoder and mir_eval metrics)
    EXCEPT the backbone is FCPE's `CFNaiveMelPE` instead of the 2D DeepUNet. This isolates the
    architecture as the only variable for an apples-to-apples comparison.

    Validation uses a single full-sequence forward (no 128-frame chunking): FCPE preserves the time
    resolution and has no 2**en_de_layers divisibility constraint, so the whole clip is run at once,
    avoiding the chunk-boundary artifacts that the RMVPE sliding-window validation introduces.
    """

    def __init__(self, config: dict):
        # Bypass RMVPETask.__init__ (which builds the DeepUNet E2E) and set up the same attributes
        # with the FCPE backbone instead.
        pl.LightningModule.__init__(self)
        self.config = config
        self.save_hyperparameters(config)

        self.generator = FCPE_E2E(config=config)

        self.alpha = config['alpha']
        self.gamma = config['gamma']
        self.learning_rate = float(config['learning_rate'])
        self.clip_grad_norm = config.get('clip_grad_norm', None)

        self.validation_step_outputs = []
        self.skip_immediate_ckpt_save = False  # Required by DsModelCheckpoint

    def validation_step(self, batch, batch_idx):
        mel = batch['mel']            # (1, n_mels, T)
        pitch_label = batch['pitch']  # (1, T, n_class)

        # Full-sequence inference (no chunking).
        _, pitch_pred = self.generator(mel)  # (1, T, n_class)

        loss = bce(pitch_pred, pitch_label)
        self.log('val/loss', loss)

        # Voicing threshold: max bin prob below this -> treated as unvoiced (F0=0).
        pitch_th = self.config.get('pitch_th', 0.03)

        cents_pred = self.to_local_average_cents(pitch_pred.squeeze(0).cpu().numpy(), None, pitch_th)
        cents_label = self.to_local_average_cents(pitch_label.squeeze(0).cpu().numpy(), None, pitch_th)

        freq_pred = np.array([10 * (2 ** (c / 1200)) if c else 0 for c in cents_pred])
        freq = np.array([10 * (2 ** (c / 1200)) if c else 0 for c in cents_label])

        hop_cent_length = self.config['hop_size'] / self.config['audio_sample_rate']
        time_slice = np.array([i * hop_cent_length for i in range(len(cents_label))])

        ref_v, ref_c, est_v, est_c = to_cent_voicing(time_slice, freq, time_slice, freq_pred)

        metrics = {
            'RPA': raw_pitch_accuracy(ref_v, ref_c, est_v, est_c),
            'RCA': raw_chroma_accuracy(ref_v, ref_c, est_v, est_c),
            'OA': overall_accuracy(ref_v, ref_c, est_v, est_c),
            'VFA': voicing_false_alarm(ref_v, est_v),
            'VR': voicing_recall(ref_v, est_v),
            'loss': loss.item(),
        }

        if batch_idx == 0:
            self.log_f0_comparison(freq, freq_pred)

        self.validation_step_outputs.append(metrics)
        return metrics
