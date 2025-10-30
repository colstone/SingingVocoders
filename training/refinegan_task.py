import logging
import os
import pathlib
import random
import sys
from typing import Dict

import lightning.pytorch as pl
import matplotlib
import numpy as np
import torch
import torch.utils.data
import torchaudio
from lightning.pytorch.utilities.rank_zero import rank_zero_debug, rank_zero_info, rank_zero_only
from matplotlib import pyplot as plt
from torch import nn

from torch.utils.data import Dataset
from torchmetrics import Metric, MeanMetric

import utils
import torch.nn.functional as F
from models.refinegan.generator import RefineGANGenerator
from models.refinegan.mpd import MultiPeriodDiscriminator
from models.refinegan.mrd import MultiResolutionDiscriminator
from models.refinegan.mbmrd import MultiBandMRD
from models.refinegan.msd import MultiScaleDiscriminator as RGMultiScaleDiscriminator
from training.base_task_gan import GanBaseTask
from utils.training_utils import (
    DsBatchSampler, DsEvalBatchSampler,
    get_latest_checkpoint_path
)
from utils.wav2F0 import get_pitch
from utils.wav2mel import PitchAdjustableMelSpectrogram


def spec_to_figure(spec, vmin=None, vmax=None):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    fig = plt.figure(figsize=(12, 9), dpi=100)
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    plt.tight_layout()
    return fig


def dynamic_range_compression_torch(x, C=1, clip_val=1e-9):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def wav_aug(wav, hop_size, speed=1):
    orig_freq = int(np.round(hop_size * speed))
    new_freq = hop_size
    return torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)(wav)


class RefineGAN_dataset(Dataset):
    def __init__(self, config: dict, data_dir, infer=False):
        super().__init__()
        self.config = config

        self.data_dir = data_dir if isinstance(data_dir, pathlib.Path) else pathlib.Path(data_dir)
        with open(self.data_dir, 'r', encoding='utf8') as f:
            fills = f.read().strip().split('\n')
        self.data_index = fills
        self.infer = infer
        self.volume_aug = self.config['volume_aug']
        self.volume_aug_prob = self.config['volume_aug_prob'] if not infer else 0
        self.key_aug = self.config.get('key_aug', False)
        self.key_aug_prob = self.config.get('key_aug_prob', 0.5)
        if self.key_aug:
            self.mel_spec_transform = PitchAdjustableMelSpectrogram(
                sample_rate=config['audio_sample_rate'],
                n_fft=config['fft_size'],
                win_length=config['win_size'],
                hop_length=config['hop_size'],
                f_min=config['fmin'],
                f_max=config['fmax'],
                n_mels=config['audio_num_mel_bins'],
            )

    def __getitem__(self, index):
        data_path = self.data_index[index]
        data = np.load(data_path)
        if self.infer:
            return {'f0': data['f0'], 'spectrogram': data['mel'], 'audio': data['audio']}

        if not self.key_aug:
            return {'f0': data['f0'], 'spectrogram': data['mel'], 'audio': data['audio']}
        else:
            if random.random() < self.key_aug_prob:
                audio = torch.from_numpy(data['audio'])
                speed = random.uniform(self.config['aug_min'], self.config['aug_max'])
                crop_mel_frames = int(np.ceil((self.config['crop_mel_frames'] + 4) * speed))
                samples_per_frame = self.config['hop_size']
                crop_wav_samples = crop_mel_frames * samples_per_frame
                if crop_wav_samples >= audio.shape[0]:
                    return {'f0': data['f0'], 'spectrogram': data['mel'], 'audio': data['audio']}
                start = random.randint(0, audio.shape[0] - 1 - crop_wav_samples)
                end = start + crop_wav_samples
                audio = audio[start:end]
                audio_aug = wav_aug(audio, self.config["hop_size"], speed=speed)
                mel_aug = dynamic_range_compression_torch(self.mel_spec_transform(audio_aug[None, :]))
                f0, uv = get_pitch(
                    audio.numpy(),
                    hparams=self.config,
                    speed=speed,
                    interp_uv=True,
                    length=mel_aug.shape[-1],
                )
                if f0 is None:
                    return {'f0': data['f0'], 'spectrogram': data['mel'], 'audio': data['audio']}
                audio_aug = audio_aug[2 * samples_per_frame: -2 * samples_per_frame].numpy()
                mel_aug = mel_aug[0, :, 2:-2].T.numpy()
                f0_aug = f0[2:-2] * speed
                return {'f0': f0_aug, 'spectrogram': mel_aug, 'audio': audio_aug}
            else:
                return {'f0': data['f0'], 'spectrogram': data['mel'], 'audio': data['audio']}

    def __len__(self):
        return len(self.data_index)

    def collater(self, minibatch):
        samples_per_frame = self.config['hop_size']
        if self.infer:
            crop_mel_frames = 0
        else:
            crop_mel_frames = self.config['crop_mel_frames']

        for record in minibatch:
            if record['spectrogram'].shape[0] < crop_mel_frames:
                del record['spectrogram']
                del record['audio']
                del record['f0']
                continue
            elif record['spectrogram'].shape[0] == crop_mel_frames:
                start = 0
            else:
                start = random.randint(0, record['spectrogram'].shape[0] - 1 - crop_mel_frames)
            end = start + crop_mel_frames
            if self.infer:
                record['spectrogram'] = record['spectrogram'].T
                record['f0'] = record['f0']
            else:
                record['spectrogram'] = record['spectrogram'][start:end].T
                record['f0'] = record['f0'][start:end]
            start *= samples_per_frame
            end *= samples_per_frame
            if self.infer:
                cty = (len(record['spectrogram'].T) * samples_per_frame)
                record['audio'] = record['audio'][:cty]
                record['audio'] = np.pad(
                    record['audio'],
                    (0, (len(record['spectrogram'].T) * samples_per_frame) - len(record['audio'])),
                    mode='constant',
                )
            else:
                record['audio'] = record['audio'][start:end]
                record['audio'] = np.pad(
                    record['audio'], (0, (end - start) - len(record['audio'])), mode='constant'
                )

        if self.volume_aug:
            for record in minibatch:
                if record.get('audio') is None:
                    continue
                audio = record['audio']
                audio_mel = record['spectrogram']
                if random.random() < self.volume_aug_prob:
                    max_amp = float(np.max(np.abs(audio))) + 1e-5
                    max_shift = min(3, np.log(1 / max_amp))
                    log_mel_shift = random.uniform(-3, max_shift)
                    audio *= np.exp(log_mel_shift)
                    audio_mel += log_mel_shift
                audio_mel = torch.clamp(torch.from_numpy(audio_mel), min=np.log(1e-5)).numpy()
                record['audio'] = audio
                record['spectrogram'] = audio_mel

        audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
        spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
        f0 = np.stack([record['f0'] for record in minibatch if 'f0' in record])

        return {
            'audio': torch.from_numpy(audio).unsqueeze(1),
            'mel': torch.from_numpy(spectrogram),
            'f0': torch.from_numpy(f0),
        }


class stftlog:
    def __init__(self, n_fft=2048, win_length=2048, hop_length=512, center=True):
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.window = torch.hann_window(self.win_length)

    @torch.no_grad()
    def exc(self, y: torch.Tensor) -> torch.Tensor:
        # y: (T,) or (B, T)
        if y.dim() == 1:
            y = y[None, :]
        spec = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(y.device),
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs()
        return spec


class RefineGAN(GanBaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.TF = PitchAdjustableMelSpectrogram(
            f_min=0,
            f_max=None,
            n_mels=256,
        )
        self.logged_gt_wav = set()
        self.stft = stftlog()

    def build_dataset(self):
        self.train_dataset = RefineGAN_dataset(
            config=self.config,
            data_dir=pathlib.Path(self.config['DataIndexPath']) / self.config['train_set_name'],
        )
        self.valid_dataset = RefineGAN_dataset(
            config=self.config,
            data_dir=pathlib.Path(self.config['DataIndexPath']) / self.config['valid_set_name'],
            infer=True,
        )

    def build_model(self):
        cfg = self.config['model_args']
        gen_kwargs = {
            'sampling_rate': self.config['audio_sample_rate'],
            'num_mels': self.config['audio_num_mel_bins'],
            'hop_length': self.config['hop_size'],
            'downsample_rates': tuple(cfg.get('downsample_rates', [2, 2, 8, 8])),
            'upsample_rates': tuple(cfg.get('upsample_rates', [8, 8, 2, 2])),
            'leaky_relu_slope': float(cfg.get('leaky_relu_slope', 0.2)),
            'start_channels': int(cfg.get('start_channels', 16)),
            'template_generator': cfg.get('template_generator', 'comb'),
        }
        self.generator = RefineGANGenerator(**gen_kwargs)
        # Discriminators
        self.mpd = MultiPeriodDiscriminator(periods=cfg.get('mpd_periods', [2, 3, 5, 7, 11]))

        # Original MRD toggle (default True for backward compatibility)
        use_mrd = cfg.get('use_mrd', True)
        # New MBMRD toggle (default False)
        use_mbmrd = cfg.get('use_mbmrd', False)
        # Optional MSD toggle (default False)
        use_msd = cfg.get('use_msd', False)

        disc_dict = {'mpd': self.mpd}
        if use_mrd:
            self.mrd = MultiResolutionDiscriminator(resolutions=cfg.get('mrd_resolutions', [
                [1024, 120, 600], [2048, 240, 1200], [512, 50, 240]
            ]))
            disc_dict['mrd'] = self.mrd
        if use_mbmrd:
            fft_sizes = cfg.get('mbmrd_fft_sizes', [2048, 1024, 512])
            hop_factor = float(cfg.get('mbmrd_hop_factor', 0.25))
            self.mbmrd = MultiBandMRD(
                fft_sizes=fft_sizes,
                hop_factor=hop_factor,
                sample_rate=self.config.get('audio_sample_rate', 44100),
                center=False,
                normalized=True,
            )
            disc_dict['mbmrd'] = self.mbmrd
        if use_msd:
            self.msd = RGMultiScaleDiscriminator()
            disc_dict['msd'] = self.msd

        # For optimizer wiring in GanBaseTask.configure_optimizers
        self.discriminator = nn.ModuleDict(disc_dict)

        # Mel transforms
        self.mel_transform = PitchAdjustableMelSpectrogram(
            sample_rate=self.config['audio_sample_rate'],
            n_fft=self.config['fft_size'],
            win_length=self.config['win_size'],
            hop_length=self.config['hop_size'],
            f_min=self.config['fmin'],
            f_max=self.config['fmax'],
            n_mels=self.config['audio_num_mel_bins'],
        )
        # Multi-scale mel transforms for generator mel loss
        ms_list = cfg.get('multi_scale_mels', [
            [self.config['fft_size'], self.config['hop_size'], self.config['win_size']],
            [2048, 270, 1080],
            [4096, 540, 2160],
        ])
        self.multi_scale_mels = [
            PitchAdjustableMelSpectrogram(
                sample_rate=self.config['audio_sample_rate'],
                n_fft=int(nf), win_length=int(wl), hop_length=int(hp),
                f_min=0, f_max=self.config['audio_sample_rate'] // 2,
                n_mels=self.config['audio_num_mel_bins'],
            ) for (nf, hp, wl) in ms_list
        ]

    def build_losses_and_metrics(self):
        # STFT loss toggle and weights (consistent with nsf-hifigan configs)
        self.use_stftloss = self.config.get('use_stftloss', False)
        self.lab_aux_melloss = float(self.config.get('lab_aux_melloss', self.config.get('lab_aux_loss', 45)))
        self.lab_aux_stftloss = float(self.config.get('lab_aux_stftloss', 2.5))
        if self.use_stftloss:
            from modules.loss.stft_loss import warp_stft
            self.stft_loss = warp_stft({
                'fft_sizes': self.config.get('loss_fft_sizes', [1024, 2048, 512]),
                'hop_sizes': self.config.get('loss_hop_sizes', [120, 240, 50]),
                'win_lengths': self.config.get('loss_win_lengths', [600, 1200, 240]),
            })

    def get_mels(self, audios, transform=None):
        if transform is None:
            transform = self.mel_transform
        # PitchAdjustableMelSpectrogram uses the input tensor's device internally
        x = transform(audios.squeeze(1))
        return dynamic_range_compression_torch(x)

    def generator_adv_loss(self, disc_outputs):
        # Logistic non-saturating loss with softplus: E[softplus(-D(G(x)))]
        losses = []
        for dg in disc_outputs:
            losses.append(F.softplus(-dg).mean())
        return sum(losses) / len(losses)

    def discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        # Logistic loss: E[softplus(-D(real))] + E[softplus(D(fake))]
        losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = F.softplus(-dr).mean()
            g_loss = F.softplus(dg).mean()
            losses.append((r_loss + g_loss) / 2)
        return sum(losses) / len(losses)

    def generator_mel_loss(self, y, y_hat):
        loss_mel = []
        for mel_transform in self.multi_scale_mels:
            y_mel = self.get_mels(y, mel_transform)
            y_g_hat_mel = self.get_mels(y_hat, mel_transform)
            loss_mel.append(torch.nn.functional.smooth_l1_loss(y_mel, y_g_hat_mel))
        return sum(loss_mel) / len(loss_mel)

    def generator_envelope_loss(self, y, y_hat):
        def extract_envelope(signal, kernel_size=100, stride=50):
            return torch.nn.functional.max_pool1d(signal, kernel_size=kernel_size, stride=stride)
        y_env = extract_envelope(y)
        y_hat_env = extract_envelope(y_hat)
        y_rev_env = extract_envelope(-y)
        y_hat_rev_env = extract_envelope(-y_hat)
        return torch.nn.functional.l1_loss(y_env, y_hat_env) + torch.nn.functional.l1_loss(y_rev_env, y_hat_rev_env)

    def _training_step(self, sample, batch_idx):
        opt_g, opt_d = self.optimizers()

        # Ground-truth audio and mel
        y: torch.Tensor = sample['audio'].float()
        # Recompute mel from audio for stability
        gt_mels = self.get_mels(y)
        # f0 expects [B, 1, T]
        pitches = sample['f0'].unsqueeze(1).float()

        # Forward generator
        y_g_hat = self.generator(gt_mels, pitches)

        # Discriminator step
        opt_d.zero_grad()
        # MPD
        y_g_hat_x, _ = self.mpd(y_g_hat.detach())
        y_x, _ = self.mpd(y)
        loss_mpd = self.discriminator_loss(y_x, y_g_hat_x)
        loss_d = loss_mpd
        log_items = {
            'train_loss_d_mpd': loss_mpd.detach(),
        }
        # MRD (if enabled)
        if hasattr(self, 'mrd'):
            y_g_hat_x, _ = self.mrd(y_g_hat.detach())
            y_x, _ = self.mrd(y)
            loss_mrd = self.discriminator_loss(y_x, y_g_hat_x)
            loss_d = loss_d + loss_mrd
            log_items['train_loss_d_mrd'] = loss_mrd.detach()
        # MBMRD (if enabled)
        if hasattr(self, 'mbmrd'):
            y_g_hat_x, _ = self.mbmrd(y_g_hat.detach())
            y_x, _ = self.mbmrd(y)
            loss_mbmrd = self.discriminator_loss(y_x, y_g_hat_x)
            loss_d = loss_d + loss_mbmrd
            log_items['train_loss_d_mbmrd'] = loss_mbmrd.detach()
        # MSD (if enabled)
        if hasattr(self, 'msd'):
            y_g_hat_x, _ = self.msd(y_g_hat.detach())
            y_x, _ = self.msd(y)
            loss_msd = self.discriminator_loss(y_x, y_g_hat_x)
            loss_d = loss_d + loss_msd
            log_items['train_loss_d_msd'] = loss_msd.detach()
        self.manual_backward(loss_d)
        opt_d.step()

        # Generator step
        opt_g.zero_grad()
        # Align lengths if slight mismatch
        corrected_length = min(y.shape[-1], y_g_hat.shape[-1])
        y = y[..., :corrected_length]
        y_g_hat = y_g_hat[..., :corrected_length]

        loss_mel = self.generator_mel_loss(y, y_g_hat)
        loss_env = self.generator_envelope_loss(y, y_g_hat)
        # Adv losses
        y_g_hat_x, _ = self.mpd(y_g_hat)
        g_loss_mpd = self.generator_adv_loss(y_g_hat_x)
        g_loss_total_disc = g_loss_mpd
        # MRD adv (if enabled)
        g_loss_mrd = None
        if hasattr(self, 'mrd'):
            y_g_hat_x, _ = self.mrd(y_g_hat)
            g_loss_mrd = self.generator_adv_loss(y_g_hat_x)
            g_loss_total_disc = g_loss_total_disc + g_loss_mrd
        # MBMRD adv (if enabled)
        g_loss_mbmrd = None
        if hasattr(self, 'mbmrd'):
            y_g_hat_x, _ = self.mbmrd(y_g_hat)
            g_loss_mbmrd = self.generator_adv_loss(y_g_hat_x)
            g_loss_total_disc = g_loss_total_disc + g_loss_mbmrd
        # MSD adv (if enabled)
        g_loss_msd = None
        if hasattr(self, 'msd'):
            y_g_hat_x, _ = self.msd(y_g_hat)
            g_loss_msd = self.generator_adv_loss(y_g_hat_x)
            g_loss_total_disc = g_loss_total_disc + g_loss_msd

        # Optional STFT loss (sc + mag)
        stft_term = 0.0
        stft_sc = None
        stft_mag = None
        if getattr(self, 'use_stftloss', False):
            sc_loss, mag_loss = self.stft_loss.loss(y_g_hat.squeeze(1), y.squeeze(1))
            stft_sc, stft_mag = sc_loss, mag_loss
            stft_term = (stft_sc + stft_mag) * self.lab_aux_stftloss

        loss_g = self.lab_aux_melloss * loss_mel + loss_env + g_loss_total_disc + (stft_term if isinstance(stft_term, torch.Tensor) else 0.0)
        self.manual_backward(loss_g)
        opt_g.step()

        # Build logs
        log_out = {
            'train_loss_d': loss_d.detach(),
            'train_loss_g': loss_g.detach(),
            'train_loss_g_mel': loss_mel.detach(),
            'train_loss_g_env': loss_env.detach(),
            'train_loss_g_mpd': g_loss_mpd.detach(),
            **({'train_loss_g_mrd': g_loss_mrd.detach()} if g_loss_mrd is not None else {}),
            **({'train_loss_g_mbmrd': g_loss_mbmrd.detach()} if g_loss_mbmrd is not None else {}),
            **({'train_loss_g_msd': g_loss_msd.detach()} if g_loss_msd is not None else {}),
            **({'train_loss_g_stft': stft_term.detach()} if isinstance(stft_term, torch.Tensor) else {}),
            **({'train_stft_sc': stft_sc.detach(), 'train_stft_mag': stft_mag.detach()} if (stft_sc is not None and stft_mag is not None) else {}),
        }
        log_out.update(log_items)
        return log_out

    def _validation_step(self, sample, batch_idx):
        y: torch.Tensor = sample['audio'].float()
        gt_mels = self.get_mels(y)
        pitches = sample['f0'].unsqueeze(1).float()
        y_g_hat = self.generator(gt_mels, pitches)

        # Compute mel L1 on default mel
        y_g_hat_mel = self.get_mels(y_g_hat)[:, :, : gt_mels.shape[2]]
        mel_l1 = torch.nn.functional.l1_loss(gt_mels, y_g_hat_mel)

        # Optional STFT metrics
        val_logs = {'mel_l1': mel_l1}
        if getattr(self, 'use_stftloss', False):
            with torch.no_grad():
                sc_loss, mag_loss = self.stft_loss.loss(y_g_hat.squeeze(1), y.squeeze(1))
            val_logs.update({'val_stft_sc': sc_loss.detach(), 'val_stft_mag': mag_loss.detach()})

        # Log audio and mel images (similar to nsf-hifigan)
        with torch.no_grad():
            if getattr(self.logger, 'experiment', None) is not None:
                # Audio
                self.logger.experiment.add_audio(
                    f'RefineGAN_{batch_idx}_', y_g_hat,
                    sample_rate=self.config['audio_sample_rate'],
                    global_step=self.global_step,
                )
                if batch_idx not in self.logged_gt_wav:
                    self.logger.experiment.add_audio(
                        f'gt_{batch_idx}_', y,
                        sample_rate=self.config['audio_sample_rate'],
                        global_step=self.global_step,
                    )
                    self.logged_gt_wav.add(batch_idx)

                # Mel figures
                try:
                    vmin = self.config['mel_vmin']
                    vmax = self.config['mel_vmax']
                    # shape (B, F, T) -> (B, T, F)
                    gt_mel_img = gt_mels.transpose(1, 2)
                    pr_mel_img = y_g_hat_mel.transpose(1, 2)
                    # Reuse plot_mel util for side-by-side diff | gt | pred
                    self.plot_mel(
                        batch_idx,
                        gt_mel_img,
                        pr_mel_img,
                        name=f'RefineGAN_mel_{batch_idx}'
                    )
                except Exception:
                    pass

        return val_logs, 1

    def plot_mel(self, batch_idx, spec, spec_out, name=None):
        name = f'mel_{batch_idx}' if name is None else name
        vmin = self.config['mel_vmin']
        vmax = self.config['mel_vmax']
        spec_cat = torch.cat([(spec_out - spec).abs() + vmin, spec, spec_out], -1)
        self.logger.experiment.add_figure(name, spec_to_figure(spec_cat[0], vmin, vmax), self.global_step)
