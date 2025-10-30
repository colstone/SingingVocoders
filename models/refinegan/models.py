import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv1d, ReflectionPad1d
from torch.nn.utils import weight_norm

from models.nsf_HiFigan.models import (
    LRELU_SLOPE,
    Generator as _NSFHiFiGANGenerator,
    init_weights,
)


class Generator(_NSFHiFiGANGenerator):
    """
    RefineGAN-style generator adapted from fish-diffusion's iSTFTNet variant.

    Differences vs NSF-HiFiGAN:
    - Final conv projects to (post_n_fft + 2) channels: [mag_bins + phase_bins].
    - Predicts log-magnitude and phase, then reconstructs waveform via iSTFT.
    - Returns waveform tensor (B, 1, T) for drop-in use in existing tasks.

    Required in h (AttrDict):
    - sampling_rate, hop_size, num_mels (existing NSF fields)
    - gen_istft_n_fft (e.g., 2048)
    - gen_istft_hop_size (usually equals hop_size)
    """

    def __init__(self, h):
        super().__init__(h)

        # iSTFT params
        self.post_n_fft: int = int(getattr(self.h, "gen_istft_n_fft", 2048))
        self.post_hop: int = int(getattr(self.h, "gen_istft_hop_size", self.h.hop_size))

        # Post conv outputs: n_fft + 2 channels (magnitude + phase)
        # Determine input channels to final projection from base Generator
        ch = int(self.conv_post.in_channels)
        # Replace final projection to output (mag+phase)
        self.conv_post = weight_norm(Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        self.conv_post.apply(init_weights)

        # Pad so that frame alignment matches iSTFT framing
        self.reflection_pad = ReflectionPad1d((1, 0))

        # Window used by iSTFT
        window = torch.hann_window(self.post_n_fft)
        self.register_buffer("hanning_window", window)

    def _istft(self, spec: Tensor, phase: Tensor) -> Tensor:
        # spec: (B, F, T_frames), phase: (B, F, T_frames)
        # Build complex STFT: mag * exp(j*phase)
        complex_stft = spec * torch.exp(phase * 1j)
        y = torch.istft(
            complex_stft,
            n_fft=self.post_n_fft,
            hop_length=self.post_hop,
            win_length=self.post_n_fft,
            window=self.hanning_window,
            center=True,
            normalized=False,
            onesided=True,
            return_complex=False,
        )
        # (B, T)
        return y[:, None, :]

    def forward(self, x: Tensor, f0: Tensor) -> Tensor:
        # Harmonic source from hn-NSF module (aligned by overall upsample factor)
        har_source = self.m_source(f0, self.upp).transpose(1, 2)

        # Conditioning projection
        x = self.conv_pre(x)

        # Upsampling + ResBlocks
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source

            xs = None
            for j in range(self.num_kernels):
                block = self.resblocks[i * self.num_kernels + j]
                xs = block(x) if xs is None else xs + block(x)
            x = xs / self.num_kernels

        # Downsample time axis from samples to STFT frames
        # Typically, self.upp == self.post_hop (e.g., 512), so this reduces
        # mel_length * upp -> mel_length frames
        x = F.avg_pool1d(x, kernel_size=self.post_hop, stride=self.post_hop, ceil_mode=True)

        # Project to spectrogram+phase, then ISTFT to waveform
        x = self.reflection_pad(x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)

        mag_bins = self.post_n_fft // 2 + 1
        spec = torch.exp(x[:, :mag_bins, :])
        phase = torch.sin(x[:, mag_bins:, :])

        y = self._istft(spec, phase)
        return y
