import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import List, Tuple, Optional, Sequence


def wn_conv2d(*args, act: bool = True, **kwargs):
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    # SELU like reference, using LeakyReLU is also reasonable
    return nn.Sequential(conv, nn.SELU(0.1))


class MRD(nn.Module):
    """
    Multi-band spectrogram discriminator operating on complex STFT
    (real/imag as 2 input channels) for a single FFT size.

    This is a self-contained implementation inspired by the user's
    reference. It does not depend on external packages.
    """

    def __init__(
        self,
        window_length: int,
        hop_factor: float = 0.25,
        sample_rate: int = 44100,
        bands: Optional[List[Tuple[float, float]]] = None,
        channels: int = 1,
        center: bool = False,
        normalized: bool = True,
    ):
        super().__init__()

        self.window_length = int(window_length)
        self.hop_length = max(1, int(round(self.window_length * float(hop_factor))))
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.center = bool(center)
        self.normalized = bool(normalized)

        # Compute band index ranges once based on expected STFT freq bins
        n_freqs = self.window_length // 2 + 1
        if bands is None:
            # 10 uniform fractional bands across [0, 1]
            bands = [
                (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),
                (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8),
                (0.8, 0.9), (0.9, 1.0),
            ]
        # Convert to frequency bin indices
        self.bands = [(int(b0 * n_freqs), int(b1 * n_freqs)) for (b0, b1) in bands]

        ch = 32
        def make_stack():
            return nn.ModuleList([
                wn_conv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
                wn_conv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                wn_conv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                wn_conv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                wn_conv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ])

        # One conv stack per band, then concatenate across frequency axis
        self.band_convs = nn.ModuleList([make_stack() for _ in range(len(self.bands))])
        self.conv_post = wn_conv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

        # Pre-create window buffer on demand
        self.register_buffer("_hann_window", torch.hann_window(self.window_length), persistent=False)

    def _spectrogram_complex(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute complex STFT and return tensor shaped (B, 2, T, F),
        where channel 0 is real, 1 is imaginary.
        """
        # Expect input x as (B, 1, T) or (B, T)
        if x.dim() == 3:
            x = x.squeeze(1)
        # STFT: (B, F, T)
        spec = torch.stft(
            x,
            n_fft=self.window_length,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self._hann_window.to(x.device),
            center=self.center,
            pad_mode="reflect",
            normalized=self.normalized,
            onesided=True,
            return_complex=True,
        )
        # (B, F, T, 2) -> (B, 2, T, F)
        spec_ri = torch.view_as_real(spec).permute(0, 3, 2, 1).contiguous()
        return spec_ri

    def forward(self, x: torch.Tensor):
        # Complex spectrogram channels: (B, 2, T, F)
        spec_ri = self._spectrogram_complex(x)

        fmap: list[torch.Tensor] = []
        band_outs: list[torch.Tensor] = []

        # Process each frequency band with its conv stack
        for (f0, f1), stack in zip(self.bands, self.band_convs):
            band = spec_ri[..., f0:f1]
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            band_outs.append(band)

        # Concatenate across frequency axis, finish with 1x1 conv_post
        x_cat = torch.cat(band_outs, dim=-1)
        x_cat = self.conv_post(x_cat)
        fmap.append(x_cat)
        logits = torch.flatten(x_cat, 1, -1)

        return logits, fmap


class MultiBandMRD(nn.Module):
    """
    Container that builds a bank of MRD discriminators for multiple FFT/window sizes.
    API matches other discriminators in this repo: returns list of logits and list of fmap lists.
    """

    def __init__(
        self,
        *,
        fft_sizes: Sequence[int],
        hop_factor: float = 0.25,
        sample_rate: int = 44100,
        bands: Optional[List[Tuple[float, float]]] = None,
        center: bool = False,
        normalized: bool = True,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList([
            MRD(
                window_length=int(fs),
                hop_factor=hop_factor,
                sample_rate=sample_rate,
                bands=bands,
                center=center,
                normalized=normalized,
            )
            for fs in fft_sizes
        ])

    def forward(self, x: torch.Tensor):
        y_d_rs = []
        fmap_rs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(x)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
        return y_d_rs, fmap_rs
