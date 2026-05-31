import torch
import torch.nn as nn
import torch.nn.functional as F

# import weight_norm from different version of pytorch
try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm

from .conformer_naive import ConformerNaiveEncoder

# `CFNaiveMelPE` is ported verbatim from FCPE (https://github.com/CNChTu/FCPE) torchfcpe/models.py.
# In this integration we only use its `forward` (produces a 360-dim sigmoid latent). Its internal
# `cent_table` / `train_and_loss` / `infer` are kept for fidelity but NOT used by FCPETask — the bin
# semantics are defined by the existing RMVPE one-hot labels and `to_local_average_cents` decoder.


class CFNaiveMelPE(nn.Module):
    """
    Conformer-based Mel-spectrogram Prediction Encoder in Fast Context-based Pitch Estimation

    Args:
        input_channels (int): Number of input channels, should be same as the number of bins of mel-spectrogram.
        out_dims (int): Number of output dimensions, also class numbers.
        hidden_dims (int): Number of hidden dimensions.
        n_layers (int): Number of conformer layers.
        f0_max (float): Maximum frequency of f0.
        f0_min (float): Minimum frequency of f0.
        use_fa_norm (bool): Whether to use fast attention norm, default False
        conv_only (bool): Whether to use only conv module without attention, default False
        conv_dropout (float): Dropout rate of conv module, default 0.
        atten_dropout (float): Dropout rate of attention module, default 0.
        use_harmonic_emb (bool): Whether to use harmonic embedding, default False
    """

    def __init__(self,
                 input_channels: int,
                 out_dims: int,
                 hidden_dims: int = 512,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 f0_max: float = 1975.5,
                 f0_min: float = 32.70,
                 use_fa_norm: bool = False,
                 conv_only: bool = False,
                 conv_dropout: float = 0.,
                 atten_dropout: float = 0.,
                 use_harmonic_emb: bool = False,
                 ):
        super().__init__()
        self.input_channels = input_channels
        self.out_dims = out_dims
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.f0_max = f0_max
        self.f0_min = f0_min
        self.use_fa_norm = use_fa_norm
        self.residual_dropout = 0.1  # 废弃代码,仅做兼容性保留
        self.attention_dropout = 0.1  # 废弃代码,仅做兼容性保留

        # Harmonic embedding
        if use_harmonic_emb:
            self.harmonic_emb = nn.Embedding(9, hidden_dims)
        else:
            self.harmonic_emb = None

        # Input stack, convert mel-spectrogram to hidden_dims
        self.input_stack = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dims, 3, 1, 1),
            nn.GroupNorm(4, hidden_dims),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_dims, hidden_dims, 3, 1, 1)
        )
        # Conformer Encoder
        self.net = ConformerNaiveEncoder(
            num_layers=n_layers,
            num_heads=n_heads,
            dim_model=hidden_dims,
            use_norm=use_fa_norm,
            conv_only=conv_only,
            conv_dropout=conv_dropout,
            atten_dropout=atten_dropout,
        )
        # LayerNorm
        self.norm = nn.LayerNorm(hidden_dims)
        # Output stack, convert hidden_dims to out_dims
        self.output_proj = weight_norm(
            nn.Linear(hidden_dims, out_dims)
        )
        # Cent table buffer
        self.cent_table_b = torch.linspace(self.f0_to_cent(torch.Tensor([f0_min]))[0],
                                           self.f0_to_cent(torch.Tensor([f0_max]))[0],
                                           out_dims).detach()
        self.register_buffer("cent_table", self.cent_table_b)
        # gaussian_blurred_cent_mask_b buffer
        self.gaussian_blurred_cent_mask_b = (1200. * torch.log2(torch.Tensor([self.f0_max / 10.])))[0].detach()
        self.register_buffer("gaussian_blurred_cent_mask", self.gaussian_blurred_cent_mask_b)

    def forward(self, x: torch.Tensor, _h_emb=None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input mel-spectrogram, shape (B, T, input_channels) or (B, T, mel_bins).
            _h_emb (int): Harmonic embedding index, like 0, 1, 2, only use in train. Default: None.
        return:
            torch.Tensor: Predicted f0 latent, shape (B, T, out_dims).
        """
        x = self.input_stack(x.transpose(-1, -2)).transpose(-1, -2)
        if self.harmonic_emb is not None:
            if _h_emb is None:
                x = x + self.harmonic_emb(torch.LongTensor([0]).to(x.device))
            else:
                x = x + self.harmonic_emb(torch.LongTensor([int(_h_emb)]).to(x.device))
        x = self.net(x)
        x = self.norm(x)
        x = self.output_proj(x)
        x = torch.sigmoid(x)
        return x  # latent (B, T, out_dims)

    @torch.no_grad()
    def latent2cents_decoder(self, y, threshold=0.05, mask=True):
        """Convert latent to cents. y: (B, T, out_dims) -> (B, T, 1)."""
        B, N, _ = y.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        rtn = torch.sum(ci * y, dim=-1, keepdim=True) / torch.sum(y, dim=-1, keepdim=True)
        if mask:
            confident = torch.max(y, dim=-1, keepdim=True)[0]
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= threshold] = float("-INF")
            rtn = rtn * confident_mask
        return rtn  # (B, T, 1)

    @torch.no_grad()
    def latent2cents_local_decoder(self, y, threshold=0.05, mask=True):
        """Convert latent to cents using local argmax. y: (B, T, out_dims) -> (B, T, 1)."""
        B, N, _ = y.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        confident, max_index = torch.max(y, dim=-1, keepdim=True)
        local_argmax_index = torch.arange(0, 9).to(max_index.device) + (max_index - 4)
        local_argmax_index[local_argmax_index < 0] = 0
        local_argmax_index[local_argmax_index >= self.out_dims] = self.out_dims - 1
        ci_l = torch.gather(ci, -1, local_argmax_index)
        y_l = torch.gather(y, -1, local_argmax_index)
        rtn = torch.sum(ci_l * y_l, dim=-1, keepdim=True) / torch.sum(y_l, dim=-1, keepdim=True)
        if mask:
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= threshold] = float("-INF")
            rtn = rtn * confident_mask
        return rtn  # (B, T, 1)

    @torch.no_grad()
    def gaussian_blurred_cent2latent(self, cents):  # cents: [B,N,1]
        """Convert cents to a gaussian-blurred latent target. (B, T, 1) -> (B, T, out_dims)."""
        mask = (cents > 0.1) & (cents < self.gaussian_blurred_cent_mask)
        B, N, _ = cents.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        return torch.exp(-torch.square(ci - cents) / 1250) * mask.float()

    @torch.no_grad()
    def infer(self, mel, decoder="local_argmax", threshold=0.05):
        latent = self.forward(mel)
        if decoder == "argmax":
            cents = self.latent2cents_decoder(latent, threshold=threshold)
        elif decoder == "local_argmax":
            cents = self.latent2cents_local_decoder(latent, threshold=threshold)
        else:
            raise ValueError(f"  [x] Unknown decoder type {decoder}.")
        f0 = self.cent_to_f0(cents)
        return f0  # (B, T, 1)

    def train_and_loss(self, mel, gt_f0, loss_scale=10):
        """Native FCPE gaussian-target BCE loss (kept for the optional second-step A/B; not used by default)."""
        if mel.shape[-2] != gt_f0.shape[-2]:
            _len = min(mel.shape[-2], gt_f0.shape[-2])
            mel = mel[:, :_len, :]
            gt_f0 = gt_f0[:, :_len, :]
        gt_cent_f0 = self.f0_to_cent(gt_f0)
        x_gt = self.gaussian_blurred_cent2latent(gt_cent_f0)
        if self.harmonic_emb is not None:
            x = self.forward(mel, _h_emb=0)
            x_half = self.forward(mel, _h_emb=1)
            x_gt_half = self.gaussian_blurred_cent2latent(gt_cent_f0 / 2)
            x_gt_double = self.gaussian_blurred_cent2latent(gt_cent_f0 * 2)
            x_double = self.forward(mel, _h_emb=2)
            loss = F.binary_cross_entropy(x, x_gt)
            loss_half = F.binary_cross_entropy(x_half, x_gt_half)
            loss_double = F.binary_cross_entropy(x_double, x_gt_double)
            loss = loss + (loss_half + loss_double) / 2
            loss = loss * loss_scale
        else:
            x = self.forward(mel)
            loss = F.binary_cross_entropy(x, x_gt) * loss_scale
        return loss

    @torch.no_grad()
    def cent_to_f0(self, cent):
        return 10. * 2 ** (cent / 1200.)

    @torch.no_grad()
    def f0_to_cent(self, f0):
        return 1200. * torch.log2(f0 / 10.)


class FCPE_E2E(nn.Module):
    """
    Thin adapter so CFNaiveMelPE plugs into the existing SingingVocoders RMVPE training task.

    - Reads its dims/hparams from the same config dict as RMVPE.
    - Accepts mel in the repo's `(B, n_mels, T)` layout and feeds CFNaiveMelPE `(B, T, n_mels)`.
    - Returns `(hidden_vec, pitch_pred)` to match the `hidden, pred = self.generator(mel)` call.

    NOTE: `audio_num_mel_bins` / `hop_size` / mel params are taken from the (unchanged) repo config —
    FCPE's own mel front-end is intentionally NOT used.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.model = CFNaiveMelPE(
            input_channels=config['audio_num_mel_bins'],
            out_dims=config['n_class'],
            hidden_dims=config.get('hidden_dims', 512),
            n_layers=config.get('n_layers', 6),
            n_heads=config.get('n_heads', 8),
            # The model's internal cent_table is unused (we decode with RMVPE's mapping); keep FCPE defaults.
            f0_max=config.get('model_f0_max', 1975.5),
            f0_min=config.get('model_f0_min', 32.70),
            use_fa_norm=config.get('use_fa_norm', True),
            conv_only=config.get('conv_only', True),
            conv_dropout=config.get('conv_dropout', 0.0),
            atten_dropout=config.get('atten_dropout', 0.0),
            use_harmonic_emb=config.get('use_harmonic_emb', False),
        )

    def forward(self, mel):
        # mel: (B, n_mels, T) -> (B, T, n_mels)
        x = mel.transpose(1, 2)
        pred = self.model(x)  # (B, T, out_dims)
        hidden_vec = 0
        return hidden_vec, pred
