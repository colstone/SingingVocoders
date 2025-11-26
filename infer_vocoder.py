import argparse
import importlib
import pathlib
import re
from typing import Tuple

import torch
import torchaudio

from utils.config_utils import read_full_config
from utils.wav2mel import PitchAdjustableMelSpectrogram
from utils.wav2F0 import get_pitch


def _dynamic_range_compression_torch(x: torch.Tensor, C: float = 1.0, clip_val: float = 1e-5) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * C)


def build_refinegan_generator(config: dict):
    from models.refinegan.generator import RefineGANGenerator

    cfg = config['model_args']
    gen_kwargs = {
        'sampling_rate': config['audio_sample_rate'],
        'num_mels': config['audio_num_mel_bins'],
        'hop_length': config['hop_size'],
        'downsample_rates': tuple(cfg.get('downsample_rates', [2, 2, 8, 8])),
        'upsample_rates': tuple(cfg.get('upsample_rates', [8, 8, 2, 2])),
        'leaky_relu_slope': float(cfg.get('leaky_relu_slope', 0.2)),
        'start_channels': int(cfg.get('start_channels', 16)),
        'template_generator': cfg.get('template_generator', 'comb'),
    }
    G = RefineGANGenerator(**gen_kwargs)
    return G


def build_nsfhifigan_generator(config: dict):
    from models.nsf_HiFigan.models import Generator, AttrDict

    cfg = dict(config['model_args'])
    cfg.update({
        'sampling_rate': config['audio_sample_rate'],
        'num_mels': config['audio_num_mel_bins'],
        'hop_size': config['hop_size'],
    })
    h = AttrDict(cfg)
    G = Generator(h)
    return G


def build_hifigan_chroma_generator(config: dict):
    from models.nsf_HiFigan_chroma.models import Generator, AttrDict

    cfg = dict(config['model_args'])
    cfg.update({
        'sampling_rate': config['audio_sample_rate'],
        'num_mels': config['audio_num_mel_bins'],
        'hop_size': config['hop_size'],
    })
    h = AttrDict(cfg)
    G = Generator(h)
    return G


def build_univnet_generator(config: dict):
    # UnivNet takes full config and a weight_norm flag
    from models.univnet.univnet import UnivNet
    G = UnivNet(config, use_weight_norm=config['model_args'].get('use_weight_norm', True))
    return G


def build_nsf_univnet_generator(config: dict):
    from models.nsf_univnet.nsfunivnet import nsfUnivNet
    G = nsfUnivNet(config, use_weight_norm=config['model_args'].get('use_weight_norm', True))
    return G


def build_ddspgan_generator(config: dict):
    from models.ddspgan.ddspgan import DDSPgan
    G = DDSPgan(config)
    return G


def build_lvc_ddspgan_generator(config: dict):
    from models.lvc_ddspgan.lvc_ddspgan import DDSPgan
    G = DDSPgan(config)
    return G


def build_ddsp_univnet_generator(config: dict):
    from models.ddsp_univnet.ddspunivnet import ddspUnivNet
    G = ddspUnivNet(config, use_weight_norm=config['model_args'].get('use_weight_norm', True))
    return G


def build_hifivae_generator(config: dict):
    from models.HiFivae.models import HiFivae, AttrDict
    cfg = dict(config['model_args'])
    cfg.update({
        'sampling_rate': config['audio_sample_rate'],
        'num_mels': config['audio_num_mel_bins'],
        'hop_size': config['hop_size'],
    })
    h = AttrDict(cfg)
    G = HiFivae(h)
    return G


def build_freev_generator(config: dict):
    from models.freev import FreeVGenerator
    G = FreeVGenerator(
        num_mels=config['audio_num_mel_bins'],
        n_fft=config['fft_size'],
        hop_length=config['hop_size'],
        win_length=config['win_size'],
    )
    return G


def build_freev_nsf_generator(config: dict):
    from models.freev.nsf_generator import FreeVNSFGenerator
    G = FreeVNSFGenerator(
        num_mels=config['audio_num_mel_bins'],
        n_fft=config['fft_size'],
        hop_length=config['hop_size'],
        win_length=config['win_size'],
    )
    return G


def load_generator_state_dict(ckpt_path: pathlib.Path, generator: torch.nn.Module, device: torch.device) -> Tuple[int, str]:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    # filter generator.*
    gen_sd = {k.split('generator.', 1)[1]: v for k, v in state_dict.items() if k.startswith('generator.')}
    if not gen_sd:
        # try direct load
        gen_sd = state_dict
    missing, unexpected = generator.load_state_dict(gen_sd, strict=False)
    if missing:
        print(f"[infer] Missing keys: {len(missing)} (showing up to 8): {missing[:8]}")
    if unexpected:
        print(f"[infer] Unexpected keys: {len(unexpected)} (showing up to 8): {unexpected[:8]}")

    step = int(ckpt.get('global_step', -1))
    if step < 0:
        m = re.search(r"steps_(\d+)", ckpt_path.stem)
        if m:
            step = int(m.group(1))
    return step, ckpt.get('task_cls', '')


def infer_one(config_path: pathlib.Path, ckpt_path: pathlib.Path, wav_path: pathlib.Path, device: torch.device) -> pathlib.Path:
    cfg = read_full_config(config_path)
    task_cls = cfg.get('task_cls', 'training.refinegan_task.RefineGAN')
    task_name = task_cls.split('.')[-1]
    pitch_extractor = cfg.get('pe', cfg.get('pitch_extractor', 'parselmouth'))

    if 'refinegan_task.RefineGAN' in task_cls:
        G = build_refinegan_generator(cfg)
    elif 'nsf_HiFigan_task.nsf_HiFigan' in task_cls:
        G = build_nsfhifigan_generator(cfg)
    elif 'nsf_HiFigan_chroma_task.nsf_HiFigan_chroma' in task_cls:
        G = build_hifigan_chroma_generator(cfg)
    elif 'univnet.univnet_task' in task_cls:
        G = build_univnet_generator(cfg)
    elif 'univnet_nsf.nsf_univnet_task' in task_cls:
        G = build_nsf_univnet_generator(cfg)
    elif 'univnet_nsf_msd.nsf_univnet_task' in task_cls:
        G = build_nsf_univnet_generator(cfg)
    elif 'ddspgan_task.ddspgan_task' in task_cls:
        G = build_ddspgan_generator(cfg)
    elif 'ddspgan_task_2.ddspgan_task' in task_cls:
        G = build_ddspgan_generator(cfg)
    elif 'lvc_ddspgan_task.ddspgan_task' in task_cls:
        G = build_lvc_ddspgan_generator(cfg)
    elif 'univnet_ddsp.ddsp_univnet_task' in task_cls:
        G = build_ddsp_univnet_generator(cfg)
    elif 'HiFivae_task.HiFivae_task' in task_cls:
        G = build_hifivae_generator(cfg)
    elif 'freev_task.FreeVTask' in task_cls:
        G = build_freev_generator(cfg)
    elif 'freev_nsf_task.FreeVNSFTask' in task_cls:
        G = build_freev_nsf_generator(cfg)
    else:
        raise NotImplementedError(f"Unsupported task_cls: {task_cls}")

    G.to(device)
    G.eval()
    step, _ = load_generator_state_dict(ckpt_path, G, device)

    # Load audio
    wav, sr = torchaudio.load(str(wav_path))
    wav = wav.mean(dim=0, keepdim=True)  # mono
    target_sr = int(cfg['audio_sample_rate'])
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    # Build mel transform
    mel_tf = PitchAdjustableMelSpectrogram(
        sample_rate=target_sr,
        n_fft=cfg['fft_size'],
        win_length=cfg['win_size'],
        hop_length=cfg['hop_size'],
        f_min=cfg['fmin'],
        f_max=cfg['fmax'],
        n_mels=cfg['audio_num_mel_bins'],
    )

    with torch.no_grad():
        wav_t = wav.to(device)
        mel = mel_tf(wav_t).to(device)  # (B=1, n_mels, T)
        mel = _dynamic_range_compression_torch(mel)

        # Prepare model-specific inputs
        task = task_cls
        if (
            'refinegan_task.RefineGAN' in task
            or 'nsf_HiFigan_task.nsf_HiFigan' in task
            or 'nsf_HiFigan_chroma_task.nsf_HiFigan_chroma' in task
            or 'univnet_nsf.nsf_univnet_task' in task
            or 'univnet_nsf_msd.nsf_univnet_task' in task
            or 'ddspgan_task.ddspgan_task' in task
            or 'ddspgan_task_2.ddspgan_task' in task
            or 'lvc_ddspgan_task.ddspgan_task' in task
            or 'univnet_ddsp.ddsp_univnet_task' in task
        ):
            # These require f0
            f0, uv = get_pitch(
                pitch_extractor,
                wav.squeeze(0).numpy(),
                length=mel.shape[-1],
                hparams=cfg,
                speed=1,
                interp_uv=True,
            )
            if f0 is None:
                raise RuntimeError("Pitch extractor returned all-UV; please choose a different audio segment.")
            f0_t = torch.from_numpy(f0)[None, None, :].to(device)
        else:
            f0_t = None

        # Run the correct generator signature
        if 'refinegan_task.RefineGAN' in task:
            y_hat = G(mel, f0_t)
        elif 'nsf_HiFigan_task.nsf_HiFigan' in task:
            y_hat = G(mel, f0_t)
        elif 'nsf_HiFigan_chroma_task.nsf_HiFigan_chroma' in task:
            y_hat = G(mel, f0_t)
        elif 'univnet.univnet_task' in task:
            # Build noise input similarly to training
            upmel = cfg['model_args'].get('upmel')
            cond_in_channels = cfg['model_args']['cond_in_channels']
            Tm = mel.shape[-1]
            Tn = Tm * upmel if upmel is not None else Tm
            x = torch.randn(1, cond_in_channels, Tn, device=device)
            y_hat = G(x=x, c=mel)
        elif 'univnet_nsf.nsf_univnet_task' in task:
            upmel = cfg['model_args'].get('upmel')
            cond_in_channels = cfg['model_args']['cond_in_channels']
            Tm = mel.shape[-1]
            Tn = Tm * upmel if upmel is not None else Tm
            x = torch.randn(1, cond_in_channels, Tn, device=device)
            y_hat, _ = G(x=x, c=mel, f0=f0_t)
        elif 'univnet_nsf_msd.nsf_univnet_task' in task:
            upmel = cfg['model_args'].get('upmel')
            cond_in_channels = cfg['model_args']['cond_in_channels']
            Tm = mel.shape[-1]
            Tn = Tm * upmel if upmel is not None else Tm
            x = torch.randn(1, cond_in_channels, Tn, device=device)
            y_hat, _ = G(x=x, c=mel, f0=f0_t)
        elif 'ddspgan_task.ddspgan_task' in task:
            y_hat = G(mel=mel, f0=f0_t)
        elif 'ddspgan_task_2.ddspgan_task' in task:
            y_hat = G(mel=mel, f0=f0_t)
        elif 'lvc_ddspgan_task.ddspgan_task' in task:
            y_hat = G(mel=mel, f0=f0_t)
        elif 'univnet_ddsp.ddsp_univnet_task' in task:
            upmel = cfg['model_args'].get('upmel')
            cond_in_channels = cfg['model_args']['cond_in_channels']
            Tm = mel.shape[-1]
            Tn = Tm * upmel if upmel is not None else Tm
            x = torch.randn(1, cond_in_channels, Tn, device=device)
            y_hat, _, _, _ = G(x=x, c=mel, f0=f0_t)
        elif 'HiFivae_task.HiFivae_task' in task:
            # HiFivae encodes waveform and reconstructs; feed raw audio as input
            y_hat, *_ = G(x=wav_t)
        elif 'freev_task.FreeVTask' in task:
            y_hat = G(mel)
        elif 'freev_nsf_task.FreeVNSFTask' in task:
            # requires f0
            f0, uv = get_pitch(
                pitch_extractor,
                wav.squeeze(0).numpy(),
                length=mel.shape[-1],
                hparams=cfg,
                speed=1,
                interp_uv=True,
            )
            if f0 is None:
                raise RuntimeError("Pitch extractor returned all-UV; please choose a different audio segment.")
            f0_t = torch.from_numpy(f0)[None, None, :].to(device)
            y_hat = G(mel, f0_t)
        else:
            raise NotImplementedError(f"Unsupported task_cls: {task_cls}")

        y_hat = y_hat.squeeze(0).cpu()

    # Name output
    step_str = str(step) if step >= 0 else 'unknown'
    out_name = f"{wav_path.stem}.{task_name}_step{step_str}.wav"
    out_path = wav_path.with_name(out_name)
    torchaudio.save(str(out_path), y_hat, sample_rate=target_sr)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Vocoder inference for RefineGAN/NSF-HiFiGAN")
    ap.add_argument('--config', required=True, help='Path to config YAML used for training')
    ap.add_argument('--ckpt', required=True, help='Path to Lightning checkpoint (.ckpt)')
    ap.add_argument('--wav', required=True, help='Path to input test wav file')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    config_path = pathlib.Path(args.config)
    ckpt_path = pathlib.Path(args.ckpt)
    wav_path = pathlib.Path(args.wav)
    device = torch.device(args.device)

    out = infer_one(config_path, ckpt_path, wav_path, device)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
