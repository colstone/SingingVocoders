import librosa
import numpy as np
import parselmouth
import pyworld as pw
import torch
import torch.nn.functional as F

from utils.wav2mel import PitchAdjustableMelSpectrogram

# from utils.pitch_utils import interp_f0

PITCH_EXTRACTORS_ID_TO_NAME = {
    1: 'parselmouth',
    2: 'harvest',
    3: 'rmvpe',
    4: 'fcpe',
}
PITCH_EXTRACTORS_NAME_TO_ID = {v: k for k, v in PITCH_EXTRACTORS_ID_TO_NAME.items()}


def to_local_average_cents(salience, center=None, thred=0.0):
    """
    find the weighted average cents near the argmax bin
    """
    if not hasattr(to_local_average_cents, 'cents_mapping'):
        # the bin number-to-cents mapping
        to_local_average_cents.cents_mapping = (
                np.linspace(0, 7180, 360) + 1997.3794084376191)

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(
            salience * to_local_average_cents.cents_mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum if np.max(salience) > thred else 0
    if salience.ndim == 2:
        return np.array([to_local_average_cents(salience[i, :], None, thred) for i in
                         range(salience.shape[0])])

    raise Exception("label should be either 1d or 2d ndarray")


def norm_f0(f0, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = np.log2(f0 + uv)  # avoid arithmetic error
    f0[uv] = -np.inf
    return f0


def denorm_f0(f0, uv, pitch_padding=None):
    f0 = 2 ** f0
    if uv is not None:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0


def interp_f0(f0, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = norm_f0(f0, uv)
    if uv.any() and not uv.all():
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return denorm_f0(f0, uv=None), uv


def get_pitch(pe, wav_data, length, hparams, speed=1, interp_uv=False, **kwargs):
    if pe == 'parselmouth':
        return get_pitch_parselmouth(wav_data, length, hparams, speed=speed, interp_uv=interp_uv)
    elif pe == 'harvest':
        return get_pitch_harvest(wav_data, length, hparams, speed=speed, interp_uv=interp_uv)
    elif pe == 'rmvpe':
        return get_pitch_rmvpe(wav=wav_data, length=length, hparams=hparams, model=kwargs.get('model'), device=kwargs.get('device'), mel=kwargs.get('mel'), pe_hparams=kwargs.get('pe_hparams'), speed=speed, interp_uv=interp_uv)
    elif pe == 'fcpe':
        return get_pitch_fcpe(wav=wav_data, length=length, hparams=hparams, model=kwargs.get('model'), device=kwargs.get('device'), mel=kwargs.get('mel'), pe_hparams=kwargs.get('pe_hparams'), speed=speed, interp_uv=interp_uv)
    else:
        raise ValueError(f" [x] Unknown pitch extractor: {pe}")


def _run_pitch_model(forward_fn, wav=None, mel=None, length=None, hparams=None, model=None, device=None,
                     pe_hparams=None, speed=1, interp_uv=False):
    """Pure shared pipeline for the model-based PE encoders (RMVPE / FCPE), with NO per-model branching:
    (optionally) wav -> mel, move to device, `pitch_pred = forward_fn(model, mel, mel_config)` (per-frame
    logits, (T, n_class)), decode to f0 (Hz), align to `length`, optionally interpolate unvoiced.
    The per-model difference (how the model is run over the mel) lives in forward_fn, supplied by
    get_pitch_rmvpe / get_pitch_fcpe.
    """
    if wav is None and mel is None:
        raise ValueError("wav and mel cannot be both None")
    if model is None:
        raise ValueError("a model must be passed to the model-based pitch extractor")
    if device is None:
        device = torch.device('cpu')

    # Use pe_hparams if available (model-specific config), else hparams (vocoder config)
    mel_config = pe_hparams if pe_hparams is not None else hparams
    if mel_config is None and mel is None:
        raise ValueError("hparams or pe_hparams must be provided if mel is None")

    if mel is None:
        # Compute mel from wav
        wav_data = wav
        if isinstance(wav_data, torch.Tensor):
            wav_data = wav_data.cpu().numpy()
        if wav_data.ndim > 1:
            wav_data = wav_data.flatten()

        hop_size = int(np.round(mel_config['hop_size'] * speed))
        mel_extractor = PitchAdjustableMelSpectrogram(
            sample_rate=mel_config['audio_sample_rate'],
            n_fft=mel_config['fft_size'],
            win_length=mel_config['win_size'],
            hop_length=hop_size,
            f_min=mel_config['fmin'],
            f_max=mel_config['fmax'],
            n_mels=mel_config['audio_num_mel_bins'],
        ).to(device)

        pad_len = hop_size // 2
        wav_data_padded = np.pad(wav_data, (pad_len, pad_len), mode='reflect')
        audio_t = torch.from_numpy(wav_data_padded).float().to(device).unsqueeze(0)
        with torch.no_grad():
            mel = mel_extractor(audio_t)  # (1, n_mels, T)
            mel = mel_extractor.dynamic_range_compression_torch(mel)

    mel = mel.to(device)
    if mel.ndim == 2:
        mel = mel.unsqueeze(0)  # (1, n_mels, T)

    with torch.no_grad():
        pitch_pred = forward_fn(model, mel, mel_config)   # (T, n_class)
        cents = to_local_average_cents(pitch_pred.cpu().numpy(), None, thred=0.03)
        f0 = np.array([10 * (2 ** (c / 1200)) if c > 0 else 0 for c in cents])

    # Align to the requested frame count (handles a PE hop != vocoder hop via time-axis interpolation)
    if length is not None and len(f0) != length:
        if pe_hparams and hparams and pe_hparams['hop_size'] != hparams['hop_size']:
            src_hop = pe_hparams['hop_size'] * speed
            t_src = np.arange(len(f0)) * src_hop / mel_config['audio_sample_rate']
            t_tgt = np.arange(length) * (hparams['hop_size'] * speed) / hparams['audio_sample_rate']
            f0 = np.interp(t_tgt, t_src, f0)
        else:
            if len(f0) < length:
                f0 = np.pad(f0, (0, length - len(f0)))
            f0 = f0[:length]
    elif length is not None:
        if len(f0) < length:
            f0 = np.pad(f0, (0, length - len(f0)))
        f0 = f0[:length]

    uv = f0 == 0
    if uv.any() and interp_uv:
        f0, uv = interp_f0(f0, uv)
    return f0, uv


def get_pitch_rmvpe(wav=None, mel=None, length=None, hparams=None, model=None, device=None, pe_hparams=None, speed=1, interp_uv=False):
    def forward(model, mel, mel_config):
        # RMVPE's 2D DeepUnet runs on fixed 128-frame chunks (its training shape): pad to a multiple,
        # run per window, concat, crop back.
        chunk_size = 128
        bs = mel_config.get('batch_size', 1)
        T = mel.shape[-1]
        pad_frames = (-T) % chunk_size
        if pad_frames:
            mel = F.pad(mel, (0, pad_frames))
        N = mel.shape[-1] // chunk_size
        chunks = mel.view(1, mel.shape[1], N, chunk_size).transpose(1, 2).squeeze(0)  # (N, n_mels, chunk)
        preds = [model(chunks[i:i + bs])[1] for i in range(0, N, bs)]                 # each (B, chunk, n_class)
        return torch.cat(preds, dim=0).reshape(-1, preds[0].shape[-1])[:T]            # (T, n_class)

    return _run_pitch_model(forward, wav=wav, mel=mel, length=length, hparams=hparams, model=model,
                            device=device, pe_hparams=pe_hparams, speed=speed, interp_uv=interp_uv)


def get_pitch_fcpe(wav=None, mel=None, length=None, hparams=None, model=None, device=None, pe_hparams=None, speed=1, interp_uv=False):
    def forward(model, mel, mel_config):
        # FCPE (CFNaiveMelPE) handles arbitrary T -> run the full mel, no chunking.
        return model(mel)[1].squeeze(0)                                              # (T, n_class)

    return _run_pitch_model(forward, wav=wav, mel=mel, length=length, hparams=hparams, model=model,
                            device=device, pe_hparams=pe_hparams, speed=speed, interp_uv=interp_uv)


def get_pitch_parselmouth(wav_data, length, hparams, speed=1, interp_uv=False):
    """

    :param wav_data: [T]
    :param length: Expected number of frames
    :param hparams:
    :param speed: Change the speed
    :param interp_uv: Interpolate unvoiced parts
    :return: f0, uv
    """
    hop_size = int(np.round(hparams['hop_size'] * speed))
    time_step = hop_size / hparams['audio_sample_rate']
    f0_min = hparams['f0_min']
    f0_max = hparams['f0_max']

    l_pad = int(np.ceil(1.5 / f0_min * hparams['audio_sample_rate']))
    r_pad = hop_size * ((len(wav_data) - 1) // hop_size + 1) - len(wav_data) + l_pad + 1
    wav_data = np.pad(wav_data, (l_pad, r_pad))

    # noinspection PyArgumentList
    s = parselmouth.Sound(wav_data, sampling_frequency=hparams['audio_sample_rate']).to_pitch_ac(
        time_step=time_step, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max)
    assert np.abs(s.t1 - 1.5 / f0_min) < 0.001
    f0 = s.selected_array['frequency'].astype(np.float32)
    if len(f0) < length:
        f0 = np.pad(f0, (0, length - len(f0)))
    f0 = f0[: length]
    uv = f0 == 0
    if uv.any() and interp_uv:
        f0, uv = interp_f0(f0, uv)
    return f0, uv

def get_pitch_harvest(wav_data, length, hparams, speed=1, interp_uv=False):
    hop_size = int(np.round(hparams['hop_size'] * speed))
    time_step = 1000 * hop_size / hparams['audio_sample_rate']
    f0_floor = hparams['f0_min']
    f0_ceil = hparams['f0_max']

    f0, _ = pw.harvest(wav_data.astype(np.float64), hparams['audio_sample_rate'], f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=time_step)
    f0 = f0.astype(np.float32)

    if f0.size < length:
        f0 = np.pad(f0, (0, length - f0.size))
    f0 = f0[:length]
    uv = f0 == 0
    if uv.any() and interp_uv:
        f0, uv = interp_f0(f0, uv)
    return f0, uv
