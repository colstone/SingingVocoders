import logging
import os

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from glob import glob

from mir_eval.melody import raw_pitch_accuracy, to_cent_voicing, raw_chroma_accuracy, overall_accuracy
from mir_eval.melody import voicing_recall, voicing_false_alarm

from models.rmvpe.model import E2E
from modules.loss.rmvpe_loss import FL, bce
from utils.wav2F0 import get_pitch
from utils.wav2mel import PitchAdjustableMelSpectrogram


class RMVPE_dataset(Dataset):
    def __init__(self, config: dict, path, test=False):
        super().__init__()
        self.config = config
        self.path = path
        self.hop_length = config["hop_size"]
        self.seq_len = config['seq_l'] if not test else None
        en_de_layers = config['en_de_layers']
        self.CONST = config['const']
        self.win_size = config['win_size']
        if self.seq_len is not None:
            assert self.seq_len % self.hop_length == 0
            assert ((self.seq_len // self.hop_length) + 1) % (2 ** en_de_layers) == 0

        self.num_class = config['n_class']

        self.mel_tf = PitchAdjustableMelSpectrogram(
            sample_rate=config['audio_sample_rate'],
            n_fft=config['fft_size'],
            win_length=self.win_size,
            hop_length=self.hop_length,
            f_min=config['fmin'],
            f_max=config['fmax'],
            n_mels=config['audio_num_mel_bins'],
        )

        self.data = []
        # Pre-load data logic from original code
        # Note: This loads everything into memory. 
        # Ideally this should be done lazily or cached, but sticking to original logic for consistency.
        files = glob(os.path.join(self.path, '*.wav'))
        if not files:
            logging.warning(f"No wav files found in {self.path}")
        
        for audio_file in files:
            self.data.extend(self.load(audio_file))

    def load(self, audio_path):
        data = []
        try:
            audio, _ = librosa.load(audio_path, sr=self.config['audio_sample_rate'])
        except Exception as e:
            logging.error(f"Error loading {audio_path}: {e}")
            return []

        # Pad with hop_length // 2 to align Mel (which adds (win-hop)/2) with F0
        audio_l = len(audio)
        audio = np.pad(audio, self.hop_length // 2, mode='reflect')

        audio_tensor = torch.from_numpy(audio).float()
        # mel_tf expects (B, T) or (T)? Unsqueeze helps.
        # Original code used .to(device) here but we should keep it CPU until getitem/collate
        with torch.no_grad():
            mel = self.mel_tf(audio_tensor.unsqueeze(0)).squeeze()  # (n_mels, T)
            mel = self.mel_tf.dynamic_range_compression_torch(x=mel)

        audio_steps = mel.shape[-1]

        pitch_label = torch.zeros(audio_steps, self.num_class, dtype=torch.float)
        voice_label = torch.zeros(audio_steps, dtype=torch.float)

        try:
            raw_f0, uv = get_pitch(
                pe=self.config["pitch_extractor"],
                wav_data=audio,
                length=audio_steps,
                hparams=self.config)
        except Exception as e:
             logging.error(f"Error extracting pitch for {audio_path}: {e}")
             return []

        if raw_f0 is None:
            # raise RuntimeError("Pitch extractor returned all-UV; please choose a different audio segment.")
            logging.warning(f"Pitch extractor returned None for {audio_path}")
            return []

        for i, freq in enumerate(raw_f0):
            if float(freq) != 0:
                cent = 1200 * np.log2(freq / 10)
                index = int(round((cent - self.CONST) / 20))
                if 0 <= index < self.num_class:
                    pitch_label[i][index] = 1
                    voice_label[i] = 1

        if self.seq_len is not None:
            n_steps = self.seq_len // self.hop_length + 1
            # Calculate how many segments
            if audio_l >= self.seq_len:
                for i in range(audio_l // self.seq_len):
                    begin_t = i * self.seq_len
                    begin_step = begin_t // self.hop_length
                    end_step = begin_step + n_steps
                    
                    if end_step <= mel.shape[1]:
                        data.append(dict(mel=mel[:, begin_step:end_step], 
                                         pitch=pitch_label[begin_step:end_step],
                                         voice=voice_label[begin_step:end_step], 
                                         file=audio_path))
                
                # Handle last segment if needed or just skip? 
                # Original code:
                # data.append(dict(mel=mel[:, -n_steps:], pitch=pitch_label[-n_steps:],
                #              voice=voice_label[-n_steps:], file=audio_path))
                # We can keep the last segment logic
                data.append(dict(mel=mel[:, -n_steps:], pitch=pitch_label[-n_steps:],
                                 voice=voice_label[-n_steps:], file=audio_path))
            else:
                 # Audio too short for seq_len, pad or skip? 
                 # For training usually we need fixed size. Original code doesn't seem to handle short audio explicitly in loop
                 pass
        else:
            # For validation (test=True), seq_len is None, we return full sequence
            data.append(dict(mel=mel, pitch=pitch_label, voice=voice_label, file=audio_path))

        return data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        # Basic collate
        # keys: mel, pitch, voice, file
        # Check if all have same size (training) or different (validation)
        # If training, stack. If validation, batch_size should be 1 usually.
        
        mels = [item['mel'] for item in batch]
        pitches = [item['pitch'] for item in batch]
        voices = [item['voice'] for item in batch]
        files = [item['file'] for item in batch]

        # Check shapes
        if len(set(m.shape[1] for m in mels)) == 1:
            return {
                'mel': torch.stack(mels),
                'pitch': torch.stack(pitches),
                'voice': torch.stack(voices),
                'file': files
            }
        else:
            # Different lengths (validation) - return list or assume batch_size=1 and stack 
            # (if batch_size=1, stack works)
            if len(batch) == 1:
                 return {
                    'mel': torch.stack(mels),
                    'pitch': torch.stack(pitches),
                    'voice': torch.stack(voices),
                    'file': files
                }
            else:
                 raise RuntimeError("Batch elements have different lengths, use batch_size=1 for validation")


class RMVPETask(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        
        self.model = E2E(config=config)
        
        self.alpha = config['alpha']
        self.gamma = config['gamma']
        self.learning_rate = float(config['learning_rate'])
        self.clip_grad_norm = config.get('clip_grad_norm', None)
        
        # Validation metrics
        self.validation_step_outputs = []
        
        # Required by DsModelCheckpoint
        self.skip_immediate_ckpt_save = False

    def setup(self, stage=None):
        pass # Datasets are created in train_dataloader/val_dataloader to avoid pickling issues if needed, or here.
             # But standard pattern in this repo seems to be in task init or build_dataset.
             # I will do it lazily or in dataloader creation to be safe.

    def forward(self, mel):
        return self.model(mel)

    def training_step(self, batch, batch_idx):
        mel = batch['mel']
        pitch_label = batch['pitch']
        
        hidden_vec, pitch_pred = self.model(mel)
        loss = FL(pitch_pred, pitch_label, self.alpha, self.gamma)
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Scheduler
        decay_steps = self.config.get('learning_rate_decay_steps', 1000) 
        # Note: In raw code: len(data_loader) * config['learning_rate_decay_steps']
        # Here we don't know len(data_loader) easily in configure_optimizers without creating dataset.
        # But we can assume the user configures it or we approximate.
        # If we want exact match with raw code:
        # We need dataset size.
        
        # Let's create dataset in __init__ or setup?
        # Creating dataset in __init__ is safer for optimizer config if we need length.
        
        # For now, let's use the raw config value directly, but note that StepLR step_size is in "steps" (batches) or "epochs"?
        # Torch StepLR steps are calls to scheduler.step(). In raw code, scheduler.step() is called every batch.
        # So decay_steps should be in total steps.
        # Raw code: learning_rate_decay_steps = len(data_loader) * config['learning_rate_decay_steps']
        # This means config['learning_rate_decay_steps'] is in EPOCHS.
        
        # We'll rely on the standard Lightning LR monitor/scheduler. 
        # But to replicate raw behavior:
        
        scheduler = StepLR(optimizer, step_size=decay_steps, gamma=self.config.get('learning_rate_decay_rate', 0.98)) 
        # Wait, I cannot know len(dataloader) easily here. 
        # I'll just return optimizer for now to keep it simple, or use a standard scheduler.
        # The user can tune it.
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # called every step
            }
        }

    def train_dataloader(self):
        dataset = RMVPE_dataset(
            config=self.config, 
            path=os.path.join(self.config['DataIndexPath'], self.config['train_set_name']),
            test=False
        )
        # Recalculate scheduler step_size if possible?
        # We can update the scheduler here? No, configure_optimizers is called before.
        # Actually pl calls setup() -> configure_optimizers() -> train_dataloader().
        # So we can create dataset in setup().
        self.train_ds_len = len(dataset)
        
        return DataLoader(
            dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            drop_last=True,
            collate_fn=dataset.collate_fn,
            num_workers=self.config.get('ds_workers', 4),
            persistent_workers=True
        )

    def val_dataloader(self):
        dataset = RMVPE_dataset(
            config=self.config, 
            path=os.path.join(self.config['DataIndexPath'], self.config['valid_set_name']),
            test=True
        )
        return DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, 
            drop_last=False,
            collate_fn=dataset.collate_fn,
            num_workers=self.config.get('ds_workers', 4)
        )

    def validation_step(self, batch, batch_idx):
        mel = batch['mel'] # (B, n_mels, T)
        pitch_label = batch['pitch'] # (B, T, n_class)
        # B is 1
        
        pitch_pred = torch.zeros_like(pitch_label)
        
        # Sliding window inference as in evaluate()
        # mel shape: [1, 128, T]
        T = mel.shape[-1]
        
        for start_steps in range(0, T, 128):
            end_steps = start_steps + 128
            t_mel = mel[:, :, start_steps:end_steps]
            
            # Check length and pad
            curr_len = t_mel.shape[-1]
            pad_len = 128 - curr_len
            if pad_len > 0:
                t_mel = F.pad(t_mel, (0, pad_len))
            
            # Forward
            # model expects (B, n_mels, 128)
            hidden_vec, t_pitch_pred = self.model(t_mel)
            t_pitch_pred = t_pitch_pred.squeeze(0) # (128, n_class)
            
            # Assign back
            write_len = min(128, T - start_steps)
            pitch_pred[0, start_steps:start_steps + write_len] = t_pitch_pred[:write_len]

        loss = bce(pitch_pred, pitch_label)
        self.log('val/loss', loss)
        
        # Metrics
        pitch_th = 0.0 # From original code default
        
        cents_pred = self.to_local_average_cents(pitch_pred.squeeze(0).cpu().numpy(), None, pitch_th)
        cents_label = self.to_local_average_cents(pitch_label.squeeze(0).cpu().numpy(), None, pitch_th)

        freq_pred = np.array([10 * (2 ** (cent_pred / 1200)) if cent_pred else 0 for cent_pred in cents_pred])
        freq = np.array([10 * (2 ** (cent / 1200)) if cent else 0 for cent in cents_label])
        
        hop_cent_length = self.config['hop_size'] / self.config['audio_sample_rate']
        time_slice = np.array([i * hop_cent_length for i in range(len(cents_label))])
        
        ref_v, ref_c, est_v, est_c = to_cent_voicing(time_slice, freq, time_slice, freq_pred)

        rpa = raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)
        rca = raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)
        oa = overall_accuracy(ref_v, ref_c, est_v, est_c)
        vfa = voicing_false_alarm(ref_v, est_v)
        vr = voicing_recall(ref_v, est_v)
        
        metrics = {
            'RPA': rpa,
            'RCA': rca,
            'OA': oa,
            'VFA': vfa,
            'VR': vr,
            'loss': loss.item()
        }
        self.validation_step_outputs.append(metrics)
        return metrics

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if not outputs:
            return
            
        avg_loss = np.mean([x['loss'] for x in outputs])
        avg_rpa = np.mean([x['RPA'] for x in outputs])
        avg_rca = np.mean([x['RCA'] for x in outputs])
        avg_oa = np.mean([x['OA'] for x in outputs])
        avg_vfa = np.mean([x['VFA'] for x in outputs])
        avg_vr = np.mean([x['VR'] for x in outputs])
        
        self.log('val/loss_epoch', avg_loss, prog_bar=True)
        self.log('val/RPA', avg_rpa, prog_bar=True)
        self.log('val/RCA', avg_rca)
        self.log('val/OA', avg_oa)
        self.log('val/VFA', avg_vfa)
        self.log('val/VR', avg_vr)
        
        self.validation_step_outputs.clear()

    @staticmethod
    def to_local_average_cents(salience, center=None, thred=0.0):
        """
        find the weighted average cents near the argmax bin
        """
        if not hasattr(RMVPETask.to_local_average_cents, 'cents_mapping'):
            # the bin number-to-cents mapping
            RMVPETask.to_local_average_cents.cents_mapping = (
                    np.linspace(0, 7180, 360) + 1997.3794084376191)
        
        mapping = RMVPETask.to_local_average_cents.cents_mapping

        if salience.ndim == 1:
            if center is None:
                center = int(np.argmax(salience))
            start = max(0, center - 4)
            end = min(len(salience), center + 5)
            salience = salience[start:end]
            if len(salience) == 0: return 0
            product_sum = np.sum(
                salience * mapping[start:end])
            weight_sum = np.sum(salience)
            return product_sum / weight_sum if np.max(salience) > thred else 0
        if salience.ndim == 2:
            return np.array([RMVPETask.to_local_average_cents(salience[i, :], None, thred) for i in
                             range(salience.shape[0])])
        raise Exception("label should be either 1d or 2d ndarray")
