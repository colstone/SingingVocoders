import importlib
import os

from training.base_task_pitch import PitchBaseTask


class FCPETask(PitchBaseTask):
    """FCPE backbone (CFNaiveMelPE: depthwise-sep Conv1D + Conformer, ~10M params).

    Pairs with configs/fcpe.yaml. All loss / validation / decode logic lives in PitchBaseTask
    (training/base_task_pitch.py); this leaf only builds the FCPE model and its datasets — same pattern
    as the vocoder leaves overriding GanBaseTask.build_model / build_dataset.
    """

    def build_model(self):
        from models.fcpe import FCPE_E2E
        self.generator = FCPE_E2E(config=self.config)

    def build_dataset(self):
        # FCPE (CFNaiveMelPE) does not downsample time -> validation clips need no rounding (time_multiple=1).
        cls_path = self.config.get('dataset_cls', 'training.base_task_pitch.CachedF0Dataset')
        module_name, cls_name = cls_path.rsplit('.', 1)
        dataset_cls = getattr(importlib.import_module(module_name), cls_name)
        di = self.config['DataIndexPath']
        self.train_dataset = dataset_cls(
            config=self.config, path=os.path.join(di, self.config['train_set_name']), test=False, time_multiple=1)
        self.valid_dataset = dataset_cls(
            config=self.config, path=os.path.join(di, self.config['valid_set_name']), test=True, time_multiple=1)
        self.train_probe_dataset = dataset_cls(
            config=self.config, path=os.path.join(di, self.config['train_set_name']), test=True, time_multiple=1,
            max_files=self.train_probe_num) if self.train_probe_num > 0 else None
