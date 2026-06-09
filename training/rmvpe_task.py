import importlib
import os

from training.base_task_pitch import PitchBaseTask


class RMVPETask(PitchBaseTask):
    """RMVPE backbone (2D DeepUNet + BiGRU, ~90M params).

    Pairs with configs/rmvpe.yaml. All loss / validation / decode logic lives in PitchBaseTask
    (training/base_task_pitch.py); this leaf only builds the RMVPE model and its datasets — same pattern
    as the vocoder leaves overriding GanBaseTask.build_model / build_dataset.
    """

    def build_model(self):
        from models.rmvpe.model import E2E
        self.generator = E2E(config=self.config)

    def build_dataset(self):
        # RMVPE's 2D DeepUnet pools time by 2 per en_de layer -> clips must be a multiple of 2**en_de_layers.
        time_multiple = 2 ** self.config['en_de_layers']
        cls_path = self.config.get('dataset_cls', 'training.base_task_pitch.CachedF0Dataset')
        module_name, cls_name = cls_path.rsplit('.', 1)
        dataset_cls = getattr(importlib.import_module(module_name), cls_name)
        di = self.config['DataIndexPath']
        self.train_dataset = dataset_cls(
            config=self.config, path=os.path.join(di, self.config['train_set_name']), test=False, time_multiple=time_multiple)
        self.valid_dataset = dataset_cls(
            config=self.config, path=os.path.join(di, self.config['valid_set_name']), test=True, time_multiple=time_multiple)
        self.train_probe_dataset = dataset_cls(
            config=self.config, path=os.path.join(di, self.config['train_set_name']), test=True, time_multiple=time_multiple,
            max_files=self.train_probe_num) if self.train_probe_num > 0 else None
