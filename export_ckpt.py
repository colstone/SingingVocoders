import pathlib
import json
import click
import torch
from tqdm import tqdm

from utils import get_latest_checkpoint_path
from utils.config_utils import read_full_config, print_config


@click.command(help='')
@click.option('--exp_name', required=False, metavar='EXP', help='Name of the experiment')
@click.option('--ckpt_path', required=False, metavar='FILE', help='Path to the checkpoint file')
@click.option('--save_path', required=True, metavar='FILE', help='Path to save the exported checkpoint')
@click.option('--work_dir', required=False, metavar='DIR', help='Working directory containing the experiments')
def export(exp_name, ckpt_path, save_path, work_dir):
    # print_config(config)
    if exp_name is None and ckpt_path is None:
        raise RuntimeError('Either --exp_name or --ckpt_path should be specified.')
    if ckpt_path is None:
        if work_dir is None:
            work_dir = pathlib.Path(__file__).parent / 'experiments'
        else:
            work_dir = pathlib.Path(work_dir)
        work_dir = work_dir / exp_name
        assert not work_dir.exists() or work_dir.is_dir(), f'Path \'{work_dir}\' is not a directory.'
        ckpt_path = get_latest_checkpoint_path(work_dir)
    
    ckpt = {}
    temp_dict = torch.load(ckpt_path)['state_dict']
    for i in tqdm(temp_dict):
        i: str
        if 'generator.' in i:
            # print(i)
            ckpt[i.replace('generator.', '')] = temp_dict[i]
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'generator': ckpt}, save_path)
    print("Export checkpoint file successfully: ", save_path)
    
    config_file = pathlib.Path(ckpt_path).with_name('config.yaml')
    config = read_full_config(config_file)
    task_cls = str(config.get('task_cls', '')).lower()
    new_config_file = pathlib.Path(save_path).with_name('config.json')
    model_args = dict(config.get('model_args', {}))  # shallow copy to avoid in-place edits
    base_audio_cfg = {
        'sampling_rate': config['audio_sample_rate'],
        'num_mels': config['audio_num_mel_bins'],
        'hop_size': config['hop_size'],
        'n_fft': config['fft_size'],
        'win_size': config['win_size'],
        'fmin': config['fmin'],
        'fmax': config['fmax'],
    }

    if 'refinegan' in task_cls:
        # Only keep generator-related fields for RefineGAN export
        refinegan_cfg = {
            'task': 'RefineGAN',
            **base_audio_cfg,
            'downsample_rates': list(model_args.get('downsample_rates', [2, 2, 8, 8])),
            'upsample_rates': list(model_args.get('upsample_rates', [8, 8, 2, 2])),
            'start_channels': int(model_args.get('start_channels', 16)),
            'leaky_relu_slope': float(model_args.get('leaky_relu_slope', 0.2)),
            'template_generator': model_args.get('template_generator', 'comb'),
        }
        export_config = refinegan_cfg
    else:
        # Fallback to original export behavior
        new_config = model_args
        new_config.update(base_audio_cfg)
        if 'pc_aug' not in config.keys():
            new_config['pc_aug'] = False
        else:
            new_config['pc_aug'] = config['pc_aug']
        if 'mini_nsf' not in new_config.keys():
            new_config['mini_nsf'] = False
        if 'noise_sigma' not in new_config.keys():
            new_config['noise_sigma'] = 0.0
        export_config = new_config

    with open(new_config_file, 'w') as json_file:
        json_file.write(json.dumps(export_config, indent=1))
        print("Export configuration file successfully: ", new_config_file)


if __name__ == '__main__':
    export()
