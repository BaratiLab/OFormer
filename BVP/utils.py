import os
import shutil
import numpy as np
import torch


def save_checkpoint(state, save_path: str, is_best: bool = False, max_keep: int = None):
    """Saves torch model to checkpoint file.
    Args:
        state (torch model state): State of a torch Neural Network
        save_path (str): Destination path for saving checkpoint
        is_best (bool): If ``True`` creates additional copy
            ``best_model.ckpt``
        max_keep (int): Specifies the max amount of checkpoints to keep
    """
    # save checkpoint
    torch.save(state, save_path)

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'latest_checkpoint.txt')

    save_path = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_path + '\n'] + ckpt_list
    else:
        ckpt_list = [save_path + '\n']

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))


def load_checkpoint(ckpt_dir_or_file: str, map_location=None, load_best=False):
    """Loads torch model from checkpoint file.
    Args:
        ckpt_dir_or_file (str): Path to checkpoint directory or filename
        map_location: Can be used to directly load to specific device
        load_best (bool): If True loads ``best_model.ckpt`` if exists.
    """
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
        else:
            with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint.txt')) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


def ensure_dir(dir_name: str):
    """Creates folder if not exists.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_arguments(parser):
    # basic training settings
    parser.add_argument(
        '--lr', type=float, default=3e-4, help='Specifies learing rate for optimizer. (default: 1e-3)'
    )
    parser.add_argument(
        '--resume', action='store_true', help='If set resumes training from provided checkpoint. (default: None)'
    )
    parser.add_argument(
        '--path_to_resume', type=str, default='', help='Path to checkpoint to resume training. (default: "")'
    )
    parser.add_argument(
        '--iters', type=int, default=100000, help='Number of training iterations. (default: 100k)'
    )
    parser.add_argument(
        '--log_dir', type=str, default='./', help='Path to log, save checkpoints. '
    )
    parser.add_argument(
        '--ckpt_every', type=int, default=5000, help='Save model checkpoints every x iterations. (default: 5k)'
    )
    parser.add_argument(
        '--use_data_parallel', action='store_true', help='Use DataParallel to train'
    )

    # ==================================
    # general option
    parser.add_argument(
        '--in_grid_size', type=int, default=64, help='Size of input spatial grid '
    )
    parser.add_argument(
        '--in_seq_len', type=int, default=10, help='Length of input sequence. (default: 10)'
    )
    # model options for encoder

    parser.add_argument(
        '--in_channels', type=int, default=3, help='Channel of input feature. (default: 3)'
    )
    parser.add_argument(
        '--encoder_emb_dim', type=int, default=128, help='Channel of token embedding in encoder. (default: 128)'
    )
    parser.add_argument(
        '--out_seq_emb_dim', type=int, default=128, help='Channel of output feature map. (default: 128)'
    )
    parser.add_argument(
        '--encoder_depth', type=int, default=2, help='Depth of transformer in encoder. (default: 2)'
    )
    parser.add_argument(
        '--encoder_heads', type=int, default=4, help='Heads of transformer in encoder. (default: 4)'
    )
    parser.add_argument(
        '--propagate_round', type=int, default=2, help='How many rounds to propagate the dynamics (default: 2)'
    )
    parser.add_argument(
        '--use_cnn_encoder', action='store_true',
    )
    parser.add_argument(
        '--use_simple_encoder', action='store_true',
    )
    parser.add_argument(
        '--use_pooling', action='store_true',
    )
    parser.add_argument(
        '--no_st', action='store_true',
    )
    parser.add_argument(
        '--no_cls', action='store_true',
    )
    parser.add_argument(
        '--use_grad', action='store_true',
    )
    parser.add_argument(
        '--use_attn_prop', action='store_true',
    )
    parser.add_argument(
        '--sampling_ratio', type=float, default=0.85, help='Sampling points during training (default: 0.5)'
    )
    parser.add_argument(
        '--eval_mode', action='store_true',
    )

    # model options for decoder
    parser.add_argument(
        '--out_channels', type=int, default=1, help='Channel of output. (default: 1)'
    )
    parser.add_argument(
        '--decoder_emb_dim', type=int, default=128, help='Channel of token embedding in decoder. (default: 128)'
    )
    parser.add_argument(
        '--out_step', type=int, default=10, help='How many steps to propagate forward each call. (default: 10)'
    )
    parser.add_argument(
        '--out_seq_len', type=int, default=50, help='Length of output sequence. (default: 190)'
    )
    parser.add_argument(
        '--propagator_depth', type=int, default=2, help='Depth of mlp in propagator. (default: 2)'
    )
    parser.add_argument(
        '--decoding_depth', type=int, default=2, help='Depth of decoding network in the phinet. (default: 2)'
    )
    parser.add_argument(
        '--use_position_encoding', action='store_true',
    )
    parser.add_argument(
        '--fourier_frequency', type=int, default=8, help='Fourier feature frequency. (default: 8)'
    )

    # ===================================
    # for dataset
    parser.add_argument(
        '--batch_size', type=int, default=16, help='Size of each batch (default: 16)'
    )
    parser.add_argument(
        '--train_dataset_path', type=str, default='path_to_data/fenics_pde/fenics_data', help='Path to dataset.'
    )
    parser.add_argument(
        '--test_dataset_path', type=str, default='path_to_data/fenics_pde/test_data', help='Path to dataset.'
    )
    parser.add_argument(
        '--train_sequence_num', type=int, default=10000, help='How many sequences in the training dataset.'
    )
    parser.add_argument(
        '--test_sequence_num', type=int, default=500, help='How many sequences in the training dataset.'
    )
    parser.add_argument(
        '--sequence_length', type=int, default=200, help='How many snapshots in each sequence.'
    )
    parser.add_argument(
        '--interval', type=int, default=4, help='The interval of when sample snapshots from sequence'
    )

    return parser
