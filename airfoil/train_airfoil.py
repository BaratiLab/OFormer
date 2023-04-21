import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import time
import os
from functools import partial
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from tensorboardX import SummaryWriter

from nn_module.encoder_module import IrregSTEncoder2D
from nn_module.decoder_module import IrregSTDecoder2D


from utils import load_checkpoint, save_checkpoint, ensure_dir
import torchvision
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import datetime
import logging
import shutil
from typing import Union
from einops import rearrange
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader, TensorDataset
from dataset_new import AirfoilData
from scipy.interpolate import griddata
from matplotlib.tri import Triangulation

# set flags / seeds
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)


def build_model():
    # currently they are hard coded
    encoder = IrregSTEncoder2D(
        input_channels=6,    # vx, vy, prs, dns, pos_x, pos_y
        time_window=4,
        in_emb_dim=128,
        out_chanels=128,
        max_node_type=3,
        heads=1,
        depth=4,
        res=200,
        use_ln=True,
        emb_dropout=0.0,
    )

    decoder = IrregSTDecoder2D(
        max_node_type=3,
        latent_channels=128,
        out_channels=4,  # vx, vy, prs, dns
        res=200,
        scale=2,
        dropout=0.1
    )

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) +\
                      sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return encoder, decoder


def make_image_grid(image: torch.Tensor,
                    coords: torch.tensor,
                    triags: torch.tensor,
                    out_path, nrow=25):
    b, t, n = image.shape
    image = rearrange(image, 'b t n-> (b t) n')

    image = image.detach().cpu().numpy()    # [bt, n]
    coords = coords.detach().cpu().numpy()  # [b, n, 2]
    triags = triags.detach().cpu().numpy()    # [b, 3]

    fig = plt.figure(figsize=(8., 4.))
    grid = ImageGrid(fig, 111, # similar to subplot(111)
                     nrows_ncols=(b*t//nrow, nrow),  # creates 2x2 grid of axes

                     )

    for ax, im_no in zip(grid, np.arange(b*t)):
        data = image[im_no]
        triag = triags[im_no//t]
        pos = coords[im_no//t]
        triag = Triangulation(pos[:, 0], pos[:, 1], triag)

        ax.tripcolor(triag, data)
        # plt.triplot(triag, 'ko-', ms=0.3, lw=0.05, alpha=0.6)
        ax.set_xlim([-0.5+20, 1.5+20])
        ax.set_ylim([-1.4+19.96, 1.4+19.96])
        ax.set_aspect(0.5)
        ax.axis('off')
        del triag
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def pointwise_rel_loss(x, y, p=2):
    #   x, y [b, t, n, c]
    assert x.shape == y.shape
    eps = 1e-5
    if p == 1:
        y_norm = 1.#y.abs() + eps
        diff = (x-y).abs()
    else:
        y_norm = 1.#y.pow(p) + eps
        diff = (x-y).pow(p)
    diff = diff / y_norm   # [b, c]
    diff = diff.sum(dim=-1)  # sum over channels
    diff = diff.mean(dim=(1, 2))  # sum over time
    diff = diff.mean()
    return diff


def roi_rel_loss(x, y, pos, p=2):
    mask = torch.logical_and(pos[..., 0:1] > -0.1 + 20, pos[..., 0:1] < 1.4 + 20)
    mask = torch.logical_and(mask, pos[..., 1:2] > -1.2 + 19.96)
    mask = torch.logical_and(mask, pos[..., 1:2] < 1.2 + 19.96)
    mask = mask.unsqueeze(1)
    mask = mask.expand(x.shape)
    if p == 1:
        y_norm = 1.#y.abs() + 1e-5
        diff = (x-y).abs()
    else:
        y_norm = 1.#y.pow(p) + 1e-5
        diff = (x-y).pow(p)
    diff = diff / y_norm
    diff = diff[mask]
    diff = diff.mean()
    return diff


def rmse_loss(y, y_gt):
    loss_fn = nn.MSELoss()
    mse = loss_fn(y, y_gt)
    return torch.sqrt(mse)


def roi_rmse(y, y_gt, pos):
    mask = torch.logical_and(pos[..., 0:1] > -0.1+20, pos[..., 0:1] < 1.4+20)
    mask = torch.logical_and(mask, pos[..., 1:2] > -1.2+19.96)
    mask = torch.logical_and(mask, pos[..., 1:2] < 1.2+19.96)
    mask = mask.unsqueeze(1)
    mask = mask.expand(y.shape)
    y = y[mask]
    y_gt = y_gt[mask]
    return rmse_loss(y, y_gt)


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

    # ===================================
    # for dataset
    parser.add_argument(
        '--batch_size', type=int, default=16, help='Size of each batch (default: 16)'
    )
    parser.add_argument(
        '--train_dataset_path', type=str, required=True, help='Path to training dataset.'
    )
    parser.add_argument(
        '--test_dataset_path', type=str, required=True, help='Path to testing dataset.'
    )
    parser.add_argument(
        '--curriculum_steps', type=int, default=8, help='at initial stage, dont rollout too long'
    )
    parser.add_argument(
        '--curriculum_ratio', type=float, default=0.2, help='how long is the initial stage?'
    )

    return parser


# Start with main code
if __name__ == '__main__':
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(
        description="Train a PDE transformer")
    parser = get_arguments(parser)
    opt = parser.parse_args()
    print('Using following options')
    print(opt)

    # if running on GPU and we want to use cuda move model there
    use_cuda = torch.cuda.is_available()

    # add code for datasets

    print('Preparing the data')
    train_data_path = opt.train_dataset_path
    test_data_path = opt.test_dataset_path

    train_set = AirfoilData(train_data_path, tw=4, use_normalized=True)
    test_set = AirfoilData(test_data_path, tw=4, use_normalized=False, return_cells=True)

    train_dataloader = DataLoader(
        train_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=1,)

    test_dataloader = DataLoader(
        test_set,
        batch_size=opt.batch_size,
        shuffle=False, num_workers=1)

    # instantiate network
    print('Building network')
    encoder, decoder = build_model()
    if use_cuda:
        encoder, decoder = encoder.cuda(), decoder.cuda()

    # typically we use tensorboardX to keep track of experiments
    writer = SummaryWriter()
    checkpoint_dir = os.path.join(opt.log_dir, 'model_ckpt')
    ensure_dir(checkpoint_dir)

    sample_dir = os.path.join(opt.log_dir, 'samples')
    ensure_dir(sample_dir)

    # save option information to the disk
    logger = logging.getLogger("LOG")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (opt.log_dir, 'logging_info'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('=======Option used=======')
    for arg in vars(opt):
        logger.info(f'{arg}: {getattr(opt, arg)}')

    # save the py script of models
    script_dir = os.path.join(opt.log_dir, 'script_cache')
    ensure_dir(script_dir)
    shutil.copy('../../nn_module/__init__.py', script_dir)
    shutil.copy('../../nn_module/attention_module.py', script_dir)
    shutil.copy('../../nn_module/cnn_module.py', script_dir)
    shutil.copy('../../nn_module/encoder_module.py', script_dir)
    shutil.copy('../../nn_module/decoder_module.py', script_dir)
    shutil.copy('../../nn_module/fourier_neural_operator.py', script_dir)
    shutil.copy('../../nn_module/gnn_module.py', script_dir)
    shutil.copy('../../train_airfoil.py', opt.log_dir)

    # create optimizers
    enc_optim = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=opt.lr,
                                  weight_decay=1e-4)
    # enc_optim = torch.optim.AdamW(list(encoder.parameters()), lr=opt.lr,
    #                               weight_decay=1e-6, amsgrad=True)
    # enc_scheduler = torch.optim.lr_scheduler.StepLR(enc_optim, opt.iters//10, gamma=0.75, last_epoch=-1)
    # dec_scheduler = torch.optim.lr_scheduler.StepLR(dec_optim, opt.iters//10, gamma=0.75, last_epoch=-1)
    enc_scheduler = OneCycleLR(enc_optim, max_lr=opt.lr, total_steps=opt.iters,
                               div_factor=1e4,
                               pct_start=0.3,
                               final_div_factor=1e4,
                               )


    # load checkpoint if needed/ wanted
    start_n_iter = 0
    if opt.resume:
        print(f'Resuming checkpoint from: {opt.path_to_resume}')
        ckpt = load_checkpoint(opt.path_to_resume)  # custom method for loading last checkpoint
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
        start_n_iter = ckpt['n_iter']

        enc_optim.load_state_dict(ckpt['enc_optim'])

        enc_scheduler.load_state_dict(ckpt['enc_sched'])
        print("last checkpoint restored")

    # now we start the main loop
    n_iter = start_n_iter

    # for loop going through dataset
    with tqdm(total=opt.iters) as pbar:
        pbar.update(n_iter)
        train_data_iter = iter(train_dataloader)

        while True:

            encoder.train()
            decoder.train()
            start_time = time.time()

            try:
                data = next(train_data_iter)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                del train_data_iter
                train_data_iter = iter(train_dataloader)
                data = next(train_data_iter)

            # data preparation
            x, y, node_type, pos = data

            if use_cuda:
                x, y, node_type, pos = x.cuda(), y.cuda(), node_type.cuda(), pos.cuda()

            input_pos = prop_pos = pos

            z = encoder.forward(x, node_type, input_pos)
            if opt.curriculum_steps > 0 and n_iter < int(opt.curriculum_ratio * opt.iters):
                progress = (n_iter * 2) / (opt.iters * opt.curriculum_ratio)
                curriculum_steps = opt.curriculum_steps + \
                                   int(max(0, progress - 1.) * ((y.shape[1] - opt.curriculum_steps) / 2.)) * 2
                y = y[:, :curriculum_steps, :]  # [b t n]
                pred = decoder.forward(z, prop_pos, node_type, curriculum_steps, input_pos)
            else:
                pred = decoder.forward(z, prop_pos, node_type, y.shape[1], input_pos)
            # pred = decoder.denormalize(pred, train_set)

            all_loss = pointwise_rel_loss(pred, y, p=2)
            roi_loss = roi_rel_loss(pred[..., :2]*pred[..., 2:3], y[..., :2]*y[..., 2:3], prop_pos, p=2)
            loss = all_loss + roi_loss*2.0
            enc_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2.0)
            # Unscales gradients and calls
            enc_optim.step()

            enc_scheduler.step()
            # with torch.no_grad():
            #     x_out = torch.cat((pot, field_x, field_y), dim=-1)
            #     pot_los, field_loss = mse_loss(x_out, y)

            # udpate tensorboardX
            writer.add_scalar('train_loss', loss, n_iter)

            pbar.set_description(
                f'Total loss (1e-4): {loss.item()*1e4:.1f}||' 
                f'All loss: {all_loss.item()*1e4:.1f}||'
                f'Roi loss: {roi_loss.item()*1e4:.1f}||'
                f'lr: {enc_optim.param_groups[0]["lr"]:.1e}|| '
                f'Seq len {y.shape[1]}||')

            pbar.update(1)
            start_time = time.time()
            n_iter += 1

            if (n_iter-1) % opt.ckpt_every == 0 or n_iter >= opt.iters:
                logger.info('Tesing')
                print('Testing')

                encoder.eval()
                decoder.eval()

                all_avg_loss = []
                all_rmse = []
                visualization_cache = {
                    'pred': [],
                    'gt': [],
                    'coords': [],
                    'cells': [],
                }
                picked = 0
                for j, data in enumerate(tqdm(test_dataloader)):
                    # data preparation
                    # x, y, x_mean, x_std = data

                    # data preparation
                    x, y, node_type, pos, cells = data

                    if use_cuda:
                        x, y, node_type, pos, cells = x.cuda(), y.cuda(), node_type.cuda(), pos.cuda(), cells.cuda()

                    input_pos = prop_pos = pos
                    with torch.no_grad():

                        z = encoder.forward(x, node_type, input_pos)
                        pred = decoder.forward(z, prop_pos, node_type, y.shape[1], input_pos)   # [b t n c]
                        pred = decoder.denormalize(pred, train_set)

                        loss = pointwise_rel_loss(pred, y, p=2)
                        all_avg_loss.append(loss.item())

                        mom_rmse = roi_rmse(pred[..., :2]*pred[..., 2:3], y[..., :2]*y[..., 2:3], prop_pos)
                        all_rmse.append(mom_rmse.item())

                    if picked < 8:
                        idx = np.arange(0, min(8 - picked, y.shape[0]))
                        # randomly pick a batch
                        interv = y.shape[1] // 8
                        y = y[idx, ::interv]
                        pred = pred[idx, ::interv]
                        pos = pos[idx]
                        cells = cells[idx]

                        visualization_cache['gt'].append(y)
                        visualization_cache['pred'].append(pred)
                        visualization_cache['coords'].append(pos)
                        visualization_cache['cells'].append(cells)
                        picked += y.shape[0]

                gt = torch.cat(visualization_cache['gt'], dim=0)
                pred = torch.cat(visualization_cache['pred'], dim=0)
                coords = torch.cat(visualization_cache['coords'], dim=0)
                cells = torch.cat(visualization_cache['cells'], dim=0)

                make_image_grid(gt[..., 0], coords, cells,
                                os.path.join(sample_dir, f'gt_vx_iter:{n_iter}_{j}.png'), nrow=gt.shape[1])

                make_image_grid(pred[..., 0], coords, cells,
                                os.path.join(sample_dir, f'pred_vx_iter:{n_iter}_{j}.png'), nrow=pred.shape[1])

                make_image_grid(gt[..., 1], coords, cells,
                                os.path.join(sample_dir, f'gt_vy_iter:{n_iter}_{j}.png'), nrow=gt.shape[1])
                make_image_grid(pred[..., 1], coords, cells,
                                os.path.join(sample_dir, f'pred_vy_iter:{n_iter}_{j}.png'), nrow=pred.shape[1])

                # make_image_grid(gt[..., 2], coords, cells,
                #                 os.path.join(sample_dir, f'gt_dns_iter:{n_iter}_{j}.png'), nrow=gt.shape[1])
                # make_image_grid(pred[..., 2], coords, cells,
                #                 os.path.join(sample_dir, f'pred_dns_iter:{n_iter}_{j}.png'), nrow=pred.shape[1])
                #
                # make_image_grid(gt[..., 3], coords, cells,
                #                 os.path.join(sample_dir, f'gt_prs_iter:{n_iter}_{j}.png'), nrow=gt.shape[1])
                # make_image_grid(pred[..., 3], coords, cells,
                #                 os.path.join(sample_dir, f'pred_prs_iter:{n_iter}_{j}.png'), nrow=pred.shape[1])

                #
                # del visualization_cache
                writer.add_scalar('testing avg loss', np.mean(all_avg_loss), global_step=n_iter)

                print(f'Testing avg loss (1e-4): {np.mean(all_avg_loss)*1e4}')
                print(f'Testing avg rmse (1e-3): {np.mean(all_rmse)*1e3}')

                logger.info(f'Current iteration: {n_iter}')
                logger.info(f'Testing avg loss (1e-4): {np.mean(all_avg_loss)*1e4}')
                logger.info(f'Testing avg rmse (1e-3): {np.mean(all_rmse)*1e3}')

                # save checkpoint if needed
                ckpt = {
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'n_iter': n_iter,
                    'enc_optim': enc_optim.state_dict(),
                    'enc_sched': enc_scheduler.state_dict(),
                }

                save_checkpoint(ckpt, os.path.join(checkpoint_dir, f'model_checkpoint{n_iter}.ckpt'))
                del ckpt
                if n_iter >= opt.iters:
                    break