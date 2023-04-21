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

from nn_module.encoder_module import IrregSpatialEncoder2D
from nn_module.decoder_module import IrregSpatialDecoder2D


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
from dataset_new import ElectroStatData
from scipy.interpolate import griddata

# set flags / seeds
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
seed = 7
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)


def build_model():
    # currently they are hard coded
    encoder = IrregSpatialEncoder2D(
        input_channels=11,   # the feature provided in the dataset
        in_emb_dim=64,
        out_chanels=96,      # vel(x, y) + prs + temp
        heads=1,
        depth=2,
        res=50,
        use_ln=False,
        emb_dropout=0.05
    )

    decoder = IrregSpatialDecoder2D(
        latent_channels=96,
        out_channels=3,      # the label provided in the dataset, potential (field will be calculated via autograd)
        res=50,
        scale=1,
        dropout=0.1,
    )


    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) +\
                      sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return encoder, decoder


def make_image_grid(image: torch.Tensor, coords: torch.tensor, pad_mask: torch.tensor,
                    out_path, nrow=25, resolution=50):
    b, n, c = image.shape
    image = rearrange(image, 'b n c -> (c b) n')
    image = image.detach().cpu().numpy()    # [b, n, c]
    coords = coords.detach().cpu().numpy()  # [b, n, 2]
    pad_mask = pad_mask.detach().cpu().numpy()  # [b, n]

    fig = plt.figure(figsize=(15., 12.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(b*c//nrow, nrow),  # creates 2x2 grid of axes
                     )

    for ax, im_no in zip(grid, np.arange(b*c)):
        data = image[im_no]
        # vmax = data.max()
        # vmin = data.min()

        mask = pad_mask[im_no % b]
        coords_ = coords[im_no % b][mask[:, 0]]
        mx, my = coords_[:, 0], coords_[:, 1]

        ax.scatter(mx, my, c=data)

        # f_x = mx.max() - mx.min()
        # f_y = my.max() - my.min()
        # resolution_x = (f_x * resolution * 2) / (f_x + f_y)
        # resolution_y = (f_y * resolution * 2) / (f_x + f_y)
        #
        # x = np.linspace(mx.min(), mx.max(), int(np.round(resolution_x)))
        # y = np.linspace(my.min(), my.max(), int(np.round(resolution_y)))
        #
        # xx, yy = np.meshgrid(x, y)
        # extent = (x.min(), x.max(), y.min(), y.max())

        # interpolation
        # grid_data = griddata(np.stack([mx, my], axis=1), data, (xx, yy), method="linear")
        # grid_data = np.nan_to_num(grid_data)
        # Iterating over the grid returns the Axes.
        # imsh = ax.imshow(
        #     grid_data,
        #     extent=extent,
        #     vmin=vmin,
        #     vmax=vmax,
        #     cmap='viridis',
        #     origin="lower",
        #     interpolation="bilinear",
        # )
        # cbar = fig.colorbar(imsh, ax=ax, shrink=0.7)
        # cbar.ax.tick_params(labelsize=45)
        ax.axis('off')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def pointwise_rel_loss(x, y, p=1):
    # assume x, y are unrolled and masked
    #   x, y [b*n, 1]
    assert x.shape == y.shape
    eps = 1e-5
    if p == 1:
        y_norm = 1#y.abs() + eps
        diff = (x-y).abs()
    else:
        y_norm = 1#y.pow(p) + eps
        diff = (x-y).pow(p)
    diff = diff / y_norm   # [b, c]
    diff = diff.mean()
    return diff


def unroll_sequence(x, pad_mask):
    # x [b, n, c]
    # pad_mask [b, n, 1]
    x = rearrange(x, 'b n c -> (b n) c')
    pad_mask = rearrange(pad_mask, 'b n 1 -> (b n)')
    x = x[pad_mask]
    return x.view([-1, x.shape[-1]])


def mse_loss(y, y_gt):
    loss_fn = nn.MSELoss()
    potential_loss = loss_fn(y[:, 0], y_gt[:, 0])
    field_loss = loss_fn(y[:, 1:], y_gt[:, 1:])

    return potential_loss, field_loss


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

    return parser


def autograd_x(u, x):
    """
    Input:
        pred: predicted points data, [B*N, 1]
        x: input points data, [B*N, 2]
    Return:
        grad_pred: gradient of pred w.r.t. x, [B*N, 2]
    """

    grad_pred = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                      create_graph=True)[0]
    # grad_pred_y = torch.autograd.grad(u, x[:, 1], grad_outputs=torch.ones_like(u),
    #                                   create_graph=True, retain_graph=True)[0]
    return -grad_pred[..., 0:1], -grad_pred[..., 1:]



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

    train_set = ElectroStatData(train_data_path)
    test_set = ElectroStatData(test_data_path)

    train_dataloader = DataLoader(
        train_set,
        batch_size=opt.batch_size,
        shuffle=True)

    test_dataloader = DataLoader(
        test_set,
        batch_size=opt.batch_size,
        shuffle=False)

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
    logger.info(f'Using seed: {seed}')
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
    shutil.copy('../../train_electro.py', opt.log_dir)

    # create optimizers
    enc_optim = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=opt.lr,
                                  weight_decay=1e-4, amsgrad=True)
    # enc_optim = torch.optim.AdamW(list(encoder.parameters()), lr=opt.lr,
    #                               weight_decay=1e-6, amsgrad=True)
    # enc_scheduler = torch.optim.lr_scheduler.StepLR(enc_optim, opt.iters//10, gamma=0.75, last_epoch=-1)
    # dec_scheduler = torch.optim.lr_scheduler.StepLR(dec_optim, opt.iters//10, gamma=0.75, last_epoch=-1)
    enc_scheduler = OneCycleLR(enc_optim, max_lr=opt.lr, total_steps=opt.iters,
                               div_factor=1e2,
                               pct_start=0.1,
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
            x, y, pos, pad_mask, bound_mask = data

            if use_cuda:
                x, y, pos, pad_mask, bound_mask = x.cuda(), y.cuda(), pos.cuda(), pad_mask.cuda(), bound_mask.cuda()

            input_pos = pos
            prop_pos = pos.clone()
            prop_pos.requires_grad = True

            z = encoder.forward(x, input_pos, pad_mask)
            pred = decoder.forward(z, prop_pos, input_pos, pad_mask, bound_mask)

            pot, field_x, field_y = pred[:, :, 0:1], pred[:, :, 1:2], pred[:, :, 2:]
            pot = unroll_sequence(pot, pad_mask)
            field_x = unroll_sequence(field_x, pad_mask)
            field_y = unroll_sequence(field_y, pad_mask)
            y = unroll_sequence(y, pad_mask)

            loss_1 = pointwise_rel_loss(pot, y[:, 0:1], p=2)
            loss_2 = pointwise_rel_loss(field_x, y[:, 1:2], p=2)
            loss_3 = pointwise_rel_loss(field_y, y[:, 2:3], p=2)
            pred_loss = loss_1 + (loss_2 + loss_3) * 1.0

            loss = pred_loss
            enc_optim.zero_grad()

            loss.backward()

            # with amp.scale_loss(loss, [enc_optim, dec_optim]) as scaled_loss:
            #     scaled_loss.backward()
            # print(torch.max(decoder.decoding_transformer.attn_module1.to_q.weight.grad))
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2.0)

            # Unscales gradients and calls
            enc_optim.step()

            enc_scheduler.step()
            # with torch.no_grad():
            #     x_out = torch.cat((pot, field_x, field_y), dim=-1)
            #     pot_los, field_loss = mse_loss(x_out, y)

            # udpate tensorboardX
            writer.add_scalar('train_loss', loss, n_iter)
            writer.add_scalar('prediction_loss', pred_loss, n_iter)
            writer.add_scalar('loss_scalar', loss_1, n_iter)
            writer.add_scalar('field_loss', loss_2+loss_3, n_iter)

            pbar.set_description(
                f'Total loss (1e-4): {loss.item()*1e4:.1f}||'
                # f'prediction (1e-4): {pred_loss.item()*1e4:.1f}||'
                # f'potential (1e-3): {pot_los.item()*1e3:.1f}||'
                # f'field (1e-3): {field_loss.item()*1e3:.1f}||'
                f'loss_scalar: {loss_1.item()*1e4:.1f}||'
                f'loss_field: {loss_2.item()*1e4 + loss_3.item()*1e4:.1f}||'
                
                f'lr: {enc_optim.param_groups[0]["lr"]:.1e}'
                f'Iters: {n_iter}/{opt.iters}')

            pbar.update(1)
            start_time = time.time()
            n_iter += 1

            if (n_iter-1) % opt.ckpt_every == 0 or n_iter >= opt.iters:
                logger.info('Tesing')
                print('Testing')

                encoder.eval()
                decoder.eval()

                all_avg_loss = []
                all_avg_pot_loss = []
                all_avg_field_loss = []
                visualization_cache = {
                    'pred': [],
                    'gt': [],
                    'coords': [],
                    'pad_mask': [],
                }
                picked = 0
                for j, data in enumerate(tqdm(test_dataloader)):
                    # data preparation
                    # x, y, x_mean, x_std = data

                    # data preparation
                    x, y, pos, pad_mask, bound_mask = data

                    if use_cuda:
                        x, y, pos, pad_mask, bound_mask = x.cuda(), y.cuda(), pos.cuda(), pad_mask.cuda(), bound_mask.cuda()

                    input_pos = pos
                    prop_pos = pos.clone()
                    prop_pos.requires_grad = True


                    z = encoder.forward(x, input_pos, pad_mask)
                    x_out = decoder.forward(z, prop_pos, input_pos, pad_mask, bound_mask)

                    # field_x, field_y = autograd_x(pot, prop_pos)
                    # field_x = field_x * scalex
                    # field_y = field_y * scaley
                    #
                    # x_out = torch.cat((pot, field_x, field_y), dim=-1)
                    pot, field_x, field_y = x_out[:, :, 0:1], x_out[:, :, 1:2], x_out[:, :, 2:]
                    pot_ = unroll_sequence(pot, pad_mask)
                    field_x = unroll_sequence(field_x, pad_mask)
                    field_y = unroll_sequence(field_y, pad_mask)

                    with torch.no_grad():
                        # x_out_unrolled = torch.cat((pot, field_x, field_y), dim=-1)
                        y_unrolled = unroll_sequence(y, pad_mask)
                        #
                        avg_loss_1 = nn.MSELoss()(pot_, y_unrolled[:, 0:1])
                        avg_loss_2 = nn.MSELoss()(field_x, y_unrolled[:, 1:2])
                        avg_loss_3 = nn.MSELoss()(field_y, y_unrolled[:, 2:3])
                        avg_loss = avg_loss_1 + avg_loss_2 + avg_loss_3
                        #
                        # avg_pot_loss, avg_field_loss = mse_loss(x_out_unrolled, y_unrolled)
                        avg_pot_loss = avg_loss_1
                        avg_field_loss = (avg_loss_2 + avg_loss_3) / 2.0

                        all_avg_loss += [avg_loss.item()]
                        all_avg_pot_loss += [avg_pot_loss.item()]
                        all_avg_field_loss += [avg_field_loss.item()]

                    if picked < 16:
                        idx = np.arange(0, min(16 - picked, y.shape[0]))
                        # randomly pick a batch
                        y = y[idx, :]
                        x_out = x_out[idx, :]
                        pos = pos[idx, :]
                        pad_mask = pad_mask[idx, :]

                        visualization_cache['gt'].append(y)
                        visualization_cache['pred'].append(x_out)
                        visualization_cache['coords'].append(pos)
                        visualization_cache['pad_mask'].append(pad_mask)
                        picked += y.shape[0]

                gt = torch.cat(visualization_cache['gt'], dim=0)
                pred = torch.cat(visualization_cache['pred'], dim=0)
                coords = torch.cat(visualization_cache['coords'], dim=0)
                pad_mask = torch.cat(visualization_cache['pad_mask'], dim=0)

                make_image_grid(gt, coords, pad_mask,
                                os.path.join(sample_dir, f'gt_iter:{n_iter}_{j}.png'), nrow=gt.shape[0])

                make_image_grid(pred, coords, pad_mask,
                                os.path.join(sample_dir, f'pred_iter:{n_iter}_{j}.png'), nrow=pred.shape[0])
                #
                # del visualization_cache
                writer.add_scalar('testing avg loss', np.mean(all_avg_loss), global_step=n_iter)

                print(f'Testing avg loss (1e-4): {np.mean(all_avg_loss)*1e4}')
                print(f'Testing avg potential (1e-3): {np.mean(all_avg_pot_loss)*1e3}')
                print(f'Testing avg field (1e-3): {np.mean(all_avg_field_loss)*1e3}')

                logger.info(f'Current iteration: {n_iter}')
                logger.info(f'Testing avg loss (1e-4): {np.mean(all_avg_loss)*1e4}')
                logger.info(f'Testing avg potential (1e-3): {np.mean(all_avg_pot_loss)*1e3}')
                logger.info(f'Testing avg field (1e-3): {np.mean(all_avg_field_loss)*1e3}')

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