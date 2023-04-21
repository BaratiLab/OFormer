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

from nn_module.encoder_module import SpatialEncoder2D
from nn_module.decoder_module import PointWiseDecoder2DSimple


from loss_fn import rel_loss, pointwise_rel_l2norm_loss
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


# set flags / seeds
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)


def build_model(res) -> (SpatialEncoder2D, PointWiseDecoder2DSimple):
    # currently they are hard coded
    encoder = SpatialEncoder2D(
        3,   # a + xy coordinates
        96,
        256,
        4,
        6,
        res=res,
        use_ln=True
    )

    decoder = PointWiseDecoder2DSimple(
        256,
        1,
        scale=0.5,
        res=res
    )

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + \
                   sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return encoder, decoder


def make_image_grid(a: torch.Tensor, u_pred: torch.Tensor, u_gt: torch.Tensor, out_path,
                    nrow=12):
    b, h, w, c = u_pred.shape   # c = 1

    a = a.detach().cpu().squeeze(-1).numpy()
    u_pred = u_pred.detach().cpu().squeeze(-1).numpy()
    u_gt = u_gt.detach().cpu().squeeze(-1).numpy()

    fig = plt.figure(figsize=(8., 8.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(b*3//nrow, nrow),  # creates 8x8 grid of axes
                     )

    for ax, im_no in zip(grid, np.arange(b*3)):
        # Iterating over the grid returns the Axes.
        if im_no % 3 == 0:
            ax.imshow(a[im_no//3], cmap='coolwarm')
        elif im_no % 3 == 1:
            ax.imshow(u_pred[im_no//3], cmap='coolwarm')
        elif im_no % 3 == 2:
            ax.imshow(u_gt[im_no//3], cmap='coolwarm')

        ax.axis('equal')
        ax.axis('off')

    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


# adapted from Galerkin Transformer
def central_diff(x: torch.Tensor, h, resolution):
    # assuming PBC
    # x: (batch, n, feats), h is the step size, assuming n = h*w
    x = rearrange(x, 'b (h w) c -> b h w c', h=resolution, w=resolution)
    x = F.pad(x,
              (0, 0, 1, 1, 1, 1), mode='constant', value=0.)  # [b c t h+2 w+2]
    grad_x = (x[:, 1:-1, 2:, :] - x[:, 1:-1, :-2, :]) / (2*h)  # f(x+h) - f(x-h) / 2h
    grad_y = (x[:, 2:, 1:-1, :] - x[:, :-2, 1:-1, :]) / (2*h)  # f(x+h) - f(x-h) / 2h

    return grad_x, grad_y


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
        '--train_dataset_path', type=str, required=True, help='Path to dataset.'
    )
    parser.add_argument(
        '--test_dataset_path', type=str, required=True, help='Path to dataset.'
    )

    parser.add_argument(
        '--train_seq_num', type=int, default=1024, help='How many sequences in the training dataset.'
    )
    parser.add_argument(
        '--test_seq_num', type=int, default=100, help='How many sequences in the training dataset.'
    )
    parser.add_argument(
        '--resolution', type=int, default=141, help='The interval of when sample snapshots from sequence'
    )

    return parser


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


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

    ntrain = opt.train_seq_num
    ntest = opt.test_seq_num
    res = opt.resolution

    sub = int((421 - 1) / (res - 1))
    dx = 1./res
    # belowing code is copied from:
    # https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
    # Data is of the shape (number of samples, grid size, grid size)
    train_data = loadmat(train_data_path)
    x_train = train_data['coeff'][:ntrain, ::sub, ::sub][:, :res, :res]   # input: a(x)
    y_train = train_data['sol'][:ntrain, ::sub, ::sub][:, :res, :res]   # solution: u(x)

    del train_data
    test_data = loadmat(test_data_path)
    x_test = test_data['coeff'][-ntest:, ::sub, ::sub][:, :res, :res]  # input: a(x)
    y_test = test_data['sol'][-ntest:, ::sub, ::sub][:, :res, :res]  # solution: u(x)
    del test_data
    print(f'Data resolution: {x_train.shape[-1]}')

    x_train = torch.as_tensor(x_train.reshape(ntrain, res, res, 1), dtype=torch.float32)
    x_test = torch.as_tensor(x_test.reshape(ntest, res, res, 1), dtype=torch.float32)
    y_train = torch.as_tensor(y_train.reshape(ntrain, res, res, 1), dtype=torch.float32)
    y_test = torch.as_tensor(y_test.reshape(ntest, res, res, 1), dtype=torch.float32)

    gridx = torch.tensor(np.linspace(0, 1, res), dtype=torch.float32)
    gridx = gridx.reshape(1, res, 1, 1).repeat([1, 1, res, 1])
    gridy = torch.tensor(np.linspace(0, 1, res), dtype=torch.float32)
    gridy = gridy.reshape(1, 1, res, 1).repeat([1, res, 1, 1])
    grid = torch.cat((gridx, gridy), dim=-1).reshape(1, -1, 2)

    x_mean = x_train.mean(dim=0)
    x_std = x_train.std(dim=0) + 1e-5

    y_mean = y_train.mean(dim=0)
    y_std = y_train.std(dim=0) + 1e-5
    y_mean = rearrange(y_mean, 'h w c -> (h w) c')
    y_std = rearrange(y_std, 'h w c -> (h w) c')

    if use_cuda:
        grid = grid.cuda()
        x_mean, x_std = x_mean.cuda(), x_std.cuda()
        y_mean, y_std = y_mean.cuda(), y_std.cuda()

    train_dataloader = DataLoader(TensorDataset(x_train, y_train),
                                   batch_size=opt.batch_size,
                                   shuffle=True)
    test_dataloader = DataLoader(TensorDataset(x_test, y_test),
                                  batch_size=opt.batch_size,
                                  shuffle=False)

    # instantiate network
    print('Building network')
    encoder, decoder = build_model(res)
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
    shutil.copy('../../train_darcy.py', opt.log_dir)

    # create optimizers
    enc_optim = torch.optim.Adam(list(encoder.parameters()), lr=opt.lr, weight_decay=1e-4)
    dec_optim = torch.optim.Adam(list(decoder.parameters()), lr=opt.lr, weight_decay=1e-4)

    # enc_scheduler = torch.optim.lr_scheduler.StepLR(enc_optim, opt.iters//10, gamma=0.75, last_epoch=-1)
    # dec_scheduler = torch.optim.lr_scheduler.StepLR(dec_optim, opt.iters//10, gamma=0.75, last_epoch=-1)
    enc_scheduler = OneCycleLR(enc_optim, max_lr=opt.lr, total_steps=opt.iters,
                              div_factor=1e2,
                              pct_start=0.2,
                              final_div_factor=1e5,
                               )
    dec_scheduler = OneCycleLR(dec_optim, max_lr=opt.lr, total_steps=opt.iters,
                               div_factor=1e2,
                               pct_start=0.2,
                               final_div_factor=1e5,
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
        dec_optim.load_state_dict(ckpt['dec_optim'])

        enc_scheduler.load_state_dict(ckpt['enc_sched'])
        dec_scheduler.load_state_dict(ckpt['dec_sched'])
        print("last checkpoint restored")

    # now we start the main loop
    n_iter = start_n_iter

    # for loop going through dataset
    with tqdm(total=opt.iters) as pbar:
        pbar.update(n_iter)
        train_data_iter = iter(train_dataloader)

        while True:

            encoder.train()
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
            x, y = data

            if use_cuda:
                x, y = x.cuda(), y.cuda()

            # standardize

            x = (x - x_mean) / x_std
            x = rearrange(x, 'b h w c -> b (h w) c')
            y = rearrange(y, 'b h w c -> b (h w) c')

            input_pos = grid.repeat([x.shape[0], 1, 1])
            prop_pos = grid.repeat([x.shape[0], 1, 1])

            x = torch.cat((x, input_pos), dim=-1)   # concat coordinates as additional feature

            # randomly create some idx
            # input_idx = torch.as_tensor(np.random.choice(input_pos.shape[1], int(0.5*input_pos.shape[1]), replace=False)).view(1, -1).cuda()
            # #prop_idx = torch.as_tensor(np.random.choice(prop_pos.shape[1], int(0.5*prop_pos.shape[1]), replace=False)).view(1, -1).cuda()
            #
            # x = index_points(x, input_idx.repeat([x.shape[0], 1]))
            # input_pos = index_points(input_pos, input_idx.repeat([x.shape[0], 1]))

            # y = index_points(y, prop_idx.repeat([x.shape[0], 1]))
            # prop_pos = index_points(prop_pos, prop_idx.repeat([x.shape[0], 1]))

            prepare_time = time.time() - start_time
            # x_out = encoder.forward(x, input_pos)
            z = encoder.forward(x, input_pos)
            x_out = decoder.forward(z, prop_pos, input_pos)

            x_out = x_out * y_std + y_mean
            x_out = rearrange(x_out, 'b (h w) c -> b c h w', h=res)
            x_out = x_out[..., 1:-1, 1:-1].contiguous()
            x_out = F.pad(x_out, (1, 1, 1, 1), "constant", 0)
            x_out = rearrange(x_out, 'b c h w -> b (h w) c')

            pred_loss = pointwise_rel_l2norm_loss(x_out, y)
            gt_grad_x, gt_grad_y = central_diff(y, dx, res)
            pred_grad_x, pred_grad_y = central_diff(x_out, dx, res)
            deriv_loss = pointwise_rel_l2norm_loss(pred_grad_x, gt_grad_x) +\
                         pointwise_rel_l2norm_loss(pred_grad_y, gt_grad_y)

            loss = pred_loss + 1e-1*deriv_loss
            enc_optim.zero_grad()
            dec_optim.zero_grad()

            loss.backward()
            # with amp.scale_loss(loss, [enc_optim, dec_optim]) as scaled_loss:
            #     scaled_loss.backward()
            # print(torch.max(decoder.decoding_transformer.attn_module1.to_q.weight.grad))
            # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2.0)
            # torch.nn.utils.clip_grad_norm_(decoder.parameters(), 2.0)

            # Unscales gradients and calls
            enc_optim.step()
            dec_optim.step()

            enc_scheduler.step()
            dec_scheduler.step()

            # udpate tensorboardX
            writer.add_scalar('train_loss', loss, n_iter)
            writer.add_scalar('prediction_loss', pred_loss, n_iter)

            # compute computation time [None,...]and *compute_efficiency*
            process_time = time.time() - start_time - prepare_time

            pbar.set_description(
                f'Total loss (1e-4): {loss.item()*1e4:.1f}||'
                f'prediction (1e-4): {pred_loss.item()*1e4:.1f}||'
                f'derivative (1e-4): {deriv_loss.item()*1e4:.1f}||'
                f'Iters: {n_iter}/{opt.iters}')

            pbar.update(1)
            start_time = time.time()
            n_iter += 1

            if (n_iter-1) % opt.ckpt_every == 0 or n_iter >= opt.iters:
                logger.info('Tesing')
                print('Testing')

                encoder.eval()
                decoder.eval()

                with torch.no_grad():
                    all_avg_loss = []
                    all_acc_loss = []
                    visualization_cache = {
                        'in_seq': [],
                        'pred': [],
                        'gt': [],
                    }
                    picked = 0
                    for j, data in enumerate(tqdm(test_dataloader)):
                        # data preparation
                        x, y = data

                        if use_cuda:
                            x, y = x.cuda(), y.cuda()

                        # standardize
                        # data_mean = torch.mean(x, dim=1, keepdim=True)
                        # data_std = torch.std(x, dim=1, keepdim=True)
                        x = (x - x_mean) / x_std
                        x = rearrange(x, 'b h w c -> b (h w) c')
                        y = rearrange(y, 'b h w c -> b (h w) c')

                        input_pos = prop_pos = grid.repeat([x.shape[0], 1, 1])
                        #x, input_pos = pad_pbc(x, input_pos, dx)
                        x = torch.cat((x, input_pos), dim=-1)  # concat coordinates as additional feature

                        prepare_time = time.time() - start_time

                        z = encoder.forward(x, input_pos)
                        x_out = decoder.forward(z, prop_pos, input_pos)

                        x_out = x_out * y_std + y_mean
                        x_out = rearrange(x_out, 'b (h w) c -> b c h w', h=res)
                        x_out = x_out[..., 1:-1, 1:-1].contiguous()
                        x_out = F.pad(x_out, (1, 1, 1, 1), "constant", 0)
                        x_out = rearrange(x_out, 'b c h w -> b (h w) c')

                        avg_loss = pointwise_rel_l2norm_loss(x_out, y)
                        accumulated_mse = torch.nn.MSELoss(reduction='sum')(x_out, y) /   \
                                          (res**2 * x.shape[0])

                        all_avg_loss += [avg_loss.item()]
                        all_acc_loss += [accumulated_mse.item()]

                        # rescale
                        x = x[:, :, :1]

                        x = rearrange(x, 'b (h w) c -> b h w c', h=res, w=res) * x_std + x_mean
                        x_out = rearrange(x_out, 'b (h w) c -> b h w c', h=res, w=res)
                        y = rearrange(y, 'b (h w) c -> b h w c', h=res, w=res)

                        if picked < 24:
                            idx = np.arange(0, min(24 - picked, x.shape[0]))
                            # randomly pick a batch
                            x = x[idx]
                            y = y[idx]
                            x_out = x_out[idx]
                            visualization_cache['gt'].append(y)
                            visualization_cache['in_seq'].append(x)
                            visualization_cache['pred'].append(x_out)
                            picked += x.shape[0]

                all_gt = torch.cat(visualization_cache['gt'], dim=0)
                all_in_seq = torch.cat(visualization_cache['in_seq'], dim=0)
                all_pred = torch.cat(visualization_cache['pred'], dim=0)

                make_image_grid(all_in_seq, all_pred, all_gt,
                                os.path.join(sample_dir, f'result_iter:{n_iter}_{j}.png'))

                del visualization_cache
                writer.add_scalar('testing avg loss', np.mean(all_avg_loss), global_step=n_iter)

                print(f'Testing avg loss (1e-4): {np.mean(all_avg_loss)*1e4}')
                print(f'Testing accumulated mse loss (1e-4): {np.mean(all_acc_loss)*1e4}')

                logger.info(f'Current iteration: {n_iter}')
                logger.info(f'Testing avg loss (1e-4): {np.mean(all_avg_loss)*1e4}')
                logger.info(f'Testing accumulated mse loss (1e-4): {np.mean(all_acc_loss)*1e4}')

                # save checkpoint if needed
                ckpt = {
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'n_iter': n_iter,
                    'enc_optim': enc_optim.state_dict(),
                    'dec_optim': dec_optim.state_dict(),
                    'enc_sched': enc_scheduler.state_dict(),
                    'dec_sched': dec_scheduler.state_dict(),
                }

                save_checkpoint(ckpt, os.path.join(checkpoint_dir, f'model_checkpoint{n_iter}.ckpt'))
                del ckpt
                if n_iter >= opt.iters:
                    break