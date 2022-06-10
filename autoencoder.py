import os
import time
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from utils import save_ae_checkpoint, print_ae_log, plot_ae_result, AverageMeter
from dataset import load_data
from modules import Encoder, Decoder



def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int, default=8, help='the batch size of training data')
    parser.add_argument('--num_workers', type=int, default=2, help='how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.')
    parser.add_argument('--max_epoches', type=int, default=50)
    parser.add_argument('--z_dim', type=int, default=100, help='the dimension of latent vector')
    parser.add_argument('--gf_dim', type=int, default=128)
    parser.add_argument('--df_dim', type=int, default=128)

    parser.add_argument('--base_lr', type=float, default=0.0002)

    parser.add_argument('--board_interval', type=int, default=50)
    parser.add_argument('--image_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=1000)

    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--save_dir', type=str, default='sample', help='the directory of generated data')
    parser.add_argument('--train_dir', type=str, default='train_dir', help='the directory of training data')

    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    cur_time = time.strftime('%Y%m%d_H%H%M%S', time.localtime())
    args.save_dir = os.path.join(args.save_dir, cur_time, 'autoencoder')

    ### Initialize result directories and folders ###
    os.makedirs(args.save_dir, exist_ok=True)
    args.trainpics_dir = os.path.join(args.save_dir, 'TrainPics')
    os.makedirs(args.trainpics_dir, exist_ok=True)
    args.checkpoints_dir = os.path.join(args.save_dir, 'CheckPoints')
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    args.best_checkpoints_dir = os.path.join(args.save_dir, 'BestResult')
    os.makedirs(args.best_checkpoints_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize([32,32]),
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    if args.dataset == 'mnist':
        args.c_dim = 1
        train_loader = load_data(args.train_dir, transform, 'mnist', args)
    elif args.dataset == 'cifar10':
        args.c_dim = 3
        train_loader = load_data(args.train_dir, transform, 'cifar10', args)
    else:
        raise ValueError("Invalid Dataset. Must be one of [mnist, cifar10]")

    encoder = Encoder(args.z_dim, args.c_dim, args.df_dim).to(args.device)
    decoder = Decoder(args.z_dim, args.c_dim, args.gf_dim).to(args.device)

    opt_enc = torch.optim.Adam(encoder.parameters(), lr=args.base_lr, betas=(0, 0.999))
    opt_dec = torch.optim.Adam(decoder.parameters(), lr=args.base_lr, betas=(0, 0.999))

    criterion = nn.MSELoss(reduction='mean')

    losses = AverageMeter()

    encoder.train()
    decoder.train()

    best_loss = None
    train_step = 0

    for epoch in range(args.max_epoches):
        for iter, (image_ori, label) in enumerate(train_loader):
            batch_size = image_ori.size(0)

            image_ori = image_ori.to(args.device)

            mu, log_sigmoid = encoder(image_ori)

            std = torch.exp(log_sigmoid/2)
            eps = torch.randn_like(std)

            z = mu + eps * std
            z = z.view(-1, args.z_dim, 1, 1)
            z = z.to(args.device)

            image_rec = decoder(z)

            loss = criterion(image_rec, image_ori)

            losses.update(loss.item())

            opt_enc.zero_grad()
            opt_dec.zero_grad()
            loss.backward()
            opt_enc.step()
            opt_dec.step()

            if (train_step+1) % args.board_interval == 0:
                print_ae_log(epoch+1, args.max_epoches, iter+1, len(train_loader), train_step+1, args.base_lr, losses)
            
            if (train_step+1) % args.image_interval == 0:
                fig = plot_ae_result(image_ori, image_rec)
                fig_dir = os.path.join(args.trainpics_dir, 'epoch_{:05d}_iter_{:05d}_step_{:05d}.png'.format(epoch, iter+1, train_step+1))
                fig.savefig(fig_dir)
                plt.close(fig)

            if (best_loss is None) or (loss < best_loss):
                best_loss = loss
                save_ae_checkpoint(encoder=encoder, decoder=decoder, args=args, epoch=epoch+1, is_best=True)
            if (train_step+1) % args.save_interval == 0:
                save_ae_checkpoint(encoder=encoder, decoder=decoder, args=args, epoch=epoch+1, is_best=False)

            train_step += 1

        print_ae_log(epoch+1, args.max_epoches, iter+1, len(train_loader), train_step+1, args.base_lr, losses)

        fig = plot_ae_result(image_ori, image_rec)
        fig_dir = os.path.join(args.trainpics_dir, 'epoch_{:05d}_iter_{:05d}_step_{:05d}.png'.format(epoch+1, iter+1, train_step+1))
        fig.savefig(fig_dir)
        plt.close(fig)

        if (best_loss is None) or (loss < best_loss):
            best_loss = loss
            save_ae_checkpoint(encoder=encoder, decoder=decoder, args=args, epoch=epoch+1, is_best=True)
        if (train_step+1) % args.save_interval == 0:
            save_ae_checkpoint(encoder=encoder, decoder=decoder, args=args, epoch=epoch+1, is_best=False)



if __name__ == '__main__':
	main()