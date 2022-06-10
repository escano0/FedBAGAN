import os
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms
import torch.distributions.multivariate_normal as mn

from utils import batch2one
from dataset import load_data
from modules import Encoder, Decoder



def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--gf_dim', type=int, default=128)
    parser.add_argument('--df_dim', type=int, default=128)

    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--save_dir', type=str, default='exp')
    parser.add_argument('--pretrained_dir', type=str, default='exp/BestResult')
    parser.add_argument('--train_dir', type=str, default='train_dir')

    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    
    ### Initialize result directories and folders ###
    args.distribution_dir = os.path.join(args.save_dir, 'Distribution')
    os.makedirs(args.distribution_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize([32,32]),
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    if args.dataset == 'mnist':
        args.c_dim = 1
        args.class_num = 10
        train_loader = load_data(args.train_dir, transform, 'mnist', args)
    elif args.dataset == 'cifar10':
        args.c_dim = 3
        args.class_num = 10
        train_loader = load_data(args.train_dir, transform, 'cifar10', args)
    else:
        raise ValueError("Invalid Dataset. Must be one of [mnist, cifar10]")

    encoder = Encoder(args.z_dim, args.c_dim, args.df_dim).to(args.device)
    encoder.load_state_dict(torch.load(os.path.join(args.pretrained_dir, 'encoder_best.pth'), map_location=args.device), strict=True)
    encoder.eval()

    decoder = Decoder(args.z_dim, args.c_dim, args.gf_dim).to(args.device)
    decoder.load_state_dict(torch.load(os.path.join(args.pretrained_dir, 'decoder_best.pth'), map_location=args.device), strict=True)
    decoder.eval()

    Z = []

    with torch.no_grad():
        for i in range(args.class_num):
            Z.append(torch.zeros((1, args.z_dim), dtype=torch.float)) # Z : [class_num, z_dim]
        
        for iter, (image_ori, label) in enumerate(train_loader):
            image_ori = image_ori.to(args.device)

            mu, log_sigmoid = encoder(image_ori)

            std = torch.exp(log_sigmoid/2)
            eps = torch.randn_like(std)

            z = mu + eps * std
            z = z.view(-1, 1, args.z_dim)

            Z = batch2one(Z, label, z)

        N = []
        for i in range(args.class_num):
            label_mean = torch.mean(Z[i][1:], dim=0)
            label_cov = torch.from_numpy(np.cov(Z[i][1:].numpy(), rowvar=False)).type(torch.float)

            m = mn.MultivariateNormal(label_mean, label_cov)

            N.append(m)

        torch.save({'distribution': N}, os.path.join(args.distribution_dir, 'class_distribution.dt'))



if __name__ == '__main__':
	main()