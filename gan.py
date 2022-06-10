import os
import math
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from utils import conditional_latent_generator, save_gan_checkpoint, print_gan_log, print_gan_acc, plot_gan_result, AverageMeter
from dataset import load_data
from modules import Generator, Discriminator



def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--max_epoches', type=int, default=50)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--c_dim', type=int, default=1)
    parser.add_argument('--gf_dim', type=int, default=128)
    parser.add_argument('--df_dim', type=int, default=128)

    parser.add_argument('--base_lr', type=float, default=0.0002)

    parser.add_argument('--board_interval', type=int, default=50)
    parser.add_argument('--image_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=1000)

    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--class_num', type=int, default=10)

    parser.add_argument('--save_dir', type=str, default='sample', help='the directory of generated data')
    parser.add_argument('--pretrained_dir', type=str, default='exp/BestResult')
    parser.add_argument('--distribution_path', type=str, default='exp/BestResult')
    parser.add_argument('--train_dir', type=str, default='train_dir', help='the directory of training data')

    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    args.save_dir = os.path.join(args.save_dir, 'adversarial')

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
        train_loader = load_data(args.train_dir, transform, 'mnist', args)
    elif args.dataset == 'cifar10':
        train_loader = load_data(args.train_dir, transform, 'cifar10', args)

    distribution = torch.load(args.distribution_path)['distribution']

    generator = Generator(args.z_dim, args.c_dim, args.gf_dim).to(args.device)
    generator.load_state_dict(torch.load(os.path.join(args.pretrained_dir, 'decoder_best.pth'), map_location=args.device), strict=True)

    discriminator = Discriminator(args.class_num, args.c_dim, args.df_dim).to(args.device)

    encoder_state = torch.load(os.path.join(args.pretrained_dir, 'encoder_best.pth'), map_location=args.device)
    del encoder_state['fc_z1.weight']
    del encoder_state['fc_z1.bias']
    del encoder_state['fc_z2.weight']
    del encoder_state['fc_z2.bias']
    encoder_state.update({'fc_aux.weight':discriminator.state_dict()['fc_aux.weight']})
    encoder_state.update({'fc_aux.bias':discriminator.state_dict()['fc_aux.bias']})
    
    discriminator.load_state_dict(encoder_state, strict=True)

    criterion = nn.NLLLoss()

    opt_gen = torch.optim.Adam(generator.parameters(), lr=args.base_lr, betas=(0, 0.999))
    opt_dis = torch.optim.Adam(discriminator.parameters(), lr=args.base_lr, betas=(0, 0.999))

    generator.train()
    discriminator.train()

    best_gen_loss = None
    gen_losses = AverageMeter()
    dis_losses = AverageMeter()
    train_step = 0

    real_label = torch.LongTensor(args.batch_size).to(args.device)
    fake_label = torch.LongTensor(args.batch_size).to(args.device)	

    for epoch in range(args.max_epoches):
        total_real = 0
        total_fake = 0
        correct_real = 0
        correct_fake = 0
        for iter, (img_real, label) in enumerate(train_loader):
            batch_size = img_real.size(0)

            ##### Update 'D' : max log(D(x)) + log(1-D(G(z))) #####
            fake_num = math.ceil(batch_size/args.class_num)
            conditional_z, z_label = conditional_latent_generator(distribution, args.class_num, batch_size)
            conditional_z = conditional_z.to(args.device)
            z_label = z_label.type(torch.LongTensor).to(args.device)

            label = label.long().squeeze() # "squeeze" : [batch, 1] --> [batch] ... e.g) [1,2,3,4...]		

            img_real = img_real.to(args.device)
            label = label.to(args.device)

            sample_feature, dis_real = discriminator(img_real)
            real_label.resize_(batch_size).copy_(label)	# "cpu" : gpu --> cpu // <<.data.cpu vs cpu>> // "resize_as" : get tensor size and resize 

            dis_loss_real = criterion(dis_real, real_label)
            
            noise = conditional_z[0:fake_num].view(-1, args.z_dim, 1, 1)

            fake_label.resize_(noise.shape[0]).fill_(args.class_num)	# fake_label = '(num_class)+1'

            img_fake = generator(noise)

            _, dis_fake = discriminator(img_fake.detach())
            dis_loss_fake = criterion(dis_fake, fake_label)

            dis_loss = dis_loss_real + dis_loss_fake
            dis_losses.update(dis_loss.item())

            opt_gen.zero_grad()
            opt_dis.zero_grad()
            dis_loss.backward()
            opt_dis.step()

            ##### Update 'G' : max log(D(G(z))) #####
            noise = conditional_z.view(-1, args.z_dim, 1, 1)
            img_fake = generator(noise)

            _, dis_fake = discriminator(img_fake.detach())

            gen_loss = criterion(dis_fake, z_label)
            gen_losses.update(gen_loss.item())
            
            opt_gen.zero_grad()
            opt_dis.zero_grad()
            gen_loss.backward()
            opt_gen.step()
            
            pred_real = torch.max(dis_real.data, 1)[1]
            pred_fake = torch.max(dis_fake.data, 1)[1]
            total_real += real_label.size(0)
            total_fake += z_label.size(0)
            correct_real += (pred_real == real_label).sum().item()
            correct_fake += (pred_fake == z_label).sum().item()

            if (train_step+1) % args.board_interval == 0:
                print_gan_log(epoch+1, args.max_epoches, iter+1, len(train_loader), train_step+1, args.base_lr, gen_losses, dis_losses)
            
            if (train_step+1) % args.image_interval == 0:
                fig = plot_gan_result(img_real, img_fake)
                fig_dir = os.path.join(args.trainpics_dir, 'epoch_{:05d}_iter_{:05d}_step_{:05d}.png'.format(epoch, iter+1, train_step+1))
                fig.savefig(fig_dir)
                plt.close(fig)

            if (best_gen_loss is None) or (gen_loss < best_gen_loss):
                best_gen_loss = gen_loss
                save_gan_checkpoint(generator=generator, discriminator=discriminator, args=args, epoch=epoch+1, is_best=True)
            if (train_step+1) % args.save_interval == 0:
                save_gan_checkpoint(generator=generator, discriminator=discriminator, args=args, epoch=epoch+1, is_best=False)

            train_step += 1

        print_gan_log(epoch+1, args.max_epoches, iter+1, len(train_loader), train_step+1, args.base_lr, gen_losses, dis_losses)

        fig = plot_gan_result(img_real, img_fake)
        fig_dir = os.path.join(args.trainpics_dir, 'epoch_{:05d}_iter_{:05d}_step_{:05d}.png'.format(epoch+1, iter+1, train_step+1))
        fig.savefig(fig_dir)
        plt.close(fig)

        if (best_gen_loss is None) or (gen_loss < best_gen_loss):
            best_gen_loss = gen_loss
            save_gan_checkpoint(generator=generator, discriminator=discriminator, args=args, epoch=epoch+1, is_best=True)
        if (train_step+1) % args.save_interval == 0:
            save_gan_checkpoint(generator=generator, discriminator=discriminator, args=args, epoch=epoch+1, is_best=False)

        real_acc = 100 * correct_real / total_real
        fake_acc = 100 * correct_fake / total_fake
        print_gan_acc(real_acc, fake_acc)





if __name__ == '__main__':
	main()