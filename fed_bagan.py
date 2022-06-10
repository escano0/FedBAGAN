import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import math
import argparse
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

from modules import Generator, Discriminator
from dataset import get_dataset, DatasetSplit, DatasetValidate
from utils import conditional_latent_generator, save_fed_checkpoint, plot_fed_result, AverageMeter



def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dis_num', type=int, default=5)
    parser.add_argument('--local_bs', type=int, default=64)
    parser.add_argument('--validate_bs', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--local_maxepoch', type=int, default=10)
    parser.add_argument('--global_maxepoch', type=int, default=50)

    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--gf_dim', type=int, default=128)
    parser.add_argument('--df_dim', type=int, default=128)

    parser.add_argument('--iid', type=int, default=1)
    parser.add_argument('--unequal', type=int, default=0)
    parser.add_argument('--num_users', type=int, default=20)
    parser.add_argument('--frac', type=float, default=0.5)
    parser.add_argument('--base_lr', type=float, default=0.0002)

    parser.add_argument('--loss', type=str, default='l2')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--optimizer', type=str, default='adam', help='')

    parser.add_argument('--save_dir', type=str, default='sample', help='the directory of generated data')
    parser.add_argument('--pretrained_dir', type=str, default='exp/BestResult')
    parser.add_argument('--distribution_path', type=str, default='exp/BestResult')
    parser.add_argument('--train_dir', type=str, default='train_dir', help='the directory of training data')

    args = parser.parse_args()
    return args



def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = w[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg



class LocalUpdate(object):
    def __init__(self, args, distribution, dataset, idxs, criterion):
        self.args = args
        self.distribution = distribution
        self.dataloader = DataLoader(
            DatasetSplit(dataset, idxs), 
            batch_size=self.args.local_bs, 
            shuffle=True
        )
        self.criterion = criterion

    def update_weights(self, model_gen, model_dis, global_epoch):
        
        model_gen.train()
        model_dis.train()

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            opt_gen = torch.optim.SGD(model_gen.parameters(), lr=self.args.base_lr, momentum=0.5)
            opt_dis = torch.optim.SGD(model_dis.parameters(), lr=self.args.base_lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            opt_gen = torch.optim.Adam(model_gen.parameters(), lr=self.args.base_lr, weight_decay=1e-4)
            opt_dis = torch.optim.Adam(model_dis.parameters(), lr=self.args.base_lr, weight_decay=1e-4)
        else:
            raise ValueError("Invalid Optimizer Model. Must be one of [sgd, adam]")
        
        gen_losses = AverageMeter()
        dis_losses = AverageMeter()

        real_label = torch.LongTensor(self.args.local_bs).to(self.args.device)
        fake_label = torch.LongTensor(self.args.local_bs).to(self.args.device)

        for local_epoch in range(self.args.local_maxepoch):
            
            for iter, (img_real, label) in enumerate(self.dataloader):
                batch_size = img_real.size(0)

                ##### Update 'D' : max log(D(x)) + log(1-D(G(z))) #####
                fake_num = math.ceil(batch_size/self.args.class_num)
                conditional_z, z_label = conditional_latent_generator(self.distribution, self.args.class_num, batch_size)
                conditional_z = conditional_z.to(self.args.device)
                z_label = z_label.type(torch.LongTensor).to(self.args.device)

                label = label.squeeze() # "squeeze" : [batch, 1] --> [batch] ... e.g) [1,2,3,4...]		

                img_real = img_real.to(self.args.device)
                label = label.to(self.args.device)

                _, dis_real = model_dis(img_real)
                real_label.resize_(batch_size).copy_(label)	# "cpu" : gpu --> cpu // <<.data.cpu vs cpu>> // "resize_as" : get tensor size and resize 

                dis_loss_real = self.criterion(dis_real, real_label)
                
                noise = conditional_z[0:fake_num].view(-1, self.args.z_dim, 1, 1)

                fake_label.resize_(noise.shape[0]).fill_(self.args.class_num)	# fake_label = '(num_class)+1'

                img_fake = model_gen(noise)

                _, dis_fake = model_dis(img_fake.detach())
                dis_loss_fake = self.criterion(dis_fake, fake_label)

                dis_loss = dis_loss_real + dis_loss_fake
                dis_losses.update(dis_loss.item())

                opt_gen.zero_grad()
                opt_dis.zero_grad()
                dis_loss.backward()
                opt_dis.step()

                ##### Update 'G' : max log(D(G(z))) #####
                noise = conditional_z.view(-1, self.args.z_dim, 1, 1)
                img_fake = model_gen(noise)

                _, dis_fake = model_dis(img_fake.detach())

                gen_loss = self.criterion(dis_fake, z_label)
                gen_losses.update(gen_loss.item())
                
                opt_gen.zero_grad()
                opt_dis.zero_grad()
                gen_loss.backward()
                opt_gen.step()
            
            print('| Global Epoch: {} | Local Epoch: {} | Generator Loss: {:.6f} | Discriminator Loss: {:.6f}'.format(global_epoch, local_epoch, gen_losses.avg, dis_losses.avg))

            ##### visulize example image #####
            conditional_z, z_label = conditional_latent_generator(self.distribution, self.args.class_num, self.args.dis_num)
            conditional_z = conditional_z.to(self.args.device)
            z_label = z_label.type(torch.LongTensor).to(self.args.device)

            noise = conditional_z.view(-1, self.args.z_dim, 1, 1)
            img_fake = model_gen(noise)

            fig = plot_fed_result(image=img_fake, label=z_label)
            fig_dir = os.path.join(self.args.trainpics_dir, 'GlobalEpoch_{:05d}_LocalEpoch_{:05d}.png'.format(global_epoch, local_epoch))
            fig.savefig(fig_dir)
            plt.close(fig)

        return model_gen.state_dict(), model_dis.state_dict(), gen_losses.avg, dis_losses.avg



def validation(args, model_gen, model_dis, dataset, distributon, criterion):
    model_gen.eval()
    model_dis.eval()

    dataloader = DataLoader(
        DatasetValidate(dataset),
        batch_size=args.validate_bs, 
        shuffle=False
    )

    real_label = torch.LongTensor(args.validate_bs).to(args.device)
    fake_label = torch.LongTensor(args.validate_bs).to(args.device)

    validate_loss = AverageMeter()

    with torch.no_grad():

        for img_real, label in dataloader:
            batch_size = img_real.size(0)
            
            fake_num = math.ceil(batch_size/args.class_num)

            conditional_z, z_label = conditional_latent_generator(distributon, args.class_num, batch_size)
            conditional_z = conditional_z.to(args.device)

            z_label = z_label.type(torch.LongTensor).to(args.device)

            label = label.squeeze() # "squeeze" : [batch, 1] --> [batch] ... e.g) [1,2,3,4...]		

            img_real = img_real.to(args.device)
            label = label.to(args.device)

            _, dis_real = model_dis(img_real)
            real_label.resize_(batch_size).copy_(label)	# "cpu" : gpu --> cpu // <<.data.cpu vs cpu>> // "resize_as" : get tensor size and resize 

            dis_loss_real = criterion(dis_real, real_label)
                    
            noise = conditional_z[0:fake_num].view(-1, args.z_dim, 1, 1)

            fake_label.resize_(noise.shape[0]).fill_(args.class_num)	# fake_label = '(num_class)+1'

            img_fake = model_gen(noise)

            _, dis_fake = model_dis(img_fake.detach())
            dis_loss_fake = criterion(dis_fake, fake_label)

            validate_loss.update(dis_loss_real.item() + dis_loss_fake.item())
    
    return validate_loss.avg



def main():
    args = parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    train_dataset, test_dataset, user_groups = get_dataset(args)

    args.save_dir = os.path.join(args.save_dir, 'fed_bagan')

    ### Initialize result directories and folders ###
    os.makedirs(args.save_dir, exist_ok=True)
    args.trainpics_dir = os.path.join(args.save_dir, 'TrainPics')
    os.makedirs(args.trainpics_dir, exist_ok=True)
    args.checkpoints_dir = os.path.join(args.save_dir, 'CheckPoints')
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    args.best_checkpoints_dir = os.path.join(args.save_dir, 'BestResult')
    os.makedirs(args.best_checkpoints_dir, exist_ok=True)
    args.log_dir = os.path.join(args.save_dir, 'Logs')
    os.makedirs(args.log_dir, exist_ok=True)

    ### Initialize logger ###
    logger = SummaryWriter(log_dir=args.log_dir)

    data_distribution = torch.load(args.distribution_path)['distribution']

    if args.dataset == 'mnist':
        args.c_dim = 1
        args.class_num = 10
    elif args.dataset == 'cifar10':
        args.c_dim = 3
        args.class_num = 10
    else:
        raise ValueError("Invalid Dataset. Must be one of [mnist, cifar10]")

    ##### generator initialize #####
    generator = Generator(args.z_dim, args.c_dim, args.gf_dim).to(args.device)
    generator.load_state_dict(torch.load(os.path.join(args.pretrained_dir, 'decoder_best.pth'), map_location=args.device), strict=True)

    ##### discriminator initialize #####
    discriminator = Discriminator(args.class_num, args.c_dim, args.df_dim).to(args.device)
    encoder_state = torch.load(os.path.join(args.pretrained_dir, 'encoder_best.pth'), map_location=args.device)
    del encoder_state['fc_z1.weight']
    del encoder_state['fc_z1.bias']
    del encoder_state['fc_z2.weight']
    del encoder_state['fc_z2.bias']
    encoder_state.update({'fc_aux.weight':discriminator.state_dict()['fc_aux.weight']})
    encoder_state.update({'fc_aux.bias':discriminator.state_dict()['fc_aux.bias']})
    discriminator.load_state_dict(encoder_state, strict=True)

    if args.loss == 'l2':
        criterion = nn.MSELoss()
    elif args.loss == 'nll':
        criterion = nn.NLLLoss()
    elif args.loss == 'cel':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid Loss Function. Must be one of [l2, nll, cel]")
    
    best_loss = None
    gen_local_loss = AverageMeter()
    dis_local_loss = AverageMeter()

    for global_epoch in range(args.global_maxepoch):
        gen_local_weight, dis_local_weight  = [], []

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local = LocalUpdate(args=args, distribution=data_distribution, dataset=train_dataset, idxs=user_groups[idx], criterion=criterion)
            gen_weight, dis_weight, gen_loss, dis_loss = local.update_weights(model_gen=generator, model_dis=discriminator, global_epoch=global_epoch)

        gen_local_weight.append(gen_weight)
        dis_local_weight.append(dis_weight)

        gen_local_loss.update(gen_loss)
        dis_local_loss.update(dis_loss)

        ##### update global weights #####
        gen_global_weight = average_weights(gen_local_weight)
        dis_global_weight = average_weights(dis_local_weight)

        generator.load_state_dict(gen_global_weight, strict=True)
        discriminator.load_state_dict(dis_global_weight, strict=True)

        # print global training loss after every 'i' rounds
        print('After {} epoch global training, averged local generator loss is: {:.4f}, averged local discrimitor loss is: {:.4f}'.format(global_epoch+1, gen_local_loss.avg, dis_local_loss.avg))
        
        validate_loss = validation(args, model_gen=generator, model_dis=discriminator, dataset=test_dataset, distributon=data_distribution, criterion=criterion)
        print('Averged validation loss is: {:.4f}'.format(validate_loss))

        if (best_loss is None) or (validate_loss < best_loss):
            best_loss = validate_loss
            save_fed_checkpoint(generator=generator, discriminator=discriminator, args=args, epoch=global_epoch+1, is_best=True)

            ##### visulize best resultd #####
            conditional_z, z_label = conditional_latent_generator(data_distribution, args.class_num, args.dis_num)
            conditional_z = conditional_z.to(args.device)

            noise = conditional_z.view(-1, args.z_dim, 1, 1)
            img_fake = generator(noise)

            fig = plot_fed_result(image=img_fake, label=z_label)
            fig_dir = os.path.join(args.best_checkpoints_dir, 'BestResults.png')
            fig.savefig(fig_dir)
            plt.close(fig)

        else:
            save_fed_checkpoint(generator=generator, discriminator=discriminator, args=args, epoch=global_epoch+1, is_best=False)
        
        logger.add_scalar("train/gen_local_loss", gen_local_loss.avg, global_epoch)
        logger.add_scalar("train/dis_local_loss", dis_local_loss.avg, global_epoch)
        logger.add_scalar("validation/validation_loss", validate_loss, global_epoch)



if __name__ == '__main__':
	main()