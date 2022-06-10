import os
from PIL import Image
import matplotlib.pyplot as plt

import torch



def batch2one(Z, y, z):
    for i in range(y.shape[0]):
        Z[y[i]] = torch.cat((Z[y[i]], z[i].cpu()), dim=0) # Z[label][0] should be deleted..
    return Z		



def conditional_latent_generator(distribution, class_num, batch):
    class_labels = torch.randint(0, class_num, (batch,), dtype=torch.int)
    fake_z = distribution[class_labels[0].item()].sample((1,))
    for c in class_labels[1:]:
        fake_z = torch.cat((fake_z, distribution[c.item()].sample((1,))), dim=0)
    return fake_z, class_labels



def save_ae_checkpoint(encoder, decoder, args, epoch, is_best):
    if is_best:
        torch.save(encoder.state_dict(), os.path.join(args.best_checkpoints_dir, 'encoder_best.pth'))
        torch.save(decoder.state_dict(), os.path.join(args.best_checkpoints_dir, 'decoder_best.pth'))
    else:
        torch.save(encoder.state_dict(), os.path.join(args.checkpoints_dir, 'encoder_epoch{:05d}.pth'.format(epoch)))
        torch.save(decoder.state_dict(), os.path.join(args.checkpoints_dir, 'decoder_epoch{:05d}.pth'.format(epoch)))



def save_gan_checkpoint(generator, discriminator, args, epoch, is_best):
    if is_best:
        torch.save(generator.state_dict(), os.path.join(args.best_checkpoints_dir, 'generator_best.pth'))
        torch.save(discriminator.state_dict(), os.path.join(args.best_checkpoints_dir, 'discriminator_best.pth'))
    else:
        torch.save(generator.state_dict(), os.path.join(args.checkpoints_dir, 'generator_epoch{:05d}.pth'.format(epoch)))
        torch.save(discriminator.state_dict(), os.path.join(args.checkpoints_dir, 'discriminator_epoch{:05d}.pth'.format(epoch)))



def save_fed_checkpoint(generator, discriminator, args, epoch, is_best):
    if is_best:
        torch.save(generator.state_dict(), os.path.join(args.best_checkpoints_dir, 'generator_best.pth'))
        torch.save(discriminator.state_dict(), os.path.join(args.best_checkpoints_dir, 'discriminator_best.pth'))
    else:
        torch.save(generator.state_dict(), os.path.join(args.checkpoints_dir, 'generator_epoch{:05d}.pth'.format(epoch)))
        torch.save(discriminator.state_dict(), os.path.join(args.checkpoints_dir, 'discriminator_epoch{:05d}.pth'.format(epoch)))



def print_ae_log(epoch, epoches, iter, iters, train_step, learning_rate, losses):
    print('epoch: [{}/{}] iteration: [{}/{}] step: {} Learning rate: {}'.format(epoch, epoches, iter, iters, train_step, learning_rate))
    print('Loss = {loss.val:.4f} (ave = {loss.avg:.4f})\n'.format(loss=losses))



def print_gan_log(epoch, epoches, iter, iters, train_step, learning_rate, dis_losses, gen_losses):
    print('epoch: [{}/{}] iteration: [{}/{}] step: {} Learning rate: {}'.format(epoch, epoches, iter, iters, train_step, learning_rate))
    print('generator_loss = {gen_loss.val:.4f} (ave = {gen_loss.avg:.4f})'.format(gen_loss=gen_losses))
    print('discriminator_loss = {dis_loss.val:.4f} (ave = {dis_loss.avg:.4f})\n'.format(dis_loss=dis_losses))



def print_gan_acc(real_acc, fake_acc):
   print('Real Accuracy : {} Fake Accuracy : {}'.format(real_acc, fake_acc))



def tensor2img(var):
    # var: 3 x 256 x 256 --> 256 x 256 x 3
    var = var.cpu().detach().numpy().transpose([1,2,0])
    #var = ((var+1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))



def plot_ae_result(image_ori, image_rec):
    dis_num = 4#len(image_ori)

    fig = plt.figure(figsize=(8, 4*dis_num))
    gs = fig.add_gridspec(nrows=dis_num, ncols=2)

    for img_idx in range(dis_num):
        fig.add_subplot(gs[img_idx, 0])
        img_ori = tensor2img(image_ori[img_idx])
        plt.imshow(img_ori)
        plt.title('Original Image')

        fig.add_subplot(gs[img_idx, 1])
        img_rec = tensor2img(image_rec[img_idx])
        plt.imshow(img_rec)
        plt.title('Reconstrcted Image')

    plt.tight_layout()
    return fig



def plot_gan_result(image_real, image_fake):
    dis_num = 4#len(image_ori)

    fig = plt.figure(figsize=(8, 4*dis_num))
    gs = fig.add_gridspec(nrows=dis_num, ncols=2)

    for img_idx in range(dis_num):
        fig.add_subplot(gs[img_idx, 0])
        img_real = tensor2img(image_real[img_idx])
        plt.imshow(img_real)
        plt.title('Real Image')

        fig.add_subplot(gs[img_idx, 1])
        img_fake = tensor2img(image_fake[img_idx])
        plt.imshow(img_fake)
        plt.title('Fake Image')

    plt.tight_layout()
    return fig



def plot_fed_result(image, label):
    img_num = len(image)

    fig = plt.figure(figsize=(4*img_num, 8))
    gs = fig.add_gridspec(nrows=1, ncols=img_num)

    for img_idx in range(img_num):

        fig.add_subplot(gs[0, img_idx])
        img_fake = tensor2img(image[img_idx])
        img_label = label[img_idx]
        plt.imshow(img_fake)
        plt.title('Fake Image(label: {})'.format(img_label))

    plt.tight_layout()
    return fig



class AverageMeter(object):
    """ Computes ans stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count