import os
import torch
import argparse
from tqdm import tqdm
from utils import tensor2img
from modules import Generator



def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--label_generated', type=str, default='[0,1,2,3,4,5]')
    parser.add_argument('--number_generated', type=str, default='[10,10,10,10,10]')
    
    parser.add_argument('--c_dim', type=int, default=3)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--gf_dim', type=int, default=128)
    parser.add_argument('--df_dim', type=int, default=128)

    parser.add_argument('--save_dir', type=str, default='sample', help='the directory of generated data')
    parser.add_argument('--pretrained_dir', type=str, default='exp/BestResult')
    parser.add_argument('--distribution_path', type=str, default='exp/BestResult')

    args = parser.parse_args()
    return args




def main():
    args = parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    args.save_dir = os.path.join(args.save_dir, 'fed_bagan_generated')
    print("[*]Exporting the generation results at {}".format(args.save_dir))
    os.makedirs(args.save_dir, exist_ok=True)

    ##### distribution initialize #####
    data_distribution = torch.load(args.distribution_path)['distribution']

    ##### generator initialize #####
    generator = Generator(args.z_dim, args.c_dim, args.gf_dim).to(args.device)
    generator.load_state_dict(torch.load(os.path.join(args.pretrained_dir, 'generator_best.pth'), map_location=args.device), strict=True)
    generator.eval()

    label_generated = args.label_generated.split(',')
    number_generated = args.number_generated.split(',')

    assert len(label_generated) == len(number_generated), "[!]Please ensure the number of elements in label_generate and number_generated is same"

    for label in label_generated:
        label = int(label)

        num_generate = int(number_generated[label])
        print("[*]Generating {} label({}) images".format(num_generate, label))

        label_dir = os.path.join(args.save_dir, '{}'.format(label))
        os.makedirs(label_dir, exist_ok=True)

        conditional_z = data_distribution[label].sample((num_generate,))
        conditional_z = conditional_z.to(args.device)

        latent_code = conditional_z.view(-1, args.z_dim, 1, 1)
        
        for idx, code in enumerate(tqdm(latent_code)):
            code = torch.unsqueeze(code, 0)
            img_fake = generator(code)

            img_fake = tensor2img(img_fake[0])

            img_fake.save(os.path.join(label_dir, '{}.png'.format(idx)))


if __name__ == '__main__':
	main()