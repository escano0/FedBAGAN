import torch
import torch.nn as nn



# Discriminator
class Encoder(nn.Module):
    def __init__(self, z_dim, c_dim, df_dim):
        super(Encoder, self).__init__()
        self.df_dim = df_dim

        self.conv0 = nn.Conv2d(c_dim, df_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu0 = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv2d(df_dim, df_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(df_dim*2)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(df_dim*2, df_dim*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(df_dim*4)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(df_dim*4, df_dim*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(df_dim*8)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.fc_z1 = nn.Linear(df_dim*8*2*2, z_dim)
        self.fc_z2 = nn.Linear(df_dim*8*2*2, z_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
    

    def forward(self, input):
        h0 = self.relu0(self.conv0(input)) # 3*32*32 -> 128*16*16
        h1 = self.relu1(self.bn1(self.conv1(h0))) # 256*8*8
        h2 = self.relu2(self.bn2(self.conv2(h1))) # 512*4*4
        h3 = self.relu3(self.bn3(self.conv3(h2))) # 1024*2*2
        
        mu = self.fc_z1(h3.view(-1, self.df_dim*8*2*2))	# (bs, 4096) -> (bs, 100)
        sigma = self.fc_z2(h3.view(-1, self.df_dim*8*2*2)) # (bs, 4096) -> (bs, 100)
        return mu, sigma




# Generator
class Decoder(nn.Module):
    def __init__(self, z_dim, c_dim, gf_dim):
        super(Decoder, self).__init__()
        self.convTrans0 = nn.ConvTranspose2d(z_dim, gf_dim*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(gf_dim*8)
        self.relu0 = nn.ReLU(inplace=True)
        
        self.convTrans1 = nn.ConvTranspose2d(gf_dim*8, gf_dim*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(gf_dim*4)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.convTrans2 = nn.ConvTranspose2d(gf_dim*4, gf_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(gf_dim*2)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.convTrans3 = nn.ConvTranspose2d(gf_dim*2, gf_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(gf_dim)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.convTrans4 = nn.ConvTranspose2d(gf_dim, c_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.tanh = nn.Tanh()
        
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
    

    def forward(self, z):
        h0 = self.relu0(self.bn0(self.convTrans0(z))) # 100*1*1 -> 1024*2*2
        h1 = self.relu1(self.bn1(self.convTrans1(h0))) # 1024*2*2 -> 512*4*4
        h2 = self.relu2(self.bn2(self.convTrans2(h1))) # 512*4*4 -> 256*8*8
        h3 = self.relu3(self.bn3(self.convTrans3(h2))) # 256*8*8 -> 128*16*16
        h4 = self.convTrans4(h3) # 128*16*16 -> 3*32*32
        output = self.tanh(h4)
        return output # 3*32*32        



class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, gf_dim):
        super(Generator, self).__init__()
        self.convTrans0 = nn.ConvTranspose2d(z_dim, gf_dim*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(gf_dim*8)
        self.relu0 = nn.ReLU(inplace=True)
        
        self.convTrans1 = nn.ConvTranspose2d(gf_dim*8, gf_dim*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(gf_dim*4)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.convTrans2 = nn.ConvTranspose2d(gf_dim*4, gf_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(gf_dim*2)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.convTrans3 = nn.ConvTranspose2d(gf_dim*2, gf_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(gf_dim)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.convTrans4 = nn.ConvTranspose2d(gf_dim, c_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.tanh = nn.Tanh()
    

    def forward(self, z):
        h0 = self.relu0(self.bn0(self.convTrans0(z))) # 100*1*1 -> 1024*2*2
        h1 = self.relu1(self.bn1(self.convTrans1(h0))) # 1024*2*2 -> 512*4*4
        h2 = self.relu2(self.bn2(self.convTrans2(h1))) # 512*4*4 -> 256*8*8
        h3 = self.relu3(self.bn3(self.convTrans3(h2))) # 256*8*8 -> 128*16*16
        h4 = self.convTrans4(h3) # 128*16*16 -> 3*32*32
        output = self.tanh(h4)
        return output # 3*32*32



class Discriminator(nn.Module):
    def __init__(self, class_num, c_dim, df_dim, ):
        super(Discriminator, self).__init__()
        self.df_dim = df_dim

        self.conv0 = nn.Conv2d(c_dim, df_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu0 = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv2d(df_dim, df_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(df_dim*2)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(df_dim*2, df_dim*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(df_dim*4)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(df_dim*4, df_dim*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(df_dim*8)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.fc_aux = nn.Linear(df_dim*8*2*2, class_num+1)
        self.softmax = nn.LogSoftmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
    

    def forward(self, input):
        h0 = self.relu0(self.conv0(input)) # 3*32*32 -> 128*16*16
        h1 = self.relu1(self.bn1(self.conv1(h0))) # 256*8*8
        h2 = self.relu2(self.bn2(self.conv2(h1))) # 512*4*4
        h3 = self.relu3(self.bn3(self.conv3(h2))) # 1024*2*2
        
        # self.fc_aux(h3.view(-1, self.df_dim*8*2*2)) # [bs, 4096]
        output = self.softmax(self.fc_aux(h3.view(-1, self.df_dim*8*2*2)))
        return h3, output
