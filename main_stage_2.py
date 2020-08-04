# -*- coding: utf-8 -*-
from main_stage_1 import Conv_k3, Upblock,G_Stage1, CondAugment_Model, D_output, weights_init
from main_stage_1 import KL_loss, cal_G_loss, cal_D_loss
from dataset import BirdDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.utils import make_grid

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class ResBlock(nn.Module):
    def __init__(self, plane):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            Conv_k3(plane, plane),
            nn.BatchNorm2d(plane),
            nn.ReLU(True),
            Conv_k3(plane, plane),
            nn.BatchNorm2d(plane)
        )
        self.relu = nn.ReLU(True)
        
    def forward(self, x):
        tmp = x
        o = self.block(x)
        o = o + tmp
        return self.relu(o)
    
class G_Stage2(nn.Module):
    def __init__(self, G_Stage1):
        super(G_Stage2, self).__init__()
        self.G1 = G_Stage1
        self.CA = CondAugment_Model()
        for p in self.G1.parameters():
            p.requires_grad = False
        self.encoder = nn.Sequential(
            Conv_k3(3, 128),
            nn.ReLU(True),
            nn.Conv2d(128, 128 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 2),
            nn.ReLU(True),
            nn.Conv2d(128 * 2, 128 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 4),
            nn.ReLU(True))
        self.combine = nn.Sequential(
            Conv_k3(640, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.residual = nn.Sequential(
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512)
        )
        self.decoder = nn.Sequential(
            Upblock(512,256),
            Upblock(256,128),
            Upblock(128,64),
            Upblock(64,32),
            Conv_k3(32,3),
            nn.Tanh()
        )
        
    def forward(self, noise, emb):
        init_image, _, _ = self.G1(noise, emb)
        encoded = self.encoder(init_image)
        
        cond, m, s = self.CA(emb)
        cond = cond.view(-1, 128, 1, 1)
        cond = cond.repeat(1, 1, 16, 16)
        
        encoded_cond = torch.cat([encoded, cond],1)
        img_feature = self.combine(encoded_cond)
        img_feature = self.residual(img_feature)
        img = self.decoder(img_feature)
        
        return init_image, img, m, s
        
class D_Stage2(nn.Module):
    def __init__(self):
        super(D_Stage2, self).__init__()
        self.img_encoder = nn.Sequential(
            # start 3 x 256 x 256
            nn.Conv2d(3, 128, 4, 2, 1, bias=False), #=> 128 x 128 x 128
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), #=> 256 x 64 x 64
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), #=> 512 x 32 x 32
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False), #=> 1024 x 16 x 16
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(1024, 2048, 4, 2, 1, bias=False), #=> 2048 x 8 x 8
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(2048, 4096, 4, 2, 1, bias=False), #=> 4096 x 4 x 4
            nn.BatchNorm2d(4096),
            nn.LeakyReLU(0.2, True),
            
            Conv_k3(4096, 2048), # 2048 x 4 x 4
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, True),
            Conv_k3(2048, 1024), # 1024 x 4 x 4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True)
        )
        
        self.condition_classifier = D_output()
        self.uncondition_classifier = D_output(have_cond=False)
        
    def forward(self, img):
        img_feature = self.img_encoder(img)
        return img_feature
    
def main():
    device = torch.device('cuda:0')
    # load dataset with size 256x256
    batch_size = 16
    transform = transforms.Compose([
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = BirdDataset(dataDir = '../data/bird_stack/', split='train', transform=transform, imgSize=256)
    tr_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    #load model Stage-I generator and put it into Stage-II generator
    G1 = G_Stage1()
    G1.load_state_dict(torch.load('./Result_stage1/netG_epoch_600.pth'))
    G1.eval()
    netG = G_Stage2(G1).to(device)
    netG.apply(weights_init)
    netD = D_Stage2().to(device)
    netD.apply(weights_init)

    lr = 0.0002
    optD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    # remove the parameter from Stage-I generator
    netG_param = []
    for p in netG.parameters():
        if p.requires_grad:
            netG_param.append(p)
    optG = optim.Adam(netG_param, lr=lr, betas=(0.5, 0.999))

    fixed_noise = torch.rand(batch_size, 100, 1, 1).to(device)

    real_labels = (torch.FloatTensor(batch_size).fill_(1)).to(device)
    fake_labels = (torch.FloatTensor(batch_size).fill_(0)).to(device)
    
    if not (os.path.isdir('./Result_stage2/')):
        os.makedirs(imageDir)
    num_epoch = 600
    iters = 0
    for epoch in range(num_epoch+1):
        if epoch % 100 == 0 and epoch > 0:
            lr = lr*0.5
            for param_group in optG.param_groups:
                param_group['lr'] = lr
            for param_group in optD.param_groups:
                param_group['lr'] = lr
        for i, data in enumerate(tr_loader,0):
            real_imgs, encoded_caps = data
            real_imgs = real_imgs.to(device)
            encoded_caps = encoded_caps.to(device)

            ##update discriminator
            netD.zero_grad()
            # generate fake image
            noise = torch.rand(batch_size, 100, 1, 1).to(device)
            init_img ,fake_imgs, m, s = netG(noise, encoded_caps)
            errD, errD_real, errD_wrong, errD_fake = cal_D_loss(netD, real_imgs, fake_imgs, real_labels, fake_labels, m)
            errD.backward()
            optD.step()

            ##update generator
            netG.zero_grad()
            errG = cal_G_loss(netD, fake_imgs, real_labels, m)
            errG += errG + KL_loss(m,s)
            errG.backward()
            optG.step()

            if i%50 == 0:
                 print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_D_R: %.4f\tLoss_D_W: %.4f\tLoss_D_F %.4f'
                      % (epoch, num_epoch, i, len(tr_loader),
                         errD.item(), errG.item(), errD_real, errD_wrong, errD_fake))
        if epoch%10==0:
            with torch.no_grad():
                _, fake, _, _  = netG(fixed_noise, encoded_caps)
                fig = plt.figure(figsize=(10,10))
                grid = make_grid(fake.detach().cpu(), nrow=8, normalize=True).permute(1,2,0).numpy()
                plt.imshow(grid)
                fig.savefig('./Result_stage2/epch-{}.png'.format(epoch))
        if epoch%25==0:
            torch.save(netG.state_dict(), './Result_stage2/netG2_epoch_{}.pth'.format(epoch))
    torch.save(netD.state_dict(), './Result_stage2/netD2_epoch_last.pth')

if __name__ == '__main__':
    main()