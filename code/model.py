import torch.nn as nn

CUDA = None
cond_dim = 128
df_dim = 96
gf_dim = 192
z_dim = 100
emb_dim = 1024

def Conv_k3(in_p, out_p, stride=1):
    return nn.Conv2d(in_p, out_p, kernel_size=3, stride=stride, padding=1, bias=False)

class Upblock(nn.Module):
    def __init__(self, inp, outp):
        super(Upblock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = Conv_k3(inp, outp)
        self.batch = nn.BatchNorm2d(outp)
        self.relu = nn.ReLU(True)
        
    def forward(self, x):
        o = self.up(x)
        o = self.relu(self.conv(o))
        o = self.batch(o)
        return o

class ResBlock(nn.Module):
    def __init__(self, plane):
        super(self, ResBlock).__init__()
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
        

class CondAugment_Model(nn.Module):
    def __init__(self):
        super(CondAugment_Model,self).__init__()
        self.fc = nn.Linear(in_features=emb_dim, out_features=cond_dim*2)
        self.relu = nn.ReLU(True)
        
    def convert(self, embed):
        x = self.relu(self.fc(embed))
        mean, sigma = x[:, : cond_dim], x[:, cond_dim:]
        return mean, sigma
    
    def forward(self, x):
        mean, sigma = self.convert(x)
        diag = torch.exp(sigma*0.5)
        if CUDA is not None:
            normal_dis = (torch.FloatTensor(diag.size()).normal_()).cuda()
        else:
            normal_dis = (torch.FloatTensor(diag.size()).normal_())
        condition = (diag*normal_dis)+mean
        return condition, mean, sigma
        
class G_Stage1(nn.Module):
    def __init__(self):
        super(G_Stage1, self).__init__()
        self.CA = CondAugment_Model()
        self.fc = nn.Sequential(
            nn.Linear(in_features=228, out_features=128*4*4, bias=False),
            nn.BatchNorm1d(128*4*4),
            nn.ReLU(inplace=True)
        )
        self.img = nn.Sequential(
            Upblock(128,64),
            Upblock(64,32),
            Upblock(32,16),
            Upblock(16,3),
            nn.Tanh()
        )
        
    def forward(self, noise, emb):
        cond, mean, sigma = self.CA(emb)
        cond = cond.view(noise.size(0), cond.shape[1], 1, 1)
        x = torch.cat((noise, cond),1)
        x = x.view(-1, 228)
        o = self.fc(x)
        h_code = o.view(-1, 128, 4, 4)
        fake_img = self.img(h_code)
        return fake_img, mean, sigma