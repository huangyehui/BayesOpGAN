import argparse
import os

import math

# import torchvision.transforms as transforms
# from torchvision.utils import save_image

from dataset_ruff import RuffDataset
# from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from bayesian_optimize import BayesianMode
import numpy as np
#from torchvision import models

os.makedirs("images", exist_ok=True)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_default_dtype(torch.float64)



parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=110000,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00002,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100,
                    help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=40,
                    help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1,
                    help="number of image channels")
parser.add_argument("--sample_interval", type=int,
                    default=400, help="interval betwen image samples")
parser.add_argument("--pretrain", type=bool, default=False,
                    help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)


cuda = True if torch.cuda.is_available() else False



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            # cancat
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        

        self.alpha = 0.5
        self.layer1 = nn.Sequential(*block(opt.latent_dim, 200, normalize=True))
        self.layer2 = nn.Sequential(*block(200, 400))
        self.layer3 = nn.Sequential(*block(400, 800))
        self.layer4 = nn.Sequential(*block(800, 1600))
        self.last = nn.Linear(1600, int(np.prod(img_shape)))
        self.active = nn.Softplus()

        self.upsample = nn.Upsample(scale_factor=2, mode='linear')


    def smooth_up_sample(self, z):
        z = z.view(z.size(0), 1, z.size(1))
        z = self.upsample(z)
        z = z.view(z.size(0), -1)
        return z

    def progress_grow(self, z, data):
        #print("原始{}", z)
        z = self.smooth_up_sample(z)
        #print("平滑{}", z)
        #print("放大{}", data)
        data = data * self.alpha + z *(1-self.alpha)
        
        #print("拼接{}", data)
        return data

    def forward(self, z):
        # 生成数据 [64, 8100]
        #data = self.model(z)

        data = self.layer1(z)
     
        z = data
      

        data = self.layer2(z)
        data = self.progress_grow(z, data)
        z = data


        data = self.layer3(z)
        data = self.progress_grow(z, data)
        z = data

        data = self.layer4(z)
        data = self.progress_grow(z, data)
        z = data
        data = self.last(data)
        data = self.active(data)

        img = data.view(data.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
        
            nn.Linear(1600, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.down_sample = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        out = self.model(img_flat)
        feature = out
        out = self.down_sample(feature)
        validity = self.sigmoid(out)
        return feature, validity


# Loss function
adversarial_loss = torch.nn.BCELoss()
generate_loss = torch.nn.MSELoss()
coordinate_x_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    coordinate_x_loss.cuda()
    

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataset = RuffDataset('dataset/Beryl')
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True
)

# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.DoubleTensor if cuda else torch.DoubleTensor

# ----------
#  Training
# ----------
if __name__ == '__main__': 
   
    epsilon = 0.1
    bayes_mode = BayesianMode()
    bayes_mode_gt = BayesianMode()
    bayesstack = {}
    bayesstack['suggest'] = []
    bayesstack['score'] = []
    bayesstack['gt_feature'] = []
    bayesstack['gt_score'] = []
    catch_path = './images/bayesgan/'
    f = open('./datas.txt', 'w')
    gen_loss = 100
    dis_loss = 100
    
    if opt.pretrain:
        generator.load_state_dict(torch.load('./model/model_gen.pkl'))
        discriminator.load_state_dict(torch.load('./model/model_dis.pkl'))

    for epoch in range(opt.n_epochs):
        for i, (imgs, coordinate_x) in enumerate(dataloader):
            print('imgs')
            print(imgs.size())
            print('coord_x')
            print(coordinate_x)
            print('========load data finish==========')
            input_x = torch.zeros(imgs.size(0)) + 200
            valid = Variable(Tensor(imgs.size(0), 1).fill_(
                1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0),
                            requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # 原来取随机数，改为贝叶斯最大值
            argmax_g_x = []
            argmax_gt_x = []
            # 获取缓存数据
            suggest = bayesstack['suggest']
            score = bayesstack['score']
            gt_feature = bayesstack['gt_feature']
            gt_score = bayesstack['gt_score']

            # Generate a batch of images
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(
                0, 1, (imgs.shape[0], opt.latent_dim))))
            gen_imgs = generator(z)
            fea, gen_imgs_res = discriminator(gen_imgs)

            #需要进一步分类
            print('coordinate_x:')
            print(coordinate_x.size())

            g_loss = adversarial_loss(gen_imgs_res, valid)
           # _, x_wave = torch.max(x_data, 1)

            bo_loss = 0
            # 计算两个分布之间的最大值差异
            if len(suggest) > 0 and len(score) > 0 and len(gt_feature) and len(gt_score):
                try:
                    # 生成样本最大值
                    gp = bayes_mode.fit(suggest, score)
                    argmax_g_x = bayes_mode.ac_func(gp)
                    # gt最大值

                    gp_gt = bayes_mode_gt.fit(gt_feature, valid)
                    argmax_gt_x = bayes_mode_gt.ac_func(gp_gt)
                  
                    bo_loss = F.mse_loss(argmax_g_x, argmax_gt_x)
                except Exception as e:
                    print("!!!!!get exception:{}".format(str(e)))

            #生成loss由三部分组成
            print('gloss:%f boloss%f'%(g_loss, bo_loss))
            gen_loss = (g_loss + 0.5 * bo_loss) 
            gen_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples

            feature, real_imgs_out = discriminator(real_imgs)
            real_loss = adversarial_loss(real_imgs_out, valid)
            gen_f, gen_validity = discriminator(gen_imgs.detach())
            fake_loss = adversarial_loss(gen_validity, fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # 生成器数据缓存
            bayesstack['suggest'] = gen_f.detach()
            bayesstack['score'] = gen_imgs_res.detach()
            # gt数据缓存
            bayesstack['gt_feature'] = feature.detach()
            bayesstack['gt_score'] = real_imgs_out.detach()
            str_bo_loss  = ''
            if bo_loss == 0:
                str_bo_loss = str(bo_loss)
            else:
                str_bo_loss =  str(bo_loss.item())

            info = "[double fea Epoch {}/{}] [Batch {}/{}] [D loss: {}] [G loss: {}] [tatol loss: {}] [bo loss: {}]  \n".format(
                epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), gen_loss.item(),str_bo_loss)
            f.write(info)
            print(info)
            
            dataloader_size = len(dataloader)
            batches_done = epoch * dataloader_size + i
            if batches_done % opt.sample_interval == 0:
                arr_raman = gen_imgs.view(gen_imgs.size(0), -1)

                for i in range(gen_imgs.size(0)):
                    filename = "./raman/%d_%d.txt" % (batches_done,i)
                    fout = open(filename, 'w')
                    
                    list_out = arr_raman[i].tolist()
                    str_raman = '\n'.join(map(str,list_out))#''.join(list_out) 
                    fout.write(str_raman)
                    fout.write('\n')
                   # fout.write(str(arr_x_data[i]) + '\n')

                if d_loss.item() < dis_loss and g_loss.item() < gen_loss:
                    gen_loss = g_loss.item()
                    dis_loss = d_loss.item()
                f.write("save:" + filename)
                torch.save(generator.state_dict(), "./out_model/" + str(epoch) + "_model_gen.pkl")
                torch.save(discriminator.state_dict(), "./out_model/" + str(epoch) + "_model_dis.pkl")
    f.close()
