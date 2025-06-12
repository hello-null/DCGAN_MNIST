import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import time
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
plt.switch_backend('tkagg')
from torchvision.utils import make_grid
from torch.optim import Adam
from torchsummary import summary

'''
《UNSUPERVISED REPRESENTATION LEARNING
WITH DEEP CONVOLUTIONAL
GENERATIVE ADVERSARIAL NETWORKS》
https://github.com/XavierJiezou/pytorch-dcgan-mnist
'''


ROOT=r'F:\NeuralNetworkModel\DCGAN_MNIST\RUN_1'

'''
RUN_1是一个文件夹，下面包含D文件夹、G文件夹、fake_imgs文件夹

RUN_1
|————/D
|————/G
|————/fake_imgs
'''


# 超参数设置
class Params:
    # Root directory for dataset
    dataroot = r'H:\datasets\MNIST'
    # Number of workers for dataloader
    workers = 1
    # Batch size during training
    batch_size = 1024
    # Spatial size of training images. All images will be resized to this size using a transformer.
    image_size = 64
    # Number of channels in the training images. For color images this is 3
    nc = 1
    # Size of z latent vector (i.e. size of generator input)
    nz = 100
    # Size of feature maps in generator
    ngf = 64
    # Size of feature maps in discriminator
    ndf = 64
    # Number of training epochs
    num_epochs = 200
    # Learning rate for optimizers
    lr = 2e-4
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1
    def __init__(self):
        super(Params, self).__init__()




train_data = datasets.MNIST(
    root=Params.dataroot,
    train=True,
    transform=transforms.Compose([
        transforms.Resize(Params.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    download=False
)
test_data = datasets.MNIST(
    root=Params.dataroot,
    train=False,
    transform=transforms.Compose([
        transforms.Resize(Params.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    download=False
)



dataloader = DataLoader(
    dataset=train_data,
    batch_size=Params.batch_size,
    shuffle=True,
    num_workers=Params.workers
)

device = torch.device('cuda:0' if (torch.cuda.is_available() and Params.ngpu > 0) else 'cpu')


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z 100x1x1 , going into a convolution
            nn.ConvTranspose2d(Params.nz, Params.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(Params.ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(Params.ngf * 8, Params.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Params.ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(Params.ngf * 4, Params.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Params.ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(Params.ngf * 2, Params.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Params.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(Params.ngf, Params.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
    def forward(self, input):
        return self.main(input)
    def init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.ConvTranspose2d):
                init.normal_(m.weight,0.0,0.02)
            elif isinstance(m,nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(Params.nc, Params.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(Params.ndf, Params.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Params.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(Params.ndf * 2, Params.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Params.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(Params.ndf * 4, Params.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Params.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(Params.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)
    def init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                init.normal_(m.weight,0.0,0.02)
            elif isinstance(m,nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':

    netG = Generator().to(device)
    netG.apply(weights_init)

    netD = Discriminator().to(device)
    netD.apply(weights_init)

    criterion = nn.BCELoss()

    optimizerD = Adam(netD.parameters(), lr=Params.lr, betas=(Params.beta1, 0.999))
    optimizerG = Adam(netG.parameters(), lr=Params.lr, betas=(Params.beta1, 0.999))

    for epoch in range(Params.num_epochs):

        tm_start = time.time()

        for data in tqdm(dataloader,desc='epoch={}/{}'.format(epoch+1,Params.num_epochs)):
            # torch.Size([32, 1, 64, 64]) torch.Size([32])

            # TODO 向鉴别器展示一个真实的数据样本，告诉它该样本的分类应该是1.0。
            optimizerD.zero_grad()
            real_data = data[0].to(device)
            b_size = real_data.size(0)
            label = torch.full((b_size,), 0.95, dtype=torch.float, device=device) # torch.Size([32])
            output = netD(
                real_data
            ).view(-1) # torch.Size([32])
            loss1 = criterion(output, label)
            loss1.backward()

            # TODO 向鉴别器显示一个生成器的输出，告诉它该样本的分类应该是0.0。
            noise = torch.randn((b_size, Params.nz, 1, 1), device=device) # torch.Size([32, 100, 1, 1])
            fake = netG(noise) # torch.Size([32, 1, 64, 64])
            label.fill_(0.05) # torch.Size([32])
            output = netD(
                fake.detach()
            ).view(-1)
            loss2 = criterion(output, label)
            loss2.backward()
            loss3=loss1+loss2
            optimizerD.step()

            # TODO 向鉴别器显示一个生成器的输出，告诉生成器结果应该是1.0。
            optimizerG.zero_grad()
            label.fill_(0.95) # torch.Size([32])
            output = netD(
                fake
            ).view(-1)
            loss4 = criterion(output, label)
            loss4.backward()
            optimizerG.step()

        tm_end = time.time()
        str_train = 'epoch={} lr={:.8f} D_loss={:.3f} G_loss={:.3f} cost_time_m={:.3f}\n'.format(
            epoch,
            optimizerD.param_groups[0]['lr'],
            loss3.item(),
            loss4.item(),
            (tm_end - tm_start) / 60,
        )
        print(str_train, end='')

        with open(ROOT+"\\INFO.txt", "a", encoding="utf-8") as f:
            f.write(str_train)  # 格式化字符串
        torch.save(netD.state_dict(), ROOT+'\\D\\dict_epoch_{}.pth'.format(epoch))
        torch.save(netG.state_dict(), ROOT+'\\G\\dict_epoch_{}.pth'.format(epoch))

        # with torch.no_grad():
        #     noise = torch.randn((64, Params.nz, 1, 1), device=device)
        #     fake = netG(noise).detach().cpu()
        # a1 = make_grid(fake * 0.5 + 0.5, nrow=8)
        # fig = plt.figure(figsize=(20,20))
        # plt.imshow(a1.permute(1, 2, 0))
        # plt.axis("off")
        # plt.show()

        with torch.no_grad():
            noise = torch.randn((64, Params.nz, 1, 1), device=device)  # torch.Size([32, 100, 1, 1])
            fake = netG(noise).detach().cpu()
        grid = make_grid(
            fake*0.5+0.5,
            nrow=8,  # 每行4张
        )
        plt.figure(figsize=(12,12))
        plt.imshow(grid.permute(1,2,0).numpy())
        plt.savefig(
            ROOT+'\\fake_imgs\\epoch_{}.jpg'.format(epoch),
            bbox_inches='tight',
            pad_inches=0.1)
        plt.close()

