import argparse
import torch
import logging
import random
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import imageio
import numpy as np
import datetime
import os
from scipy.stats import norm
from pathlib import Path
import torch.functional as F
from model.GAN import Generator, Discriminator
from model.cGAN import Generator, Discriminator
from data_utils.FaceDataSet import load_data, FaceDataset
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

parser = argparse.ArgumentParser('AutoEncoder')
parser.add_argument('--gpu', type=str, default='0', help="number of gpu")
parser.add_argument('--dataset', type=str, default='MNIST', help='LFW/MNIST')
parser.add_argument('--method', type=str, default='GAN', help='GAN')
parser.add_argument('--N', default=10, type=int, help='number of images to show')
parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
parser.add_argument('--epoch',  default=100, type=int, help='number of epoch in training')
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
args = parser.parse_args()
N = args.N
DOWNLOAD_MNIST = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

global W, H, latent_dim, Cin, n_class

results_dir = Path('./results/')
results_dir.mkdir(exist_ok=True)
logs_dir = Path('./results/logs/')
logs_dir.mkdir(exist_ok=True)
model_dir = Path('./results/logs/{}/'.format(args.method))
model_dir.mkdir(exist_ok=True)
imgs_dir = Path('./results/imgs/')
imgs_dir.mkdir(exist_ok=True)
imgs_method_dir = Path('./results/imgs/{}'.format(args.method))
imgs_method_dir.mkdir(exist_ok=True)

# logger
logger = logging.getLogger("Generative Model")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('./results/logs/{}/train_{}_'.format(args.method, args.method)+ str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M'))+'.txt')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.info('---------------------------------------------------TRANING---------------------------------------------------')
logger.info('Model Paramter ...')
logger.info(args)

# load data
logger.info('Load dataset ...')
if args.dataset == 'LFW':
    W = 87
    H = 65
    Cin = 1
    latent_dim = 2
    X_train, X_test, y_train, y_test, img, label = load_data()
    total_data = FaceDataset(img, label)
    train_data = FaceDataset(X_train, y_train)
    test_data = FaceDataset(X_test, y_test)
else :
    W = 28
    H = 28
    Cin = 2
    latent_dim = 90
    n_class = 10
    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(0, 1),
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    train_data = torchvision.datasets.MNIST(
        root='dataset',
        train=True,
        transform=transform,
        download=DOWNLOAD_MNIST,
    )
    test_data = torchvision.datasets.MNIST(
        root='dataset',
        train=False,
        transform=transform,
    )
train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)

# load model
logger.info('Load model...')
if args.method == 'GAN':
    generator = Generator(latent_dim, n_class, W, H).cuda()
    discriminator = Discriminator(W, H, n_class).cuda()
else:
    None
optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.learning_rate)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate)

criteria = torch.nn.BCELoss()

# traning process
def train():
    logger.info('Start training...')
    fig, ax = plt.subplots(n_class, N)
    ln, = plt.plot([], [], animated=True)

    # begin to train
    images = []
    generator.train()
    discriminator.train()
    for epoch in range(args.epoch):
        loss = torch.zeros((2))
        turn_g = 1
        turn_d = 1
        t = 0
        for id, (x, label) in enumerate(train_loader):
            if args.method == '12' :
                b_x = torch.Tensor(x.unsqueeze(1)).float().cuda()
            else:
                b_x = torch.Tensor(x.view(-1, W*H)).float().cuda()
            label = label.float().cuda()

            B, _= b_x.shape

            z = torch.randn(B, latent_dim).cuda()
            z_y = torch.randint(0, 10, (B,)).cuda()

            if id % turn_g == 0:
                fake = generator(z, z_y)
                fake_score = discriminator(fake, z_y)
                g_loss = criteria(fake_score, torch.ones_like(fake_score))

                optimizer_g.zero_grad()
                g_loss.backward()
                optimizer_g.step()
                loss[0] += g_loss.data * turn_g

            fake = generator(z, z_y)
            fake_score = discriminator(fake, z_y)
            real_score = discriminator(b_x, label)

            d_fake_loss = criteria(fake_score, torch.zeros_like(fake_score))
            d_real_loss = criteria(real_score, torch.ones_like(fake_score))
            d_loss = (d_fake_loss + d_real_loss) / 2

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            loss[1] += d_loss.data

            t += 1

        logger.info('Epoch %d : G Loss %.4f D Loss %.4f' % (epoch, loss[0] / t , loss[1] / t ))
        print('Epoch: ', epoch, "| G Loss %.4f D Loss %.4f" % (loss[0] / t, loss[1] / t ))
        if epoch % 10 == 0:
            for i in range(n_class):
                z = torch.randn((N, latent_dim)).cuda()
                z_y = torch.tensor([i] * N).view(N).cuda()
                img = generator(z, z_y)
                for j in range(N):
                    ax[i][j].clear()
                    ax[i][j].imshow(np.reshape(img.cpu().data.numpy()[j], (W, H)), cmap='gray')
                    ax[i][j].set_xticks(())
                    ax[i][j].set_yticks(())
                # plt.show()
            plt.savefig("results/imgs/{}/{}.png".format(args.method, epoch))
            images.append(imageio.imread("results/imgs/{}/{}.png".format(args.method, epoch)))
    imageio.mimsave("results/imgs/{}/{}.gif".format(args.method, args.method), images, duration=0.5)



if __name__ == '__main__':
    train()
