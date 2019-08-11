import argparse
import torch
import logging
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import imageio
import numpy as np
import datetime
import os
from pathlib import Path
from model.DCGAN import Generator, Discriminator


from data_utils.FaceDataSet import load_data, FaceDataset
from data_utils.faces import load_cartoon, FaceCartoonDataset
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

parser = argparse.ArgumentParser('AutoEncoder')
parser.add_argument('--gpu', type=str, default='0', help="number of gpu")
parser.add_argument('--dataset', type=str, default='cartoon', help='LFW/MNIST/cartoon')
parser.add_argument('--method', type=str, default='DCGAN', help='GAN')
parser.add_argument('--N', default=5, type=int, help='number of images to show')
parser.add_argument('--batch_size', type=int, default=128, help='batch size in training')
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
elif args.dataset == 'MNIST':
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
elif args.dataset == 'cartoon':
    W = 96
    H = 96
    Cin = 3
    latent_dim = 100
    n_class = 5
    path = 'dataset/anime'
    file_index = load_cartoon(path)
    train_data = FaceCartoonDataset(path)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[-1], std=[2]),
    ]
    )

train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)

# load model
logger.info('Load model...')
if args.method == 'DCGAN':
    generator = Generator(latent_dim, W, H).cuda()
    discriminator = Discriminator(W, H).cuda()
else:
    None
optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.learning_rate)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate)

criteria = torch.nn.BCELoss()

# traning process
def train():
    logger.info('Start training...')
    fig, ax = plt.subplots(n_class, N)

    # begin to train
    images = []
    generator.train()
    discriminator.train()
    for epoch in range(args.epoch):
        loss = torch.zeros((2))
        turn_g = 1
        turn_d = 1
        t = 0
        for id, x in enumerate(train_loader):
            if args.method == 'DCGAN':
                b_x = torch.Tensor(x).float().cuda()
            else:
                b_x = torch.Tensor(x.view(-1, W*H)).float().cuda()
            B = b_x.shape[0]

            z = torch.randn(B, latent_dim).view(B, latent_dim, 1, 1).cuda()

            fake = generator(z)
            fake_score = discriminator(fake)
            g_loss = criteria(fake_score, torch.ones_like(fake_score))

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()
            loss[0] += g_loss.data * turn_g

            fake = generator(z)
            fake_score = discriminator(fake)
            real_score = discriminator(b_x)

            d_fake_loss = criteria(fake_score, torch.zeros_like(fake_score))
            d_real_loss = criteria(real_score, torch.ones_like(fake_score))
            d_loss = d_fake_loss + d_real_loss

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            loss[1] += d_loss.data

            t += 1

        logger.info('Epoch %d : G Loss %.4f D Loss %.4f' % (epoch, loss[0] / t , loss[1] / t ))
        print('Epoch: ', epoch, "| G Loss %.4f D Loss %.4f" % (loss[0] / t, loss[1] / t ))
        if epoch % 20 == 0:
            for i in range(n_class):
                z = torch.randn(B, latent_dim).view(B, latent_dim, 1, 1).cuda()
                img = generator(z).permute(0, 2, 3, 1)
                img = (img + 1) * 127.5
                for j in range(N):
                    ax[i][j].clear()
                    ax[i][j].imshow(np.reshape(img.cpu().data.numpy()[j], (W, H, Cin)))
                    ax[i][j].set_xticks(())
                    ax[i][j].set_yticks(())
                # plt.show()
            plt.savefig("results/imgs/{}/{}.png".format(args.method, epoch))
            # images.append(imageio.imread("results/imgs/{}/{}.png".format(args.method, epoch)))
    # imageio.mimsave("results/imgs/{}/{}.gif".format(args.method, args.method), images, duration=0.5)


if __name__ == '__main__':
    train()
