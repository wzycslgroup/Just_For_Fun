import argparse
import torch
import logging
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import datetime
import os
from pathlib import Path
from model.DCGAN import Generator, Discriminator


from data_utils.FaceDataSet import load_data, FaceDataset
from data_utils.faces import FaceCartoonDataset
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

parser = argparse.ArgumentParser('Deep Generative Model')
parser.add_argument('--gpu', type=str, default='1', help="number of gpu")
parser.add_argument('--dataset', type=str, default='cartoon', help='LFW/MNIST/cartoon')
parser.add_argument('--method', type=str, default='DCGAN', help='GAN')
parser.add_argument('--gen_num', type=int, default=64, help='GAN')
parser.add_argument('--N', default=5, type=int, help='number of images to show')
parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training')
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
    path = "dataset/faces"
    train_data = FaceCartoonDataset(path)

train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)

# load model
logger.info('Load model...')

generator = Generator(latent_dim, W, H).cuda()
discriminator = Discriminator(W, H).cuda()
criteria = torch.nn.BCELoss()
optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))


# traning process
def train():
    logger.info('Start training...')
    fig, ax = plt.subplots(n_class, N)

    # begin to train
    generator.train()
    discriminator.train()
    turn_g = 1
    turn_d = 1
    for epoch in range(args.epoch):
        loss = torch.zeros((2))
        t = 0
        for id, imgs in enumerate(train_loader):
            b_x = torch.Tensor(imgs).float().cuda()

            B = b_x.shape[0]
            optimizer_d.zero_grad()

            z = torch.randn(B, latent_dim).view(B, latent_dim, 1, 1).cuda()
            fake = generator(z)
            fake_score = discriminator(fake.detach())
            real_score = discriminator(b_x)
            d_fake_loss = criteria(fake_score, torch.zeros_like(fake_score))
            d_real_loss = criteria(real_score, torch.ones_like(fake_score))
            d_loss = d_fake_loss + d_real_loss
            d_loss.backward()
            optimizer_d.step()

            loss[1] += d_loss.data.cpu() * turn_d

            # 固定判别器，更新生成器
            fake_score = discriminator(fake)
            g_loss = criteria(fake_score, torch.ones_like(fake_score))
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()
            loss[0] += g_loss.data.cpu() * turn_g
            t += 1

        logger.info('Epoch %d : G Loss %.4f D Loss %.4f' % (epoch, loss[0] / t , loss[1] / t ))
        print('Epoch: ', epoch, "| G Loss %.4f D Loss %.4f" % (loss[0] / t, loss[1] / t ))
        if epoch % 5 == 0:
            torchvision.utils.save_image(fake.data,
                                         filename="results/imgs/{}/{}.png".format(args.method, epoch),
                                         normalize=True,
                                         nrow=8)


if __name__ == '__main__':
    train()
