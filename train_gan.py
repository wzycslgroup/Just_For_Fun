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
from pathlib import Path
import torch.functional as F
from model.GAN import Generator, Discriminator
from data_utils.FaceDataSet import load_data, FaceDataset
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

parser = argparse.ArgumentParser('AutoEncoder')
parser.add_argument('--gpu', type=str, default='0', help="number of gpu")
parser.add_argument('--dataset', type=str, default='MNIST', help='LFW/MNIST')
parser.add_argument('--method', type=str, default='GAN', help='GAN')
parser.add_argument('--N', default=5, type=int, help='number of images to show')
parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
parser.add_argument('--epoch',  default=50, type=int, help='number of epoch in training')
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
args = parser.parse_args()
N = args.N
DOWNLOAD_MNIST = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

global W, H, latent_dim, Cin

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
    latent_dim = 10
    X_train, X_test, y_train, y_test, img, label = load_data()
    total_data = FaceDataset(img, label)
    train_data = FaceDataset(X_train, y_train)
    test_data = FaceDataset(X_test, y_test)
else :
    W = 28
    H = 28
    Cin = 2
    latent_dim = 30
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
    fig, ax = plt.subplots(2, N)
    ln, = plt.plot([], [], animated=True)

    # randomly sample N images to visualize
    # train_num = train_data.data.shape[0]
    # test_num = test_data.data.shape[0]
    # tot_num = total_data.data.shape[0]
    # random.seed(tot_num)
    # slice = random.sample(list(range(0, tot_num)), N)
    # slice = list(range(N))
    # origin_image = torch.empty(N, W*H)
    # # origin_label = torch.empty(N, 1)
    # for i in range(N):
    #     origin_image[i] = total_data.train_data[slice[i]].view(-1, W*H)\
    #                     .type(torch.FloatTensor)
    #     # origin_label[i] = test_data.train_labels[slice[i]].type(torch.FloatTensor)
    #     ax[0][i].imshow(np.reshape(origin_image.data.numpy()[i], (W, H)))
    #     ax[0][i].set_xticks([])
    #     ax[0][i].set_yticks([])

    # begin to train
    images = []
    turn = 1
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
                b_y = torch.Tensor(x.unsqueeze(1)).float().cuda()
            else:
                b_x = torch.Tensor(x.view(-1, W*H)).float().cuda()
                b_y = torch.Tensor(x.view(-1, W*H)).float().cuda()
            label = label.float().cuda()




            B, _= b_x.shape

            if id % turn_g == 0:
                z = torch.randn(B, latent_dim).cuda()
                fake = generator(z)
                fake_score = discriminator(fake)
                g_loss = criteria(fake_score, torch.ones_like(fake_score))

                optimizer_g.zero_grad()
                g_loss.backward()
                optimizer_g.step()
                loss[0] += g_loss.data * turn_g

            z = torch.randn(B, latent_dim).cuda()
            fake = generator(z)
            fake_score = discriminator(fake)
            real_score = discriminator(b_x)

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
        if epoch % 5 == 0:
            # if args.method == 'ConvAE' or args.method == 'ConvVAE' or args.method == 'ConvVAE_v2':
            #     org_img = origin_image.reshape(-1, 1, W, H)
            #     _, decoded_data = model(org_img.cuda())
            # else:
            #     _, decoded_data = model(origin_image.cuda())
            z = torch.randn((N, latent_dim)).cuda()
            img = generator(z)
            for i in range(N):
                ax[1][i].clear()
                ax[1][i].imshow(np.reshape(img.cpu().data.numpy()[i], (W, H)), cmap='gray')
                ax[1][i].set_xticks(())
                ax[1][i].set_yticks(())
                # plt.show()
            plt.savefig("results/imgs/{}/{}.png".format(args.method, epoch))
            images.append(imageio.imread("results/imgs/{}/{}.png".format(args.method, epoch)))
    imageio.mimsave("results/imgs/{}/{}.gif".format(args.method, args.method), images, duration=0.5)

    # # 测试数字在隐含空间的分布
    # test = train_data.data.view(-1, 28*28).float() / 255.
    # x_test_encoded, _ = model(test.cuda())
    # plt.figure(figsize=(6, 6))
    # plt.scatter(x_test_encoded[:, 0].cpu().detach().numpy(), x_test_encoded[:, 1].cpu().detach().numpy(),
    #             c=test_data.targets.cpu().detach().numpy())
    # plt.colorbar()
    # plt.savefig("fig/number1.png")
    # #plt.show()
    #
    # # 观察隐变量的两个维度变化是如何影响输出结果的
    # n = 15  # figure with 15x15 digits
    # digit_size = 28
    # figure = np.zeros((digit_size * n, digit_size * n))
    #
    # # 用正态分布的分位数来构建隐变量对
    # grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    # grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    #
    # print(grid_x)
    #
    # for i, yi in enumerate(grid_x):
    #     for j, xi in enumerate(grid_y):
    #         z_sample = torch.Tensor(np.array([[xi, yi]])).float().cuda()
    #         x_decoded = model.generator(z_sample).cpu()
    #         digit = x_decoded[0].reshape(digit_size, digit_size).detach().numpy()
    #         figure[i * digit_size: (i + 1) * digit_size,
    #         j * digit_size: (j + 1) * digit_size] = digit
    #
    # plt.figure(figsize=(10, 10))
    # plt.imshow(figure, cmap='Greys_r')
    # #plt.show()
    # plt.savefig("fig/number2.png")


if __name__ == '__main__':
    train()
