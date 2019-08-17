import torch
import torchvision
from torch.utils.data import Dataset
import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_cartoon(path):
    file_list = os.listdir(path)
    file_index = [i for i in range(len(file_list))]
    return file_index


class FaceCartoonDataset(Dataset):
    def __init__(self, path):
        super(FaceCartoonDataset, self).__init__()
        self.path = path
        self.file_list = os.listdir(self.path)
        self.transform = torchvision.transforms.Compose([
            # torchvision.transforms.ToTensor(),
            # # torchvision.transforms.Normalize(0, 1),
            # torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
            # torchvision.transforms.Scale(96),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        file_path = os.path.join(self.path, self.file_list[item])
        image = cv2.imread(file_path)
        # image = image[:, :, ::-1]
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        image = self.transform(image)
        return image

if __name__ == '__main__':
    lc = FaceCartoonDataset('../dataset/faces')
    cc = lc[26]
    print(cc)
    image = cv2.imread('../dataset/faces/0000fdee4208b8b7e12074c920bc6166-0.jpg')
    plt.imshow(image)
    plt.savefig('test.png')
    plt.show()
    print(image)