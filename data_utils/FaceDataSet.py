import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people


def translate(X):
    return torch.from_numpy(X)


def load_data():
    people = fetch_lfw_people(min_faces_per_person=100, resize=0.7)
    img = people.images
    img = img / 255
    label = people.target
    X_train, X_test, y_train, y_test = train_test_split(img, label)
    img = translate(img)
    label = translate(label)
    X_train = translate(X_train)
    X_test = translate(X_test)
    y_train = translate(y_train)
    y_test = translate(y_test)
    return X_train, X_test, y_train, y_test, img, label


class FaceDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    @property
    def train_data(self):
        return self.data

    @property
    def train_labels(self):
        return self.labels
