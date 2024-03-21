from datasets import load_dataset
import numpy as np
import torch


class Cifar10:
    def __init__(self):
        self.trainset = load_dataset("cifar10", split="train", streaming=True)
        self.testset = load_dataset("cifar10", split="test", streaming=True)

    def get_data_n_labels(self, batch):
        data = np.concatenate([np.expand_dims(img, axis=0) for img in batch['img']], axis=0).transpose(
            (0, 3, 1, 2))
        print(data.shape)
        labels = torch.as_tensor(batch['label'], dtype=torch.long)
        return data, labels


class Cifar100:
    def __init__(self):
        self.trainset = load_dataset("cifar100", split="train", streaming=True)
        self.testset = load_dataset("cifar100", split="test", streaming=True)

    def get_data_n_labels(self, batch):
        data = np.concatenate([np.expand_dims(img, axis=0) for img in batch['img']], axis=0).transpose(
            (0, 3, 1, 2))
        labels = torch.as_tensor(batch['fine_label'], dtype=torch.long)
        return data, labels


class Food101:
    def __init__(self):
        self.trainset = load_dataset("food101", split="train", streaming=True)
        self.testset = load_dataset("food101", split="validation", streaming=True)

    def get_data_n_labels(self, batch):
        data = [np.array(img) for img in batch['image']]
        width = 0
        height = 0
        for datum in data:  # need the max image height and width in order to make the array not jagged
            if datum.shape[0] > width:  # probably 528
                width = datum.shape[0]
            if datum.shape[1] > height:  # probably 528
                height = datum.shape[1]

        non_jagged_data = np.zeros((len(data), width, height, 3))  # 3 for RGB

        for i, datum in enumerate(data):
            non_jagged_data[i, :datum.shape[0], :datum.shape[1], :] += datum

        data = non_jagged_data.transpose((0, 3, 1, 2))  # shape b, c, w, h
        print(data.shape)

        labels = torch.as_tensor(batch['label'], dtype=torch.long)

        return data, labels


class DeepPlantAGMC:
    def __init__(self):
        self.trainset = load_dataset("deep-plants/AGM", split="train[:80%]", streaming=True)  # breaks on the split
        self.testset = load_dataset("deep-plants/AGM", split="train[80%:]", streaming=True)
        self.labels = {"tu3": 0, "by": 1, "zx1": 2, "ida": 3, "idb": 4, "wh7": 5, "tu1": 6, "m1a": 7, "m1b": 8,
                       "y2": 9, "y1": 10, "bx": 11, "rx3": 12, "bz": 13, "tu4": 14, "tu2": 15, "j1": 16, "zx3": 17}

    def get_data_n_labels(self, batch):
        data = np.concatenate([np.expand_dims(img, axis=0) for img in batch['image']], axis=0).transpose(
            (0, 3, 1, 2))

        string_labels = batch['crop_type']
        labels = torch.as_tensor([self.labels[label] for label in string_labels], dtype=torch.long)
        return data, labels


class AmazonianFish:
    def __init__(self):
        self.trainset = load_dataset("davanstrien/amazonian_fish_classifier_data", split="train[:80%]", streaming=True)
        self.testset = load_dataset("davanstrien/amazonian_fish_classifier_data", split="train[80%:]", streaming=True)

    def get_data_n_labels(self, batch):
        data = np.concatenate([np.expand_dims(img, axis=0) for img in batch['image']], axis=0).transpose(
            (0, 3, 1, 2))
        labels = torch.as_tensor(batch['label'], dtype=torch.long)
        return data, labels


class Imagenet1k:
    def __init__(self):
        self.trainset = load_dataset("imagenet-1k", split="train", streaming=True)
        self.testset = load_dataset("imagenet-1k", split="validation", streaming=True)

    def get_data_n_labels(self, batch):
        data = np.concatenate([np.expand_dims(img, axis=0) for img in batch['img']], axis=0).transpose(
            (0, 3, 1, 2))
        labels = torch.as_tensor(batch['label'], dtype=torch.long)
        return data, labels
