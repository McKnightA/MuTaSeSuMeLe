from datasets import load_dataset
import numpy as np
import torch


def scale_down_image_batch(images, new_size):
    for img in images:
        img.thumbnail((new_size, new_size))
    data = [np.array(img) for img in images]
    for i, datum in enumerate(data):
        if len(datum.shape) < 3:  # some images are black and white
            # raise ValueError("dataset is broken")
            data[i] = np.expand_dims(datum, -1)

    return data


def pad_jagged_image_batch(images, padded_size):
    non_jagged_data = np.zeros((len(images), padded_size[0], padded_size[1], 3))  # 3 for RGB

    for i, datum in enumerate(images):
        image_w_pad_size = padded_size[0] - datum.shape[0]
        image_h_pad_size = padded_size[1] - datum.shape[1]
        non_jagged_data[i, image_w_pad_size // 2: datum.shape[0] + image_w_pad_size // 2,
                        image_h_pad_size // 2:datum.shape[1] + image_h_pad_size // 2, :] += datum

    return non_jagged_data


class Cifar10:
    def __init__(self):
        self.trainset = load_dataset("D:\\cifar10", split="train", streaming=True)
        self.testset = load_dataset("D:\\cifar10", split="test", streaming=True)
        self.name = "Cifar10"
        self.class_num = 10

    def get_data_n_labels(self, batch):
        data = np.concatenate([np.expand_dims(img, axis=0) for img in batch['img']], axis=0).transpose(
            (0, 3, 1, 2))
        labels = torch.as_tensor(batch['label'], dtype=torch.long)

        return data, labels


class Cifar100:
    def __init__(self):
        self.trainset = load_dataset("D:\\cifar100", split="train", streaming=True)
        self.testset = load_dataset("D:\\cifar100", split="test", streaming=True)
        self.name = "Cifar100"
        self.class_num = 100

    def get_data_n_labels(self, batch):
        data = np.concatenate([np.expand_dims(img, axis=0) for img in batch['img']], axis=0).transpose(
            (0, 3, 1, 2))
        labels = torch.as_tensor(batch['fine_label'], dtype=torch.long)

        return data, labels


class Food101:
    def __init__(self):
        self.trainset = load_dataset("D:\\food101", split="train", streaming=True)
        self.testset = load_dataset("D:\\food101", split="validation", streaming=True)
        self.name = "Food101"
        self.class_num = 101

    def get_data_n_labels(self, batch):
        # in order to match cifar image sizes
        data = scale_down_image_batch(batch["image"], 32)

        data = pad_jagged_image_batch(data, (32, 32)).transpose((0, 3, 1, 2))  # shape b, c, w, h

        labels = torch.as_tensor(batch['label'], dtype=torch.long)

        return data, labels


class Beans:
    def __init__(self):
        self.trainset = load_dataset("D:\\beans", split="train", streaming=True)
        self.testset = load_dataset("D:\\beans", split="validation", streaming=True)
        self.name = "Beans"
        self.class_num = 3

    def get_data_n_labels(self, batch):
        data = [np.expand_dims(img, 0) for img in scale_down_image_batch(batch["image"], 32)]
        data = np.concatenate(data, 0).transpose((0, 3, 1, 2))

        labels = torch.as_tensor(batch['labels'], dtype=torch.long)
        return data, labels


class Svhn:
    def __init__(self):
        self.trainset = load_dataset("D:\\svhn", "cropped_digits", split="train", streaming=True)
        self.testset = load_dataset("D:\\svhn", "cropped_digits", split="train", streaming=True)
        self.name = "SVHN"
        self.class_num = 10

    def get_data_n_labels(self, batch):
        data = np.concatenate([np.expand_dims(img, axis=0) for img in batch['image']], axis=0).transpose(
            (0, 3, 1, 2))
        labels = torch.as_tensor(batch['label'], dtype=torch.long)

        return data, labels


class Imagenet1k:
    def __init__(self):
        self.trainset = load_dataset("imagenet-1k", split="train", streaming=True)
        self.testset = load_dataset("imagenet-1k", split="validation", streaming=True)
        self.name = "Imagenet1k"
        self.class_num = 1000

    def get_data_n_labels(self, batch):
        pass


class Coco:
    """
    https://huggingface.co/datasets/detection-datasets/coco
    """
    def __init__(self):
        self.trainset = load_dataset("detection-datasets/coco", split="train", streaming=True)
        self.testset = load_dataset("detection-dataset/coco", split="val", streaming=True)
        self.name = "Coco"

    def get_data_n_labels(self, batch):
        pass

