import random
import numpy as np
import torch
import torchvision.transforms.functional as trfm


# data augmentations should be functions that receive raw data
# and return the data and information about what modification happened

def rotate(input_data):
    """
    applies random rotations in increments of 90 degrees to images
    :param input_data: expecting image data in the shape (batch, channels, height, width)
    :return: rotated images and the labels of the amount of rotation
    """

    # generate labels
    rotation_angles = np.random.randint(0, 4, input_data.shape[0])
    # perform the rotations
    rotated_data = []
    for i in range(input_data.shape[0]):
        rotated_data.append(np.expand_dims(np.rot90(input_data[i], rotation_angles[i], axes=(1, 2)), axis=0))

    rotated_data = np.concatenate(rotated_data)

    return rotated_data, rotation_angles


def horizontal_flip(input_data):
    """
    randomly flips half of the images in a batch horizontally
    :param input_data: expecting image data in the shape (batch, channels, height, width)
    :return:
    """

    indices = np.arange(input_data.shape[0])
    np.random.shuffle(indices)

    flipped = np.zeros(input_data.shape[0])
    for index in indices[:int(input_data.shape[0] / 2)]:
        input_data[index] = trfm.hflip(torch.as_tensor(input_data[index]))
        flipped[index] = 1

    return input_data, flipped


def cropping(input_data):
    """
    applies random cropping to the images and resizes to the original image size
    :param input_data: expecting image data in the shape (batch, channels, height, width)
    :return:
    """
    top = np.random.randint(0, input_data.shape[2] // 2)
    left = np.random.randint(0, input_data.shape[3] // 2)
    height = np.random.randint(input_data.shape[2] // 2, input_data.shape[2] - top)
    width = np.random.randint(input_data.shape[3] // 2, input_data.shape[3] - left)

    cropped = trfm.crop(torch.as_tensor(input_data), top, left, height, width)
    cropped_resized = trfm.resize(torch.as_tensor(cropped), [input_data.shape[2], input_data.shape[3]], antialias=True)
    # todo (potentially) replace with torchvision.transforms.RandomResizedCrop()
    return cropped_resized, (top, left, height, width)


def gauss_blur(input_data):
    """
    applies a gaussian blur to the images in the batch
    :param input_data: expecting image data in the shape (batch, channels, height, width)
    :return:
    """
    gau_h = random.randrange(3, 10, 2)
    gau_w = random.randrange(3, 10, 2)
    return trfm.gaussian_blur(torch.as_tensor(input_data), [gau_h, gau_w]), (gau_h, gau_w)


def color_distortions(input_data):
    """
    applies random color distortions to the images in the batch
    :param input_data: expecting image data in the shape (batch, channels, height, width)
    :return:
    """
    # todo (potentially) replace with torchvision.transforms.ColorJitter()
    #  but color distortion is used in SimCLR so that images can't be identified based on color histogram alone
    #  and equalize "Equalize the histogram of an image" so it should achieve the same effect
    return trfm.equalize(torch.as_tensor(input_data, dtype=torch.uint8)), (0, 0)


def masking(input_data):
    """
    splits the image into 16x 16 segments then applies a mask to the images covering 75% of each image
    :param input_data: expecting image data in the shape (batch, channels, height, width)
    :return: the data with portions removed, a mask with ones where the data was removed
    """

    patch_indices = np.arange(256)
    np.random.shuffle(patch_indices)

    patch_height = input_data.shape[2]//16
    patch_width = input_data.shape[3]//16

    mask = np.ones((input_data.shape[0], 1, input_data.shape[2], input_data.shape[3]))
    masked_data = np.zeros_like(input_data)

    for patch_index in patch_indices[:int(len(patch_indices) * .25)]:
        row = patch_index // 16 * patch_height
        col = patch_index % 16 * patch_width

        mask[:, :, row:row + patch_height, col:col+patch_width] = 0
        masked_data[:, :, row:row + patch_height, col:col+patch_width] = \
            input_data[:, :, row:row + patch_height, col:col+patch_width]

    return masked_data, mask
