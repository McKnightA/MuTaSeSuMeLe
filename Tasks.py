import torch
import torch.nn as nn
import torchmetrics
from skimage.color import rgb2lab, lab2rgb
import DataAugments as da

# TODO:
#  (eventually) recreate less naive multi-task tasks

# Self Supervised Tasks ----------------------------------------------


class Rotation(nn.Module):
    """
    inspired by https://arxiv.org/abs/1803.07728
    """

    def __init__(self, embed_features, task_head, device="cpu", *args, **kwargs):
        """

        :param embed_features:
        :param task_head:
        :param device:
        """
        super().__init__(*args, **kwargs)
        self.name = 'Rotation'
        self.device = device

        self.task_head = task_head(embed_features, 4, device)
        self.loss = nn.CrossEntropyLoss()
        self.labels = None

    def pretreat(self, input_data):
        """
        rotate the image by a random angle and save the angle as a label
        :param input_data: a batch of raw images in np array
        :return: the rotated images as tensors
        """

        rotated, self.labels = da.rotate(input_data)

        rotated = torch.tensor(rotated, dtype=torch.float, device=self.device, requires_grad=True)
        self.labels = torch.as_tensor(self.labels, dtype=torch.long, device=self.device)

        return rotated

    def generate_loss(self, embedded_data, clear_labels=True):
        """

        :param embedded_data:
        :param clear_labels:
        :return:
        """

        predictions = self.task_head.forward(embedded_data)
        loss = self.loss(predictions, self.labels)

        if clear_labels:
            self.labels = None

        return loss

    def check_performance(self, input_data, backbone):
        """

        :param input_data:
        :param backbone:
        :return:
        """
        treated = self.pretreat(input_data)  # forward pass the data
        embedded = backbone.forward(treated)
        predictions = self.task_head.forward(embedded)
        accuracy = torchmetrics.functional.accuracy(predictions, self.labels, task="multiclass", num_classes=4)

        return accuracy.cpu()

    def forward(self, input_data, backbone):
        treated = self.pretreat(input_data)
        embedded = backbone.forward(treated)
        loss = self.generate_loss(embedded)
        return loss


class Colorization(nn.Module):
    """
    inspired by https://arxiv.org/abs/1603.08511
    """

    def __init__(self, embed_features, task_head, device="cpu", *args, **kwargs):
        """

        :param embed_features:
        :param task_head:
        :param device:
        """
        super().__init__(*args, **kwargs)
        self.name = 'Colorization'
        self.device = device

        self.desired_precision = 128

        self.harmonization = nn.Conv2d(1, 3, (1, 1), device=device)
        self.task_head = task_head(embed_features, self.desired_precision * 2, device)

        self.loss = nn.CrossEntropyLoss()
        self.labels = None

    def pretreat(self, input_data):
        """

        :param input_data: a batch of raw images
        :return:
        """

        input_data = input_data / 255  # skimage expects float values to be between -1 and 1 or 0 and 1 I opt for [0, 1]
        # convert to lab and maintain (b, c, h, w) shape
        lab_data = rgb2lab(input_data.transpose(0, 2, 3, 1)) / 110  # source paper uses this 110 normalization value
        lab_data = torch.tensor(lab_data.transpose(0, 3, 1, 2), dtype=torch.float, device=self.device)
        l_data = lab_data[:, 0, :, :].unsqueeze(1)
        self.labels = lab_data[:, 1:, :, :]

        return self.harmonization(l_data)

    def generate_loss(self, embedded_data, clear_labels=True):
        """

        :param embedded_data:
        :param clear_labels:
        :return:
        """

        output_shape = list(self.labels.shape)
        output_shape[1] = self.desired_precision * 2

        # labels are between -1 and 1
        # turning ab channel values into indices for label values to be used in cross entropy
        self.labels = nn.functional.sigmoid(self.labels)
        self.labels = (self.labels * self.desired_precision) // 1

        # up-scaling the embedded data into ab value predictions for each pixel
        output = self.task_head.forward(embedded_data, output_shape)

        # generate for a and b values
        loss_a = self.loss(output[:, :self.desired_precision], self.labels[:, 0].long())
        loss_b = self.loss(output[:, self.desired_precision:], self.labels[:, 1].long())

        loss = (loss_a + loss_b) / 2

        if clear_labels:
            self.labels = None

        return loss

    def check_performance(self, input_data, backbone):
        """

        :param input_data:
        :param backbone:
        :return:
        """
        batch = self.pretreat(input_data)
        latent = backbone.forward(batch)
        predictions = self.task_head.forward(latent, (input_data.shape[0], self.desired_precision * 2,
                                                      input_data.shape[2], input_data.shape[3]))

        predictions_a = torch.argmax(predictions[:, :self.desired_precision], dim=1, keepdim=True)
        predictions_a = torch.special.logit(predictions_a / self.desired_precision) * 110

        predictions_b = torch.argmax(predictions[:, self.desired_precision:], dim=1, keepdim=True)
        predictions_b = torch.special.logit(predictions_b / self.desired_precision) * 110

        base_l = torch.tensor(rgb2lab(input_data.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)[:, 0], device=self.device)
        base_l = base_l.unsqueeze(1)

        reconstructed = torch.concatenate((base_l, predictions_a, predictions_b), dim=1).permute(0, 2, 3, 1)
        reconstructed = lab2rgb(reconstructed.cpu())

        return reconstructed[0]

    def forward(self, input_data, backbone):
        treated = self.pretreat(input_data)
        embedded = backbone.forward(treated)
        loss = self.generate_loss(embedded)
        return loss


class Contrastive(nn.Module):
    """
    inspired by https://arxiv.org/abs/2002.05709
    """

    def __init__(self, embed_features, task_head, device="cpu", *args, **kwargs):
        """

        :param embed_features:
        :param task_head:
        :param device:
        """
        super().__init__(*args, **kwargs)
        self.name = "Contrastive"
        self.device = device

        # "As shown in Section 3, the combination of random crop and
        # color distortion is crucial to achieve a good performance"

        self.augments = [da.horizontal_flip, da.cropping, da.color_distortions, da.gauss_blur]  # maybe masking cutting

        self.task_head = task_head(embed_features, 256, device)
        self.temperature = 0.1

    def pretreat(self, input_data):
        """

        :param input_data:
        :return:
        """

        # gauss shows bad results in their augmentation heatmap,
        # but cutout shows good performance so maybe use the masking augmentation
        # potentially ideal augment order hflip, crop, color_distort, blur, mask

        aug_data_1 = input_data.copy()  # I've seen drastically worse results when I / 255
        for augment in self.augments:  # I was expecting the data to stay as a np.array, but it doesn't
            aug_data_1, scrap_info = augment(aug_data_1)

        aug_data_2 = input_data.copy()
        for augment in self.augments:
            aug_data_2, scrap_info = augment(aug_data_2)

        aug_data = torch.concatenate((aug_data_1, aug_data_2), dim=0).float().requires_grad_(True).to(self.device)

        return aug_data

    def generate_loss(self, embedded_data, clear_labels=True):
        """

        :param embedded_data:
        :param clear_labels:
        :return:
        """
        """
        def nt_xent(out1, out2, temp):
            out = torch.concatenate((out1, out2), dim=0)
            n_samples = out.shape[0]

            sim = torch.matmul(out, out.T)
            scaled_sim = torch.exp(sim / temp)

            mask = ~torch.eye(n_samples)
            neg = torch.sum(scaled_sim * mask, dim=-1)

            pos = torch.exp(torch.sum(out1 * out2, dim=-1) / temp)
            pos = torch.concatenate((pos, pos), dim=0)

            loss = -torch.log(pos / neg).mean()
            return loss"""

        output = self.task_head(embedded_data)

        aug1 = nn.functional.normalize(output[:output.shape[0] // 2])
        aug2 = nn.functional.normalize(output[output.shape[0] // 2:])
        out = torch.concatenate((aug1, aug2), dim=0)
        n_samples = out.shape[0]

        sim = torch.matmul(out, out.T)
        scaled_sim = torch.exp(sim / self.temperature)

        mask = ~torch.eye(n_samples, dtype=torch.bool, device=self.device)
        neg = torch.sum(scaled_sim * mask, dim=-1)

        pos = torch.exp(torch.sum(aug1 * aug2, dim=-1) / self.temperature)
        pos = torch.concatenate((pos, pos), dim=0)

        loss = -torch.log(pos / neg).mean()

        return loss

    def check_performance(self, input_data, backbone):
        """

        :param input_data:
        :param backbone:
        :return:
        """
        embedded = backbone.forward(torch.tensor(input_data, dtype=torch.float, device=self.device))
        vector_representation = self.task_head.forward(embedded)

        similarity = torch.matmul(vector_representation.data.detach(), vector_representation.T.data.detach())

        similarity = torch.exp(similarity / self.temperature)

        return similarity.unsqueeze(-1).cpu()

    def forward(self, input_data, backbone):
        treated = self.pretreat(input_data)
        embedded = backbone.forward(treated)
        loss = self.generate_loss(embedded)
        return loss


class MaskedAutoEncoding(nn.Module):
    """
    inspired by https://arxiv.org/abs/2111.06377
    """

    def __init__(self, embed_features, task_head, device="cpu", *args, **kwargs):
        """

        :param embed_features:
        :param task_head:
        :param device:
        """
        super().__init__(*args, **kwargs)
        self.name = "Masked Auto Encoding"
        self.device = device

        self.harmonization = nn.Conv2d(4, 3, (1, 1), device=device)
        self.task_head = task_head(embed_features, 3, device)

        self.loss = nn.MSELoss()
        self.labels = None

    def pretreat(self, input_data):
        """

        :param input_data:
        :return:
        """

        # normalizing the input is helpful
        # augmentations have been shown to be potentially helpful but not necessary
        # no color jitter, yes cropping and horizontal flipping

        input_data = input_data / 255
        masked_image, mask = da.masking(input_data)

        self.labels = torch.tensor(mask, dtype=torch.float, device=self.device, requires_grad=True), \
                      torch.tensor(input_data, dtype=torch.float, device=self.device, requires_grad=True)

        combo = torch.concatenate((torch.tensor(masked_image, dtype=torch.float, device=self.device),
                                   self.labels[0]), dim=1).requires_grad_(True)

        pretreated = self.harmonization(combo)

        return pretreated

    def generate_loss(self, embedded_data, clear_labels=True):
        """

        :param embedded_data:
        :param clear_labels:
        :return:
        """

        output_shape = self.labels[1].shape

        output = self.task_head.forward(embedded_data, output_shape)
        output = nn.functional.sigmoid(output)

        result = self.loss(output * self.labels[0], self.labels[1] * self.labels[0])

        if clear_labels:
            self.labels = None

        return result

    def check_performance(self, input_data, backbone):
        """

        :param input_data:
        :param backbone:
        :return:
        """
        batch = self.pretreat(input_data)
        embedded = backbone.forward(batch)  # wtf why do you use 15GB
        predictions = torch.nn.functional.sigmoid(self.task_head.forward(embedded, batch.shape))
        # visualize the prediction
        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(input_data[0].transpose(1, 2, 0) / 255)  # original data
        # ax[1].imshow(self.labels[0][0].permute(1, 2, 0).detach())  # applied mask
        # ax[2].imshow(predictions[0].permute(1, 2, 0).detach())  # recreated data

        return predictions.data.detach().cpu().permute(0, 2, 3, 1)[0]

    def forward(self, input_data, backbone):
        treated = self.pretreat(input_data)
        embedded = backbone.forward(treated)
        loss = self.generate_loss(embedded)
        return loss


# Supervised Tasks ----------------------------------------------------


class Cifar10Classification(nn.Module):
    """

    """

    def __init__(self, embed_dim, task_head, device="cpu", *args, **kwargs):
        """

        :param embed_dim:
        :param task_head:
        :param device:
        """
        super().__init__(*args, **kwargs)
        self.name = "Cifar10 Classification"
        self.device = device

        self.task_head = task_head(embed_dim, 10, device)
        self.loss = nn.CrossEntropyLoss()

    def pretreat(self, input_data):
        """

        :param input_data:
        :return:
        """
        return torch.tensor(input_data, dtype=torch.float, device=self.device, requires_grad=True) / 255

    def generate_loss(self, embed_data, labels):
        """

        :param embed_data:
        :param labels:
        :return:
        """
        prediction = self.task_head.forward(embed_data)

        loss = self.loss(prediction, labels.to(self.device))

        return loss

    def check_performance(self, input_data, labels, backbone):
        """

        :param input_data:
        :param labels:
        :param backbone:
        :return:
        """
        treated = self.pretreat(input_data)
        embedded = backbone.forward(treated)
        predictions = self.task_head.forward(embedded)
        accuracy = torchmetrics.functional.accuracy(predictions, labels.to(self.device), task="multiclass", num_classes=10)

        return accuracy.cpu()

    def forward(self, input_data, labels, backbone):
        treated = self.pretreat(input_data)
        embedded = backbone.forward(treated)
        loss = self.generate_loss(embedded, labels)
        return loss


class Classification(nn.Module):
    """

    """

    def __init__(self, embed_dim, task_head, classes, name, device="cpu", *args, **kwargs):
        """

        :param embed_dim:
        :param task_head:
        :param classes:
        :param name:
        :param device:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.name = name + "Classification"
        self.device = device

        self.num_classes = classes
        self.task_head = task_head(embed_dim, classes, device)
        self.loss = nn.CrossEntropyLoss()

    def pretreat(self, input_data):
        """

        :param input_data:
        :return:
        """
        return torch.tensor(input_data, dtype=torch.float, device=self.device, requires_grad=True) / 255

    def generate_loss(self, embed_data, labels):
        """

        :param embed_data:
        :param labels:
        :return:
        """
        prediction = self.task_head.forward(embed_data)

        loss = self.loss(prediction, labels.to(self.device))

        return loss

    def check_performance(self, input_data, labels, backbone):
        """

        :param input_data:
        :param labels:
        :param backbone:
        :return:
        """
        treated = self.pretreat(input_data)
        embedded = backbone.forward(treated)
        predictions = self.task_head.forward(embedded)
        accuracy = torchmetrics.functional.accuracy(predictions, labels.to(self.device),
                                                    task="multiclass", num_classes=self.num_classes)

        return accuracy.cpu()

    def forward(self, input_data, labels, backbone):
        treated = self.pretreat(input_data)
        embedded = backbone.forward(treated)
        loss = self.generate_loss(embedded, labels)
        return loss


# Multi-Task Tasks ----------------------------------------------------


class AveragedLossMultiTask(nn.Module):
    """
    runs each task then uses an unweighted average of the losses of the set as the final loss
    """

    def __init__(self, tasks, device, *args, **kwargs):
        """

        :param tasks: a list of tasks
        :param device:
        """
        super().__init__(*args, **kwargs)
        self.name = ""
        for task in tasks:
            self.name += task.name + "+"
        self.tasks = tasks
        self.device = device

        self.batch_shapes = []

        self.params = []
        for task in self.tasks:
            self.params += list(task.parameters())

    def pretreat(self, input_data):
        """

        :param input_data:
        :return:
        """
        batch = [task.pretreat(input_data) for task in self.tasks]
        self.batch_shapes = [treated.shape[0] for treated in batch]
        batch = torch.concatenate(batch, dim=0)
        return batch

    def generate_loss(self, embedded_data, labels=None, clear_labels=True):
        """

        :param embedded_data:
        :param labels:
        :param clear_labels:
        :return:
        """

        total_loss = []
        first_index = 0  # tasks like contrastive change the batch size, so I count through each batch length
        for k in range(len(self.tasks)):
            try:
                loss = self.tasks[k].generate_loss(embedded_data[first_index:first_index + self.batch_shapes[k]],
                                                   clear_labels)
            except TypeError:  # ready for a supervised training task
                # this breaks because supervised tasks don't have a clear_labels field
                loss = self.tasks[k].generate_loss(embedded_data[first_index:first_index + self.batch_shapes[k]],
                                                   labels, clear_labels)
            total_loss.append(loss)

            first_index += self.batch_shapes[k]

        averaged_task_loss = torch.mean(torch.tensor(total_loss, device=self.device, requires_grad=True))

        return averaged_task_loss

    def check_performance(self, input_data, backbone):
        """

        :param input_data:
        :param backbone:
        :return:
        """
        return 0

    def forward(self, input_data, backbone):
        losses = []
        for task in self.tasks:
            losses.append(task.forward(input_data, backbone))

        if len(losses) > 1:
            averaged_task_loss = torch.mean(torch.stack(losses))
        elif len(losses) == 1:
            averaged_task_loss = torch.mean(losses[0])
        else:
            print("huh")
            averaged_task_loss = 0

        return averaged_task_loss
