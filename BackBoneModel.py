import torch
import torch.nn as nn


class SimpleConvEncode(nn.Module):
    """

    """
    def __init__(self, embed_dim, device="cpu", *args, **kwargs):
        """

        """
        super().__init__(*args, **kwargs)
        self.l1 = nn.Conv2d(3, 16, (3, 3), device=device)
        self.l2 = nn.Conv2d(16, embed_dim, (3, 3), device=device)

    def forward(self, input_data):
        """

        :param input_data:
        :return:
        """
        out = self.l1(input_data)
        out = nn.functional.silu(out)
        out = self.l2(out)
        out = torch.mean(out, dim=(2, 3))

        return out


class SimpleTaskHead(nn.Module):
    """
    should have more than one layer for a tail, at least needs nonlinearity,
    some evidence says increasing tail length allows backbone output to be more general
    """
    def __init__(self, embed_dim, output_dim, device="cpu", *args, **kwargs):
        """

        :param embed_dim:
        :param output_dim:
        """
        super().__init__(*args, **kwargs)
        self.l1 = nn.Linear(embed_dim, 1024, device=device)
        self.l2 = nn.Linear(1024, 1024, device=device)
        self.l3 = nn.Linear(1024, output_dim, device=device)

    def forward(self, embed_data):
        """

        :param embed_data:
        :return:
        """
        out = self.l1(embed_data)
        out = nn.functional.silu(out)
        out = self.l2(out)
        out = nn.functional.silu(out)
        out = self.l3(out)

        return out


class SimpleConvDecode(nn.Module):
    """

    """
    def __init__(self, embed_dim, output_channels, device='cpu', *args, **kwargs):
        """

        :param embed_dim:
        :param output_channels:
        """
        super().__init__(*args, **kwargs)

        self.up1 = nn.Linear(embed_dim, 1024, device=device)
        self.up2 = nn.Conv2d(1, 3, (3, 3), padding=0, device=device)
        self.up3 = nn.ConvTranspose2d(3, output_channels, (3, 3), device=device)

    def forward(self, embed_data, output_shape):
        """

        :param embed_data:
        :param output_shape:
        :return:
        """
        output = self.up1(embed_data)
        output = nn.functional.silu(output)
        output = output.reshape((embed_data.shape[0], 1, 32, 32))  # 32 = sqrt(1024)
        output = self.up2(output)
        output = nn.functional.silu(output)
        output = self.up3(output, output_size=output_shape)

        return output


# todo flesh out
class Cifar10Encoder(nn.Module):
    """
    from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8745428
    """
    def __init__(self):
        """

        """
        # conv1
        # batchnorm
        # conv2
        # batchnorm
        # pool1
        # dropout
        # conv3
        # batchnorm
        # conv4
        # batchnorm
        # pool2
        # dropout
        # conv5
        # batchnorm
        # conv6
        # batchnorm
        # pool3
        # dropout
        # flatten


# todo flesh out
class Cifar10Classifier(nn.Module):
    """
    from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8745428
    """
    def __init__(self):
        """

        """

        # dense
        # batchnorm
        # dropout
        # dense
        # batchnorm
        # dense

