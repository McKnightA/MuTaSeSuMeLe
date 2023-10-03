import torch
import torch.nn as nn


class SimpleConvEncode(nn.Module):
    """

    """
    def __init__(self, embed_dim, *args, **kwargs):
        """

        """
        super().__init__(*args, **kwargs)
        self.l1 = nn.Conv2d(3, 16, (3, 3))
        self.l2 = nn.Conv2d(16, embed_dim, (3, 3))

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


class ResNetLikeEncode(nn.Module):
    """

    """

    def __init__(self, embed_dim, *args, **kwargs):
        """

        :param embed_dim:
        """
        super().__init__(*args, **kwargs)
        # todo:
        #  read ResNet paper cause this isn't right
        #  figure out all the needed layers

        self.block1 = nn.Sequential(nn.Conv2d(3, 32, (3, 3)),
                                    nn.SiLU(),
                                    nn.Conv2d(32, 64, (3, 3), stride=(2, 2)),
                                    nn.SiLU(),
                                    nn.BatchNorm2d(64))
        self.block2 = nn.Sequential(nn.Conv2d(64, 128, (3, 3)),
                                    nn.SiLU(),
                                    nn.Conv2d(128, 128, (3, 3), stride=(2, 2)),
                                    nn.SiLU(),
                                    nn.BatchNorm2d(128))
        self.block3 = nn.Sequential(nn.Conv2d(128, 256, (3, 3)),
                                    nn.SiLU(),
                                    nn.Conv2d(256, 256, (3, 3)),
                                    nn.SiLU(),
                                    nn.Conv2d(256, embed_dim, (3, 3), stride=(2, 2)),
                                    nn.SiLU(),
                                    nn.BatchNorm2d(embed_dim))

    def forward(self, input_data):
        """

        :param input_data:
        :return:
        """
        out = self.block1(input_data)
        out = self.block2(out)
        out = self.block3(out)
        return out


class SimpleTaskHead(nn.Module):
    """
    should have more than one layer for a tail, at least needs nonlinearity,
    some evidence says increasing tail length allows backbone output to be more general
    """
    def __init__(self, embed_dim, output_dim, *args, **kwargs):
        """

        :param embed_dim:
        :param output_dim:
        """
        super().__init__(*args, **kwargs)
        self.l1 = nn.Linear(embed_dim, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, output_dim)

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
    def __init__(self, embed_dim, output_channels, *args, **kwargs):
        """

        :param embed_dim:
        :param output_channels:
        """
        super().__init__(*args, **kwargs)

        self.up1 = nn.Linear(embed_dim, 1024)
        self.up2 = nn.Conv2d(1, 3, (3, 3), padding=0)
        self.up3 = nn.ConvTranspose2d(3, output_channels, (3, 3))

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


