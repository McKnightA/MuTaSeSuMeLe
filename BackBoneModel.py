import torch
import torch.nn as nn


class SimpleConvEncode(nn.Module):
    """
    small and fast with no concern for performance
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
    small and fast with no concern for performance
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


class Cifar10Encoder(nn.Module):
    """
    from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8745428
    """
    def __init__(self, embed_dim, device='cpu', *args, **kwargs):
        """

        :param embed_dim:
        :param device:
        """
        super().__init__(*args, **kwargs)

        def create_conv_packet(starting_channels, ending_channels):
            packet = nn.Sequential(nn.Conv2d(starting_channels, ending_channels, (3, 3), (1, 1), 1, device=device),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(ending_channels, device=device),
                                   nn.Conv2d(ending_channels, ending_channels, (3, 3), (1, 1), device=device),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(ending_channels, device=device),
                                   nn.MaxPool2d(2, 2),
                                   nn.Dropout(0.25))

            return packet

        self.layers = nn.Sequential(create_conv_packet(3, 32),
                                    create_conv_packet(32, 64),
                                    create_conv_packet(64, 128),
                                    nn.Flatten(),
                                    nn.Linear(512, embed_dim, device=device),
                                    nn.BatchNorm1d(embed_dim, device=device),
                                    nn.Dropout(0.25))

    def forward(self, input_data):
        """

        :param input_data:
        :return:
        """

        return self.layers.forward(input_data)


class Cifar10Classifier(nn.Module):
    """
    from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8745428
    """
    def __init__(self, embed_dim, output_dim, device="cpu", *args, **kwargs):
        """

        :param embed_dim:
        :param output_dim:
        :param device:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(nn.Linear(embed_dim, 512, device=device),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(512, device=device),
                                    nn.Linear(512, output_dim, device=device))

    def forward(self, input_data):
        """

        :param input_data:
        :return:
        """
        return self.layers.forward(input_data)


class Cifar10Decoder(nn.Module):
    """
    reversing https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8745428
    """
    def __init__(self, embed_dim, output_channels, device='cpu', *args, **kwargs):
        """

        :param embed_dim:
        :param output_channels:
        :param device:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

        def create_conv_packet(starting_channels, ending_channels):
            packet = nn.Sequential(nn.ConvTranspose2d(starting_channels, starting_channels, (3, 3), device=device),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(starting_channels, device=device),
                                   nn.ConvTranspose2d(starting_channels, ending_channels, (3, 3), padding=(1, 1), device=device),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(ending_channels, device=device),
                                   nn.Dropout(0.25))

            return packet

        self.group1 = nn.Sequential(nn.Linear(embed_dim, 512, device=device),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(512, device=device))
        # reshape 512 -> 2 x 2 x 128
        # nn.functional.interpolate()

        self.group2 = create_conv_packet(128, 64)

        # nn.functional.interpolate()

        self.group3 = create_conv_packet(64, 32)

        # nn.functional.interpolate()

        self.group4 = nn.Sequential(nn.ConvTranspose2d(32, 32, (3, 3), device=device),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(32, device=device),
                                    nn.ConvTranspose2d(32, output_channels, (3, 3), padding=(1, 1), device=device))

    def forward(self, input_data, output_shape):
        """

        :param input_data:
        :return:
        """

        out = self.group1(input_data)
        out = torch.reshape(out, (out.shape[0], 128, 2, 2))
        out = nn.functional.interpolate(out, (4, 4))

        out = self.group2(out)
        out = nn.functional.interpolate(out, (13, 13))

        out = self.group3(out)
        out = nn.functional.interpolate(out, (30, 30))

        out = self.group4(out)

        return out
