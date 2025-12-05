import torch.nn as nn
from torch.nn.utils import spectral_norm


class ConvInstanceNormLeakyReLUBlock(nn.Module):
    """
    Small convolutional block: Conv2d -> InstanceNorm2d -> LeakyReLU.

    This helper module encapsulates a 2D convolution followed by
    instance normalization and a LeakyReLU activation. It is used inside
    the `Discriminator` to build a sequence of progressively deeper
    feature extractors.

    Parameters
    ----------
    in_channels : int
        Number of input channels for the convolution.
    out_channels : int
        Number of output channels for the convolution.
    **kwargs : dict
        Additional keyword arguments forwarded to `nn.Conv2d` (e.g.
        `kernel_size`, `stride`, `padding`).

    Attributes
    ----------
    block : torch.nn.Sequential
        The sequential module performing conv -> norm -> activation.
    """

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()

        self.block = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    """
    PatchGAN-style discriminator for image patches.

    The discriminator is implemented as a sequence of convolutional
    layers that reduce spatial resolution and increase the number of
    feature channels, finishing with a single-channel output map which
    represents real/fake scores for image patches.

    Parameters
    ----------
    in_channels : int
        Number of input image channels.
    features : list of int, optional
        Sequence of channel sizes for intermediate layers (default:
        [64, 128, 256, 512]).

    Attributes
    ----------
    model : torch.nn.Sequential
        The composed discriminator network.

    Raises
    ------
    TypeError
        If `in_channels` is not an `int` or `features` is not a list-like
        of integers.
    """

    def __init__(self, in_channels: int, features: list = [64, 128, 256, 512]):
        super().__init__()
        if not isinstance(in_channels, int):
            raise TypeError(
                f"Expected type of in_channels is int. Got {type(in_channels)}."
            )
        if not isinstance(features, list):
            raise TypeError(f"Expected type of features is list. Got {type(features)}.")

        layers = [
            spectral_norm(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=features[0],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    padding_mode="reflect",
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_ch = features[0]
        for feature in features[1:]:
            layers.append(
                ConvInstanceNormLeakyReLUBlock(
                    in_ch,
                    feature,
                    kernel_size=4,
                    stride=1 if feature == features[-1] else 2,
                    padding=1,
                    padding_mode="reflect",
                )
            )
            in_ch = feature

        layers.append(
            spectral_norm(
                nn.Conv2d(
                    in_channels=features[-1],
                    out_channels=1,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    padding_mode="reflect",
                )
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
