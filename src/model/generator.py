import torch
import torch.nn as nn

from src.model.convolution import ConvolutionalBlock
from src.model.residual import ResidualBlock


class Generator(nn.Module):
    """
    Image-to-image generator network with downsampling, residual, and upsampling stages.

    This generator follows a common architecture for image translation:
    an initial convolutional layer, two downsampling convolutional blocks,
    a stack of residual blocks operating at a bottleneck resolution, and
    two upsampling blocks followed by a final convolution. The output is
    passed through a `tanh` activation to produce values in [-1, 1].

    Parameters
    ----------
    img_channels_in : int
        Number of input image channels.
    img_channels_out : int
        Number of output image channels.
    num_features : int, optional
        Number of feature maps in the first convolution (default: 64).
    num_residuals : int, optional
        Number of residual blocks at the bottleneck (default: 6).

    Attributes
    ----------
    initial_layer : torch.nn.Sequential
        Initial conv -> norm -> activation layer.
    downsampling_layers : torch.nn.ModuleList
        Two convolutional blocks that reduce spatial resolution.
    residual_layers : torch.nn.Sequential
        Stack of `ResidualBlock` instances operating at bottleneck channels.
    upsampling_layers : torch.nn.ModuleList
        Two transpose-convolutional blocks that restore spatial resolution.
    last_layer : torch.nn.Conv2d
        Final convolution producing the output image channels.

    Raises
    ------
    TypeError
        If any of the constructor arguments are of the wrong type.
    """

    def __init__(
        self,
        img_channels_in: int,
        img_channels_out: int,
        num_features: int = 64,
        num_residuals: int = 6,
    ):
        super().__init__()
        if not isinstance(img_channels_in, int):
            raise TypeError(
                f"Expected type of img_channels_in is int. Got {type(img_channels_in)}."
            )
        if not isinstance(img_channels_out, int):
            raise TypeError(
                f"Expected type of img_channels_out is int. Got {type(img_channels_out)}."
            )
        if not isinstance(num_features, int):
            raise TypeError(
                f"Expected type of num_features is int. Got {type(num_features)}."
            )
        if not isinstance(num_residuals, int):
            raise TypeError(
                f"Expected type of num_residuals is int. Got {type(num_residuals)}."
            )
        self.initial_layer = nn.Sequential(
            nn.Conv2d(
                img_channels_in,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )

        self.downsampling_layers = nn.ModuleList(
            [
                ConvolutionalBlock(
                    num_features,
                    num_features * 2,
                    is_downsampling=True,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                ConvolutionalBlock(
                    num_features * 2,
                    num_features * 4,
                    is_downsampling=True,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        self.residual_layers = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )

        self.upsampling_layers = nn.ModuleList(
            [
                ConvolutionalBlock(
                    num_features * 4,
                    num_features * 2,
                    is_downsampling=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ConvolutionalBlock(
                    num_features * 2,
                    num_features * 1,
                    is_downsampling=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )

        self.last_layer = nn.Conv2d(
            num_features * 1,
            img_channels_out,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial_layer(x)
        for layer in self.downsampling_layers:
            x = layer(x)
        x = self.residual_layers(x)
        for layer in self.upsampling_layers:
            x = layer(x)
        return torch.tanh(self.last_layer(x))
