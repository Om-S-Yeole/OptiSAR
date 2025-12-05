import torch.nn as nn

from src.model.convolution import ConvolutionalBlock


class ResidualBlock(nn.Module):
    """
    Residual block that adds input to a small convolutional subnetwork.

    The block implements the classical residual connection used in many
    image-to-image networks: a pair of convolutional sub-blocks is applied
    to the input and the result is added back to the input tensor.

    Parameters
    ----------
    channels : int
        Number of input and output channels for the residual block.

    Attributes
    ----------
    block : torch.nn.Sequential
        The sequential subnetwork that computes the residual to be added
        to the input.

    Raises
    ------
    TypeError
        If `channels` is not an `int`.
    """

    def __init__(self, channels: int):
        super().__init__()
        if not isinstance(channels, int):
            raise TypeError(f"Expected type of channels is int. Got {type(channels)}.")
        self.block = nn.Sequential(
            ConvolutionalBlock(
                channels, channels, add_activation=True, kernel_size=3, padding=1
            ),
            ConvolutionalBlock(
                channels, channels, add_activation=False, kernel_size=3, padding=1
            ),
        )

    def forward(self, x):
        return x + self.block(x)
