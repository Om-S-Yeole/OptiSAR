import torch.nn as nn


class ConvolutionalBlock(nn.Module):
    """
    Convolutional block used for downsampling or upsampling layers.

    This block is a thin wrapper around either `nn.Conv2d` (for
    downsampling) or `nn.ConvTranspose2d` (for upsampling), followed by
    instance normalization and an optional activation. It is commonly used
    as a building block in encoder/decoder generator networks.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    is_downsampling : bool, optional
        If True (default), uses `nn.Conv2d`. If False, uses
        `nn.ConvTranspose2d` for upsampling.
    add_activation : bool, optional
        If True (default), appends a ReLU activation after normalization.
    **kwargs : dict
        Additional keyword arguments forwarded to the convolution
        constructor (for example `kernel_size`, `stride`, `padding`,
        `output_padding`).

    Attributes
    ----------
    conv : torch.nn.Sequential
        The sequential module performing conv -> norm -> activation.

    Notes
    -----
    The block uses `padding_mode="reflect"` for regular convolutions
    when that argument is provided in `**kwargs` by the caller.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_downsampling: bool = True,
        add_activation: bool = True,
        **kwargs,
    ):
        super().__init__()
        if not isinstance(in_channels, int):
            raise TypeError(
                f"Expected type of in_channels is int. Got {type(in_channels)}."
            )
        if not isinstance(out_channels, int):
            raise TypeError(
                f"Expected type of out_channels is int. Got {type(out_channels)}."
            )
        if not isinstance(is_downsampling, bool):
            raise TypeError(
                f"Expected type of is_downsampling is bool. Got {type(is_downsampling)}."
            )
        if not isinstance(add_activation, bool):
            raise TypeError(
                f"Expected type of add_activation is bool. Got {type(add_activation)}."
            )
        if is_downsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if add_activation else nn.Identity(),
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if add_activation else nn.Identity(),
            )

    def forward(self, x):
        return self.conv(x)
