import os
from pathlib import Path

from PIL import Image
from torch.utils.data import IterableDataset
from torchvision.transforms import transforms as T


class PairIterator:
    """
    Iterator that yields paired SAR and RGB image tensors.

    This iterator wraps a directory iterator (like the one returned by
    `os.scandir`) and, on each iteration, loads a SAR image and the
    corresponding RGB image (the RGB path is derived from the SAR path by
    replacing `_s1_` with `_s2_` and `sar` with `rgb`). It applies the
    appropriate torchvision transforms to return normalized tensors.

    Parameters
    ----------
    it : iterator
        An iterator yielding directory entries or objects with a `path`
        attribute (for example, the iterator returned by `os.scandir`).

    Attributes
    ----------
    it : iterator
        The underlying directory iterator.
    sar_transform : torchvision.transforms.Compose
        Transform applied to SAR PIL images to produce a float tensor in
        range [0, 1].
    rgb_transform : torchvision.transforms.Compose
        Transform applied to RGB PIL images to produce a float tensor in
        range [0, 1].

    Methods
    -------
    __iter__()
        Returns the iterator object itself.
    __next__()
        Loads the next SAR/RGB image pair and returns a tuple
        `(sar_tensor, rgb_tensor)`.
    """

    def __init__(self, it):
        self.it = it
        self.sar_transform = T.Compose(
            [T.Grayscale(), T.ToTensor(), lambda x: x / 255.0]
        )
        self.rgb_transform = T.Compose([T.PILToTensor(), lambda x: x / 255.0])

    def __iter__(self):
        return self

    def __next__(self):
        sar_path = next(self.it).path
        rgb_path = sar_path.replace("_s1_", "_s2_").replace("sar", "rgb")

        sar_img = Image.open(sar_path)
        rgb_img = Image.open(rgb_path)

        return (self.sar_transform(sar_img), self.rgb_transform(rgb_img))


class OptiSARDataset(IterableDataset):
    """
    Iterable dataset that yields paired SAR and RGB image tensors from disk.

    This dataset implements the iterable dataset interface from
    `torch.utils.data`. It expects a root directory and the relative
    subdirectory names for SAR and RGB training images. Iteration scans the
    SAR directory and yields paired tensors produced by `PairIterator`.

    Parameters
    ----------
    root_dir : str
        Path to the dataset root directory.
    train_sar : str
        Name of the subdirectory under `root_dir` containing SAR images.
    train_rgb : str
        Name of the subdirectory under `root_dir` containing RGB images.

    Attributes
    ----------
    root_dir : pathlib.Path
        Resolved path to the dataset root directory.
    train_sar_dir : pathlib.Path
        Path to the SAR images directory.
    train_rgb_dir : pathlib.Path
        Path to the RGB images directory.

    Raises
    ------
    TypeError
        If any of the constructor arguments are not of type `str`.

    Notes
    -----
    The dataset uses `PairIterator` internally and assumes SAR filenames
    contain `_s1_` and their corresponding RGB filenames can be derived by
    replacing `_s1_` with `_s2_` and `sar` with `rgb`.
    """

    def __init__(self, root_dir: str, train_sar: str, train_rgb: str):
        super().__init__()
        if not isinstance(root_dir, str):
            raise TypeError(f"Expected type of root_dir is str. Got {type(root_dir)}.")
        if not isinstance(train_sar, str):
            raise TypeError(
                f"Expected type of train_sar is str. Got {type(train_sar)}."
            )
        if not isinstance(train_rgb, str):
            raise TypeError(
                f"Expected type of train_rgb is str. Got {type(train_rgb)}."
            )
        self.root_dir = Path(root_dir)
        self.train_sar_dir = self.root_dir / train_sar
        self.train_rgb_dir = self.root_dir / train_rgb

    def __iter__(self):
        with (os.scandir(self.train_sar_dir) as it_sar,):
            yield from PairIterator(it_sar)
