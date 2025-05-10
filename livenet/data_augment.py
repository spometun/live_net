import numpy as np
from scipy import ndimage
import torch


def _elastic_transform(img: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    """
    Apply a smooth 2-D "elastic" shift so that the image centre moves by `shift`,
    while all four edges remain unmoved.

    Parameters
    ----------
    img : ndarray, H×W×C
        Input image array.
    shift : (shift_x, shift_y)
        Horizontal and vertical shift (integer pixels) at the image center.

    Returns
    -------
    warped : ndarray
        Transformed image, same shape and dtype as `img`.
    """
    assert img.ndim == 3, "Input must be H×W×C"
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2, (h - 1) / 2

    # 1D "tent" weights: 1 at centre, 0 at edges
    x = np.arange(w)
    y = np.arange(h)
    wx = 1 - np.abs((x - cx) / cx)
    wy = 1 - np.abs((y - cy) / cy)
    WX, WY = np.meshgrid(wx, wy)

    # per-pixel offset fields
    dx = WX * shift[1]
    dy = WY * shift[0]

    # original sampling grid
    XX, YY = np.meshgrid(np.arange(w), np.arange(h))
    coords = [YY + dy, XX + dx]  # row_coords, col_coords

    warped = np.empty_like(img)
    for c in range(img.shape[2]):
        warped[..., c] = ndimage.map_coordinates(
            img[..., c], coords, order=1, mode='reflect'
        )
    return warped


class RandomElasticShift:
    """
    A torchvision-style transform that applies a smooth 2D "elastic" shift
    so that the image centre moves randomly within [-max_shift, max_shift] on each axis,
    while keeping the edges fixed.

    Input: torch.Tensor, shape CxH×W
    Output: torch.Tensor, same shape and dtype as input
    """
    def __init__(self, max_shift: int, seed: int = 0):
        """
        Parameters
        ----------
        max_shift : int
            Maximum absolute pixel shift at the centre (in pixels). Must be integer.
        seed : int
            Seed for reproducible random shifts. Defaults to 0.
        """
        if not isinstance(max_shift, int) or max_shift < 0:
            raise ValueError("max_shift must be a non-negative integer")
        self.max_shift = max_shift
        self.rng = np.random.default_rng(seed)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply the random elastic shift.

        Parameters
        ----------
        img : torch.Tensor, shape H×W×C
            Input image. Any dtype (float or integer) is preserved.

        Returns
        -------
        torch.Tensor
            Elastically shifted image, same shape, dtype, and device as input.
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError("RandomElasticShift only accepts torch.Tensor")
        if img.ndim != 3:
            raise ValueError("Input tensor must have shape H×W×C")

        # sample random integer shifts in [-max_shift, max_shift], inclusive
        shift_x = self.rng.integers(-self.max_shift, self.max_shift, endpoint=True)
        shift_y = self.rng.integers(-self.max_shift, self.max_shift, endpoint=True)

        # numpy round-trip
        device, dtype = img.device, img.dtype
        arr = img.detach().cpu().numpy()

        arr = arr.transpose(1, 2, 0)
        warped = _elastic_transform(arr, (shift_y, shift_x))
        warped = warped.transpose(2, 0, 1)

        # back to torch, same dtype & device
        out = torch.from_numpy(warped).to(device=device, dtype=dtype)
        return out


class RandomRotateCrop:
    """
    A transform that applies a random rotation within [-max_rotation, max_rotation] degrees (using SciPy's ndimage.rotate),
    then crops back to the original size by selecting one of five positions: four corners or the center.

    Input: torch.Tensor of shape C×H×W
    Output: torch.Tensor of shape C×H×W, same dtype and device.
    """

    def __init__(self, max_rotation: float, seed: int = 0):
        """
        Parameters
        ----------
        max_rotation : float
            Maximum absolute rotation angle in degrees.
        seed : int
            Seed for reproducible randomness.
        """
        self.max_rotation = max_rotation
        self.rng = np.random.default_rng(seed)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply the random rotation and fixed-size crop.

        Parameters
        ----------
        img : torch.Tensor, shape C×H×W
            Input image.

        Returns
        -------
        torch.Tensor
            Transformed image, same shape, dtype, and device.
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if img.ndim != 3:
            raise ValueError("Input tensor must be C×H×W")
        C, H, W = img.shape

        angle = float(self.rng.uniform(-self.max_rotation, self.max_rotation))

        # convert to numpy for ndimage
        device, dtype = img.device, img.dtype
        arr = img.detach().cpu().numpy()

        # rotate around axes 1 (H) and 2 (W), expanding output
        rotated = ndimage.rotate(arr, angle, axes=(1, 2), order=1, mode="reflect")

        _, H2, W2 = rotated.shape
        # integer margins for center crop back to (H, W)
        margin_x = (W2 - W) // 2
        margin_y = (H2 - H) // 2

        # crop positions: top-left, top-right, bottom-left, bottom-right, center
        positions = [ (0, 0), (0, 2 * margin_x), (2 * margin_y, 0), (2 * margin_y, 2 * margin_x), (margin_y, margin_x) ]
        idx = int(self.rng.integers(0, len(positions)))
        y0, x0 = positions[idx]

        cropped = rotated[:, y0: y0 + H, x0: x0 + W]
        # back to torch
        out = torch.from_numpy(cropped).to(device=device, dtype=dtype)
        return out

