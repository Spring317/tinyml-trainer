from typing import Tuple
from torchvision.transforms import functional  # type: ignore


class CentralCropResize:
    """
    A custom image transformation that performs a central crop followed by resizing and normalization.

    This transform first crops the central region of the input image according to a specified 
    fraction of the original size, then resizes the cropped region to a target size. 
    After resizing, pixel values are normalized to the range [-1, 1].

    Args:
        central_fraction (float, optional): 
            The fraction of the image to retain around the center.
            Defaults to 0.875 (keep 87.5% of the center).
        size (Tuple[int, int], optional): 
            The target (height, width) size to which the image will be resized.
            Defaults to (224, 224).

    Notes:
        - Input `img` is expected to be a PIL Image.
        - Output is a normalized torch.Tensor.
        - The final normalization scales pixel values from [0, 1] (after ToTensor) 
          to [-1, 1] using `img = (img - 0.5) * 2.0`.
    """
    def __init__(self, central_fraction=0.875, size: Tuple[int, int] = (224, 224)):
        self.central_fraction = central_fraction
        self.size = list(size)

    def __call__(self, img):
        img = functional.to_tensor(img)
        _, h, w = img.shape
        crop_h = int(h * self.central_fraction)
        crop_w = int(w * self.central_fraction)
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2

        img = functional.crop(img, top, left, crop_h, crop_w)
        img = functional.resize(
            img, self.size, interpolation=functional.InterpolationMode.BILINEAR
        )

        return img