from torchvision import transforms  # type: ignore


class ColorDistorter:
    """
    A custom image transformation that applies randomized color distortions (brightness, contrast, saturation, hue) using `ColorJitter` in different operation orders.

    Args:
        ordering (int):
            An integer specifying the order in which color transformations are applied.
            The order is determined modulo 4, supporting 4 distinct patterns.
        brightness (float, optional):
            Maximum change in brightness. Defaults to 32.0/255.0 (~0.125).
        contrast (float, optional):
            Maximum change in contrast. Defaults to 0.5.
        saturation (float, optional):
            Maximum change in saturation. Defaults to 0.5.
        hue (float, optional):
            Maximum change in hue shift. Defaults to 0.2.

    Transformation Orders:
        Based on `ordering % 4`, the following operation sequences are used:
            - 0 → Brightness → Saturation → Hue → Contrast
            - 1 → Saturation → Brightness → Contrast → Hue
            - 2 → Contrast → Hue → Brightness → Saturation
            - 3 → Hue → Saturation → Contrast → Brightness
    """

    def __init__(
        self,
        ordering: int,
        brightness: float = 32.0 / 255.0,
        contrast: float = 0.5,
        saturation: float = 0.5,
        hue: float = 0.2,
    ):
        self.ordering = ordering
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self._brightness_jitter = transforms.ColorJitter(brightness=self.brightness)
        self._contrast_jitter = transforms.ColorJitter(saturation=self.saturation)
        self._saturation_jitter = transforms.ColorJitter(saturation=self.saturation)
        self._hue_jitter = transforms.ColorJitter(hue=self.hue)

    def __call__(self, img):
        ops = {
            0: [self._brightness, self._saturation, self._hue, self._contrast],
            1: [self._saturation, self._brightness, self._contrast, self._hue],
            2: [self._contrast, self._hue, self._brightness, self._saturation],
            3: [self._hue, self._saturation, self._contrast, self._brightness],
        }
        for fn in ops[self.ordering % 4]:
            img = fn(img)
        return img

    def _brightness(self, img):
        return self._brightness_jitter(img)

    def _saturation(self, img):
        return self._saturation_jitter(img)

    def _contrast(self, img):
        return self._contrast_jitter(img)

    def _hue(self, img):
        return self._hue_jitter(img)
