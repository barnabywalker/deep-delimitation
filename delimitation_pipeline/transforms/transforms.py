from torchvision import transforms
from PIL import Image

__all__ = [
    "CropAspect"
]

class CropAspect(object):
    """Centre crop an image along its longest dimension to achieve a particular aspect ratio.

    Args:
        aspect (float): Desired aspect ratio of output image. Image dimensions are
        integers, so the realised aspect ratio might not match this exactly.
    """
    def __init__(self, aspect=1):
        assert isinstance(aspect, (float, int))
        self.output_aspect = aspect

    def __call__(self, img):
        if isinstance(img, Image.Image):
            w, h = img.size
        else:
            h, w = img.shape[:2]

        new_h = (w // self.output_aspect) if h > w else h
        new_w = (h // self.output_aspect) if w > h else w

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        return transforms.functional.crop(img, top, left, new_h, new_w)
        