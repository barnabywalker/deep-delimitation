from torchvision import transforms
from PIL import Image

__all__ = [
    "CropAspect",
    "SimCLRTransforms"
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

class SimCLRTransforms(object):
    def __init__(self, input_height, guassian_blur=True, jitter_strength=1):
        tfms = [
            CropAspect(aspect=1),
            transforms.ToTensor()
        ]

        jitter = transforms.ColorJitter(
            0.8 * jitter_strength,
            0.8 * jitter_strength,
            0.8 * jitter_strength,
            0.2 * jitter_strength
        )

        tfms.extend([
            transforms.RandomResizedCrop(size=input_height),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ])

        if guassian_blur:
            kernel_size = int(0.1 * input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1
            tfms.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size)], p=0.5))

        self.transform = transforms.Compose(tfms)

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform(sample)

        return x1, x2
        