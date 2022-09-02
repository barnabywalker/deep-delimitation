from torchvision import transforms
from ..transforms import CropAspect

class SimCLRTransforms:
    def __init__(self, input_height=256, guassian_blur=True, jitter_strength=1):
        tfms = [
            CropAspect(aspect=1),
            transforms.ToTensor()
        ]

        jitter = transforms.ColorJitter([
            0.8 * jitter_strength,
            0.8 * jitter_strength,
            0.8 * jitter_strength,
            0.2 * jitter_strength
        ])

    
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
            tfms.append(transforms.GaussianBlur(kernel_size=kernel_size, p=0.5))

        self.transform = transforms.Compose(tfms)

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform(sample)

        return x1, x2

