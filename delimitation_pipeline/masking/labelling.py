import numpy as np

from PIL import Image
from matplotlib.colors import rgb_to_hsv
from sklearn.cluster import KMeans

__all__ = [
    "load_img_arr",
    "cluster_img"
]

def load_img_arr(path):
    """Load an image from a path and convert to a numpy array.
    """
    return np.array(Image.open(path))


def cluster_img(img):
    """Apply k-means clustering with 3 clusters to separate out parts of a specimen image.
    """
    img_hsv = rgb_to_hsv(img / 255)
    kmeans = KMeans(n_clusters=3)

    img_arr = img_hsv.reshape(img.shape[0] * img.shape[1], 3)
    img_labels = kmeans.fit_predict(img_arr)

    return img_labels, kmeans
