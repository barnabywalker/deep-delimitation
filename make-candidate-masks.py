import os

from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image

from delimitation_pipeline.masking import load_img_arr, cluster_img

def make_masks(img_paths, outdir):
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for i, path in enumerate(tqdm(img_paths, desc="saving masks")):
        if i % 100 == 1:
            print(f"{i} masks created")

        name = path.split("/")[-1]
        img = load_img_arr(path)
        clusters, _ = cluster_img(img)
        
        for i in range(3):
            mask = img.reshape(img.shape[0] * img.shape[1], 3).copy()
            mask[clusters != i] = 0
            mask[clusters == i] = 255

            n = name.split(".")[0]
            ext = name.split(".")[-1]
            im = Image.fromarray(mask.reshape(img.shape))
            im.save(os.path.join(outdir, f"{n}_{i}.{ext}"))


def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--imgdir", default="mask-images", type=str)
    parser.add_argument("--outdir", default="mask-candidates", type=str)

    args = parser.parse_args()

    paths = [os.path.join(args.imgdir, p) for p in os.listdir(args.imgdir)]
    make_masks(paths, args.outdir)


if __name__ == "__main__":
    cli_main()
