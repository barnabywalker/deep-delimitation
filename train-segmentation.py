#' training script for fastai based UNet model, for specimen segmentation.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import wandb

from fastcore.xtras import Path
from fastai.callback.schedule import lr_find, fit_one_cycle
from fastai.callback.progress import CSVLogger
from fastai.callback.training import MixedPrecision
from fastai.callback.wandb import *
from fastai.data.transforms import get_image_files, FuncSplitter, RandomSplitter

from fastai.vision.data import SegmentationDataLoaders
from fastai.vision.augment import aug_transforms, Resize, RandomResizedCrop, IntToFloatTensor
from fastai.vision.learner import unet_learner

from fastai.metrics import Dice
from fastai.optimizer import ranger

from torchvision.models.resnet import resnet34, resnet18

from argparse import ArgumentParser

def clear_pyplot_memory():
    plt.clf()
    plt.cla()
    plt.close()


def create_dls(root, img_dir, mask_dir, size=256, batch_size=32, splitter=None):
    if splitter is None:
        splitter = RandomSplitter(valid_pct=0.2)

    dls = SegmentationDataLoaders.from_label_func(
        root,
        fnames=get_image_files(img_dir),
        label_func=lambda o: mask_dir/f"{o.stem}_mask{o.suffix}",
        splitter=splitter,
        codes=["background", "specimen"],
        item_tfms=[Resize(int(size*2), method="pad", pad_mode="zeros")],
        batch_tfms=[
            *aug_transforms(pad_mode="zeros"),
            RandomResizedCrop(size=size, min_scale=0.25, max_scale=0.9, ratio=(1, 1)),
            IntToFloatTensor(div_mask=255)
        ],
        bs=batch_size
    )

    return dls

def main():
    parser = ArgumentParser()
    parser.add_argument("cmd", choices=["lr_find", "train"])
    parser.add_argument("name", type=str)
    
    parser.add_argument("--root", default="data/masking", type=str)
    parser.add_argument("--images", default="images/original_hires_images", type=str)
    parser.add_argument("--masks", default="masks/processed", type=str)
    parser.add_argument("--metadata", default=None, type=str)
    parser.add_argument("--save_dir", default="output", type=str)
    
    parser.add_argument("--valid_pct", default=0.2, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--img_size", default=256, type=int)

    parser.add_argument("--backbone", default="resnet34", type=str, choices=["resnet18", "resnet34"])
    parser.add_argument('--self-attention', dest='attention', action='store_true')
    parser.set_defaults(attention=False)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--epochs", default=10)
    parser.add_argument("--two-stage", dest="two_stage", action="store_true")
    parser.set_defaults(two_stage=False)

    args = parser.parse_args()

    root = Path(args.root)
    img_dir = root/args.images
    mask_dir = root/args.masks

    if args.metadata is not None:
        metadata = pd.read_csv(root/args.metadata, sep="\t")
        valid_barcodes = metadata[metadata.valid_set_equals_1 == 1].CatBarcode.values
        splitter = FuncSplitter(lambda o: int(Path(o).stem) in valid_barcodes)
    else:
        splitter = RandomSplitter(valid_pct=args.valid_pct)

    
    dls = create_dls(root, img_dir, mask_dir, size=args.img_size, splitter=splitter, batch_size=args.batch_size)

    dls.show_batch(show=True)
    plt.savefig("segmentation-batch.png")
    clear_pyplot_memory()

    if args.backbone == "resnet18":
        encoder = resnet18
    elif args.backbone == "resnet34":
        encoder = resnet34

    out_path = Path(args.save_dir)
    outdir = out_path/args.name
    modeldir = outdir/"model"

    model = unet_learner(dls, encoder, metrics=Dice(), self_attention=args.attention, cbs=CSVLogger(),
                         path=outdir, model_dir=modeldir)

    if args.cmd == "lr_find":
        model.lr_find(show_plot=True)
        plt.savefig(outdir/"lr-plot.png")
        clear_pyplot_memory()

    elif args.cmd == "train":
        wandb.init(project='unet-segmenter')

        callbacks = [MixedPrecision(), WandbCallback()]
        model.fit_one_cycle(args.epochs, slice(args.lr), cbs=callbacks)

        if args.two_stage:
            lrs = slice(args.lr / 400, args.lr / 4)
            model.unfreeze()

            model.fit_one_cycle(args.epochs, lrs, cbs=callbacks)

        model.show_results(show_plot=True)
        plt.savefig(outdir/"results-plot.png")
        clear_pyplot_memory()

        model.save("final-model")
    

if __name__ == "__main__":
    main()