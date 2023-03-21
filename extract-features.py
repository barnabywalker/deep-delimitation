# Extract features from an image dataset using a pre-trained model.
import torch
import yaml
import json
import os
import pytorch_lightning as pl
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from torchvision import transforms
from torchvision.models import convnext_tiny, resnet18, resnet50
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from delimitation_pipeline.models import Classifier, SimCLR
from delimitation_pipeline.transforms import CropAspect
from delimitation_pipeline.datasets import HalfEarthDataset, SpecimenDataset, HalfEarthModule


def cli_main():
    #----------
    # args    |
    #----------
    parser = ArgumentParser()
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--root_dir", type=str, default=None)

    parser.add_argument("--project", dest="project", action="store_true")
    parser.set_defaults(project=False)
    
    parser.add_argument('--half_earth', dest='half_earth', action='store_true')
    parser.add_argument('--not_half_earth', dest='half_earth', action='store_false')
    parser.set_defaults(half_earth=False)

    parser.add_argument("--seed", type=int, default=46290)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    #----------
    # dirs    |
    #----------
    root_dir = args.data_dir if args.root_dir is None else args.root_dir

    version = args.version

    versions = os.listdir("_".join([args.name, "species-delimitation"]))
    if version is None:
        version = int(versions[-1].split("_")[0])
    
    run_id = [v.split("_")[-1] for v in versions if v.startswith(str(version))][0]

    logs = os.listdir("wandb")
    log_name = [l for l in logs if l.endswith(run_id)][0]

    model_dir = os.path.join("_".join([args.name, "species-delimitation"]), f"{version}_{run_id}")
    log_dir = os.path.join("wandb", log_name)
    output_dir = os.path.join("output", args.name, f"version_{version}")

    with open(os.path.join(log_dir, "files", "config.yaml"), "r") as infile:
        model_params = yaml.safe_load(infile)

    with open(os.path.join(log_dir, "files", "wandb-metadata.json"), "r") as infile:
        model_args = json.load(infile)["args"]
    
    model_args = {a.split("=")[0].strip("--"): a.split("=")[-1] for a in model_args}
    #----------
    # data    |
    #----------
    device = torch.device("cuda:0" if (args.device == "gpu") else "cpu")
    ds = HalfEarthDataset(root=root_dir, target_type=model_params["target"]["value"])

    tfms = transforms.Compose([
        transforms.ToTensor(),
        CropAspect(aspect=1),
        transforms.Resize(args.image_size)
    ])

    if args.half_earth:
        dm = HalfEarthModule(data_dir=args.data_dir, target_type="name", balanced=False, shuffle=False,
                             batch_size=args.batch_size, num_workers=args.num_workers)
        
        dm.setup(stage="fit")
        dls = [("train", dm.train_dataloader()), ("val", dm.val_dataloader())]

        dm.setup(stage="test")
        test_dls = dm.test_dataloader()
        dls.extend([
            ("test-herb", test_dls[0]),
            ("test-taxon", test_dls[1]),
            ("test-specimens", test_dls[2])
        ])
        feature_ds = ds
    else:
        feature_ds = SpecimenDataset(args.data_dir, transform=tfms)
        dls = [("specimens", DataLoader(feature_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True))]
    
    print(f"Extracting features from {len(feature_ds)} images in {[len(dl) for dl in dls]} batches (of {args.batch_size} images)")

    #-----------
    # model    |
    #-----------
    if model_params["backbone"]["value"] == "convnext":
        encoder = convnext_tiny(pretrained=True)
    elif model_params["backbone"]["value"] == "resnet18":
        encoder = resnet18(pretrained=True)
    elif model_params["backbone"]["value"] == "resnet50":
        encoder = resnet50(pretrained=True)

    encoder = nn.Sequential(*list(encoder.children())[:-1], nn.Flatten())

    hparams = {
        "encoder": encoder,
        "feat_dim": model_params["feat-dim"]["value"],
        "proj_layers": model_params["proj-layers"]["value"],
        "lr": model_params["learning-rate"]["value"]
    }
    if model_params["model-type"]["value"] == "classifier":
        hparams = {
            **hparams, 
            "class_index": ds.categories_index,
            "class_map": ds.categories_map, 
            "target": model_params["target"]["value"],
            "hierarchical": model_params["hierarchical-loss"]["value"],
            "nsteps": int(len(ds) / model_params["batch-size"]["value"]) * int(model_args["max_epochs"]),
            "scheduler": model_params["scheduler"]["value"],
            "max_lr": model_params["max-learning-rate"]["value"]
        }

        model = Classifier(**hparams)
    elif model_params["model-type"]["value"] == "simclr":
        hparams = {
            **hparams,
            "temperature": model_params["temperature"]["value"],
            "max_epochs": int(model_args["max_epochs"]),
            "train_iters_per_epoch": int(len(ds) / model_params["batch-size"]["value"]),
            "scheduler": model_params["scheduler"]["value"]
        }

        model = SimCLR(**hparams)
    
    latest_chkpt = os.listdir(os.path.join(model_dir, "checkpoints"))[-1]
    model = model.load_from_checkpoint(os.path.join(model_dir, "checkpoints", latest_chkpt),
                                       strict=False, **hparams)

    extractor = model.encoder.to(device).eval()
    projector = model.projection.to(device).eval()

    print(f"Recreated model {args.name} from {latest_chkpt}...")

    #----------
    # latents |
    #----------
    print(f"saving features from {args.data_dir} to {output_dir}...")
    for name, dl in dls:
        features = []
        labels = []
        for img, label in tqdm(dl, desc="generating features"):
            feats = extractor(img.to(device))
            if args.project:
                feats = projector(feats)
            
            features.append(feats.detach().cpu().numpy())
            labels.extend(label)

        features = np.vstack(features)
        with open(os.path.join(output_dir, f"features_{args.save_name}-{name}.npy"), "wb") as outfile:
            np.save(outfile, features)

        pd.DataFrame(labels, columns=["label"]).to_csv(os.path.join(output_dir, f"feature-labels_{args.save_name}.csv"), index=False)

    print("finished!")

if __name__ == "__main__":
    cli_main()

