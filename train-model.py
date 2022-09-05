from datetime import datetime
import os
import pytorch_lightning as pl

from argparse import ArgumentParser
from datetime import datetime
from torchvision.models import resnet18, resnet50, convnext_tiny
from torch import nn

from delimitation_pipeline.datasets import HalfEarthModule
from delimitation_pipeline.models import Classifier, SimCLR
from delimitation_pipeline.transforms import SimCLRTransforms
from delimitation_pipeline.utils import check_uncommitted, get_commit_hash

class UnimplementedError(Exception):
    pass

def cli_main():
    pl.seed_everything(420)

    #----------
    # args    |
    #----------
    parser = ArgumentParser()
    parser.add_argument("--model_type", default="classifier", type=str, choices=["classifier", "simclr"])
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--input_height", default=32, type=int)
    parser.add_argument("--feat_dim", default=128, type=int)
    parser.add_argument("--proj_layers", default=1, type=int)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--data", default="data/herbarium-2019-fgvc6", type=str)
    parser.add_argument("--target", default="name", type=str)
    parser.add_argument('--balanced', dest='balanced', action='store_true')
    parser.add_argument('--not-balanced', dest='balanced', action='store_false')
    parser.set_defaults(balanced=False)
    parser.add_argument('--hierarchical', dest='hierarchical', action='store_true')
    parser.add_argument('--not-hierarchical', dest='hierarchical', action='store_false')
    parser.set_defaults(hierarchical=False)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--max_lr", default=None, type=float)
    parser.add_argument("--scheduler", default=None, type=str, choices=[None, "one_cycle", "plateau"])
    parser.add_argument("--backbone", default="resnet18", type=str, choices=["resnet18", "resnet50", "convnext"])
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    print("parsed cli:")
    print(vars(args))

    #----------
    # setup   |
    #----------
    check_uncommitted(warn=False)

    latest_hash = get_commit_hash()
    name = f"{args.model_type}-{latest_hash}"

    print("setting up loggers...")
    csv_logger = pl.loggers.CSVLogger("lightning_logs", name=name)
    version = csv_logger.version
    
    if not os.path.exists(os.path.join("output", name)):
        os.mkdir(os.path.join("output", name))

    outdir = os.path.join("output", name, f"version_{version}")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    print(f"set up folders, saving to {outdir}")

    wandb_logger = pl.loggers.WandbLogger(project="species-delimitation", name=name)
    # log experiment settings

    wandb_logger.experiment.config.update({
        "batch-size": args.batch_size,
        "image-size": args.input_height,
        "feat-dim": args.feat_dim,
        "proj-layers": args.proj_layers,
        "temperature": args.temperature,
        "balanced-sampling": args.balanced,
        "hierarchical-loss": args.hierarchical if args.model_type == "classifier" else False,
        "learning-rate": args.lr,
        "target": args.target,
        "max-learning-rate": args.max_lr,
        "scheduler": args.scheduler if args.model_type != "simclr" else "LambdaLR",
        "backbone": args.backbone,
        "model-type": args.model_type
    })
    print("set up loggers!")

    #----------
    # data    |
    #----------
    dm = HalfEarthModule(data_dir=args.data, target_type=args.target, num_workers=args.num_workers, balanced=args.balanced)
    if args.model_type == "simclr":
        dm.train_transform = SimCLRTransforms(args.input_height)
        dm.val_transform = SimCLRTransforms(args.input_height)
    dm.setup(stage="fit")
    
    ds = dm.train_dataloader().dataset.dataset

    print(f"created dataloader from {dm.data_dir}")

    #----------
    # model   |
    #----------
    
    if args.backbone == "resnet18":
        encoder = resnet18(pretrained=True)
    elif args.backbone == "resnet50":
        encoder = resnet50(pretrained=True)
    elif args.backbone == "convnext":
        encoder = convnext_tiny(pretrained=True)
        
    encoder = nn.Sequential(*list(encoder.children())[:-1], nn.Flatten())

    if args.model_type == "classifier":
        model = Classifier(
            encoder, 
            dm.target, 
            ds.categories_index, 
            ds.categories_map, 
            feat_dim=args.feat_dim,
            proj_layers=args.proj_layers,
            hierarchical=args.hierarchical,
            lr=args.lr, 
            max_lr=args.max_lr, 
            scheduler=args.scheduler, 
            nsteps=len(dm.train_dataloader())*args.max_epochs
        )
    elif args.model_type == "simclr":
        model = SimCLR(
            encoder,
            feat_dim=args.feat_dim,
            proj_layers=args.proj_layers, 
            temperature=args.temperature, 
            max_epochs=args.max_epochs,
            train_iters_per_epoch=len(dm.dataset_train)
        )

    print("set up model")

    #-----------
    # training |
    #-----------
    wandb_logger.watch(model)
    trainer = pl.Trainer.from_argparse_args(args, logger=[csv_logger, wandb_logger])
    trainer.fit(model, dm)

    #----------
    # testing |
    #----------
    dm.setup(stage="test")
    trainer.test(datamodule=dm)

if __name__ == "__main__":
    cli_main()
