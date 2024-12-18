import os

import click
import torch
import util
import wandb
import yaml
from lightning.fabric import Fabric
from training_loop import training_loop


def wandb_set_startup_timeout(seconds: int):
    assert isinstance(seconds, int)
    os.environ["WANDB__SERVICE_WAIT"] = f"{seconds}"


# ----------------------------------------------------------------------------
# Parse an integer with optional power-of-two suffix:
# 'Ki' = kibi = 2^10
# 'Mi' = mebi = 2^20
# 'Gi' = gibi = 2^30


def parse_ndata(s):
    if isinstance(s, int):
        return s
    if s.endswith("Ki"):
        return int(s[:-2]) << 10
    if s.endswith("Mi"):
        return int(s[:-2]) << 20
    if s.endswith("Gi"):
        return int(s[:-2]) << 30
    return int(s)


# fmt: off
@click.command()
# General
@click.option('--run-dir',          help='Where to save the results', metavar='DIR',                    type=str, required=True)
@click.option('--run-id',              help='Unique identifier for the run', metavar='STR',             type=str, required=True)
@click.option('--desc',             help='String to include in result dir name', metavar='STR',         type=str)
# Device
@click.option('--accelerator',      help='Which accelerator to use', metavar='STR',                     type=click.Choice(["cpu", "gpu", "cuda", "tpu", "auto"]), default="cuda")
@click.option('--devices',          help='Number of devices to use', metavar='INT',                     type=click.IntRange(min=1), default=1)
@click.option('--num-nodes',        help='Number of nodes to use', metavar='INT',                       type=click.IntRange(min=1), default=1)
@click.option('--strategy',         help='Which strategy to use', metavar='STR',                        type=click.Choice(["auto", "dp", "ddp", "ddp_spawn", "ddp_cpu", "xla", "deepspeed", "fsdp"]), default="auto")
# Data
@click.option('--train-data',      help='Path to the training dataset', metavar='ZIP|DIR|FILE',         type=str, required=True)
@click.option('--valid-data',      help='Path to the validation dataset', metavar='ZIP|DIR|FILE',       type=str, required=False)
@click.option('--spatial-res',     help='Spatial size of the data', metavar='INT',                      type=click.IntRange(min=4), required=True)
@click.option('--num-features',    help='Number of features of the data', metavar='INT',                type=click.IntRange(min=1), required=True)
@click.option('--cache-data/--no-cache-data',      help='Cache dataset in CPU memory',                  default=False, show_default=True)
# Model
@click.option('--markov-order',    help='Order of the Markov chain', metavar='INT',                     type=click.IntRange(min=1), default=3, show_default=True)
# Training
@click.option('--lr',               help='Max. learning rate', metavar='FLOAT',                         type=click.FloatRange(min=0.0, min_open=True), default=2e-4, show_default=True)
@click.option('--lr-decay-start',   help='Learning rate decay', metavar='NBATCHES',                     type=click.FloatRange(min=0.0), default=35000)
@click.option('--lr-rampup',        help='Learning rate rampup', metavar='MDATA',                       type=click.FloatRange(min=0.0), default=10)

@click.option('--total-ndata',      help='Total number of data points', metavar='NDATA',                type=parse_ndata, default="15Mi", show_default=True)
@click.option('--batch',            help='Total batch size', metavar='NDATA',                           type=click.IntRange(min=1), default=128, show_default=True)
@click.option('--batch-gpu',        help='Limit batch size per GPU', metavar='NDATA',                   type=click.IntRange(min=1), required=False)
# Logging
@click.option('--status',           help='Interval of status prints', metavar='NDATA',                  type=parse_ndata, default='20Ki', show_default=True)
@click.option('--snapshot',         help='Interval of network snapshots', metavar='NDATA',              type=parse_ndata, default='1Mi', show_default=True)
@click.option('--checkpoint',       help='Interval of training checkpoints', metavar='NDATA',           type=parse_ndata, default='2Mi', show_default=True)
@click.option('--logging',          help='Interval of logging', metavar='NDATA',                        type=parse_ndata, default='5Ki', show_default=True)
@click.option('--valid',            help='Interval of validation', metavar='NDATA',                     type=parse_ndata, default='1Mi', show_default=True)
@click.option('--log-alldevices/--log-firstdevice', help='Log on all devices',                          default=False, show_default=True)
@click.option('--slice-data',       help='Slice the data to a smaller size', metavar='NDATA',           type=parse_ndata, required=False)
@click.option('--seed',             help='Random seed', metavar='INT',                                  type=int, default=0, show_default=True)
@click.option('--wandb/--no-wandb', help='Enable/disable Weights & Biases',                             default=False, show_default=True)


# fmt: on
def main(**opts):
    wandb_set_startup_timeout(600)
    fabric, cfg, logger = setup_training(**opts)
    launch_training(fabric, cfg, logger)


def setup_training(
    run_dir,
    run_id,
    accelerator,
    devices,
    num_nodes,
    strategy,
    **opts,
):
    opts = util.EasyDict(opts)

    fabric = Fabric(
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision="16-mixed",
    )
    fabric.launch()

    # Set up run directory
    with fabric.rank_zero_first():
        # Create top level directory (as provided in CLI) if it not already exists
        if fabric.is_global_zero:
            if not os.path.exists(run_dir):
                print(f"Creating run directory: {run_dir}")
                os.makedirs(run_dir)

        # Set (and create) sub-directory for this run
        cur_run_dir = str(run_id)
        if opts.desc is not None:
            cur_run_dir += f"-{opts.desc}"
        cur_run_dir = os.path.join(run_dir, cur_run_dir)

        if fabric.is_global_zero and not os.path.isdir(cur_run_dir):
            print(f"Setting up run directory: {cur_run_dir}")
            os.makedirs(cur_run_dir)
            # Save opts to yaml file
            with open(os.path.join(cur_run_dir, "opts.yaml"), "w") as f:
                yaml.dump(dict(opts), f)

    """
        SET UP CONFIG
    """

    # Process command line options into config
    cfg = util.EasyDict()

    # Dataset
    window = 2 * opts.markov_order + 1
    common_dataset_kwargs = dict(
        class_name="dataset.COSMODataset",
        num_features=opts.num_features,
        spatial_res=opts.spatial_res,
        cached=opts.cache_data,
        window=window,
        flatten=True,
    )
    cfg.dataset_kwargs = util.EasyDict()
    cfg.dataset_kwargs.train = util.EasyDict(
        data_path=opts.train_data,
        **common_dataset_kwargs,
    )
    if opts.valid_data is not None:
        cfg.dataset_kwargs.valid = util.EasyDict(
            data_path=opts.valid_data,
            **common_dataset_kwargs,
        )

    cfg.total_ndata = opts.total_ndata
    cfg.batch_size = opts.batch
    cfg.batch_gpu = opts.batch_gpu
    cfg.log_ndata = opts.logging
    cfg.valid_ndata = opts.valid
    cfg.snapshot_ndata = opts.snapshot
    cfg.checkpoint_ndata = opts.checkpoint
    cfg.status_ndata = opts.status
    cfg.slice_ndata = opts.slice_data

    cfg.seed = opts.seed

    # Network
    with open("configs/sda_unet.yml", "r") as yf:
        _mdl_conf = yaml.full_load(yf)

    cfg.network_kwargs = util.EasyDict(
        class_name="model.score.ScoreUNet",
        channels=opts.num_features * window,
        spatial=2,
        activation=torch.nn.SiLU,
        **_mdl_conf,
    )

    # Optimizer
    cfg.optimizer_kwargs = util.EasyDict(
        class_name="torch.optim.AdamW",
        lr=opts.lr,
        weight_decay=1e-3,
        betas=[0.9, 0.999],
    )

    # Pipeline
    cfg.pipeline_kwargs = util.EasyDict(class_name="thor.pipelines.SDAPipeline")

    # EMA
    cfg.ema_kwargs = util.EasyDict(class_name="thor.ema.StandardEMA")

    cfg.lr_kwargs = util.EasyDict(
        func_name="thor.lr.linear_learning_rate_schedule",
        ref_lr=opts.lr,
        total_ndata=opts.total_ndata,
    )

    # --
    cfg.run_dir = cur_run_dir

    if fabric.is_global_zero:
        with open(os.path.join(cur_run_dir, "config.yaml"), "w") as f:
            yaml.dump(dict(cfg), f)

    # Set up logger
    if opts.wandb:
        WANDB_PROJECT_NAME = os.environ.get("WANDB_PROJECT_NAME", None)
        if WANDB_PROJECT_NAME is None:
            raise RuntimeError(
                "You chose to enable logging with Weights & Biases. "
                "Please set the WANDB_PROJECT_NAME environment variable to "
                "the w&b project that you want the training to be logged in."
            )
        if opts.log_alldevices:
            logger = wandb.init(
                project=WANDB_PROJECT_NAME,
                group=run_id,
                id=f"{run_id}-{fabric.global_rank}",
                config=cfg,
                resume="allow",
            )
        else:
            logger = (
                wandb.init(
                    project=WANDB_PROJECT_NAME,
                    group=run_id,
                    id=f"{run_id}-{fabric.global_rank}",
                    config=cfg,
                    resume="allow",
                )
                if fabric.is_global_zero
                else None
            )

        if logger is not None:
            logger.define_metric("train/kdata")
            # set all other train/ metrics to use this step
            logger.define_metric("train/*", step_metric="train/kdata")

    else:
        logger = None

    return fabric, cfg, logger


def launch_training(fabric, cfg, logger):

    training_loop(fabric=fabric, logger=logger, **cfg)

    fabric.print("Training complete.")


if __name__ == "__main__":
    main()
