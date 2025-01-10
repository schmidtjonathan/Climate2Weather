import copy
import os
import pickle
import time

import numpy as np
import torch
import util
import wandb
from dataset import InfiniteSampler
from lightning.fabric import Fabric

from thor.checkpoint import CheckpointIO
from thor.score import DefaultScoreFunction

# The training procedure is largely based on
# https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/training_loop.py#L25
# and adapted for distributed training using pytorch lightning.


def training_loop(
    fabric: Fabric,
    run_dir,
    #
    dataset_kwargs,
    network_kwargs,
    pipeline_kwargs,
    optimizer_kwargs,
    lr_kwargs,
    #
    batch_size,
    batch_gpu,
    total_ndata,
    log_ndata,
    status_ndata,
    snapshot_ndata,
    checkpoint_ndata,
    valid_ndata,
    #
    ema_kwargs=None,
    slice_ndata=None,
    seed=0,
    loss_scaling=1,
    cudnn_benchmark=True,
    logger=None,
):
    # Initialize.
    prev_status_time = time.time()
    util.set_random_seed(seed, fabric.global_rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    DO_LOG = logger is not None and log_ndata is not None

    # Validate batch size.
    batch_gpu_total = batch_size // fabric.world_size
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * fabric.world_size
    assert total_ndata % batch_size == 0
    assert slice_ndata is None or slice_ndata % batch_size == 0
    assert log_ndata is None or log_ndata % batch_size == 0
    assert status_ndata is None or status_ndata % batch_size == 0
    assert snapshot_ndata is None or (
        snapshot_ndata % batch_size == 0 and snapshot_ndata % 1024 == 0
    )
    assert checkpoint_ndata is None or (
        checkpoint_ndata % batch_size == 0 and checkpoint_ndata % 1024 == 0
    )

    # ==| Dataset(s)
    fabric.print("Setting up datasets...")
    train_dataset = util.construct_class_by_name(**dataset_kwargs.train)
    DO_VALIDATION = False
    if "valid" in dataset_kwargs:
        fabric.print(
            "WARNING: Validation dataset provided but currently not supported."
        )
        DO_VALIDATION = True
        valid_dataset = util.construct_class_by_name(**dataset_kwargs.valid)

    # ==| Network
    fabric.print("Setting up network...")

    with fabric.init_module():
        net = util.construct_class_by_name(**network_kwargs)

    net = fabric.to_device(net)
    net.train()

    # --| Print model summary
    if fabric.is_global_zero:
        ref_data = train_dataset[0]
        fabric.print(f"Data shape: {ref_data.shape}")
        with fabric.autocast():
            util.print_module_summary(
                net,
                [
                    torch.zeros(
                        [1, *ref_data.shape],
                        device=fabric.device,
                    ),
                    torch.ones([1], device=fabric.device),
                ],
                max_nesting=6,
            )

    # ==| Setup training state.
    fabric.print("Setting up training state...")
    state = util.EasyDict(cur_ndata=0, total_elapsed_time=0)
    # Prepare the model for distributed training.
    # The model is moved automatically to the right device.
    ddp = fabric.setup_module(net)  # NOTE: `net is ddp.module` ~> `True`

    pipeline = util.construct_class_by_name(**pipeline_kwargs)
    optimizer = util.construct_class_by_name(
        params=net.parameters(), **optimizer_kwargs
    )
    # Prepare the optimizer for distributed training.
    optimizer = fabric.setup_optimizers(optimizer)

    ema = (
        util.construct_class_by_name(net=net, **ema_kwargs)
        if ema_kwargs is not None
        else None
    )

    # Load previous checkpoint and decide how long to train.
    checkpoint = CheckpointIO(
        state=state,
        net=net,
        pipeline=pipeline,
        optimizer=optimizer,
        ema=ema,
    )
    checkpoint.load_latest(fabric, run_dir)
    stop_at_ndata = total_ndata
    if slice_ndata is not None:
        granularity = (
            checkpoint_ndata
            if checkpoint_ndata is not None
            else snapshot_ndata
            if snapshot_ndata is not None
            else batch_size
        )
        slice_end_ndata = (
            (state.cur_ndata + slice_ndata) // granularity * granularity
        )  # round down
        stop_at_ndata = min(stop_at_ndata, slice_end_ndata)
    assert stop_at_ndata > state.cur_ndata
    fabric.print(
        f"Training from {state.cur_ndata // 1000} kdata to {stop_at_ndata // 1000} kdata:"
    )
    fabric.print()
    fabric.print(
        f"Batch size: {batch_size} (per device: {batch_gpu}; number of accumulation rounds: {num_accumulation_rounds})"
    )
    fabric.print()

    # ==| Main training loop
    dataset_sampler = InfiniteSampler(
        dataset=train_dataset,
        rank=fabric.global_rank,
        num_replicas=fabric.world_size,
        shuffle=True,
        seed=seed,
        start_idx=state.cur_ndata,
    )

    # ==| Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=dataset_sampler,
        batch_size=batch_gpu,
        pin_memory=True,
        num_workers=2,
        prefetch_factor=2,
    )
    if DO_VALIDATION:
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            shuffle=False,
            batch_size=batch_gpu,
            pin_memory=True,
            num_workers=2,
            prefetch_factor=2,
            drop_last=False,
        )
        valid_loader = fabric.setup_dataloaders(valid_loader)

    train_loader = fabric.setup_dataloaders(train_loader, use_distributed_sampler=False)
    dataset_iterator = iter(train_loader)

    prev_status_ndata = state.cur_ndata
    cumulative_training_time = 0
    start_ndata = state.cur_ndata

    # def sampling_proc_x0(x):
    #     return torch.clamp(x, -1.5, 1.5)

    losses_accum = []

    while True:
        done = state.cur_ndata >= total_ndata

        # Report status.
        if (
            status_ndata is not None
            and (done or state.cur_ndata % status_ndata == 0)
            and (state.cur_ndata != start_ndata or start_ndata == 0)
        ):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            fabric.print(
                " +++ ".join(
                    [
                        "Status:",
                        f"{state.cur_ndata} / {total_ndata} ({state.cur_ndata / total_ndata:.2%})",
                        f"{state.total_elapsed_time:.2f} sec total",
                        f"{cur_time - prev_status_time:.2f} sec/tick",
                        f"{cumulative_training_time / max(state.cur_ndata - prev_status_ndata, 1) * 1e3:.3f} sec/kdata",
                    ]
                )
            )

            cumulative_training_time = 0
            prev_status_ndata = state.cur_ndata
            prev_status_time = cur_time

        # Save network snapshot.
        if (
            snapshot_ndata is not None
            and state.cur_ndata % snapshot_ndata == 0
            and (state.cur_ndata != start_ndata)
            and fabric.is_global_zero
        ):
            ema_list = (
                ema.get()
                if ema is not None
                else optimizer.get_ema(net)
                if hasattr(optimizer, "get_ema")
                else net
            )
            ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, "")]

            for ema_net, ema_suffix in ema_list:
                snap_data = util.EasyDict(
                    dataset_kwargs=dataset_kwargs, pipeline=pipeline
                )
                snap_data.ema = (
                    copy.deepcopy(ema_net)
                    .cpu()
                    .eval()
                    .requires_grad_(False)
                    .to(torch.float16)
                )
                fname = (
                    f"network-snapshot-{state.cur_ndata // 1000:07d}{ema_suffix}.pkl"
                )
                fabric.print(f"Saving {fname} ... ", end="", flush=True)
                with open(os.path.join(run_dir, fname), "wb") as f:
                    pickle.dump(snap_data, f)
                fabric.print("done")
                del snap_data  # conserve memory

        # Validation
        if (
            valid_ndata is not None
            and state.cur_ndata % valid_ndata == 0
            and (state.cur_ndata != start_ndata or start_ndata == 0)
        ):
            # Log samples
            if fabric.is_global_zero:
                ema_list = (
                    ema.get()
                    if ema is not None
                    else (
                        optimizer.get_ema(net) if hasattr(optimizer, "get_ema") else net
                    )
                )
                ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, "")]
                with fabric.autocast():
                    noisevec = torch.randn(
                        dataset_kwargs.train.window,
                        dataset_kwargs.train.num_features,
                        dataset_kwargs.train.spatial_res,
                        dataset_kwargs.train.spatial_res,
                    ).to(device=fabric.device)

                for ema_net, ema_suffix in ema_list:
                    _ema_module_train = ema_net.training
                    ema_net.train(False)  # Set to eval mode
                    score_function = DefaultScoreFunction(
                        ema_net,
                        markov_order=dataset_kwargs.train.window // 2,
                        noise_process=pipeline,
                    )
                    with torch.no_grad():
                        with fabric.autocast():
                            gen_sample = pipeline.sample(
                                score_function,
                                noisevec,
                                steps=100,
                                # proc_x0=sampling_proc_x0,  # 1
                                device=fabric.device,
                            ).cpu()

                    ema_net.train(_ema_module_train)  # Restore training/eval mode

                    img_array = util.trajectory_to_imgrid(gen_sample)
                    if DO_LOG:
                        _hist_fig = wandb.Image(util.value_histogram(gen_sample))
                        imgs = wandb.Image(
                            img_array, caption="Samples [time x features]"
                        )
                        logger.log(
                            {
                                f"gen_sample{ema_suffix}": imgs,
                                f"value_histogram{ema_suffix}": _hist_fig,
                            },
                            commit=False,
                        )

            if DO_VALIDATION:
                fabric.print(
                    "WARNING: Validation dataset provided but currently not supported."
                )

        # Logging
        if (
            DO_LOG
            and log_ndata is not None
            and (done or state.cur_ndata % log_ndata == 0)
            and (state.cur_ndata != start_ndata)
        ):
            logger.log(
                {
                    "train/loss": np.mean(losses_accum),
                    "train/kdata": state.cur_ndata // 1000,
                    "train/elapsed_time": state.total_elapsed_time,
                    **{
                        f"train/lr-{i}": g["lr"]
                        for i, g in enumerate(optimizer.param_groups, 1)
                    },
                },
            )
            losses_accum = []

        # Save state checkpoint.
        if (
            checkpoint_ndata is not None
            and (done or state.cur_ndata % checkpoint_ndata == 0)
            and state.cur_ndata != start_ndata
        ):
            checkpoint.save(
                fabric,
                os.path.join(
                    run_dir, f"training-state-{state.cur_ndata // 1000:07d}.ckpt"
                ),
            )

        # Done?
        if done:
            break

        # Evaluate loss and accumulate gradients.
        batch_start_time = time.time()
        # util.set_random_seed(seed, fabric.global_rank, state.cur_ndata)
        optimizer.zero_grad()
        for round_idx in range(num_accumulation_rounds):
            is_accumulating = round_idx != num_accumulation_rounds - 1
            with fabric.no_backward_sync(ddp, enabled=is_accumulating):
                data = next(dataset_iterator)
                loss = pipeline.loss(net=ddp, x=data).mean().mul(loss_scaling)
                fabric.backward(loss)

        lr = util.call_func_by_name(cur_ndata=state.cur_ndata, **lr_kwargs)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.step()
        losses_accum.append(loss.detach().item())

        # Update EMA and training state.
        state.cur_ndata += batch_size
        if ema is not None:
            ema.update(cur_ndata=state.cur_ndata, batch_size=batch_size)
        cumulative_training_time += time.time() - batch_start_time
