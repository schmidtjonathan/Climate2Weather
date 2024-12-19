import os
import pathlib
import pickle
from datetime import datetime
from typing import Optional, Sequence, Union

import fire
import numpy as np
import torch
import yaml
from data import pipeline as data_pipeline
from lightning.fabric import Fabric
from util import set_random_seed

import thor


def run(
    save_path: str,
    config_path: str,
    devices: Optional[int] = None,
    **kwargs,
):

    torch.set_float32_matmul_precision("medium")

    devices = devices if devices is not None else "auto"
    fabric = Fabric(
        accelerator="cuda",
        devices=devices,
        num_nodes=1,
        precision="16-mixed",
    )
    fabric.launch()

    fabric.print(f"Loading config from {config_path}")
    config_path = pathlib.Path(config_path)
    save_path = pathlib.Path(save_path)

    if save_path.exists():
        subdir_i = len([s for s in save_path.iterdir() if s.is_dir()]) + 1
    else:
        subdir_i = 1

    save_path = save_path / f"{subdir_i:03d}_{config_path.stem}"

    if not (
        config_path.exists()
        and config_path.is_file()
        and config_path.suffix.lower() in [".yaml", ".yml"]
    ):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.unsafe_load(f)

    for k, v in config.items():
        if k in kwargs:
            fabric.print(f">>> CONFIG: Overwriting value for {k}: {v} -> {kwargs[k]}")
            config[k] = kwargs[k]

    if fabric.is_global_zero:
        save_path.mkdir(parents=True, exist_ok=False)
        with open(save_path / "config_freeze.yaml", "w") as f:
            yaml.dump(config, f)

    _run_impl(fabric, save_path=save_path, **config)
    fabric.print("Done. \n")


def _run_impl(
    fabric: Fabric,
    save_path: pathlib.Path,
    model_path: str,
    data_path: str,
    quantile_path: str,
    start_time: str,
    num_hours: int,
    data_norm_mode: str,
    use_exact_grad: bool = False,
    observation_path: Optional[str] = None,
    data_vars: tuple = ("psl", "tas", "uas", "vas"),
    num_sampling_steps: int = 256,
    num_samples: int = 1,
    num_corrections: int = 2,
    likelihood_std: Union[float, Sequence[float]] = 1e-2,
    likelihood_gamma: Union[float, Sequence[float]] = 1e-2,
    correction_tau: float = 0.5,
    seed: int = 0,
    t_step: int = 6,
    s_step: int = 16,
    batch_size: int = 16,
    # **absorb,
):

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    assert (
        num_samples % fabric.world_size == 0
    ), "num_samples must be divisible by world_size"
    num_samples_per_gpu = num_samples // fabric.world_size

    data_vars = list(sorted(data_vars))

    set_random_seed(seed, fabric.global_rank)

    fabric.print(f"STARTING EXPERIMENT `exp000` AT {run_timestamp} >>> \n\n\n")
    fabric.print(f"Running on {fabric.world_size} GPUs")
    fabric.print(f"Saving results to {save_path}")

    # Load setup from snapshot
    with open(model_path, "rb") as f:
        snapshot_data = pickle.load(f)
    # 1. dataset
    markov_window = snapshot_data["dataset_kwargs"]["train"]["window"]
    markov_order = markov_window // 2

    # 2. forward process
    pipeline = snapshot_data["pipeline"]
    fabric.print(f"Loaded pipeline of type {type(pipeline)}")

    # 3. trained network
    fabric.print(
        f"Loading score network from {model_path}\n >  trained on window size: {snapshot_data['dataset_kwargs']['train']['window']}, order: {markov_order}"
    )

    net = fabric.to_device(snapshot_data["ema"])
    net.eval()

    ## The measurement model
    spatial_obs_fn = torch.nn.AvgPool2d(s_step, stride=s_step, padding=0)

    def select_spatiotemporal(x):
        return spatial_obs_fn(x[..., ::t_step, :, :, :])

    # >>> LOAD THE GT DATA (COSMO)
    unnormed_cosmo_dataset = data_pipeline.load_processed(
        data_path,
        data_vars,
        start_time,
        num_hours,
        do_nan_check=False,
    )
    if fabric.is_global_zero:
        unnormed_cosmo_dataset.to_netcdf(
            os.path.join(save_path, "ground_truth.nc"),
            encoding={k: {"dtype": "float32"} for k in data_vars},
        )

    fabric.print(f"Quantile path: {quantile_path}")

    cosmo_dataset = data_pipeline.normalize_ds(
        unnormed_cosmo_dataset,
        quantile_ds=quantile_path,
        mode=data_norm_mode,
    )

    ground_truth = torch.from_numpy(
        data_pipeline.ds_to_sorted_np(cosmo_dataset, data_vars)
    )
    L, C, H, W = ground_truth.shape
    # <<<

    # Create observation from it
    DO_CONDITION = True
    if observation_path is None:
        DO_CONDITION = False
        fabric.print("No observation provided. Sampling without conditioning.")
    elif observation_path == data_path:
        fabric.print(
            f"Conditioning on observations of the ground truth at {observation_path}"
        )
        observation = select_spatiotemporal(
            ground_truth
        ).clone()  # average_pooling_4d(ground_truth, s_step)[::t_step, ...]
        observation_ds = (
            cosmo_dataset.coarsen(rlat=s_step, rlon=s_step)
            .mean()
            .isel(time=slice(0, num_hours, t_step))
        )
    else:
        fabric.print(f"Conditioning on provided observation at {observation_path}")
        observation_ds = data_pipeline.load_processed(
            observation_path,
            data_vars,
            start_time,
            num_hours,
            do_nan_check=False,
        )
        observation_ds = data_pipeline.normalize_ds(
            observation_ds,
            quantile_ds=quantile_path,
            mode=data_norm_mode,
        )
        observation = torch.from_numpy(
            data_pipeline.ds_to_sorted_np(observation_ds, data_vars)
        )

    if DO_CONDITION and fabric.is_global_zero:
        observation_ds = data_pipeline.unnormalize_ds(
            observation_ds,
            quantile_ds=quantile_path,
            mode=data_norm_mode,
        )
        observation_ds.to_netcdf(
            os.path.join(save_path, "observation.nc"),
            encoding={k: {"dtype": "float32"} for k in data_vars},
        )

    score_function = thor.score.BatchedScoreFunction(
        net,
        markov_order=markov_order,
        noise_process=pipeline,
        batch_size=batch_size,
        device=fabric.device,
    )

    if DO_CONDITION:
        fabric.print(
            f"Observation shape: {observation.shape}, type: {type(observation)}"
        )

        if isinstance(likelihood_std, (list, tuple)):
            sigma = torch.zeros(1, C, 1, 1)
            for c in range(C):
                sigma[:, c, ...] = likelihood_std[c]

        else:
            sigma = likelihood_std

        if isinstance(likelihood_gamma, (list, tuple)):
            gamma = torch.zeros(1, C, 1, 1)
            for c in range(C):
                gamma[:, c, ...] = likelihood_gamma[c]
        else:
            gamma = likelihood_gamma

        score_function = score_function.condition_on(
            A=select_spatiotemporal,
            y=observation,
            std=sigma,
            gamma=gamma,
            exact_grad=use_exact_grad,
        )

    # draw initial noise to generate sample from

    fabric.print("Starting sampling...")

    for nsmpl in range(num_samples_per_gpu):
        sample_id = fabric.global_rank * num_samples_per_gpu + nsmpl
        print(f"[Rank {fabric.global_rank}] Generating sample {sample_id}...")
        noise_vec = torch.randn(L, C, H, W)
        with fabric.autocast():
            cur_gen_sample = (
                pipeline.sample(
                    score_function,
                    noise_vec,
                    steps=num_sampling_steps,
                    corrections=num_corrections,
                    tau=correction_tau,
                    show_progressbar=fabric.is_global_zero,
                )
                .to(torch.float32)
                .cpu()
                .numpy()
            )

        sample_ds = data_pipeline.np_to_ds(
            cur_gen_sample,
            reference_ds=cosmo_dataset,
            data_vars=data_vars,
        )

        # De-Normalize & Save results
        sample_ds = data_pipeline.unnormalize_ds(
            sample_ds,
            quantile_ds=quantile_path,
            mode=data_norm_mode,
        )

        sample_ds.to_netcdf(
            save_path / f"gen_sample_{sample_id:03d}.nc",
            encoding={k: {"dtype": "float32"} for k in data_vars},
        )

    fabric.print(f"Saved results to {save_path}")

    return save_path


def sweep_likelihood_hparams(save_path, config_path, devices, trials, base_seed=99):

    set_random_seed(base_seed, 0)

    sigma_dist = np.logspace(-4, 0.2, 100)
    gamma_dist = np.logspace(-4, 0.2, 100)

    cur_sigma = float(np.random.choice(sigma_dist))
    cur_gamma = float(np.random.choice(gamma_dist))

    for n in range(1, trials + 1):
        set_random_seed(base_seed, n)
        cur_sigma_draw = np.random.choice(sigma_dist, size=3, replace=True)
        cur_sigma = tuple(
            ([float(s) for s in cur_sigma_draw] + [float(cur_sigma_draw[-1])])
        )
        cur_gamma = float(np.random.choice(gamma_dist))
        print(f"Trial {n+1}: sigma = {cur_sigma}, gamma = {cur_gamma}")
        try:
            run(
                save_path=save_path,
                config_path=config_path,
                devices=devices,
                likelihood_std=cur_sigma,
                likelihood_gamma=cur_gamma,
                num_samples=10,
                num_hours=49,
                num_corrections=0,
                seed=base_seed,
            )
        except Exception as e:
            print(f"Trial {n+1} failed: {e}")


if __name__ == "__main__":
    fire.Fire(
        {
            "run": run,
            "hparam_sweep": sweep_likelihood_hparams,
        }
    )
