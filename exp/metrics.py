import pathlib
import pickle

import numpy as np
import ot
import tqdm.auto as tqdm
from dask.diagnostics import ProgressBar
from scipy.stats import wasserstein_distance_nd

import exp.exputil as exputil


def compute_wasserstein_nd(sample_da, gt_da, sliced_wd=True):
    # Calculate Wasserstein distance

    def wd_fn(a, b):
        if sliced_wd:
            return ot.sliced_wasserstein_distance(a, b, n_projections=100, seed=0)
        else:
            return wasserstein_distance_nd(a, b)

    num_times = sample_da.sizes["time"]
    assert num_times == gt_da.sizes["time"]
    num_space = sample_da.sizes["rlat"] * sample_da.sizes["rlon"]
    assert num_space == gt_da.sizes["rlat"] * gt_da.sizes["rlon"]

    gt_vals = gt_da.values
    assert gt_vals.shape == (num_times, gt_da.sizes["rlat"], gt_da.sizes["rlon"])
    gt_vals = gt_vals.reshape(num_times, num_space)

    num_samples = sample_da.sizes["sample_id"] if "sample_id" in sample_da.dims else 1
    wasserstein_array = np.zeros((num_samples,))

    for smpl_id in tqdm.trange(num_samples, desc="Samples"):
        wasserstein_array[smpl_id] = wd_fn(
            (
                sample_da.isel(sample_id=smpl_id).values.reshape(num_times, num_space)
                if "sample_id" in sample_da.dims
                else sample_da.values.reshape(num_times, num_space)
            ),
            gt_vals,
        )

    return wasserstein_array


# --------------------------------------------------------------------------------------


def rapsd(sample_da, gt_da, obs_da, varname):
    import pysteps

    d = 6

    sample_rapsd_over_time = []
    gt_rapsd_over_time = []
    obs_rapsd_over_time = []

    # Calculate RAPSD
    bar = tqdm.tqdm(sample_da.time, desc=f"RAPSD {varname}")
    for t_time in bar:
        samples_rapsd = []
        bar.set_description(
            f"RAPSD {varname} ({t_time.dt.strftime('%Y-%m-%d %H:%M').item()})"
        )
        for s in range(sample_da.sizes["sample_id"]):
            _srapsd, _ = pysteps.utils.rapsd(
                sample_da.isel(sample_id=s).sel(time=t_time).values,
                fft_method=np.fft,
                return_freq=True,
                d=d,
                normalize=True,
            )
            samples_rapsd.append(_srapsd)
        samples_rapsd = np.stack(samples_rapsd)  # (samples, wavenumbers)

        gt_rapsd, wavenumbers = pysteps.utils.rapsd(
            gt_da.sel(time=t_time, method="nearest").values,
            fft_method=np.fft,
            return_freq=True,
            d=d,
            normalize=True,
        )

        sample_rapsd_over_time.append(samples_rapsd)  # (samples, wavenumbers, )
        gt_rapsd_over_time.append(gt_rapsd)  # (wavenumbers, )

        obs_rapsd, obs_wavenumbers = pysteps.utils.rapsd(
            obs_da.sel(time=t_time, method="nearest").values,
            fft_method=np.fft,
            return_freq=True,
            d=d * 16,
            normalize=True,
        )
        obs_rapsd_over_time.append(obs_rapsd)  # (wavenumbers, )

    sample_rapsd_over_time = np.stack(
        sample_rapsd_over_time, axis=1
    )  # (samples, time, wavenumbers)
    gt_rapsd_over_time = np.stack(gt_rapsd_over_time)  # (time, wavenumbers)
    obs_rapsd_over_time = np.stack(obs_rapsd_over_time)  # (time, wavenumbers)

    # Save results
    out = dict(
        wavelengths=1.0 / wavenumbers,
        obs_wavelengths=1.0 / obs_wavenumbers,
        sample_rapsd_over_time=sample_rapsd_over_time,
        gt_rapsd_over_time=gt_rapsd_over_time,
        obs_rapsd_over_time=obs_rapsd_over_time,
    )

    return out


def melr(
    sample_da,
    gt_da,
    varname,
    rapsd_dir,
    obs_da=None,
    do_weighted: bool = False,
    do_max: bool = False,
):
    assert (
        int(do_weighted) + int(do_max) < 2
    ), "At most one of do_weighted and do_max must be True"

    rapsd_path = pathlib.Path(rapsd_dir)

    # Load RAPSD results
    try:
        rapsd_results = np.load(rapsd_path / f"{varname}_rapsd.npz")
    except FileNotFoundError:
        print(f"RAPSD results at {rapsd_path} not found for variable {varname}")
        if obs_da is None:
            raise ValueError("obs_da must be provided if rapsd is not provided")
        print("Running RAPSD...")
        rapsd_results = rapsd(
            sample_da=sample_da, gt_da=gt_da, obs_da=obs_da, varname=varname
        )
        rapsd_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            rapsd_path / f"{varname}_rapsd.npz",
            **rapsd_results,
        )

    wavelengths = rapsd_results["wavelengths"]
    sample_rapsd_over_time = rapsd_results[
        "sample_rapsd_over_time"
    ]  # (samples, time, wavenumbers)
    gt_rapsd_over_time = rapsd_results["gt_rapsd_over_time"]  # (time, wavenumbers)

    N, T, num_wave = sample_rapsd_over_time.shape
    assert gt_rapsd_over_time.shape == (T, num_wave)
    assert len(wavelengths) == num_wave

    melr_over_time = []
    for t in tqdm.tqdm(range(T)):
        if do_max:
            max_energy_idx = np.argmax(gt_rapsd_over_time[t])
        elif do_weighted:
            weights = gt_rapsd_over_time[t] / np.sum(gt_rapsd_over_time[t])
        else:
            weights = np.full_like(gt_rapsd_over_time[t], 1 / num_wave)  # avg

        melr_per_sample = []
        for s in range(N):
            s_log_ratio = np.abs(
                np.log(sample_rapsd_over_time[s, t] / gt_rapsd_over_time[t])
            )

            if do_max:
                s_melr = s_log_ratio[max_energy_idx]
            else:
                s_melr = np.sum(s_log_ratio * weights)

            melr_per_sample.append(s_melr)
        melr_over_time.append(np.array(melr_per_sample))
    melr_over_time = np.stack(melr_over_time, axis=1)  # (samples, time)

    return melr_over_time.mean(axis=1)


# --------------------------------------------------------------------------------------


def ssim(sample_da, gt_da):
    from skimage.metrics import structural_similarity as ssim

    sample_arr = sample_da.values
    gt_arr = gt_da.values

    num_samples, num_timesteps = sample_arr.shape[:2]
    data_range = float(
        max(gt_arr.max(), sample_arr.max()) - min(gt_arr.min(), sample_arr.min())
    )

    ssim_values = np.zeros((num_samples, num_timesteps))
    for s in tqdm.tqdm(range(num_samples), desc="sample"):
        for t in tqdm.tqdm(range(num_timesteps), leave=False, desc="time"):
            ssim_values[s, t] = ssim(
                sample_arr[s, t],
                gt_arr[t],
                data_range=data_range,
                win_size=15,
                # gaussian_weights=True,
                # sigma=1.5,
                # use_sample_covariance=False,
                full=False,
            )

    return ssim_values.mean(1)


# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------


def run(exp_dir):

    dask_pbar = ProgressBar(minimum=5, dt=1.0)
    dask_pbar.register()

    exp_dir = pathlib.Path(exp_dir)

    print(f"Running metrics on experiment {exp_dir}")

    out_dir = exp_dir / "metrics"
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / "run"
    save_path.mkdir(exist_ok=True)

    sample_ds, gt_ds, obs_ds = exputil.setup(exp_dir)
    obs_ds = obs_ds.sel(time=slice(sample_ds.time.min(), sample_ds.time.max()))

    FEATURE_NAMES = list(sorted(gt_ds.data_vars))

    # Can only compare to baselines on observation grid, since only our approach
    # downscales temporally
    sample_ds = sample_ds.sel(dict(time=obs_ds.time))
    gt_ds = gt_ds.sel(dict(time=obs_ds.time))

    METRICS = {}
    METRICS["wasserstein"] = {}
    METRICS["melr"] = {}
    METRICS["ssim"] = {}
    for v in FEATURE_NAMES:
        # WASSERSTEIN

        METRICS["wasserstein"][v] = {}

        sample_da = sample_ds[v].compute()
        gt_da = gt_ds[v].compute()

        gtmean = gt_da.mean()
        gtstd = gt_da.std()

        sample_da_norm = (sample_da - gtmean) / gtstd
        gt_da_norm = (gt_da - gtmean) / gtstd

        wasserstein_2d = compute_wasserstein_nd(
            sample_da=sample_da_norm, gt_da=gt_da_norm, sliced_wd=True
        )

        METRICS["wasserstein"][v]["global"] = wasserstein_2d

        # MELR
        obs_da = obs_ds[v].compute()
        METRICS["melr"][v] = {}
        melr_array = melr(
            sample_da=sample_da,
            gt_da=gt_da,
            varname=v,
            rapsd_dir=exp_dir / "metrics" / "rapsd",
            obs_da=obs_da,
            do_weighted=False,
            do_max=False,
        )
        METRICS["melr"][v]["global"] = melr_array

        # SSIM
        METRICS["ssim"][v] = {}
        ssim_out = ssim(sample_da=sample_da, gt_da=gt_da)
        METRICS["ssim"][v]["global"] = ssim_out

    print(METRICS)
    for metrictype in METRICS:
        for datavar in FEATURE_NAMES:
            print(f"{metrictype} {datavar}")
            for k, v in METRICS[metrictype][datavar].items():
                try:
                    print(f"{k}: {v.mean():.4f} \\pm {v.std():.4f}")
                except:
                    print(f"{k}: {v:.4f}")

    with open(save_path / "metrics.pickle", "wb") as filehandle:
        pickle.dump(METRICS, filehandle)


def load(exp_dir):
    save_path = pathlib.Path(exp_dir) / "metrics" / "run" / "metrics.pickle"
    with open(save_path, "rb") as filehandle:
        loaded = pickle.load(filehandle)

    FEATURE_NAMES = ("psl", "tas", "uas", "vas")
    METRIC_TYPES = list(loaded.keys())
    METRICS = {}
    for k in METRIC_TYPES:
        METRICS[k] = dict(loaded[k])

    for metrictype in METRICS:
        print(f"{metrictype}")
        for datavar in FEATURE_NAMES:
            print(f"  {datavar}")
            for k, v in METRICS[metrictype][datavar].items():
                try:
                    print(f"    {v.mean():.4f} \pm {v.std():.4f}")
                except:
                    print(f"    {v:.4f}")
        print("\n")
