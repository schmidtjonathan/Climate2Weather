import re

import xarray as xr


def setup(exp_dir, load_samples=True, load_gt=True, load_obs=True):
    def preproc(ds):
        fn = ds.encoding["source"]
        pattern = r"^.*(\d{3}).nc$"
        re_match = re.match(pattern, fn)
        if re_match is None:
            raise ValueError(f"Could not match pattern {pattern} to {fn}")
        sample_id = int(re_match.group(1))
        return ds.expand_dims("sample_id").assign_coords(sample_id=[sample_id])

    # Load sample(s), ground truth
    sample_path = list(sorted(exp_dir.glob("gen_sample*.nc")))
    print(f"Found {len(sample_path)} sample files")
    gt_path = exp_dir / "ground_truth.nc"
    obs_path = exp_dir / "observation.nc"

    if load_samples:
        sample_ds = xr.open_mfdataset(
            sample_path,
            compat="override",
            preprocess=preproc,
            data_vars="minimal",
            coords="minimal",
            join="override",
            combine_attrs="drop",
        )
        sample_ds["psl"] = sample_ds["psl"] / 100.0
    else:
        sample_ds = None

    if load_gt:
        gt_ds = xr.open_dataset(gt_path, chunks="auto")
        gt_ds["psl"] = gt_ds["psl"] / 100.0
    else:
        gt_ds = None

    if load_obs:
        try:
            obs_ds = xr.open_dataset(obs_path)
            obs_ds["psl"] = obs_ds["psl"] / 100.0
        except FileNotFoundError:
            obs_ds = None
    else:
        obs_ds = None

    return sample_ds, gt_ds, obs_ds
