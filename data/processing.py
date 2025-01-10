import os
import pathlib
import re

import fire
import h5py
import numpy as np
import tqdm
import xarray as xr
from dask.diagnostics import ProgressBar
from pipeline import normalize_ds

##


def _filter_files_by_years(file_list, start_year, end_year):
    # Regular expression to match the required file format
    file_pattern = re.compile(r"^.+_(?P<start_time>\d{12})-.+\.nc$")

    # List to store the filtered file names
    filtered_files = []

    for f in file_list:
        match = file_pattern.match(f.name)
        if not match:
            raise ValueError(f"File name '{f.name}' does not match the required format")

        start_time = match.group("start_time")
        start_year_file = int(start_time[:4])  # Extract the year part of the start_time

        # Check if the variable name matches and the start year is within the specified range
        if start_year <= start_year_file <= end_year:
            filtered_files.append(f)

    return filtered_files


def _analyze_nan(da):
    is_null_mask = da.isnull()
    if not is_null_mask.any().item():
        return None

    dim_ordering = da.dims
    time_dim = tuple(dim_ordering).index("time")
    nr_of_times = is_null_mask.sum(dim="time").max().item()
    tlatlon_null_indices = np.argwhere(is_null_mask.to_numpy())
    which_times = np.unique(tlatlon_null_indices[:, time_dim])
    assert len(which_times) == nr_of_times

    print(f"Found {nr_of_times} times with NaNs.")

    return {
        "nr_of_times": nr_of_times,
        "which_times": which_times,
        "null_indices": tlatlon_null_indices,
        "dim_ordering": dim_ordering,
    }


def full_cosmo_to_interpolated_patch(var_folder, out_folder, years=None):
    repo_path = os.environ.get("REPO_ROOT_DIR", None)
    if repo_path is None:
        raise RuntimeError("REPO_ROOT_DIR environment variable not set.")
    repo_path = pathlib.Path(repo_path)
    patch_mask_path = repo_path / "data" / "COSMO_patch_index-ranges.npz"

    var_folder = pathlib.Path(var_folder)
    assert var_folder.exists() and var_folder.is_dir(), (
        f"{var_folder} is not a directory."
    )
    varname = var_folder.name
    print(var_folder)
    print(varname)

    var_patch_folder = pathlib.Path(out_folder)
    var_patch_folder.mkdir(exist_ok=True, parents=True)

    file_list = list(sorted(var_folder.glob("*.nc")))
    if len(file_list) == 0:
        raise ValueError(
            f"No netCDF files found in {var_folder}. Remember that this script has to be called per variable."
        )
    print(f"Originally found {len(file_list)} files:")
    print("  > ", file_list[0].name, "  -->  ", file_list[-1].name)
    if years is not None:
        assert isinstance(years, str)
        years = tuple(map(int, years.split("-")))
        assert len(years) == 2
        start_year, end_year = years
        assert isinstance(start_year, int) and isinstance(end_year, int)
        assert 1995 <= start_year <= end_year <= 2019

        file_list = _filter_files_by_years(file_list, start_year, end_year)
        print(f"Found {len(file_list)} files that match the provided years {years}")
        print(f"  >  {file_list[0].name}  -->  {file_list[-1].name}")

    print(f"Saving data to {var_patch_folder}")
    patch_idx = np.load(patch_mask_path)
    lats = tuple(patch_idx["lats"])
    lons = tuple(patch_idx["lons"])

    to_netcdf_kwargs = {
        "engine": "h5netcdf",
        "encoding": {varname: {"chunksizes": (24, 128, 128)}},
    }

    for var_file in tqdm(file_list):
        var_patch_file = var_patch_folder / var_file.name
        # Load patch

        print(
            f"Processing file {var_file.name} with variable {varname}. Saving to {var_patch_file.name}"
        )

        ds = xr.open_dataset(var_file, engine="h5netcdf")
        try:
            ds = ds.isel(
                rlat=slice(*lats),
                rlon=slice(*lons),
            )

            da = ds[varname]

            found_null = _analyze_nan(da)
            if found_null is not None:
                print(f"Found NaNs in {var_file}")

                da.to_netcdf(
                    var_patch_file.with_name(
                        var_patch_file.stem + "_with_missing"
                    ).with_suffix(".nc"),
                    **to_netcdf_kwargs,
                )

                null_stats_file = var_patch_file.with_name(
                    var_patch_file.stem + "_nan_stats"
                ).with_suffix(".npz")
                np.savez(null_stats_file, **found_null)
                interp_da = da.interpolate_na(dim="time", method="pchip")

                found_null_after_interp = _analyze_nan(interp_da)
                if found_null_after_interp is None:
                    interp_da.to_netcdf(
                        var_patch_file.with_name(
                            var_patch_file.stem + "_interpolated"
                        ).with_suffix(".nc"),
                        **to_netcdf_kwargs,
                    )
                else:
                    print(f"Found NaNs after interpolation in {var_file}. Giving up.")
            else:
                da.to_netcdf(var_patch_file, **to_netcdf_kwargs)
        except Exception as e:
            ds.close()
            raise e
        ds.close()


##


def ds_to_sorted_np(ds, data_vars, ordering="LCHW"):
    assert ordering in ["LCHW", "CLHW"], f"Invalid ordering: {ordering}"

    data_vars = list(sorted(data_vars))
    data = np.stack(
        [ds[v].values for v in data_vars], axis=0 if ordering == "CLHW" else 1
    )
    return data


def _day_iter(ds):
    num_days = len(ds.time) // 24
    for i in range(num_days):
        yield ds.isel(time=slice(i * 24, (i + 1) * 24))


def merged_nc_to_normed_h5(
    merged_file_path, quantiles_filepath, h5_out_file, norm_mode: str
):
    nc_file = pathlib.Path(merged_file_path)
    quantiles_file = pathlib.Path(quantiles_filepath)
    h5_out_file = pathlib.Path(h5_out_file)

    dask_pbar = ProgressBar(minimum=5, dt=1.0)
    dask_pbar.register()

    print(f"Normalizing using mode {norm_mode}")

    merged_ds = xr.open_dataset(nc_file).transpose("time", "rlat", "rlon")
    quantiles_ds = xr.load_dataset(quantiles_file)

    print("Quantiles:")
    print(quantiles_ds)

    data_vars = list(sorted(merged_ds.data_vars))

    print(merged_ds)

    num_days = len(merged_ds.time) // 24

    print(f"Processing data variables {data_vars} for {num_days} days")

    h5_dataset = None
    with h5py.File(h5_out_file, "w") as f:
        f.create_dataset(
            "vars",
            data=data_vars,
            dtype=h5py.string_dtype(),
            shape=len(data_vars),
        )
        f.create_dataset(
            "norm_mode",
            data=[norm_mode],
            dtype=h5py.string_dtype(),
            shape=1,
        )

        for chunk_24h in _day_iter(merged_ds):
            normed_chunk = normalize_ds(
                chunk_24h,
                quantiles_ds,
                mode=norm_mode,
            )
            data2write = ds_to_sorted_np(normed_chunk, data_vars, ordering="LCHW")

            if np.any(np.isnan(data2write)):
                raise ValueError("NaNs in data. Aborting.")

            data_shape = data2write.shape
            if h5_dataset is None:
                h5_dataset = f.create_dataset(
                    "x",
                    data=data2write,
                    dtype=np.float32,
                    chunks=tuple(data_shape),
                    maxshape=(None, *data_shape[1:]),
                )
                print(f"Inital shape: {h5_dataset.shape} [data: {data_shape}]")
            else:
                h5_dataset.resize(h5_dataset.shape[0] + data_shape[0], axis=0)
                h5_dataset[-data_shape[0] :] = data2write
                print(f"Resized shape: {h5_dataset.shape} [data: {data_shape}]")

    merged_ds.close()
    quantiles_ds.close()


if __name__ == "__main__":
    fire.Fire(
        {
            "to_normed_h5": merged_nc_to_normed_h5,
            "extract_patch": full_cosmo_to_interpolated_patch,
        }
    )
