import pathlib
import re
from datetime import datetime, timedelta

import numpy as np
import xarray as xr


def filter_files_by_yearmonth(file_list, start_yearmonth, end_yearmonth):
    YEAR_DIGITS = 4
    MONTH_DIGITS = 2

    start_yearmonth = int(start_yearmonth)
    end_yearmonth = int(end_yearmonth)

    # Regular expression to match the required file format
    file_pattern = re.compile(r"^.+_(?P<start_time>\d{12})-.+\.nc$")

    # List to store the filtered file names
    filtered_files = []

    for f in file_list:
        match = file_pattern.match(f.name)
        if not match:
            continue

        start_time = match.group("start_time")
        start_yearmonth_file = int(start_time[: YEAR_DIGITS + MONTH_DIGITS])

        # Check if the variable name matches and the start year is within the specified range
        if start_yearmonth <= start_yearmonth_file <= end_yearmonth:
            filtered_files.append(f)

    return filtered_files


def convert_to_datetime(date_str):
    # Convert the date string into a datetime object
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d-%H")
    except ValueError:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt


def analyze_nan(da):
    is_null_mask = da.isnull()
    if not is_null_mask.any().compute().item():
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


def load_and_process_single_var_ds(
    var_name, folder_path, start_time, num_hours, do_nan_check
):
    start_datetime = convert_to_datetime(start_time)
    end_datetime = start_datetime + timedelta(hours=num_hours - 1)
    print(f"Loading data for {var_name} from {start_datetime} to {end_datetime}")

    start_yearmonth = str(start_datetime.year) + str(start_datetime.month).zfill(2)
    end_yearmonth = str(end_datetime.year) + str(end_datetime.month).zfill(2)

    # List all files in the specified folder
    files = list(folder_path.glob("*.nc"))
    files = filter_files_by_yearmonth(files, start_yearmonth, end_yearmonth)
    print(f"Files for {var_name}: {files[0].name} ... {files[-1].name}")
    print(f"Number of files for {var_name}: {len(files)}")

    # Open the dataset using xarray
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )

    if do_nan_check:
        found_null = analyze_nan(ds[var_name])
        if found_null is not None:
            print("Found missing values in dataset:")
            print(found_null)
            raise RuntimeError("Aborting")

    ds = ds.sel(time=slice(start_datetime, end_datetime))

    # Keep only relevant stuff
    vars_to_keep = set([var_name])
    dims_to_keep = set(["time", "rlat", "rlon"])
    vars_to_drop = set(ds.data_vars) - vars_to_keep
    dims_to_drop = set(ds.dims) - dims_to_keep

    ds = ds.drop_vars(vars_to_drop).drop_dims(dims_to_drop)
    if len(vars_to_drop) > 0:
        print(f"Variables dropped: {vars_to_drop}, keeping {vars_to_keep}")
    if len(dims_to_drop) > 0:
        print(f"Dimensions dropped: {dims_to_drop}, keeping {dims_to_keep}")
    ds = ds.transpose("time", "rlat", "rlon")

    return ds


def load_and_process(data_folder, data_vars, start_time, num_hours, do_nan_check=False):
    data_vars = list(sorted(data_vars))
    data_folder = pathlib.Path(data_folder)

    # List all folders in the specified data folder
    data_folders = list(
        sorted(
            d.name
            for d in data_folder.glob("*")
            if d.is_dir() and not d.name.startswith("_")
        )
    )
    data_folders = list(sorted(data_folders))
    for dv in data_vars:
        assert dv in data_folders, f"Data folder not found: {dv}"

    print(f"Found data folders: {data_folders} ...")
    print(f"... of which we need: {data_vars}")

    # Load and process each variable dataset
    all_ds_list = []
    for var_name in data_vars:
        var_ds = load_and_process_single_var_ds(
            var_name, data_folder / var_name, start_time, num_hours, do_nan_check
        )
        all_ds_list.append(var_ds)

    # Stack all requested data into single dataset
    result = xr.merge(all_ds_list, join="exact")

    return result


def load_processed(ds_path, data_vars, start_time, num_hours, do_nan_check=False):
    data_vars = list(sorted(data_vars))
    start_datetime = convert_to_datetime(start_time)
    end_datetime = start_datetime + timedelta(hours=num_hours - 1)

    ds = xr.open_dataset(ds_path)
    ds = ds.sel(time=slice(start_datetime, end_datetime))

    # Keep only relevant stuff
    vars_to_keep = set(data_vars)
    dims_to_keep = set(["time", "rlat", "rlon"])
    vars_to_drop = set(ds.data_vars) - vars_to_keep
    dims_to_drop = set(ds.dims) - dims_to_keep

    ds = ds.drop_vars(vars_to_drop).drop_dims(dims_to_drop)
    if len(vars_to_drop) > 0:
        print(f"Variables dropped: {vars_to_drop}, keeping {vars_to_keep}")
    if len(dims_to_drop) > 0:
        print(f"Dimensions dropped: {dims_to_drop}, keeping {dims_to_keep}")
    ds = ds.transpose("time", "rlat", "rlon")

    if do_nan_check:
        for var_name in data_vars:
            found_null = analyze_nan(ds[var_name])
            if found_null is not None:
                print(f"Found missing values in dataset for variable {var_name}:")
                print(found_null)
                raise RuntimeError("Aborting")

    return ds


def normalize_ds(ds, quantile_ds, mode):
    if isinstance(quantile_ds, str):
        quantile_ds = xr.load_dataset(quantile_ds)
    if isinstance(ds, str):
        ds = xr.open_dataset(ds)

    if mode == "minmax":
        min_ds = quantile_ds.sel(quantile=0.0)
        max_ds = quantile_ds.sel(quantile=1.0)
        range_ds = max_ds - min_ds
        return (ds - min_ds) / range_ds
    elif mode == "robust":
        iqr_ds = quantile_ds.sel(quantile=0.75) - quantile_ds.sel(quantile=0.25)
        median_ds = quantile_ds.sel(quantile=0.5)
        return (ds - median_ds) / iqr_ds
    elif mode == "robust95":
        iqr_ds = quantile_ds.sel(quantile=0.95) - quantile_ds.sel(quantile=0.05)
        median_ds = quantile_ds.sel(quantile=0.5)
        return (ds - median_ds) / iqr_ds
    elif mode == "quant95":
        iqr_ds = quantile_ds.sel(quantile=0.95) - quantile_ds.sel(quantile=0.05)
        lower_ds = quantile_ds.sel(quantile=0.05)
        return (ds - lower_ds) / iqr_ds
    elif mode == "quant99":
        iqr_ds = quantile_ds.sel(quantile=0.99) - quantile_ds.sel(quantile=0.01)
        lower_ds = quantile_ds.sel(quantile=0.01)
        return (ds - lower_ds) / iqr_ds
    else:
        raise ValueError(f"Invalid mode: {mode}")


def unnormalize_ds(ds, quantile_ds, mode):
    if isinstance(quantile_ds, str):
        quantile_ds = xr.load_dataset(quantile_ds)
    if isinstance(ds, str):
        ds = xr.open_dataset(ds)

    if mode == "minmax":
        min_ds = quantile_ds.sel(quantile=0.0)
        max_ds = quantile_ds.sel(quantile=1.0)
        range_ds = max_ds - min_ds
        unnormed_ds = ds * range_ds + min_ds
    elif mode == "robust":
        iqr_ds = quantile_ds.sel(quantile=0.75) - quantile_ds.sel(quantile=0.25)
        median_ds = quantile_ds.sel(quantile=0.5)
        unnormed_ds = ds * iqr_ds + median_ds
    elif mode == "robust95":
        iqr_ds = quantile_ds.sel(quantile=0.95) - quantile_ds.sel(quantile=0.05)
        median_ds = quantile_ds.sel(quantile=0.5)
        unnormed_ds = ds * iqr_ds + median_ds
    elif mode == "quant95":
        iqr_ds = quantile_ds.sel(quantile=0.95) - quantile_ds.sel(quantile=0.05)
        lower_ds = quantile_ds.sel(quantile=0.05)
        unnormed_ds = ds * iqr_ds + lower_ds
    elif mode == "quant99":
        iqr_ds = quantile_ds.sel(quantile=0.99) - quantile_ds.sel(quantile=0.01)
        lower_ds = quantile_ds.sel(quantile=0.01)
        unnormed_ds = ds * iqr_ds + lower_ds
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return unnormed_ds


def ds_to_sorted_np(ds, data_vars, ordering="LCHW"):
    assert ordering in ["LCHW", "CLHW"], f"Invalid ordering: {ordering}"
    print(
        f"Converting dataset with dimensions {ds.dims} to sorted numpy array with data vars: {data_vars}"
    )
    print(f"Ordering of dimensions: {ordering}")
    data_vars = list(sorted(data_vars))
    data = np.stack(
        [ds[v].values for v in data_vars], axis=0 if ordering == "CLHW" else 1
    )
    print(f"Resulting shape: {data.shape}")
    return data


def np_to_ds(np_arr, reference_ds, data_vars):
    assert np_arr.shape[0] == len(reference_ds.time)
    assert np_arr.shape[1] == len(data_vars)
    assert np_arr.shape[2] == len(reference_ds.rlat)
    assert np_arr.shape[3] == len(reference_ds.rlon)

    data_vars = list(sorted(data_vars))
    data_dict = {
        v: (("time", "rlat", "rlon"), np_arr[:, i]) for i, v in enumerate(data_vars)
    }
    ds = xr.Dataset(data_dict, coords=dict(reference_ds.coords))
    return ds
