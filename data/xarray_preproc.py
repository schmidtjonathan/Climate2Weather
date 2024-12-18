import fire
import xarray as xr


def compute_quantiles(
    infile, outfile, quantile_list=[0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
):
    print(f"Computing quantiles for {infile}: {quantile_list}")
    xds = xr.load_dataset(infile)
    quantiles = xds.quantile(quantile_list, dim=["time", "rlat", "rlon"])
    quantiles.to_netcdf(outfile)
    print(f"Saved quantiles to {outfile}")
    return None


def mean_climatology(infile, outfile, varname):
    print(f"Computing mean climatology for {infile}")
    clim = xr.load_dataset(infile)
    mean_clim_values = clim[varname].mean(dim=["rlat", "rlon"])
    mean_clim_values = mean_clim_values.expand_dims(
        {"rlat": clim.sizes["rlat"]}, 1
    ).expand_dims({"rlon": clim.sizes["rlon"]}, 2)

    mean_clim = clim.copy()
    mean_clim[varname] = mean_clim_values
    mean_clim.to_netcdf(outfile)
    print(f"Saved mean climatology to {outfile}")
    return None


if __name__ == "__main__":
    fire.Fire(
        {
            "quantiles": compute_quantiles,
            "mean_climatology": mean_climatology,
        }
    )
