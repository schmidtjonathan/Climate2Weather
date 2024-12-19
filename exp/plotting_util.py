import warnings

import cartopy as ctp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import animation
from PIL import Image, ImageDraw, ImageOps

DATE_FMT_STR = "%-d %b %-I%p"

COLOR_SCHEME = {
    "gt": "tab:green",
    "obs": "tab:purple",
    "pred": "black",
}

PLOT_KWARGS = {
    "gt": {
        "color": COLOR_SCHEME["gt"],
        "linestyle": "-",
        "linewidth": 2,
        "zorder": 30,
    },
    "obs": {
        "color": COLOR_SCHEME["obs"],
        "linestyle": "-",
        "linewidth": 1.5,
    },
    "pred": {
        "color": COLOR_SCHEME["pred"],
        "linestyle": "-",
        "linewidth": 1.0,
        "alpha": 0.4,
        "zorder": 29,
    },
}

PLOT_MAP_KWARGS = {
    "psl": {
        "cmap": "inferno",
    },
    "tas": {
        "cmap": "coolwarm",
    },
    "uas": {
        "cmap": sns.color_palette("Spectral_r", as_cmap=True),
    },
    "vas": {
        "cmap": sns.color_palette("Spectral_r", as_cmap=True),
    },
    "ws": {
        "cmap": sns.color_palette("YlOrBr", as_cmap=True),
    },
}


def add_borders(ax, countries=True, sea=False):
    if countries:
        ax.add_feature(
            ctp.feature.BORDERS,
            linestyle="-",
            color="black",
            linewidth=2.0,
            alpha=0.4,
            zorder=15,
        )
        ax.add_feature(
            ctp.feature.BORDERS,
            linestyle="-",
            color="white",
            linewidth=0.5,
            alpha=1.0,
            zorder=16,
        )
    if sea:
        ax.add_feature(
            ctp.feature.COASTLINE,
            linestyle="-",
            color="black",
            linewidth=2.0,
            alpha=0.4,
            zorder=15,
        )
        ax.add_feature(
            ctp.feature.COASTLINE,
            linestyle="-",
            color="white",
            linewidth=0.5,
            alpha=1.0,
            zorder=16,
        )
    return ax


def var2name(
    var, with_units=False, with_linebreak=False, with_abbrv=False, force_unit=None
):
    maybe_linebreak = "\n" if with_linebreak else " "
    long_var = {
        "uas": "Zonal Wind",
        "vas": f"{'Mer.' if with_abbrv else 'Meridional'}{maybe_linebreak}Wind",
        "tas": "Temperature",
        "psl": "Pressure",
        "ws": "Wind Speed",
    }[var]
    units = {
        "uas": "m/s",
        "vas": "m/s",
        "tas": "K",
        "psl": "hPa",
        "ws": "m/s",
    }
    if with_units:
        if force_unit is not None:
            return f"{long_var} [{force_unit}]"
        return f"{long_var} [{units[var]}]"
    return long_var


def plot_map(xda, fig, ax, add_coastlines=False, **kwargs):
    transform = ctp.crs.PlateCarree()

    add_borders(ax, countries=True, sea=add_coastlines)

    cb_ax = kwargs.pop("cb_ax", None)
    do_plot_cb = cb_ax is not None
    gcs = ax.pcolormesh(
        xda.longitude,
        xda.latitude,
        xda,
        zorder=11,
        transform=transform,
        shading=kwargs.pop("shading", "auto"),
        edgecolor=kwargs.pop("edgecolor", "face"),
        **kwargs,
    )

    if do_plot_cb:
        fig.colorbar(gcs, ax=cb_ax)
    return fig, ax, gcs


def setup_animation(
    times,
    samples,
    gt=None,
    data=None,
    feature_labels=None,
    data_labels=None,
    align_clims=True,
    imshow_kwargs={},
    axfn=None,
):

    try:
        num_samples = samples.sizes["sample_id"]
    except KeyError:
        samples = samples.expand_dims("sample_id")
        num_samples = samples.sizes["sample_id"]

    if num_samples > 3:
        num_samples = 3
        samples = samples.isel(
            sample_id=np.random.choice(samples.sizes["sample_id"], 3)
        )

    print(f"Number of samples: {num_samples}")

    F = len(gt.data_vars) if gt is not None else len(samples.data_vars)
    H = len(samples.rlat)
    W = len(samples.rlon)
    assert feature_labels is not None and len(feature_labels) == F
    gt_cols = 1 if gt is not None else 0
    data_cols = 1 if data is not None else 0
    if data is not None:
        data_F = len(data.data_vars)
        data_H = data.rlat.size
        data_W = data.rlon.size
        assert data_labels is not None and len(data_labels) == data_F

    assert len(feature_labels) == F and (data is None or (len(feature_labels) == F))

    for var in feature_labels:
        if var in imshow_kwargs:
            continue
        else:
            imshow_kwargs[var] = dict()

    proj = ctp.crs.Mollweide()
    transform = ctp.crs.PlateCarree()

    num_cols = num_samples + gt_cols + data_cols
    num_rows = F if data is None else max(F, data_F)

    fig = plt.figure(figsize=(num_cols * 3, num_rows * 2))
    rect = plt.Rectangle(
        (0.01, 0.01),
        num_samples / num_cols,
        1,
        fill=True,
        color="#eee",
        zorder=-1,
        transform=fig.transFigure,
        figure=fig,
    )
    fig.patches.append(rect)

    data_axs = []  # data_F axes
    gt_axs = []  # F axes
    smpl_axs = []  # num_samples lists of F axes

    for i in range(1, num_samples + 1):
        cur_smpl_axs = []
        for f in range(F):
            smpls_ax_f = fig.add_subplot(
                num_rows,
                num_cols,
                f * num_cols + i,
                aspect=H / W,
                frameon=False,
                projection=proj,
            )
            smpls_ax_f.set_axis_off()
            smpls_ax_f.add_feature(
                ctp.feature.BORDERS, linestyle="-", color="orange", linewidth=1.0
            )
            smpls_ax_f.add_feature(
                ctp.feature.COASTLINE, linestyle="-", color="red", linewidth=1.0
            )
            cur_smpl_axs.append(smpls_ax_f)
        smpl_axs.append(cur_smpl_axs)
        smpl_axs[i - 1][0].set_title(f"sample #{i}")

    if gt is not None:
        for f in range(F):
            gt_ax_f = fig.add_subplot(
                num_rows,
                num_cols,
                f * num_cols + num_samples + 1,
                aspect=H / W,
                frameon=False,
                projection=proj,
            )
            gt_ax_f.set_axis_off()
            gt_ax_f.add_feature(
                ctp.feature.BORDERS, linestyle="-", color="orange", linewidth=1.0
            )
            gt_ax_f.add_feature(
                ctp.feature.COASTLINE, linestyle="-", color="red", linewidth=1.0
            )
            gt_axs.append(gt_ax_f)
        gt_axs[0].set_title("truth")

    if data is not None:
        for f in range(data_F):
            data_axs.append(
                fig.add_subplot(
                    num_rows,
                    num_cols,
                    f * num_cols + num_samples + gt_cols + 1,
                    aspect=data_H / data_W,
                    frameon=False,
                    projection=proj,
                )
            )
            data_axs[-1].set_axis_off()
        data_axs[0].set_title("data")

    # -----

    smpl_ims = []
    smpl_cbs = []

    for n in range(num_samples):
        cur_smpl_ims = []
        cur_smpl_cbs = []
        for f in range(F):
            smpl_i = smpl_axs[n][f].pcolormesh(
                samples.isel(sample_id=n).longitude,
                samples.isel(sample_id=n).latitude,
                samples.isel(sample_id=n).isel({"time": 0})[feature_labels[f]],
                transform=transform,
                facecolor="white",
                **imshow_kwargs[feature_labels[f]],
            )
            cur_smpl_ims.append(smpl_i)
            cur_smpl_cbs.append(fig.colorbar(smpl_i, ax=smpl_axs[n][f]))
        smpl_ims.append(cur_smpl_ims)
        smpl_cbs.append(cur_smpl_cbs)

    for f in range(F):
        smpl_cbs[0][f].set_label(feature_labels[f])

    if gt is not None:
        gt_ims = [] if gt is not None else None
        gt_cbs = [] if gt is not None else None
        for f in range(F):
            gt_i = gt_axs[f].pcolormesh(
                gt.longitude,
                gt.latitude,
                gt.isel({"time": 0})[feature_labels[f]],  # TODO
                transform=transform,
                facecolor="white",
                **imshow_kwargs[feature_labels[f]],
            )
            gt_ims.append(gt_i)
            gt_cbs.append(fig.colorbar(gt_i, ax=gt_axs[f]))

    if data is not None:
        data_ims = []
        data_cbs = []
        for f in range(data_F):
            # data_i = data_axs[f].imshow(np.zeros((data_H, data_W)), **imshow_kwargs[feature_labels[f]])
            data_i = data_axs[f].pcolormesh(
                data.longitude,
                data.latitude,
                data.isel({"time": 0})[feature_labels[f]],  # TODO
                transform=transform,
                facecolor="white",
                **imshow_kwargs[feature_labels[f]],
            )
            data_ims.append(data_i)
            data_cbs.append(fig.colorbar(data_i, ax=data_axs[f]))
            data_cbs[-1].set_label(data_labels[f])

    fig.tight_layout()

    # ----

    if not align_clims or (gt is None):
        for n in range(num_samples):
            for f in range(F):
                s_min = samples.isel(sample_id=n)[feature_labels[f]].min()
                s_max = samples.isel(sample_id=n)[feature_labels[f]].max()
                v_min = s_min  # np.maximum(s_min, -0.1)
                v_max = s_max  # np.minimum(s_max, 1.1)
                smpl_ims[n][f].set_clim(vmin=v_min, vmax=v_max)
                smpl_cbs[n][f].mappable.set_clim(vmin=v_min, vmax=v_max)

    if data is not None:
        for f in range(data_F):
            data_min = data[feature_labels[f]].min()
            data_max = data[feature_labels[f]].max()
            v_min = data_min  # np.maximum(data_min, -0.1)
            v_max = data_max  # np.minimum(data_max, 1.1)
            data_ims[f].set_clim(vmin=v_min, vmax=v_max)
            data_cbs[f].mappable.set_clim(vmin=v_min, vmax=v_max)

    if gt is not None:
        for f in range(F):
            gt_min = gt[feature_labels[f]].min()
            gt_max = gt[feature_labels[f]].max()
            if feature_labels[f] in ["uas", "vas"]:
                max_magnitude = np.maximum(np.abs(gt_min), np.abs(gt_max))
                gt_min = -max_magnitude
                gt_max = max_magnitude
            v_min = gt_min  # np.maximum(gt_min, -0.1)
            v_max = gt_max  # np.minimum(gt_max, 1.1)
            gt_ims[f].set_clim(vmin=v_min, vmax=v_max)
            gt_cbs[f].mappable.set_clim(vmin=v_min, vmax=v_max)
            if align_clims:
                for n in range(num_samples):
                    smpl_ims[n][f].set_clim(vmin=v_min, vmax=v_max)
                    smpl_cbs[n][f].mappable.set_clim(vmin=v_min, vmax=v_max)
                # data_ims[f].set_clim(vmin=v_min, vmax=v_max)
                # data_cbs[f].mappable.set_clim(vmin=v_min, vmax=v_max)

    def animate(i):
        time = times[i]
        # fig.suptitle(f"{samples.time[t].dt.strftime('%Y-%m-%d %H:00').item()}", y=1)
        fig.suptitle(f"{time.strftime('%Y-%m-%d %H:00')}", y=1)

        # Update samples
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for n in range(num_samples):
                for f in range(F):
                    if axfn is not None:
                        axfn(smpl_axs[n][f])
                    smpl_ims[n][f].set_array(
                        samples.isel(sample_id=n).sel({"time": time})[feature_labels[f]]
                    )

            if gt is not None:
                for f in range(F):
                    if axfn is not None:
                        axfn(gt_axs[f])
                    gt_ims[f].set_array(gt.sel({"time": time})[feature_labels[f]])

        if data is not None:
            for f in range(data_F):
                try:
                    data_ims[f].set_array(data.sel({"time": time})[feature_labels[f]])
                except KeyError:
                    data_ims[f].set_array(
                        np.nan
                        * data.sel({"time": time}, method="nearest")[feature_labels[f]]
                    )

        return_val = []
        for n in range(num_samples):
            for f in range(F):
                return_val.append(smpl_ims[n][f])
        if gt is not None:
            for f in range(F):
                return_val.append(gt_ims[f])
        if data is not None:
            for f in range(data_F):
                return_val.append(data_ims[f])

        for n in range(num_samples):
            for f in range(F):
                return_val.append(smpl_cbs[n][f])
        if gt is not None:
            for f in range(F):
                return_val.append(gt_cbs[f])
        if data is not None:
            for f in range(data_F):
                return_val.append(data_cbs[f])

        return tuple(return_val)

    return fig, animate


def create_animation(fig, animate_fn, num_frames, fps=6):
    anim = animation.FuncAnimation(
        fig,
        func=animate_fn,
        frames=num_frames,
        interval=1000 // fps,
    )
    return anim


def assign_overlapping_values(large_ds, small_ds, variables=None):
    """
    Assign values from a smaller dataset to a larger dataset where spatial coordinates overlap.

    Parameters
    ----------
    large_ds : xarray.Dataset
        The dataset with larger spatial extent that will be updated
    small_ds : xarray.Dataset
        The dataset with smaller spatial extent whose values will be assigned
    variables : list, optional
        List of variable names to assign. If None, assigns all variables from small_ds

    Returns
    -------
    xarray.Dataset
        A copy of large_ds with values from small_ds assigned in overlapping regions
    """
    # Create a copy of the large dataset to avoid modifying the original
    result_ds = large_ds.copy(deep=True)

    # Get the overlapping coordinates
    lat_overlap = np.intersect1d(large_ds.rlat, small_ds.rlat)
    lon_overlap = np.intersect1d(large_ds.rlon, small_ds.rlon)

    # If no variables specified, use all data variables from small_ds
    if variables is None:
        variables = list(small_ds.data_vars)

    # Assign values for each variable
    for var in variables:
        if var not in small_ds:
            raise ValueError(f"Variable {var} not found in small dataset")
        if var not in large_ds:
            raise ValueError(f"Variable {var} not found in large dataset")

        # Get the overlapping subset of the small dataset
        small_subset = small_ds[var].sel(rlat=lat_overlap, rlon=lon_overlap)

        # Assign values to the result dataset
        result_ds[var].loc[{"rlat": lat_overlap, "rlon": lon_overlap}] = small_subset

    return result_ds


# Adapted from:
# https://github.com/francois-rozet/sda/blob/c10dacb7025295fd6d9f8f65b28f9a9e02d71315/experiments/kolmogorov/utils.py#L148
def sandwich(
    da,
    offset: int = 6,
    border: int = 2,
    **kwargs,
):

    def arr2rgb(x, vmin=None, vmax=None):
        if vmin is None:
            vmin = x.min()
        if vmax is None:
            vmax = x.max()
        x = x[..., ::-1, ::-1]
        x = (x - vmin) / (vmax - vmin)
        x = 2 * x - 1
        x = np.sign(x) * np.abs(x) ** 0.8
        x = (x + 1) / 2
        x = sns.color_palette("Spectral", as_cmap=True)(x)

        return (
            np.concatenate(
                [
                    x[..., :3] * 256,
                    255 * np.ones(x.shape[:-1] + (1,)),
                ],
                axis=-1,
            )
        ).astype(np.uint8)

    mode = "coarse" if da.sizes["rlat"] < 128 else "fine"
    w = arr2rgb(da.values)
    N = da.sizes["time"]
    H = 128  # da.sizes["rlat"]
    W = 128  # da.sizes["rlon"]

    print(w.shape)

    if mode == "coarse":
        w = np.repeat(w, 16, axis=1).repeat(16, axis=2)

    print(w.shape)

    img = Image.new(
        "RGBA",
        size=(
            W + (N - 1) * offset,
            H + (N - 1) * offset,
        ),
        color=(255, 255, 255, 255),
    )

    draw = ImageDraw.Draw(img)

    for i in range(N):
        draw.rectangle(
            (i * offset - border, i * offset - border, img.width, img.height),
            (255, 255, 255, 255),
        )
        img.paste(Image.fromarray(w[i]), (i * offset, i * offset))

    return ImageOps.mirror(img)
