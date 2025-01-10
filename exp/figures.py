import pathlib
from datetime import timedelta

import cartopy as ctp
import fire
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from dask.diagnostics import ProgressBar
from matplotlib.colors import TwoSlopeNorm
from matplotlib.legend_handler import HandlerTuple
from scipy.stats import gaussian_kde

from data import pipeline as data_pipeline
from exp import exputil, plotting_util

plt.rcParams.update({"figure.dpi": 150})


def kde_and_pmf():
    N = 1000

    exp_dir = pathlib.Path("/path/to/experiments/run/")
    out_dir = exp_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    sample_ds, gt_ds, obs_ds = exputil.setup(exp_dir)
    sample_ds = sample_ds.compute()
    gt_ds = gt_ds.compute()
    obs_ds = obs_ds.compute() if obs_ds is not None else None
    print(sample_ds)
    FEATURE_NAMES = list(sorted(sample_ds.data_vars))

    try:
        num_samples = sample_ds.sizes["sample_id"]
    except KeyError:
        sample_ds = sample_ds.expand_dims("sample_id")
        num_samples = sample_ds.sizes["sample_id"]

    kde_precomputed_path = exp_dir / "metrics" / "kde" / f"kde_{N}.npz"
    is_kde_precomputed = (
        kde_precomputed_path.exists() and kde_precomputed_path.is_file()
    )
    if is_kde_precomputed:
        kde_dict = np.load(kde_precomputed_path)
    else:
        kde_precomputed_path.parent.mkdir(exist_ok=True, parents=True)
        kde_dict = {}
        kde_x = np.stack(
            [
                np.linspace(
                    min(gt_ds[vname].min().item(), sample_ds[vname].min().item()),
                    max(gt_ds[vname].max().item(), sample_ds[vname].max().item()),
                    N,
                )
                for vname in FEATURE_NAMES
            ]
        )
        kde_gt = np.stack(
            [
                gaussian_kde(gt_ds[vname].values.reshape(-1))(kde_x[i])
                for i, vname in enumerate(FEATURE_NAMES)
            ]
        )
        kde_samples = np.stack(
            [
                np.stack(
                    [
                        gaussian_kde(
                            sample_ds[vname].isel(sample_id=s).values.reshape(-1)
                        )(kde_x[i])
                        for i, vname in enumerate(FEATURE_NAMES)
                    ]
                )
                for s in range(num_samples)
            ]
        )
        kde_dict["x"] = kde_x
        kde_dict["gt"] = kde_gt
        kde_dict["samples"] = kde_samples
        np.savez(kde_precomputed_path, **kde_dict)

    pmf = ((sample_ds - gt_ds) <= 0).sum(dim="sample_id").load() / (num_samples)

    fig, axd = plt.subplot_mosaic(
        [
            [f"kde_{v}" for v in FEATURE_NAMES] + ["legend"],
            [f"pmf_{v}" for v in FEATURE_NAMES] + ["pmf_allvars"],
        ],
        figsize=(9, 3),
        gridspec_kw=dict(height_ratios=[1.0, 0.6]),
        subplot_kw=dict(),
    )

    sample_ds = sample_ds.compute()
    gt_ds = gt_ds.compute()
    obs_ds = obs_ds.compute() if obs_ds is not None else None

    map_axis_to_label = {
        "kde_psl": "a",
        "kde_tas": "b",
        "kde_uas": "c",
        "kde_vas": "d",
        "pmf_psl": "e",
        "pmf_tas": "f",
        "pmf_uas": "g",
        "pmf_vas": "h",
        "pmf_allvars": "i",
    }

    for i, vname in enumerate(FEATURE_NAMES):
        kde_ax = axd[f"kde_{vname}"]
        if obs_ds is not None:
            kde_twax = kde_ax  # kde_ax.twinx()
            # kde_twax = kde_ax
            obs_handle = kde_twax.hist(
                obs_ds[vname].values.reshape(-1),
                # bins=50,
                alpha=0.3,
                color=plotting_util.COLOR_SCHEME["obs"],
                density=True,
            )
            kde_twax.set_yticks([])
            kde_twax.set_yticklabels([])
            kde_twax.spines["top"].set_visible(False)
            kde_twax.spines["right"].set_visible(False)
            kde_twax.spines["bottom"].set_visible(True)
            kde_twax.spines["left"].set_visible(False)

        gt_handle = kde_ax.plot(
            kde_dict["x"][i],
            kde_dict["gt"][i],
            color=plotting_util.COLOR_SCHEME["gt"],
            linewidth=2.0,
            zorder=32,
        )
        kde_ax.set_xlim(gt_ds[vname].min().item(), gt_ds[vname].max().item())
        for s in range(sample_ds.sample_id.size):
            samples_handle = kde_ax.plot(
                kde_dict["x"][i],
                kde_dict["samples"][s][i],
                color=plotting_util.COLOR_SCHEME["pred"],
                alpha=0.3,
                linewidth=1.0,
                zorder=31,
            )
            kde_ax.set_xlabel(
                plotting_util.var2name(
                    vname, with_units=True, with_linebreak=False, with_abbrv=True
                )
            )

        if vname == "psl":
            kde_ax.set_xlim([1000, 1030])
            kde_ax.set_xticks([1000, 1015, 1030])
            kde_ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        elif vname == "tas":
            kde_ax.set_xlim([260, 300])
            # kde_ax.set_xticks([266, 282, 298])
            kde_ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
            kde_ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(2))
            for label in kde_ax.get_xticklabels(which="major"):
                label.set(visible=(int(label.get_text()) in [260, 280, 300]))
        elif vname == "uas":
            kde_ax.set_xlim([-5.0, 15.0])
            kde_ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
            kde_ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
            # kde_ax.set_xticklabels([-5, 0, 5, 10, 15])
        elif vname == "vas":
            kde_ax.set_xlim([-10.0, 10.0])
            kde_ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
            kde_ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
            # kde_ax.set_xticklabels([-10, -5, 0, 5, 10])

        pmf_ax = axd[f"pmf_{vname}"]
        cur_pmf_vals = pmf[vname].values.flatten()
        pmf_ax.hist(
            cur_pmf_vals,
            label="predictions",
            alpha=0.4,
            density=True,
            bins=np.linspace(
                -1 / (2 * num_samples),
                (1 / (2 * num_samples)) + 1,
                num_samples + 2,
                endpoint=True,
            ),
            color="black",
        )

        pmf_ax.set_yticks([])
        pmf_ax.set_yticklabels([])
        pmf_ax.set_xlim((0, 1))
        pmf_ax.set_xticks([0.0, 0.5, 1.0])
        pmf_ax.set_xticklabels([0, 0.5, 1])
        pmf_ax.spines["top"].set_visible(False)
        pmf_ax.spines["right"].set_visible(False)
        pmf_ax.spines["bottom"].set_visible(True)
        pmf_ax.spines["left"].set_visible(False)

    axd[f"pmf_{FEATURE_NAMES[0]}"].set_ylabel("Density", labelpad=15)

    fig.supxlabel(
        "Probability Integral Transform",
        va="center",
        fontsize="medium",
        y=0.05,
    )

    axd[f"kde_{FEATURE_NAMES[0]}"].set_ylabel("Density", labelpad=15)
    for ax_type in ["kde", "pmf"]:
        for i, vname in enumerate(FEATURE_NAMES):
            ax = axd[f"{ax_type}_{vname}"]
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ymargin(0.0)
            # if ax_type == "kde":
            #     ax.set_frame_on(False)

    for axname in [f"pmf_{v}" for v in FEATURE_NAMES[1:]] + ["pmf_allvars"]:
        axd[axname].sharey(axd[f"pmf_{FEATURE_NAMES[0]}"])

    ax = axd["pmf_allvars"]
    allvars_values = np.concatenate([pmf[v].values.flatten() for v in FEATURE_NAMES])

    ax.hist(
        allvars_values,
        label="predictions",
        alpha=0.4,
        density=True,
        bins=np.linspace(
            -1 / (2 * num_samples),
            (1 / (2 * num_samples)) + 1,
            num_samples + 2,
            endpoint=True,
        ),
        color="black",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_title("All variables", fontsize="medium", y=1.16)

    for axname, label in map_axis_to_label.items():
        axd[axname].annotate(
            label,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(-0.75, -0.35),
            textcoords="offset fontsize",
            fontsize="medium",
            fontweight="bold",
            verticalalignment="top",
        )

    axd["legend"].set_axis_off()
    axd["legend"].legend(
        [gt_handle[0], samples_handle[0], obs_handle[-1]],
        ["Reanalysis", "Predictions", "Coarse input"],
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncols=1,
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        frameon=False,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(out_dir / "kde_pmf.png")
    fig.savefig(out_dir / "kde_pmf.pdf")


def timeseries():
    proj = ctp.crs.Mollweide()
    transform = ctp.crs.PlateCarree()
    exp_dir = pathlib.Path("/path/to/experiments/run/")
    out_dir = exp_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    generated_samples, gt, obs = exputil.setup(exp_dir)
    FEATURE_NAMES = list(sorted(generated_samples.data_vars))
    generated_samples = generated_samples.isel(
        sample_id=np.random.choice(
            generated_samples.sizes["sample_id"], 10, replace=False
        )
    )

    random_loc = {"rlat": 31, "rlon": 16}
    neighbor_loc = {"rlat": 16, "rlon": 31}

    fig, axd = plt.subplot_mosaic(
        """
        ab
        cd
        """,
        figsize=(8, 4),
    )
    gt_loc_1 = gt.isel(rlat=random_loc["rlat"], rlon=random_loc["rlon"])
    samples_loc_1 = generated_samples.isel(
        rlat=random_loc["rlat"], rlon=random_loc["rlon"]
    )
    gt_loc_2 = gt.isel(rlat=neighbor_loc["rlat"], rlon=neighbor_loc["rlon"])
    samples_loc_2 = generated_samples.isel(
        rlat=neighbor_loc["rlat"], rlon=neighbor_loc["rlon"]
    )
    if obs is not None:
        obs_loc_1 = obs.sel(
            rlat=gt_loc_1.rlat,
            rlon=gt_loc_1.rlon,
            method="nearest",
        )
        obs_loc_2 = obs.sel(
            rlat=gt_loc_2.rlat,
            rlon=gt_loc_2.rlon,
            method="nearest",
        )

    for i, (letter, vname) in enumerate(zip("abcd", FEATURE_NAMES)):
        ax = axd[letter]
        gt1_handle = ax.plot(
            gt.time,
            gt_loc_1[vname],
            color="green",
            zorder=10,
            linewidth=2.0,
        )
        gt2_handle = ax.plot(
            gt.time,
            gt_loc_2[vname],
            color="green",
            linestyle=":",
            zorder=10,
            linewidth=2.0,
        )
        obs1_handle = ax.plot(
            obs.time,
            obs_loc_1[vname],
            color="purple",
            marker="o",
            linestyle="",
            markersize=5,
            markerfacecolor="none",
            markeredgewidth=1,
            markeredgecolor="purple",
            zorder=11,
        )
        obs2_handle = ax.plot(
            obs.time,
            obs_loc_2[vname],
            color="purple",
            marker="x",
            linestyle="",
            markersize=5,
            markeredgewidth=1,
            markerfacecolor="none",
            markeredgecolor="purple",
            zorder=11,
        )

        for s in range(generated_samples.sample_id.size):
            sample1_handle = ax.plot(
                generated_samples.time,
                samples_loc_1[vname].isel(sample_id=s),
                color="black",
                alpha=0.6,
                linewidth=0.5,
            )
            sample2_handle = ax.plot(
                generated_samples.time,
                samples_loc_2[vname].isel(sample_id=s),
                label="Sample" if (i == 0 and s == 0) else None,
                color="black",
                alpha=0.6,
                linewidth=0.5,
                linestyle=":",
            )

            if letter == "a":
                ax.set_ylim([1010, 1030])
                ax.set_yticks([1010, 1020, 1030])
                ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
            elif letter == "b":
                ax.set_ylim([265, 300])
                # ax.set_yticks([266, 282, 298])
                ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
                ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
                # for label in ax.get_yticklabels(which="major"):
                #     label.set(visible=(int(label.get_text()) in [265, 280, 300]))
            elif letter == "c":
                ax.set_ylim([-2.5, 7.5])
                ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
                ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2.5))
            elif letter == "d":
                ax.set_ylim([-5.0, 5.0])
                ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
                ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2.5))

        if i // 2 == 1:
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter(plotting_util.DATE_FMT_STR)
            )
            # ax.xaxis.set_minor_locator(mdates.HourLocator(interval=12))
            ax.set_xticks(gt.time[::12])
            # ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
            for label in ax.get_xticklabels(which="major"):
                label.set(rotation=30, horizontalalignment="right")
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])
        ax.set_ylabel(plotting_util.var2name(vname, with_units=True))
        ax.annotate(
            letter,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(+0.25, -0.35),
            textcoords="offset fontsize",
            fontsize="medium",
            fontweight="bold",
            verticalalignment="top",
        )

    fig.tight_layout()
    fig.subplots_adjust(right=0.8, wspace=0.4)
    map_ax = fig.add_axes([0.75, 0.5, 0.3, 0.3], projection=proj, frameon=False)

    plotting_util.plot_map(
        obs["tas"].isel(time=0),
        fig=fig,
        ax=map_ax,
        alpha=0.3,
        cmap="coolwarm",
        edgecolor="black",
        linewidths=0.4,
    )
    map_ax.plot(
        gt_loc_1.longitude,
        gt_loc_1.latitude,
        color="purple",
        transform=transform,
        marker="o",
        linestyle="",
        markersize=5,
        markerfacecolor="none",
        markeredgewidth=1.25,
        markeredgecolor="purple",
        zorder=30,
    )
    map_ax.plot(
        gt_loc_2.longitude,
        gt_loc_2.latitude,
        transform=transform,
        color="purple",
        marker="x",
        linestyle="",
        markersize=5,
        markeredgewidth=1.25,
        markerfacecolor="none",
        markeredgecolor="purple",
        zorder=30,
    )

    map_ax.annotate(
        "e",
        xy=(-0.05, 1.1),
        xycoords="axes fraction",
        xytext=(+0.5, 0.1),
        textcoords="offset fontsize",
        fontsize="medium",
        fontweight="bold",
        verticalalignment="top",
    )

    fig.legend(
        [
            (gt1_handle[0], gt2_handle[0]),
            (sample1_handle[0], sample2_handle[0]),
            (obs1_handle[0], obs2_handle[0]),
        ],
        ["Reanalysis", "Predictions", "Coarse input"],
        loc="lower left",
        bbox_to_anchor=(0.81, 0.2),
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        frameon=False,
    )

    fig.align_ylabels([axd["a"], axd["c"]])
    fig.align_ylabels([axd["b"], axd["d"]])

    fig.savefig(out_dir / "timeseries.png")
    fig.savefig(out_dir / "timeseries.pdf")


def storm_grid():
    start_time = "2018-01-18-14"
    time_step = 1
    num_times = 7
    data_vars = ("vas",)

    proj = ctp.crs.Mollweide()

    exp_dir = pathlib.Path("/path/to/experiments/run/")
    out_dir = exp_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    generated_samples, gt, obs = exputil.setup(exp_dir)

    FEATURE_NAMES = data_vars

    num_T = num_times
    start_dt = data_pipeline.convert_to_datetime(start_time)
    dts = [start_dt + timedelta(hours=i * time_step) for i in range(num_T)]

    num_samples = generated_samples.sizes["sample_id"]

    if num_samples > 3:
        print(f"Found {num_samples} samples, selecting 3 at random.")
        num_samples = 3
        random_sample_ids = np.random.choice(
            generated_samples["sample_id"].values, num_samples, replace=False
        )
    else:
        num_samples = 1
        random_sample_ids = [0]

    generated_samples = generated_samples.compute()
    gt = gt.compute()
    obs = obs.compute()

    vmin = {v: gt.sel(time=dts)[v].quantile(0.01).item() for v in FEATURE_NAMES}
    vmax = {v: gt.sel(time=dts)[v].quantile(0.99).item() for v in FEATURE_NAMES}

    imshow_kwargs = dict(
        uas=(
            dict(
                cmap=sns.color_palette("Spectral_r", as_cmap=True),
                norm=TwoSlopeNorm(
                    0,
                    vmin=vmin["uas"],
                    vmax=vmax["uas"],
                ),
            )
            if "uas" in FEATURE_NAMES
            else {}
        ),
        vas=(
            dict(
                cmap=sns.color_palette("Spectral_r", as_cmap=True),
                norm=TwoSlopeNorm(
                    0,
                    vmin=vmin["vas"],
                    vmax=vmax["vas"],
                ),
            )
            if "vas" in FEATURE_NAMES
            else {}
        ),
        tas=(
            dict(
                cmap="coolwarm",
                vmin=vmin["tas"],
                vmax=vmax["tas"],
            )
            if "tas" in FEATURE_NAMES
            else {}
        ),
        psl=(
            dict(
                cmap="inferno",
                vmin=vmin["psl"],
                vmax=vmax["psl"],
            )
            if "psl" in FEATURE_NAMES
            else {}
        ),
    )

    for var_name in data_vars:
        fig, axs = plt.subplots(
            num_samples + 2,
            num_T,
            figsize=(9, 6),
            subplot_kw=dict(projection=proj, frameon=False),
        )

        for ip in range(num_samples + 2):
            for it, timepoint in enumerate(dts):
                if ip == num_samples:
                    cond = gt
                elif ip == num_samples + 1:
                    cond = obs
                else:
                    cond = generated_samples.isel(sample_id=random_sample_ids[ip])
                try:
                    _, _, gcs = plotting_util.plot_map(
                        cond[var_name].sel(time=timepoint),
                        fig=fig,
                        ax=axs[ip, it],
                        **imshow_kwargs.get(var_name, {}),
                        rasterized=True,
                    )
                except KeyError as e:
                    print(e)

                if ip == 0 and it % 1 == 0:
                    loc_kwargs = dict(x=0.33) if it == 0 else dict(loc="center")
                    axs[0, it].set_title(
                        (
                            f"{timepoint.strftime(plotting_util.DATE_FMT_STR)}"
                            if it > 0
                            else f"{timepoint.strftime(f'%Y {plotting_util.DATE_FMT_STR}')}"
                        ),
                        fontsize="medium",
                        y=1.1,
                        **loc_kwargs,
                    )

            if ip == num_samples:
                axs[ip, 0].yaxis.set_visible(True)
                axs[ip, 0].set_yticks([])
                axs[ip, 0].spines["left"].set_visible(False)
                axs[ip, 0].set_ylabel("Reanalysis", fontsize="medium")
            elif ip == num_samples + 1:
                axs[ip, 0].yaxis.set_visible(True)
                axs[ip, 0].set_yticks([])
                axs[ip, 0].spines["left"].set_visible(False)
                axs[ip, 0].set_ylabel("Coarse input", fontsize="medium")
            else:
                axs[ip, 0].yaxis.set_visible(True)
                axs[ip, 0].set_yticks([])
                axs[ip, 0].spines["left"].set_visible(False)
                axs[ip, 0].set_ylabel(
                    f"Sample {'#'}{random_sample_ids[ip]}", fontsize="medium"
                )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.05, wspace=0.05, hspace=0.2)
        cbar_ax = fig.add_axes([0.3, 0.09, 0.4, 0.015])
        cbar = fig.colorbar(
            gcs,
            cax=cbar_ax,
            extend="max" if var_name == "ws" else "both",
            shrink=0.5,
            orientation="horizontal",
        )
        cbar.set_label(
            plotting_util.var2name(var_name, with_units=True, with_linebreak=False)
        )

        cticks = [-4.5, -2.5, 0, 3.5, 7]
        cbar.set_ticks(cticks)
        cbar.ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))

        fig.align_ylabels(axs[:, 0])

        save_path = out_dir / f"{var_name}_storm.png"
        fig.savefig(save_path)
        fig.savefig(save_path.with_suffix(".pdf"))

        plt.close("all")


def climate_grid():
    dask_pbar = ProgressBar(minimum=5, dt=1.0)
    dask_pbar.register()

    y = "2014"
    m = str(np.random.randint(12) + 1).zfill(2)
    d = str(np.random.randint(28) + 1).zfill(2)
    h = "06"  # str(np.random.choice([0, 6, 12, 18])).zfill(2)
    start_time = f"{y}-{m}-{d}-{h}"
    time_step = 3
    num_times = 4
    data_vars = ("psl", "tas", "uas", "vas")

    proj = ctp.crs.Mollweide()

    exp_dir = pathlib.Path("/path/to/experiments/run/")
    out_dir = exp_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    generated_samples, _, obs = exputil.setup(exp_dir, load_gt=False)

    FEATURE_NAMES = data_vars

    num_T = num_times
    start_dt = data_pipeline.convert_to_datetime(start_time)
    dts = [start_dt + timedelta(hours=i * time_step) for i in range(num_T)]

    generated_samples = generated_samples.sel(time=dts).compute()

    num_samples = generated_samples.sizes["sample_id"]

    if num_samples > 3:
        print(f"Found {num_samples} samples, selecting 3 at random.")
        num_samples = 3
        random_sample_ids = np.random.choice(
            generated_samples["sample_id"].values, num_samples, replace=False
        )
    else:
        num_samples = 1
        random_sample_ids = [0]

    vmin = {v: generated_samples[v].min().compute().item() for v in FEATURE_NAMES}
    vmax = {v: generated_samples[v].max().compute().item() for v in FEATURE_NAMES}

    imshow_kwargs = dict(
        uas=(
            dict(
                cmap=sns.color_palette("Spectral_r", as_cmap=True),
                norm=TwoSlopeNorm(
                    0,
                    vmin=vmin["uas"],
                    vmax=vmax["uas"],
                ),
            )
            if "uas" in FEATURE_NAMES
            else {}
        ),
        vas=(
            dict(
                cmap=sns.color_palette("Spectral_r", as_cmap=True),
                norm=TwoSlopeNorm(
                    0,
                    vmin=vmin["vas"],
                    vmax=vmax["vas"],
                ),
            )
            if "vas" in FEATURE_NAMES
            else {}
        ),
        tas=(
            dict(
                cmap="coolwarm",
                vmin=vmin["tas"],
                vmax=vmax["tas"],
            )
            if "tas" in FEATURE_NAMES
            else {}
        ),
        psl=(
            dict(
                cmap="inferno",
                vmin=vmin["psl"],
                vmax=vmax["psl"],
            )
            if "psl" in FEATURE_NAMES
            else {}
        ),
    )

    for var_name in data_vars:
        fig, axs = plt.subplots(
            num_samples + 1,
            num_T,
            figsize=(5, 5.5),
            subplot_kw=dict(projection=proj, frameon=False),
        )

        for ip in range(num_samples + 1):
            for it, timepoint in enumerate(dts):
                if ip == num_samples:
                    cond = obs
                else:
                    cond = generated_samples.isel(sample_id=random_sample_ids[ip])
                try:
                    _, _, gcs = plotting_util.plot_map(
                        cond[var_name].sel(time=timepoint),
                        fig=fig,
                        ax=axs[ip, it],
                        **imshow_kwargs.get(var_name, {}),
                        rasterized=True,
                    )
                except KeyError:
                    print(f"Skipping {var_name} at time {timepoint}")

                if ip == 0 and it % 1 == 0:
                    loc_kwargs = dict(x=0.33) if it == 0 else dict(loc="center")
                    if True:  # var_name in ["psl", "tas"]:
                        axs[0, it].set_title(
                            (
                                f"{timepoint.strftime(f'%Y {plotting_util.DATE_FMT_STR}')}"
                                if var_name in ["psl", "uas"] and it == 0
                                else f"{timepoint.strftime(plotting_util.DATE_FMT_STR)}"
                            ),
                            fontsize="medium",
                            y=1.1,
                            **loc_kwargs,
                        )

            if True:  # var_name in ["psl", "uas"]:
                if ip == num_samples:
                    axs[ip, 0].yaxis.set_visible(True)
                    axs[ip, 0].set_yticks([])
                    axs[ip, 0].spines["left"].set_visible(False)
                    axs[ip, 0].set_ylabel("BC ESM", fontsize="medium")
                else:
                    axs[ip, 0].yaxis.set_visible(True)
                    axs[ip, 0].set_yticks([])
                    axs[ip, 0].spines["left"].set_visible(False)
                    axs[ip, 0].set_ylabel(
                        f"Sample {'#'}{random_sample_ids[ip]}", fontsize="medium"
                    )

        var2lbl = {"psl": "a", "tas": "b", "uas": "c", "vas": "d"}

        fig.tight_layout()
        fig.subplots_adjust(top=0.91, bottom=0.135, wspace=0.05, hspace=0.05)
        cbar_ax = fig.add_axes([0.3, 0.09, 0.4, 0.015])
        cbar = fig.colorbar(
            gcs,
            cax=cbar_ax,
            extend="both",
            shrink=0.5,
            orientation="horizontal",
        )
        cbar.set_label(
            plotting_util.var2name(var_name, with_units=True, with_linebreak=False)
        )

        if var_name in ["uas", "vas"]:
            cticks = np.concatenate(
                [
                    np.linspace(vmin[var_name], 0, num=4, endpoint=True),
                    np.linspace(0, vmax[var_name], num=4, endpoint=True)[1:],
                ]
            )
            cbar.set_ticks(cticks)

        if var_name == "psl":
            # cticks = [1015, 1020, 1025]
            # cbar.set_ticks(cticks)
            cbar.ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        elif var_name == "tas":
            # cticks = [270, 265, 280]
            # cbar.set_ticks(cticks)
            cbar.ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        elif var_name == "uas":
            # cticks = [-4, -2, 0, 6, 12]
            # cbar.set_ticks(cticks)
            cbar.ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
        elif var_name == "vas":
            # cticks = [-10, -5, 0, 4.5, 9]
            # cbar.set_ticks(cticks, labels=["-10", "-5", "0", "4.5", "9"])
            cbar.ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))

        fig.align_ylabels(axs[:, 0])
        axs[0, 0].annotate(
            var2lbl[var_name],
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(-2.5, +3.85),
            textcoords="offset fontsize",
            fontsize="medium",
            fontweight="bold",
            verticalalignment="top",
        )

        save_path = out_dir / f"{var_name}_clim.png"
        fig.savefig(save_path)
        fig.savefig(save_path.with_suffix(".pdf"))

        plt.close("all")


def downscaled_clim_dist():
    dask_pbar = ProgressBar(minimum=5, dt=1.0)
    dask_pbar.register()

    density = True
    biased_hadgem_dir = pathlib.Path(
        "/path/to/experiments/raw_vs_qm/biased_climate_hadgem"
    )
    biased_mpi_dir = pathlib.Path("/path/to/experiments/raw_vs_qm/biased_climate_mpi")
    debiased_hadgem_dir = pathlib.Path(
        "/path/to/experiments/raw_vs_qm/debiased_climate_hadgem"
    )
    debiased_mpi_dir = pathlib.Path(
        "/path/to/experiments/raw_vs_qm/debiased_climate_mpi"
    )

    out_dir = pathlib.Path("/path/to/experiments/raw_vs_qm/")
    out_file = out_dir / "downscaled_clim_dist.png"

    _, _, biased_hadgem_ds = exputil.setup(
        biased_hadgem_dir,
        load_gt=False,
        load_samples=False,
    )
    _, _, biased_mpi_ds = exputil.setup(
        biased_mpi_dir, load_gt=False, load_samples=False
    )
    downscaled_debiased_hadgem_ds, _, debiased_hadgem_ds = exputil.setup(
        debiased_hadgem_dir, load_gt=False
    )
    downscaled_debiased_mpi_ds, gt_ds, debiased_mpi_ds = exputil.setup(
        debiased_mpi_dir, load_gt=True
    )

    FEATURE_NAMES = ["psl", "tas", "uas", "vas"]
    num_samples = downscaled_debiased_hadgem_ds.sizes["sample_id"]

    # # -------------------------------------------------------------------------
    # # Only for determining the color scale for the plot
    # # -------------------------------------------------------------------------
    # min_for_var = {}
    # max_for_var = {}
    # for v in FEATURE_NAMES:
    #     mi = min(
    #         gt_ds[v].min().compute().item(),
    #         downscaled_debiased_hadgem_ds[v].min().compute().item(),
    #         downscaled_debiased_mpi_ds[v].min().compute().item(),
    #     )
    #     ma = max(
    #         gt_ds[v].max().compute().item(),
    #         downscaled_debiased_hadgem_ds[v].max().compute().item(),
    #         downscaled_debiased_mpi_ds[v].max().compute().item(),
    #     )
    #     print(f"{v}: {mi} - {ma}")
    #     min_for_var[v] = mi
    #     max_for_var[v] = ma

    # For de-biased MPI:
    # psl: 980.3825073242188 - 1040.5511474609375
    # tas: 249.89857482910156 - 311.8726501464844
    # uas: -15.963351249694824 - 18.637008666992188
    # vas: -18.00084686279297 - 19.4628849029541

    min_for_var = {
        "psl": 980.3825073242188,
        "tas": 249.89857482910156,
        "uas": -15.963351249694824,
        "vas": -18.00084686279297,
    }
    max_for_var = {
        "psl": 1040.5511474609375,
        "tas": 311.8726501464844,
        "uas": 18.637008666992188,
        "vas": 19.4628849029541,
    }

    # # -------------------------------------------------------------------------

    N = 250
    nbins_clim = 75
    kde_gt = None
    kdes = {}
    for name, exp_dir, sds in zip(
        ["hadgem", "mpi"],
        [debiased_hadgem_dir, debiased_mpi_dir],
        [downscaled_debiased_hadgem_ds, downscaled_debiased_mpi_ds],
    ):
        kde_precomputed_path = exp_dir / "metrics" / "kde" / f"kde_{N}.npz"
        is_kde_precomputed = (
            kde_precomputed_path.exists() and kde_precomputed_path.is_file()
        )
        if is_kde_precomputed:
            kde_dict = np.load(kde_precomputed_path)
        else:
            print(f"Computing KDE for {exp_dir}")
            kde_precomputed_path.parent.mkdir(exist_ok=True, parents=True)
            kde_dict = {}
            kde_x = np.stack(
                [
                    np.linspace(
                        min_for_var[vname],
                        max_for_var[vname],
                        N,
                    )
                    for vname in FEATURE_NAMES
                ]
            )
            if kde_gt is None:
                kde_gt = np.stack(
                    [
                        gaussian_kde(gt_ds[vname].values.reshape(-1))(kde_x[i])
                        for i, vname in enumerate(FEATURE_NAMES)
                    ]
                )
            kde_samples = np.stack(
                [
                    np.stack(
                        [
                            gaussian_kde(
                                sds[vname].isel(sample_id=s).values.reshape(-1)
                            )(kde_x[i])
                            for i, vname in enumerate(FEATURE_NAMES)
                        ]
                    )
                    for s in range(num_samples)
                ]
            )
            kde_dict["x"] = kde_x
            kde_dict["gt"] = kde_gt
            kde_dict["samples"] = kde_samples
            np.savez(kde_precomputed_path, **kde_dict)
        kdes[name] = kde_dict

    fig, axs = plt.subplots(3, 4, figsize=(9, 5), sharex="col", sharey="col")
    for i, vname in enumerate(FEATURE_NAMES):
        print(f"Variable {vname}")
        # BIASED
        (gt_handle,) = axs[0, i].plot(
            kde_dict["x"][i],
            kde_dict["gt"][i],
            color="green",
            linewidth=2,
            # linestyle="--",
            zorder=-1,
        )

        _, hbins, _ = axs[0, i].hist(
            biased_hadgem_ds[vname].values.reshape(-1),
            bins=nbins_clim,
            alpha=0.3,
            label="HadGEM" if i == 0 else None,
            color="orange",
            density=density,
        )
        _, mbins, _ = axs[0, i].hist(
            biased_mpi_ds[vname].values.reshape(-1),
            bins=nbins_clim,
            alpha=0.3,
            label="MPI" if i == 0 else None,
            color="blue",
            density=density,
        )

        # DEBIASED
        (gt_handle,) = axs[1, i].plot(
            kde_dict["x"][i],
            kde_dict["gt"][i],
            color="green",
            linewidth=2,
            # linestyle="--",
            zorder=-1,
        )
        bcH_counts, bcH_bins, legendH = axs[1, i].hist(
            debiased_hadgem_ds[vname].values.reshape(-1),
            bins=nbins_clim,
            alpha=0.3,
            # label="HadGEM (BC)",
            color="orange",
            density=density,
        )
        bcM_counts, bcM_bins, legendM = axs[1, i].hist(
            debiased_mpi_ds[vname].values.reshape(-1),
            bins=nbins_clim,
            alpha=0.3,
            # label="MPI (BC)",
            color="blue",
            density=density,
        )

        # DOWNSCALED
        axs[2, i].hist(
            bcH_bins[:-1],
            bcH_bins,
            weights=bcH_counts,
            alpha=0.3,
            # label="HadGEM (BC)",
            color="orange",
        )
        axs[2, i].hist(
            bcM_bins[:-1],
            bcM_bins,
            weights=bcM_counts,
            alpha=0.3,
            # label="MPI (BC)",
            color="blue",
        )

        for s in range(num_samples):
            print(
                f" > Sample {s + 1}/{downscaled_debiased_hadgem_ds.sizes['sample_id']}"
            )
            (samples_hadgem_handle,) = axs[2, i].plot(
                kdes["hadgem"]["x"][i],
                kdes["hadgem"]["samples"][s, i],
                color="black",
                linestyle=":",
                linewidth=1.0,
                # color="orange",
                alpha=0.3,
            )

            (samples_mpi_handle,) = axs[2, i].plot(
                kdes["mpi"]["x"][i],
                kdes["mpi"]["samples"][s, i],
                color="black",
                linewidth=1.0,
                # color="blue",
                alpha=0.3,
            )

    for r in range(3):
        for c in range(4):
            axs[r, c].spines["top"].set_visible(False)
            axs[r, c].spines["right"].set_visible(False)
            axs[r, c].spines["bottom"].set_visible(True)
            axs[r, c].spines["left"].set_visible(False)
            axs[r, c].set_yticks([])

    for lbl, ax in zip("abcdefghijkl", axs.flatten()):
        ax.annotate(
            lbl,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(+0.5, -0.5),
            textcoords="offset fontsize",
            fontsize="medium",
            fontweight="bold",
            verticalalignment="top",
        )

    axs[0, 0].set_ylabel("ESM vs.\nCOSMO REA")
    axs[1, 0].set_ylabel("BC ESM vs.\nCOSMO REA")
    axs[2, 0].set_ylabel("BC ESM vs.\nDownscaled")

    # PSL
    axs[-1, 0].set_xlim((min_for_var["psl"], max_for_var["psl"]))
    axs[-1, 0].xaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
    axs[-1, 0].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))

    # TAS
    axs[-1, 1].set_xlim((min_for_var["tas"], max_for_var["tas"]))
    axs[-1, 1].xaxis.set_major_locator(mpl.ticker.MultipleLocator(20, 10))
    axs[-1, 1].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))

    # UAS
    axs[-1, 2].set_xlim((min_for_var["uas"], max_for_var["uas"]))
    axs[-1, 2].xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    axs[-1, 2].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
    for label in axs[-1, 2].get_xticklabels(which="major")[::2]:
        label.set(visible=False)

    # VAS
    axs[-1, 3].set_xlim((min_for_var["vas"], max_for_var["vas"]))
    axs[-1, 3].xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    axs[-1, 3].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
    for label in axs[-1, 3].get_xticklabels(which="major")[::2]:
        label.set(visible=False)

    fig.align_ylabels(axs[:, 0])

    for c in range(4):
        axs[-1, c].set_xlabel(plotting_util.var2name(FEATURE_NAMES[c], with_units=True))

    fig.supylabel("Density", x=0.01)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.175)
    fig.legend(
        [gt_handle, (samples_mpi_handle, samples_hadgem_handle), (legendM, legendH)],
        ["Reanalysis (COSMO REA6)", "Predictions (MPI / HadGEM)", "ESM (MPI / HadGEM)"],
        loc="center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.025),
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        frameon=False,
    )
    fig.savefig(out_file)
    fig.savefig(out_file.with_suffix(".pdf"))


def _calc_windpower(exp_dir, single_location: tuple = None):
    import windpowerlib

    exp_dir = pathlib.Path(exp_dir)
    out_dir = exp_dir / "metrics"
    save_path = out_dir / "windpower"
    save_path.mkdir(exist_ok=True, parents=True)

    # Load sample(s), ground truth
    sample_ds, gt_ds, obs_ds = exputil.setup(exp_dir)

    hubheight = 100
    reference_height = 10
    density_at_hubheight = 1

    def wind_speed_to_hub_height(wind):
        alpha = 1 / 7
        speed_hub_height = wind * (hubheight / reference_height) ** alpha
        return speed_hub_height

    def compute_windpower(wind):
        enercon_e126 = {
            "turbine_type": "E-115/3000",
            "hub_height": hubheight,
        }
        e126 = windpowerlib.WindTurbine(**enercon_e126)
        mc_e126 = windpowerlib.ModelChain(e126)
        # write power output time series to WindTurbine object
        power = mc_e126.calculate_power_output(wind, density_at_hubheight)
        return power

    # Load sample(s), ground truth
    num_samples = sample_ds.sizes["sample_id"]
    if single_location is not None:
        single_rlat, single_rlon = single_location
        sample_ds = sample_ds.isel(rlat=single_rlat, rlon=single_rlon)
        gt_ds = gt_ds.isel(rlat=single_rlat, rlon=single_rlon)
        obs_ds = obs_ds.isel(rlat=single_rlat // 16, rlon=single_rlon // 16)
    else:
        single_rlat = None
        single_rlon = None

    sample_windspeed_100 = wind_speed_to_hub_height(
        np.sqrt(sample_ds["uas"] ** 2 + sample_ds["vas"] ** 2).values
    ).astype(np.float32, copy=False)
    gt_windspeed_100 = wind_speed_to_hub_height(
        np.sqrt(gt_ds["uas"] ** 2 + gt_ds["vas"] ** 2).values
    ).astype(np.float32, copy=False)
    obs_windspeed_100 = wind_speed_to_hub_height(
        np.sqrt(obs_ds["uas"] ** 2 + obs_ds["vas"] ** 2).values
    ).astype(np.float32, copy=False)

    del sample_ds, gt_ds, obs_ds  # save memory

    sample_windpower = compute_windpower(sample_windspeed_100).astype(
        np.float32, copy=False
    )
    gt_windpower = compute_windpower(gt_windspeed_100).astype(np.float32, copy=False)
    obs_windpower = compute_windpower(obs_windspeed_100).astype(np.float32, copy=False)

    ws_value_range = np.linspace(0.0, 30.0, 300)
    wp_value_range = np.linspace(0.0, 3000000.0, 300)

    gt_kde_windspeed = gaussian_kde(gt_windspeed_100.reshape(-1))(ws_value_range)
    gt_kde_windpower = gaussian_kde(gt_windpower.reshape(-1))(wp_value_range)
    obs_kde_windspeed = gaussian_kde(obs_windspeed_100.reshape(-1))(ws_value_range)
    obs_kde_windpower = gaussian_kde(obs_windpower.reshape(-1))(wp_value_range)
    sample_kde_windspeed = np.stack(
        [
            gaussian_kde(sample_windspeed_100[i].reshape(-1))(ws_value_range)
            for i in range(num_samples)
        ]
    )
    sample_kde_windpower = np.stack(
        [
            gaussian_kde(sample_windpower[i].reshape(-1))(wp_value_range)
            for i in range(num_samples)
        ]
    )

    power_curve = compute_windpower(ws_value_range)  # e126.power_curve["value"]  #

    gt_power_gen = power_curve * gt_kde_windspeed
    obs_power_gen = power_curve * obs_kde_windspeed
    sample_power_gen = np.stack(
        [power_curve * sample_kde_windspeed[i] for i in range(num_samples)]
    )

    to_save = dict(
        sample_windspeed_100=sample_windspeed_100,
        sample_windpower=sample_windpower,
        sample_windpower_kde=sample_kde_windpower,
        sample_windspeed_kde=sample_kde_windspeed,
        gt_windspeed_100=gt_windspeed_100,
        gt_windpower=gt_windpower,
        gt_windpower_kde=gt_kde_windpower,
        gt_windspeed_kde=gt_kde_windspeed,
        gt_power_gen=gt_power_gen,
        obs_windspeed_100=obs_windspeed_100,
        obs_windpower=obs_windpower,
        obs_windpower_kde=obs_kde_windpower,
        obs_windspeed_kde=obs_kde_windspeed,
        obs_power_gen=obs_power_gen,
        sample_power_gen=sample_power_gen,
        windpower_range=wp_value_range,
        windspeed_range=ws_value_range,
        power_curve=power_curve,
    )
    if single_location is not None:
        to_save["rlat_rlon"] = (single_rlat, single_rlon)
        savefilename = f"windpower_{single_rlat}_{single_rlon}.npz"
    else:
        savefilename = "windpower.npz"
    np.savez(save_path / savefilename, **to_save)
    return to_save


def windpowers(rlat1=17, rlon1=17, rlat2=30, rlon2=30):
    wp_loc_dir = pathlib.Path("/path/to/experiments/run/metrics/windpower")

    _, gt_ds, obs_ds = exputil.setup(
        wp_loc_dir.parent.parent, load_gt=True, load_samples=False
    )

    try:
        wp_loc1 = np.load(wp_loc_dir / f"windpower_{rlat1}_{rlon1}.npz")
        wp_loc2 = np.load(wp_loc_dir / f"windpower_{rlat2}_{rlon2}.npz")
    except:
        wp_loc_dir.mkdir(exist_ok=True, parents=True)
        wp_loc_dir.mkdir(exist_ok=True, parents=True)
        print(f"Calculating wind power for location {rlat1}, {rlon1}")
        wp_loc1 = _calc_windpower(wp_loc_dir.parent.parent, (rlat1, rlon1))
        wp_loc2 = _calc_windpower(wp_loc_dir.parent.parent, (rlat2, rlon2))

    fig, axs = plt.subplots(
        2,
        3,
        figsize=(9, 4),
        sharex="col",
        sharey="col",
    )

    axs[0, 0].plot(
        np.linspace(
            wp_loc1["gt_windspeed_100"].min(),
            wp_loc1["gt_windspeed_100"].max(),
            len(wp_loc1["gt_windspeed_kde"]),
        ),
        wp_loc1["gt_windspeed_kde"],
        color=plotting_util.COLOR_SCHEME["gt"],
        lw=2,
        zorder=30,
    )
    axs[1, 0].plot(
        np.linspace(
            wp_loc2["gt_windspeed_100"].min(),
            wp_loc2["gt_windspeed_100"].max(),
            len(wp_loc1["gt_windspeed_kde"]),
        ),
        wp_loc2["gt_windspeed_kde"],
        color=plotting_util.COLOR_SCHEME["gt"],
        lw=2,
        zorder=30,
    )

    if not np.allclose(
        wp_loc1["obs_windspeed_100"], wp_loc2["obs_windspeed_100"], rtol=1e-3
    ):
        raise ValueError("Windspeeds do not match")
    if not np.allclose(wp_loc1["obs_windpower"], wp_loc2["obs_windpower"], rtol=1e-3):
        raise ValueError("Windpowers do not match")
    if not np.allclose(
        wp_loc1["obs_windpower_kde"], wp_loc2["obs_windpower_kde"], rtol=1e-3
    ):
        raise ValueError("Windpower KDEs do not match")
    if not np.allclose(
        wp_loc1["obs_windspeed_kde"], wp_loc2["obs_windspeed_kde"], rtol=1e-3
    ):
        raise ValueError("Windspeed KDEs do not match")

    axs[0, 0].plot(
        np.linspace(
            wp_loc1["obs_windspeed_100"].min(),
            wp_loc1["obs_windspeed_100"].max(),
            len(wp_loc1["obs_windspeed_kde"]),
        ),
        wp_loc1["obs_windspeed_kde"],
        color=plotting_util.COLOR_SCHEME["obs"],
        lw=2,
        zorder=30,
    )
    axs[1, 0].plot(
        np.linspace(
            wp_loc2["obs_windspeed_100"].min(),
            wp_loc2["obs_windspeed_100"].max(),
            len(wp_loc2["obs_windspeed_kde"]),
        ),
        wp_loc2["obs_windspeed_kde"],
        color=plotting_util.COLOR_SCHEME["obs"],
        lw=2,
        zorder=30,
    )

    axs[0, 0].spines["top"].set_visible(False)
    axs[0, 0].spines["right"].set_visible(False)
    axs[0, 0].spines["bottom"].set_visible(True)
    axs[0, 0].spines["left"].set_visible(False)
    axs[0, 0].set_yticks([])
    axs[0, 0].set_yticklabels([])
    axs[1, 0].spines["top"].set_visible(False)
    axs[1, 0].spines["right"].set_visible(False)
    axs[1, 0].spines["bottom"].set_visible(True)
    axs[1, 0].spines["left"].set_visible(False)
    axs[1, 0].set_yticks([])
    axs[1, 0].set_yticklabels([])

    sample_windspeed_kde1 = wp_loc1["sample_windspeed_kde"]
    sample_windspeed_kde2 = wp_loc2["sample_windspeed_kde"]
    num_samples = sample_windspeed_kde1.shape[0]
    for s in range(num_samples):
        axs[0, 0].plot(
            np.linspace(
                wp_loc1["sample_windspeed_100"].min(),
                wp_loc1["sample_windspeed_100"].max(),
                len(sample_windspeed_kde1[s]),
            ),
            sample_windspeed_kde1[s],
            color=plotting_util.COLOR_SCHEME["pred"],
            lw=0.5,
            zorder=31,
            alpha=0.4,
        )
        axs[1, 0].plot(
            np.linspace(
                wp_loc2["sample_windspeed_100"].min(),
                wp_loc2["sample_windspeed_100"].max(),
                len(sample_windspeed_kde2[s]),
            ),
            sample_windspeed_kde2[s],
            color=plotting_util.COLOR_SCHEME["pred"],
            lw=0.5,
            zorder=31,
            alpha=0.4,
        )

    axs[0, 1].plot(
        np.linspace(
            wp_loc1["gt_windspeed_100"].min(),
            wp_loc1["gt_windspeed_100"].max(),
            len(wp_loc1["gt_windspeed_kde"]),
        ),
        wp_loc1["gt_power_gen"] / 1e3,
        color=plotting_util.COLOR_SCHEME["gt"],
        lw=2,
        zorder=30,
    )
    (gt_handle,) = axs[1, 1].plot(
        np.linspace(
            wp_loc2["gt_windspeed_100"].min(),
            wp_loc2["gt_windspeed_100"].max(),
            len(wp_loc2["gt_windspeed_kde"]),
        ),
        wp_loc2["gt_power_gen"] / 1e3,
        color=plotting_util.COLOR_SCHEME["gt"],
        lw=2,
        zorder=30,
    )
    axs[0, 1].plot(
        np.linspace(
            wp_loc1["obs_windspeed_100"].min(),
            wp_loc1["obs_windspeed_100"].max(),
            len(wp_loc1["obs_windspeed_kde"]),
        ),
        wp_loc1["obs_power_gen"] / 1e3,
        color=plotting_util.COLOR_SCHEME["obs"],
        lw=2,
        zorder=30,
    )
    (obs_handle,) = axs[1, 1].plot(
        np.linspace(
            wp_loc2["obs_windspeed_100"].min(),
            wp_loc2["obs_windspeed_100"].max(),
            len(wp_loc2["obs_windspeed_kde"]),
        ),
        wp_loc2["obs_power_gen"] / 1e3,
        color=plotting_util.COLOR_SCHEME["obs"],
        lw=2,
        zorder=30,
    )

    print(gt_ds.time.values)
    print(obs_ds.time.values)

    print(len(wp_loc1["gt_windpower"]))
    print(len(wp_loc1["obs_windpower"]))
    print(len(wp_loc1["sample_windpower"]))

    axs[0, 2].plot(
        gt_ds.time.values,
        np.cumsum(wp_loc1["gt_windpower"]) / len(wp_loc1["gt_windpower"]) / 1e3,
        color=plotting_util.COLOR_SCHEME["gt"],
        lw=2,
        zorder=30,
    )
    axs[1, 2].plot(
        gt_ds.time.values,
        np.cumsum(wp_loc2["gt_windpower"] / len(wp_loc2["gt_windpower"])) / 1e3,
        color=plotting_util.COLOR_SCHEME["gt"],
        lw=2,
        zorder=30,
    )
    axs[0, 2].plot(
        obs_ds.time.values,
        np.cumsum(wp_loc1["obs_windpower"]) / len(wp_loc1["obs_windpower"]) / 1e3,
        color=plotting_util.COLOR_SCHEME["obs"],
        lw=2,
        marker="o",
        markersize=1,
        zorder=30,
    )
    axs[1, 2].plot(
        obs_ds.time.values,
        np.cumsum(wp_loc2["obs_windpower"]) / len(wp_loc2["obs_windpower"]) / 1e3,
        color=plotting_util.COLOR_SCHEME["obs"],
        lw=2,
        marker="o",
        markersize=1,
        zorder=30,
    )

    for s in range(num_samples):
        (sample_handle,) = axs[0, 1].plot(
            np.linspace(
                wp_loc1["sample_windspeed_100"].min(),
                wp_loc1["sample_windspeed_100"].max(),
                len(wp_loc1["sample_power_gen"][s]),
            ),
            wp_loc1["sample_power_gen"][s] / 1e3,
            color=plotting_util.COLOR_SCHEME["pred"],
            lw=0.5,
            alpha=0.4,
            zorder=31,
        )
        (sample_handle,) = axs[1, 1].plot(
            np.linspace(
                wp_loc2["sample_windspeed_100"].min(),
                wp_loc2["sample_windspeed_100"].max(),
                len(wp_loc2["sample_power_gen"][s]),
            ),
            wp_loc2["sample_power_gen"][s] / 1e3,
            color=plotting_util.COLOR_SCHEME["pred"],
            lw=0.5,
            alpha=0.4,
            zorder=31,
        )
        axs[0, 2].plot(
            gt_ds.time.values,
            np.cumsum(wp_loc1["sample_windpower"][s])
            / len(wp_loc1["sample_windpower"][s])
            / 1e3,
            color=plotting_util.COLOR_SCHEME["pred"],
            lw=0.5,
            alpha=0.4,
            zorder=31,
        )
        axs[1, 2].plot(
            gt_ds.time.values,
            np.cumsum(wp_loc2["sample_windpower"][s])
            / len(wp_loc2["sample_windpower"][s])
            / 1e3,
            color=plotting_util.COLOR_SCHEME["pred"],
            lw=0.5,
            alpha=0.4,
            zorder=31,
        )

    for ax in axs[:, -1]:
        ax.set_xticks(obs_ds.time.values[(0, -1),])
        ax.xaxis.set_major_formatter(mdates.DateFormatter(plotting_util.DATE_FMT_STR))

    axs[0, 1].set_ylabel("Windpower [kW]")
    axs[1, 1].set_ylabel("Windpower [kW]")

    axs[0, 0].annotate(
        "a",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-0.5, -0.45),
        textcoords="offset fontsize",
        fontsize="medium",
        fontweight="bold",
        verticalalignment="top",
    )
    axs[0, 1].annotate(
        "b",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(+0.35, -0.25),
        textcoords="offset fontsize",
        fontsize="medium",
        fontweight="bold",
        verticalalignment="top",
    )
    axs[0, 2].annotate(
        "c",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(+0.35, -0.25),
        textcoords="offset fontsize",
        fontsize="medium",
        fontweight="bold",
        verticalalignment="top",
    )
    axs[1, 0].annotate(
        "d",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-0.5, -0.45),
        textcoords="offset fontsize",
        fontsize="medium",
        fontweight="bold",
        verticalalignment="top",
    )
    axs[1, 1].annotate(
        "e",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(+0.35, -0.35),
        textcoords="offset fontsize",
        fontsize="medium",
        fontweight="bold",
        verticalalignment="top",
    )
    axs[1, 2].annotate(
        "f",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(+0.35, -0.35),
        textcoords="offset fontsize",
        fontsize="medium",
        fontweight="bold",
        verticalalignment="top",
    )

    axs[1, 0].set_xlabel("Wind Speed [m/s]")
    axs[1, 1].set_xlabel("Wind Speed [m/s]")
    axs[1, 2].set_xlabel("Time")
    axs[0, 0].set_ylabel("Density")
    axs[1, 0].set_ylabel("Density")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2, wspace=0.5)
    fig.legend(
        [gt_handle, obs_handle, sample_handle],
        ["Reanalysis", "ESM", "Predictions"],
        loc="center",
        bbox_to_anchor=(0.5, 0.025),
        ncols=4,
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        frameon=False,
        framealpha=0.6,
    )

    save_path = pathlib.Path(
        f"/path/to/experiments/run/plots/windpower_{rlat1}-{rlon1}_{rlat2}-{rlon2}.png"
    )
    print(f"Saving to {save_path}")

    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".pdf"))

    # -----

    loaded = np.load(wp_loc_dir / "windpower.npz")
    sample_windpower = loaded["sample_windpower"]
    sample_windpower_kde = loaded["sample_windpower_kde"]
    sample_windspeed_kde = loaded["sample_windspeed_kde"]
    gt_windpower = loaded["gt_windpower"]
    gt_windspeed_kde = loaded["gt_windspeed_kde"]
    obs_windpower = loaded["obs_windpower"]
    obs_windpower_kde = loaded["obs_windpower_kde"]
    obs_windspeed_kde = loaded["obs_windspeed_kde"]
    windspeed_range = loaded["windspeed_range"]
    gt_power_gen = loaded["gt_power_gen"]
    sample_power_gen = loaded["sample_power_gen"]
    obs_power_gen = loaded["obs_power_gen"]

    num_samples = sample_windpower_kde.shape[0]

    fig, ax = plt.subplots(1, 3, figsize=(9, 2))
    ax[0].plot(
        windspeed_range,
        gt_windspeed_kde,
        color=plotting_util.COLOR_SCHEME["gt"],
        lw=2,
        zorder=30,
    )
    ax[0].plot(
        windspeed_range,
        obs_windspeed_kde,
        color=plotting_util.COLOR_SCHEME["obs"],
        lw=2,
        zorder=30,
    )
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["bottom"].set_visible(True)
    ax[0].spines["left"].set_visible(False)
    ax[0].set_yticks([])
    ax[0].set_yticklabels([])
    for s in range(sample_windspeed_kde.shape[0]):
        ax[0].plot(
            windspeed_range,
            sample_windspeed_kde[s],
            color=plotting_util.COLOR_SCHEME["pred"],
            lw=0.5,
            zorder=31,
            alpha=0.4,
        )

    ax[0].set_xlabel("Wind Speed [m/s]")
    ax[1].set_xlabel("Wind Speed [m/s]")
    ax[2].set_xlabel("Time")
    ax[0].set_ylabel("Density")

    (gt_handle,) = ax[1].plot(
        windspeed_range,
        gt_power_gen / 1e3,
        color=plotting_util.COLOR_SCHEME["gt"],
        lw=2,
        zorder=30,
    )
    (obs_handle,) = ax[1].plot(
        windspeed_range,
        obs_power_gen / 1e3,
        color=plotting_util.COLOR_SCHEME["obs"],
        lw=2,
        zorder=30,
    )

    print(gt_windpower.shape)
    print(obs_windpower.shape)
    print(sample_windpower.shape)

    gwp = np.cumsum(gt_windpower.mean((-2, -1)) / 1e3)
    ax[2].plot(
        gt_ds.time.values,
        gwp / len(gwp),
        color=plotting_util.COLOR_SCHEME["gt"],
        lw=2,
        zorder=30,
    )
    owp = np.cumsum(obs_windpower.mean((-2, -1)) / 1e3)
    ax[2].plot(
        obs_ds.time.values,
        owp / len(owp),
        color=plotting_util.COLOR_SCHEME["obs"],
        linestyle="",
        marker="o",
        markersize=1,
        zorder=30,
    )

    print(obs_windpower_kde / 1e3)
    print(obs_windspeed_kde)
    for s in range(num_samples):
        (sample_handle,) = ax[1].plot(
            windspeed_range,
            sample_power_gen[s] / 1e3,
            color=plotting_util.COLOR_SCHEME["pred"],
            lw=0.5,
            alpha=0.4,
            zorder=31,
        )
        swp = np.cumsum(sample_windpower[s].mean((-2, -1)) / 1e3)
        ax[2].plot(
            gt_ds.time.values,
            swp / len(swp),
            color=plotting_util.COLOR_SCHEME["pred"],
            lw=0.5,
            alpha=0.4,
            zorder=31,
        )

    ax[1].set_ylabel("Windpower [kW]")
    ax[0].annotate(
        "a",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-0.5, -0.35),
        textcoords="offset fontsize",
        fontsize="medium",
        fontweight="bold",
        verticalalignment="top",
    )
    ax[1].annotate(
        "b",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(+0.25, -0.35),
        textcoords="offset fontsize",
        fontsize="medium",
        fontweight="bold",
        verticalalignment="top",
    )
    ax[2].annotate(
        "c",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(+0.25, -0.35),
        textcoords="offset fontsize",
        fontsize="medium",
        fontweight="bold",
        verticalalignment="top",
    )
    ax[2].set_xticks(obs_ds.time.values[(0, -1),])
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter(plotting_util.DATE_FMT_STR))

    fig.tight_layout()
    save_path = pathlib.Path("/path/to/experiments/run/plots/windpower_globalavg.png")
    print(f"Saving to {save_path}")

    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".pdf"))


def big_grid():
    from matplotlib.markers import MarkerStyle
    from matplotlib.transforms import Affine2D

    start_time = "2018-01-18-14"
    time_step = 1
    num_times = 7
    data_vars = (
        "psl",
        "tas",
        "uas",
        "vas",
    )

    proj = ctp.crs.Mollweide()
    transform = ctp.crs.PlateCarree()

    exp_dir = pathlib.Path("/path/to/experiments/run/")
    out_dir = exp_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    sample_ds, _, obs = exputil.setup(exp_dir, load_gt=False)

    FEATURE_NAMES = data_vars

    num_T = num_times
    start_dt = data_pipeline.convert_to_datetime(start_time)
    dts = [start_dt + timedelta(hours=i * time_step) for i in range(num_T)]

    num_samples = sample_ds.sizes["sample_id"]

    if num_samples > 3:
        print(f"Found {num_samples} samples, selecting 3 at random.")
        num_samples = 3
        random_sample_ids = np.random.choice(
            sample_ds["sample_id"].values, num_samples, replace=False
        )
    else:
        num_samples = 1
        random_sample_ids = [0]

    ##

    full_data_root_dir = pathlib.Path("/path/to/data/COSMO-full-spatial-region/")
    full_data_vars = sum(
        [
            list(sorted(full_data_root_dir.glob(f"{vr}/{vr}_*.nc")))
            for vr in FEATURE_NAMES
        ],
        [],
    )
    gt_ds = xr.open_mfdataset(
        full_data_vars,
        parallel=True,
        compat="override",
        data_vars="minimal",
        coords="minimal",
        join="override",
        combine_attrs="drop",
    )

    gt_ds = gt_ds.drop_vars([v for v in gt_ds.data_vars if v not in FEATURE_NAMES])

    # crop to region that encloses the studied spatial patch
    # (can be changed to view larger spatial context around the patch)
    gt_ds = gt_ds.sel(rlat=slice(-6.5, 4.55), rlon=slice(-10, 1))
    gt_ds = gt_ds.sel(time=sample_ds.time)
    gt_ds = gt_ds.load()
    if "psl" in data_vars:
        gt_ds["psl"] = gt_ds["psl"] / 100.0  # convert to hPa

    big_sample_ds = (
        gt_ds.copy()
        .expand_dims(dim={"sample_id": sample_ds.sizes["sample_id"]})
        .copy(deep=True)
    )
    big_sample_ds = plotting_util.assign_overlapping_values(
        big_sample_ds, sample_ds, variables=FEATURE_NAMES
    )

    gt_ds = gt_ds.drop_vars([v for v in gt_ds.data_vars if v not in FEATURE_NAMES])

    ##

    sample_ds = sample_ds.compute()

    obs = obs.compute()

    vmin = {v: gt_ds.sel(time=dts)[v].quantile(0.01).item() for v in FEATURE_NAMES}
    vmax = {v: gt_ds.sel(time=dts)[v].quantile(0.99).item() for v in FEATURE_NAMES}

    imshow_kwargs = dict(
        uas=(
            dict(
                cmap=sns.color_palette("Spectral_r", as_cmap=True),
                norm=TwoSlopeNorm(
                    0,
                    vmin=vmin["uas"],
                    vmax=vmax["uas"],
                ),
            )
            if "uas" in FEATURE_NAMES
            else {}
        ),
        vas=(
            dict(
                cmap=sns.color_palette("Spectral_r", as_cmap=True),
                norm=TwoSlopeNorm(
                    0,
                    vmin=vmin["vas"],
                    vmax=vmax["vas"],
                ),
            )
            if "vas" in FEATURE_NAMES
            else {}
        ),
        tas=(
            dict(
                cmap="coolwarm",
                vmin=vmin["tas"],
                vmax=vmax["tas"],
            )
            if "tas" in FEATURE_NAMES
            else {}
        ),
        psl=(
            dict(
                cmap="inferno",
                vmin=vmin["psl"],
                vmax=vmax["psl"],
            )
            if "psl" in FEATURE_NAMES
            else {}
        ),
    )

    for var_name in data_vars:
        fig, axs = plt.subplots(
            num_samples + 2,
            num_T,
            figsize=(9, 6),
            subplot_kw=dict(projection=proj, frameon=False),
        )

        for ip in range(num_samples + 2):
            for it, timepoint in enumerate(dts):
                if ip == num_samples:
                    cond = gt_ds
                elif ip == num_samples + 1:
                    cond = obs
                else:
                    cond = big_sample_ds.isel(sample_id=random_sample_ids[ip])
                try:
                    _, _, gcs = plotting_util.plot_map(
                        cond[var_name].sel(time=timepoint),
                        fig=fig,
                        ax=axs[ip, it],
                        **imshow_kwargs.get(var_name, {}),
                        rasterized=True,
                        add_coastlines=ip <= num_samples,
                    )
                    edgemarkers = {
                        (0, 0): MarkerStyle(
                            "P",
                            # fillstyle="full",
                            transform=Affine2D().rotate_deg(10),
                        ),
                        (0, -1): MarkerStyle(
                            "P",
                            # fillstyle="full",
                            transform=Affine2D().rotate_deg(10),
                        ),
                        (-1, 0): MarkerStyle(
                            "P",
                            # fillstyle="full",
                            transform=Affine2D().rotate_deg(100),
                        ),
                        (-1, -1): MarkerStyle(
                            "P",
                            # fillstyle="full",
                            transform=Affine2D().rotate_deg(10),
                        ),
                    }
                    for rla in [0, -1]:
                        for rlo in [0, -1]:
                            axs[ip, it].plot(
                                sample_ds.longitude.isel(rlat=rla, rlon=rlo),
                                sample_ds.latitude.isel(rlat=rla, rlon=rlo),
                                marker=edgemarkers[(rla, rlo)],
                                markerfacecolor="red",
                                markeredgecolor="white",
                                markeredgewidth=1.0,
                                transform=transform,
                                markersize=8,
                                zorder=300,
                                linestyle="",
                                clip_on=False,
                            )
                except KeyError as e:
                    print(e)

                if ip == 0 and it % 1 == 0:
                    loc_kwargs = dict(x=0.33) if it == 0 else dict(loc="center")
                    axs[0, it].set_title(
                        (
                            f"{timepoint.strftime(plotting_util.DATE_FMT_STR)}"
                            if it > 0
                            else f"{timepoint.strftime(f'%Y {plotting_util.DATE_FMT_STR}')}"
                        ),
                        fontsize="medium",
                        y=1.1,
                        **loc_kwargs,
                    )

            if ip == num_samples:
                axs[ip, 0].yaxis.set_visible(True)
                axs[ip, 0].set_yticks([])
                axs[ip, 0].spines["left"].set_visible(False)
                axs[ip, 0].set_ylabel("Reanalysis", fontsize="medium")
            elif ip == num_samples + 1:
                axs[ip, 0].yaxis.set_visible(True)
                axs[ip, 0].set_yticks([])
                axs[ip, 0].spines["left"].set_visible(False)
                axs[ip, 0].set_ylabel("Coarse input", fontsize="medium")
            else:
                axs[ip, 0].yaxis.set_visible(True)
                axs[ip, 0].set_yticks([])
                axs[ip, 0].spines["left"].set_visible(False)
                axs[ip, 0].set_ylabel(
                    f"Sample {'#'}{random_sample_ids[ip]}", fontsize="medium"
                )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.05, wspace=0.05, hspace=0.2)
        cbar_ax = fig.add_axes([0.3, 0.09, 0.4, 0.015])
        cbar = fig.colorbar(
            gcs,
            cax=cbar_ax,
            extend="max" if var_name == "ws" else "both",
            shrink=0.5,
            orientation="horizontal",
        )
        cbar.set_label(
            plotting_util.var2name(var_name, with_units=True, with_linebreak=False)
        )
        if var_name == "psl":
            cbar.ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10, -5))
            cbar.ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        elif var_name == "tas":
            # cticks = [250, 265, 280]
            cbar.ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
            cbar.ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        elif var_name == "uas":
            # cticks = [-4, -2, 0, 6, 12]
            cbar.ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5.0))
            cbar.ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
        elif var_name == "vas":
            # cticks = [-4.5, -2.5, 0, 3.5, 7]
            # cbar.set_ticks(cticks)
            cbar.ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2.5))
            cbar.ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))

        fig.align_ylabels(axs[:, 0])

        save_path = out_dir / f"{var_name}_big_grid.png"
        fig.savefig(save_path)

        plt.close("all")


if __name__ == "__main__":
    fire.Fire(
        dict(
            onmodel_dist=kde_and_pmf,
            onmodel_storm=storm_grid,
            onmodel_biggrid=big_grid,
            onmodel_timeseries=timeseries,
            clim_dist=downscaled_clim_dist,
            clim_grid=climate_grid,
            clim_windpowers=windpowers,
        )
    )
