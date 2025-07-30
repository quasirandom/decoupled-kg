"""Functions to generate results plots in the paper"""

import copy
import math

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm

PALETTE = sns.color_palette("colorblind")
MARKERS = ["o", "v", "^", "d", "s", "*", "<", ">"]

ACQF_META = {
    "kg-discrete": {"color": PALETTE[0], "marker": MARKERS[0], "label": "C-MOKG"},
    "kg-discrete-random": {
        "color": PALETTE[1],
        "marker": MARKERS[1],
        "label": "C-MOKG (random)",
    },
    "jes": {"color": PALETTE[2], "marker": MARKERS[2], "label": "JES-LB"},
    "hvkg": {"color": PALETTE[3], "marker": MARKERS[3], "label": "HVKG"},
}

DECOUPLED_META = {
    "decoupled": {"fillstyle": "none", "linestyle": "dashed", "label": "Decoupled"},
    "fulleval": {"fillstyle": "full", "linestyle": "solid", "label": "Coupled"},
}

ALGO_META = {
    (acqf, eval_mode): d1 | d2 | {"label": f"{d2['label']} ({d1['label']})"}
    for acqf, d2 in ACQF_META.items()
    for eval_mode, d1 in DECOUPLED_META.items()
}
_label_overrides = {
    ("kg-discrete", "fulleval"): "maKG",
    ("kg-discrete", "decoupled"): "C-MOKG",
    ("kg-discrete-random", "fulleval"): "maKG (random)",
    ("kg-discrete-random", "decoupled"): "C-MOKG (random)",
    ("jes", "fulleval"): "JES-LB-c",
    ("jes", "decoupled"): "JES-LB-d",
    ("hvkg", "fulleval"): "HVKG-c",
    ("hvkg", "decoupled"): "HVKG-d",
}
for k, v in _label_overrides.items():
    ALGO_META[k]["label"] = v

METRIC_NAMES = {
    "scalarperf_regret": "Expected\nBayesian regret",
    "hv_regret": "Expected\nHypervolume regret",
}


def plot_mean_metrics_comparison(
    stats_df,
    metric,
    *,
    algorithms=None,
    ax=None,
    legend=True,
    legend_cols=None,
    legend_outside=None,
):
    """Generate a plot of mean performance metrics

    Args:
        stats_df: The dataframe of performance statistics
        metric: The metric to plot, either "scalarperf_regret" or "hv_regret"
        algorithms: The algorithms to include in the plot (if None, then all algorithms
            will be plotted)
        ax: An axis on which to plot the figure (if None, then a new figure will be
            created)
        legend: If True, then add a legend to the axes
        legend_cols: The number of columns to include in the legend
        legend_outside: Whether to place the legend inside or outside the axes
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)

    if algorithms is not None:
        algo_meta = {alg: ALGO_META[alg] for alg in algorithms}
    else:
        algorithms = ALGO_META.keys()
        algo_meta = ALGO_META

    # Plot a span for the initial sample
    c1 = stats_df[metric]["mean"].dropna().reset_index()["cost_cum"].min()
    ax.axvspan(0, c1, color="lightgray", alpha=0.2)

    for (alg, eval_mode), meta in algo_meta.items():
        df = (
            stats_df[metric].xs(alg, level="algorithm").xs(eval_mode, level="eval_mode")
        )
        ax.fill_between(
            df.index,
            df["mean_minus_2sem"],
            df["mean_plus_2sem"],
            color=meta["color"],
            alpha=0.2,
            step="post",
            zorder=1.1 if alg == "kg-discrete" else None,
        )
        ax.plot(
            df.index,
            df["mean"],
            color=meta["color"],
            linestyle=meta["linestyle"],
            marker=meta["marker"],
            fillstyle=meta["fillstyle"],
            markevery=20,
            label=meta["label"],
            drawstyle="steps-post",
            zorder=2.1 if alg == "kg-discrete" else None,
        )

    if legend:
        if legend_outside is None:
            legend_outside = False
        _add_legend(ax, legend_outside, algorithms, ncols=legend_cols)

    ax.set_xlabel("Cumulative evaluation cost")
    ax.set_ylabel(METRIC_NAMES[metric])

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    return ax


def _add_legend(ax, outside, algorithms, *, ncols=None):
    """Add a legend to the axes showing all algorithms

    This supports using different names for the coupled and decoupled versions
    of an algorithm.
    """
    from matplotlib.lines import Line2D

    if outside:
        if ncols is None:
            ncols = 1
        leg_kwargs = dict(loc="upper left", bbox_to_anchor=(1, 1))
        parent = ax.figure
    else:
        if ncols is None:
            ncols = 3
        leg_kwargs = dict(loc="upper right")
        parent = ax

    legend_items = [
        (Line2D([0, 1], [0, 0], **ALGO_META[alg]), ALGO_META[alg]["label"])
        for alg in algorithms
    ]

    return parent.legend(*zip(*legend_items), ncols=ncols, **leg_kwargs)


def plot_pareto_fronts_for_several_iterations(
    true_pareto, posterior_pareto, problem, iterations, *, ncol=3
):
    """Generate a plot of the true and predicted Pareto fronts at several iterations"""
    pset_image = {}
    for i in tqdm(iterations):
        _pset_pt = torch.from_numpy(posterior_pareto[i]["pareto_set"])
        _pset_image_pt = problem(_pset_pt, noise=False)
        pset_image[i] = _pset_image_pt.numpy()

    nrow = math.ceil(len(iterations) / ncol)
    fig, axs = plt.subplots(
        nrow,
        ncol,
        figsize=(2 + 3 * ncol, 0.5 + 2 * nrow),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    for i, it in enumerate(iterations):
        style = dict(marker=".", linestyle="", markersize=3)
        axs.flat[i].plot(
            *true_pareto["pareto_front"].T,
            **style,
            color="black",
            label="True Pareto front",
        )
        axs.flat[i].plot(
            *posterior_pareto[it]["pareto_front"].T,
            **style,
            color=PALETTE[1],
            label="Predicted Pareto front",
        )
        axs.flat[i].plot(
            *pset_image[it].T,
            **style,
            color=PALETTE[0],
            label="Image of predicted Pareto set",
        )
        axs.flat[i].set_title(
            "Initial samples only" if it == 0 else f"{it} BO samples",
            loc="left",
        )

    handles, labels = axs.flat[0].get_legend_handles_labels()
    handles = [copy.copy(h) for h in handles]
    for h in handles:
        h.set(markersize=10)
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0, 1),
        ncol=3,
    )

    for ax in axs.flat:
        ax.tick_params(left=True, labelleft=True, bottom=True, labelbottom=True)
        ax.set_xlabel("First objective, $f_1$")
        ax.set_ylabel("Second objective, $f_2$")

    return fig


def plot_pareto_front(true_pareto, posterior_pareto, problem, *, legend=True):
    """Plot the true and predicted Pareto fronts for a single iteration"""
    pset_pt = torch.from_numpy(posterior_pareto["pareto_set"])
    pset_image_pt = problem(pset_pt, noise=False)
    pset_image = pset_image_pt.numpy()

    fig, ax = plt.subplots(figsize=(3, 2.5))
    style = dict(marker=".", linestyle="", markersize=3)
    ax.plot(
        *true_pareto["pareto_front"].T,
        **style,
        color="black",
        label="True Pareto front",
    )
    ax.plot(
        *posterior_pareto["pareto_front"].T,
        **style,
        color=PALETTE[1],
        label="Pareto front\nof posterior mean",
    )
    ax.plot(
        *pset_image.T,
        **style,
        color=PALETTE[0],
        label="Image of Pareto set",
    )

    if legend:
        leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1))
        for h in leg.legend_handles:
            h.set(markersize=10)

    ax.set_xlabel("First objective, $f_1$")
    ax.set_ylabel("Second objective, $f_2$")

    return ax


def plot_acqf_optim_timing_medians(
    timing_stats_df,
    ax=None,
    eval_mode="decoupled",
    title=True,
    markevery=4,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5), dpi=120, constrained_layout=True)

    # Filter to iterations where no runs have terminated
    timing_stats_df = pd.merge(
        timing_stats_df.reset_index(),
        timing_stats_df.sort_index()
        .xs(0, level="iteration")[[("total", "size")]]
        .rename(columns={"size": "init_size"}, level=1)
        .reset_index(),
        how="left",
        on=("algorithm", "eval_mode"),
        validate="many_to_one",
    )
    timing_stats_df = timing_stats_df[
        timing_stats_df["total", "size"] == timing_stats_df["total", "init_size"]
    ]
    timing_stats_df = timing_stats_df.drop(
        columns=[("total", "size"), ("total", "init_size")]
    )
    timing_stats_df = timing_stats_df.set_index(
        ["algorithm", "eval_mode", "iteration"], verify_integrity=True
    )

    # Plot each algorithm
    algorithms = []
    for alg, _ in ALGO_META:
        if alg not in algorithms:
            algorithms.append(alg)

    for alg in algorithms:
        try:
            df = timing_stats_df.xs(alg, level="algorithm").xs(
                eval_mode, level="eval_mode"
            )["bo"]
        except KeyError:
            continue

        ax.fill_between(
            df.index,
            df["ci95lo"],
            df["ci95hi"],
            color=ALGO_META[alg, eval_mode]["color"],
            alpha=0.2,
        )
        ax.plot(
            df.index,
            df["median"],
            label=ACQF_META[alg]["label"],
            color=ALGO_META[alg, eval_mode]["color"],
            marker=ALGO_META[alg, eval_mode]["marker"],
            fillstyle=ALGO_META[alg, eval_mode]["fillstyle"],
            markevery=markevery,
            linestyle=ALGO_META[alg, eval_mode]["linestyle"],
        )

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    if title is True:
        title = f"Median time to optimize {DECOUPLED_META[eval_mode]['label'].lower()} acquisition functions,\nwith 2.5%-97.5% quantiles\n\n"
    if title:
        ax.set_title(title, x=0, ha="left", size="x-large")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Seconds")

    ax.legend(bbox_to_anchor=(0, 1), loc="lower left", ncol=len(algorithms))

    ax.spines[["top", "right"]].set_visible(False)

    return ax
