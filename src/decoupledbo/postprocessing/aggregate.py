"""Functions to interpolate, calculate regret and aggregate metrics"""

import numpy as np
import pandas as pd


def interpolate_metrics(df):
    """Forward fill metrics

    Different runs will choose the objectives in different orders, so costs will not be
    aligned. Therefore, we need to interpolate to all integer cost values.

    Args:
        df: A dataframe containing metrics for all runs of an algorithm variant on a
            particular test problem family

    Returns:
        A dataframe of interpolated metrics indexed by `("run_name", "cost_cum")`
    """
    interp_df = df.groupby("run_name").apply(_interpolate_metrics_single_run)
    return interp_df


def _interpolate_metrics_single_run(df):
    """Forward fill metrics for a single run

    Different runs will choose the objectives in different orders, so costs will not be
    aligned. Therefore, we need to interpolate to all integer cost values.
    """

    max_cost = df["cost_cum"].max()
    cost = np.arange(max_cost + 1)

    interp_df = (
        df.set_index("cost_cum")[
            [
                "pfront_hv_lo",
                "pfront_hv_hi",
                "pset_hv_lo",
                "pset_hv_hi",
                "predicted_scalarperf",
                "actual_scalarperf",
            ]
        ]
        .reindex(cost)
        .ffill()
    )

    return interp_df


def calculate_regret(interp_df, max_possible_scalar_perf, max_possible_hv, tol=0.01):
    """Calculate regret from the performance metrics and the max possible

    Args:
        interp_df: A dataframe containing interpolated metrics for all runs, indexed by
            `("run_name", "cost_cum")`
        max_possible_scalar_perf: A series containing the max possible scalarised
            objective for the test problem instances (in expectation over
            scalarisations)
        max_possible_hv: A series containing the max possible hypervolume for the test
            problem instances
        tol: The function will raise an error if any regret is more negative than
            `-tol`. We allow for small negative values due to numerical precision.

    Returns:
        A dataframe with columns "scalarperf_regret" and "hv_regret", and with index
            `("run_name", "cost_cum")`
    """
    merged_df = (
        interp_df.reset_index()
        .merge(
            max_possible_scalar_perf.reset_index(),
            how="left",
            on="run_name",
            validate="many_to_one",
            suffixes=(False, False),
        )
        .merge(
            max_possible_hv,
            how="left",
            on="run_name",
            validate="many_to_one",
            suffixes=(False, False),
        )
        .set_index(interp_df.index.names, verify_integrity=True)
    )

    merged_df["scalarperf_regret"] = (
        merged_df["max_possible_scalarperf"] - merged_df["actual_scalarperf"]
    )
    merged_df["hv_regret"] = (
        merged_df["max_possible_hypervolume"] - merged_df["pset_hv_lo"]
    )

    # Because of issues with numerical tolerance, the regret is sometimes
    # negative. We fix this by clipping to zero, but raising an error if the
    # unclipped regret is "too negative".
    regret_cols = ["scalarperf_regret", "hv_regret"]
    is_below_tol = (merged_df[regret_cols] < -tol).any(axis="columns")
    if is_below_tol.any():
        raise ValueError(
            f"Found a significantly negative regret value:\n"
            f"{merged_df.loc[is_below_tol, regret_cols]}"
        )

    merged_df[regret_cols] = merged_df[regret_cols].clip(lower=0)

    return merged_df[regret_cols]


def calculate_stats(regret_df, ci=0.9):
    """Calculate stats on the regret

    Care is taken to only compute statistics up to the largest cumulative cost for which
    we have data from every run.

    Args:
        A dataframe with columns "scalarperf_regret" and "hv_regret", and with index
            `("run_name", "cost_cum")`

    Returns:
        A dataframe indexed by cumulative cost (`"cost_cum"`), and with multi-index
            columns. The first level in the columns indexes the metric (`"hv_regret"` or
            `"scalarperf_regret"`), and the second level indexes the statistic
            (`"mean"`, `"sem"`, ...).
    """

    # We want to ensure that we only report stats for runs with an entry for
    # every cost_cum
    max_cost = (
        regret_df.index.to_frame(index=False)
        .groupby("run_name")["cost_cum"]
        .max()
        .min()
    )

    regret_df = regret_df[regret_df.index.get_level_values("cost_cum") <= max_cost]

    grp = regret_df.groupby("cost_cum")
    df = pd.concat(
        {
            "mean": grp.mean(),
            "sem": grp.sem(),
            "median": grp.median(),
            "cilo": grp.quantile((1 - ci) / 2),
            "cihi": grp.quantile(1 - (1 - ci) / 2),
        },
        axis="columns",
    )

    for c in regret_df.columns:
        df["mean_minus_2sem", c] = df["mean", c] - 2 * df["sem", c]
        df["mean_plus_2sem", c] = df["mean", c] + 2 * df["sem", c]

    df = df.swaplevel(axis="columns").sort_index(
        axis="columns", level=0, sort_remaining=False
    )

    return df


def calculate_timing_stats(timings_df):
    """Calculate statistics on the timings for one problem family and algorithm variant

    These are the timings to fit the GP, optimise the acquisition function, and to
    recommend and evaluate a solution.

    Args:
        timings_df: A dataframe containing timings for all runs in one problem family
            and algorithm variant. Columns should include `"run_name"`, `"iteration"`,
            `"fit"`, `"bo"` and `"metrics"`.

    Returns:
        A dataframe of timing statistics indexed by `"iteration"` and with multi-index
            columns. The first level of the columns indexes the step who's time is
            measured (`"fit"`, `"bo"` or `"metrics"`), and the second level indexes the
            statistic (`"mean"`, `"sem"`, ...).
    """
    # Assumes every algorithm ran with the same num_eval sequence
    grp = timings_df.groupby("iteration")  # ["bo"]

    # Check we aren't accidentally aggregating over extra cols
    assert (timings_df.groupby(["run_name", "iteration"]).size() == 1).all()

    timing_stats_df = (
        pd.concat(
            {
                "mean": grp.mean(),
                "sem": grp.sem(),
                "median": grp.median(),
                "ci95lo": grp.quantile(0.05 / 2),
                "ci95hi": grp.quantile(1 - 0.05 / 2),
                "size": grp.size().rename("total"),
            },
            axis="columns",
        )
        .swaplevel(axis="columns")
        .sort_index(axis="columns", level=0, sort_remaining=False)
    )

    return timing_stats_df
