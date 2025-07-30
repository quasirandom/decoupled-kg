"""Defines the data loader which acts like the DataCatalog for the postprocessing step"""

import os
import re
import sys
import warnings
from contextlib import contextmanager
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from botorch.exceptions import InputDataWarning
from tqdm.autonotebook import tqdm

from decoupledbo.pipeline.data_catalog import DataCatalog
from decoupledbo.pipeline.main import load_and_construct_gp_test_problem


@contextmanager
def suppress_botorch_standardisation_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"Input data is not standardized .*\. Please consider scaling the "
                r"input to zero mean and unit variance."
            ),
            category=InputDataWarning,
            module="botorch.models.utils.assorted",
        )
        yield


def validate_data_dir_dict(data_dirs: dict[Any, str]):
    """Raise if any of the values in the dictionary are not directories"""
    errs = []
    for d in data_dirs.values():
        if not os.path.isdir(d):
            errs.append(d)
    if errs:
        raise NotADirectoryError(errs)


def load_npz(fpath):
    with np.load(fpath) as f:
        return dict(f)


class DataLoader:
    def __init__(self, data_dirs, problem_dirs):
        """Create a DataLoader for experiment results

        Args:
            data_dirs: A dictionary mapping tuples of the form
                `(problem_family, algorithm)` to the directory containing repeated runs
                of that problem-algorithm combination. Both coupled and decoupled
                variants of the algorithm should be in the same directory.
            problem_dirs: A separate dictionary mapping `problem_family` to a directory
                containing the shared data for that test problem family.
        """
        validate_data_dir_dict(data_dirs)
        validate_data_dir_dict(problem_dirs)

        self.data_dirs = data_dirs
        self.problems_dirs = problem_dirs

    def load_and_concat_all_metrics(self, problem_alg_pair):
        """Load metrics for all repeats of one algorithm on one test problem family

        Repeats of the problem family are identified by using all subdirectories of the
        experiment directory in `self.data_dirs`. Both coupled and decoupled variants of
        the algorithm in question are loaded.

        Args:
            problem_alg_pair: A tuple of the form `(problem_family, algorithm)`

        Returns:
            A tuple `(metrics_concat, missing)`, where `metrics_concat` is a dictionary
                with keys "decoupled" and "fulleval" containing results for all runs
                of the `(problem_family, algorithm)` combination, and `missing` is a
                list of runs for which data could not be loaded.
        """
        all_run_names = self._get_runs(problem_alg_pair)

        metrics = {"decoupled": [], "fulleval": []}
        missing = []
        for run_name in tqdm(all_run_names, desc=f"Loading {problem_alg_pair}"):
            try:
                m = self.load_metrics(problem_alg_pair, run_name)
            except FileNotFoundError:
                tqdm.write(f"Could not load run {run_name}", file=sys.stderr)
                missing.append(run_name)
                continue

            assert m.keys() == metrics.keys()
            for k in metrics.keys():
                metrics[k].append(
                    m[k]
                    .assign(run_name=run_name)
                    .rename_axis(index="iteration")
                    .reset_index()
                )

        metrics_concat = {k: pd.concat(metrics[k], ignore_index=True) for k in metrics}

        if missing:
            print(
                f"Failed to load {len(missing)} out of {len(all_run_names)} runs "
                f"in total",
                file=sys.stderr,
            )

        return metrics_concat, missing

    def load_metrics(
        self, problem_alg_pair: tuple[str], run_name: str
    ) -> Dict[str, pd.DataFrame]:
        """Load metrics for one algorithm and one instance (run) of a problem family

        Both coupled and decoupled variants of the algorithm in question are loaded.

        Args:
            problem_alg_pair: A tuple of the form `(problem_family, algorithm)`
            run_name: The run name specifying which repeat of the problem family to load
                (typically a string representation of an integer)

        Returns:
            A dictionary with keys "decoupled" and "fulleval" containing results
                dataframes for both algorithm variants
        """

        metrics_decoupled_df = self._load_df(
            problem_alg_pair, run_name, "metrics", "metrics_eval_separate.pqt"
        )
        metrics_fulleval_df = self._load_df(
            problem_alg_pair, run_name, "metrics", "metrics_eval_full.pqt"
        )

        metrics_decoupled_df["cost_cum"] = metrics_decoupled_df["cost"].cumsum()
        metrics_fulleval_df["cost_cum"] = metrics_fulleval_df["cost"].cumsum()

        return {
            "decoupled": metrics_decoupled_df,
            "fulleval": metrics_fulleval_df,
        }

    def load_timings(self, problem_alg_pair, alg_variant) -> pd.DataFrame:
        """Load timings for one algorithm and one instance (run) of a problem family

        Both coupled and decoupled variants of the algorithm in question are loaded.

        Args:
            problem_alg_pair: A tuple of the form `(problem_family, algorithm)`
            alg_variant: A string representing the algorithm variant (either
                "eval_separate" or "eval_full")

        Returns:
            A dataframe containing timings for both algorithm variants
        """

        all_run_names = sorted(
            os.listdir(self.data_dirs[problem_alg_pair]),
            key=lambda s: int(re.match(r"(.*[_-])?(\d+)$", s).group(2)),
        )

        missing = []
        timings = []
        for run_name in tqdm(
            all_run_names, desc=f"Loading {problem_alg_pair} | {alg_variant}"
        ):
            try:
                df = self._load_df(
                    problem_alg_pair, run_name, "timings", f"timings_{alg_variant}.pqt"
                )
            except FileNotFoundError:
                tqdm.write(f"Could not load run {run_name}", file=sys.stderr)
                missing.append(run_name)
                continue

            df["run_name"] = run_name
            timings.append(df)

        timings_df = pd.concat(timings)

        if missing:
            print(
                f"Failed to load {len(missing)} out of {len(all_run_names)} runs "
                f"in total",
                file=sys.stderr,
            )

        return timings_df

    def _load_df(self, key, *args):
        return pd.read_parquet(os.path.join(self.data_dirs[key], *args))

    def load_max_possible_scalar_perfs(self, *problem_alg_pair):
        """Load the maximum possible scalarised performance for each test problem

        Args:
            problem_alg_pair: A tuple of the form `(problem_family, algorithm)`

        Returns:
            A pandas series containing the maximum possible scalarised performance for
                each problem in the test problem family (there is a different instance
                of the test problem for each 'run')
        """
        runs = self._get_runs(problem_alg_pair)

        max_possible_scalar_perfs = {}
        for run_name in runs:
            fpath = os.path.join(
                self.data_dirs[problem_alg_pair],
                run_name,
                "max_possible_scalarisation_metric.npy",
            )
            max_possible_scalar_perfs[run_name] = np.load(fpath).item()

        max_possible_scalar_perfs = pd.Series(max_possible_scalar_perfs)
        max_possible_scalar_perfs.index.name = "run_name"
        max_possible_scalar_perfs.name = "max_possible_scalarperf"
        return max_possible_scalar_perfs

    def load_max_possible_hypervolume(self, problems_key):
        """Load the maximum possible hypervolume for each test problem"""

        all_problem_ids = self._get_problem_instances(problems_key)
        max_possible_hv = {}
        for problem_id in all_problem_ids:
            fpath = os.path.join(self.problems_dirs[problems_key], f"{problem_id}.pt")
            problem_data = torch.load(fpath)
            max_possible_hv[problem_id] = problem_data["max_hv"]

        return pd.Series(max_possible_hv, name="max_possible_hypervolume").rename_axis(
            index="run_name"
        )

    def load_test_problem(self, problem_family, alg, variant, run, num_iterations):
        """Load a test problem and pareto front information

        Args:
            problem_family: The problem family (e.g. "lengthscales")
            alg: The algorithm (e.g. "kg-discrete")
            variant: The algorithm variant (either "eval_separate" or "eval_full")
            run: The run (e.g. "1")

        Returns:
            A tuple `(true_pareto, posterior_pareto, test_problem)`, where
                - `true_pareto` is a dict of numpy arrays with `"pareto_set"` and
                    `"pareto_front"`
                - `posterior_pareto` is a dict with keys indexing the iteration and
                    values being dictionarys of the same format as `true_pareto`
                - `test_problem` is a dict with keys `"problem"` and `"hparams"`
                    containing a `GPTestProblem` and the dictionary of hyperparameters
                    defining that problem family
        """
        problem_name = self.problems_dirs[problem_family] + f"/{run}"

        with suppress_botorch_standardisation_warning():
            _problem, _hparams = load_and_construct_gp_test_problem(
                problem_name, DataCatalog, noise_stds=None
            )
        test_problem = {"problem": _problem, "hparams": _hparams}

        true_pareto = load_npz(
            os.path.join(self.data_dirs[problem_family, alg], run, "true_pareto.npz")
        )

        posterior_pareto = {
            iteration: load_npz(
                os.path.join(
                    self.data_dirs[problem_family, alg],
                    run,
                    "posterior_pareto",
                    variant,
                    f"posterior_pareto_{iteration:02d}.npz",
                )
            )
            for iteration in range(num_iterations)
        }

        return true_pareto, posterior_pareto, test_problem

    def _get_runs(self, problem_alg_pair):
        all_run_names = sorted(
            os.listdir(self.data_dirs[problem_alg_pair]),
            key=lambda s: int(re.match(r"(.*[_-])?(\d+)$", s).group(2)),
        )
        return all_run_names

    def _get_problem_instances(self, problem_family) -> List[int]:
        return sorted(
            [
                re.match(r"(\d+)\.pt$", fname).group(1)
                for fname in os.listdir(self.problems_dirs[problem_family])
            ],
            key=int,
        )
