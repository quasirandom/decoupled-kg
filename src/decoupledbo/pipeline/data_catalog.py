import json
import logging
import os
import re
import shutil
import tarfile
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor

logger = logging.getLogger(__name__)

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../..", "data"))

SHARED_DNAME = "shared"
GP_PROBLEM_DNAME = "gp-problem"
GP_PROBLEM_FNAME_FMT = "{name}.pt"

LOGS_DNAME = "logs"
COMMANDLINE_ARGS_FNAME = "commandline_args.json"
CONFIG_FNAME = "config.yaml"
TRUE_PARETO_FNAME = "true_pareto.npz"
MAX_SCALARISED_PERFORMANCE_FNAME = "max_possible_scalarisation_metric.npy"
INITIAL_DATA_FNAME = "initial_data.pt"
HYPERPARAMETERS_FNAME = "hyperparameters.pt"
SCALARISATIONS_FNAME = "scalarisations.pt"
BO_RUN_DNAME = "bo_runs"
BO_RUN_FNAME_FMT = "bo_run_{run_key}.pqt"
POSTERIOR_PARETO_DNAME = "posterior_pareto"
POSTERIOR_PARETO_FNAME_FMT = "posterior_pareto_{:02d}.npz"
CHECKPOINTS_DNAME = "checkpoints"
CHECKPOINT_FNAME_FMT = "checkpoint_{:02d}.pt"
CHECKPOINTS_COMPRESSED_FNAME_FMT = "checkpoints-{run_key}.tgz"
METRICS_DNAME = "metrics"
METRICS_FNAME_FMT = "metrics_{run_key}.pqt"
TIMINGS_DNAME = "timings"
TIMINGS_FNAME_FMT = "timings_{run_key}.pqt"


class DataCatalog:
    @classmethod
    def save_shared_gp_test_problem_data(
        cls,
        name: str,
        bounds: List[Tuple[float, float]],
        fixed_hyperparams: Dict[str, Any],
        model_state_dict: Dict[str, Any],
        train_x: Tensor,
        train_y: Tensor,
        ref_point: List[float],
        max_hv: float,
        negate: bool,
    ):
        """Save a shared GP test problem

        This problem is shared between namespaces and is saved in a special directory
        specified by the global variable `SHARED_DNAME`.

        Args:
            name: The name identifying the GP test problem
            bounds: The bounds of the GP test problem (one list element per objective)
            fixed_hyperparams: The hyper-parameter dictionary used to generate the GP
                (The hyper-parameters are saved in two ways - this argument should be
                the dictionary of hyper-parameters used to generate the test problem via
                 `create_gp_problem_model`; it is more compact than the state_dict and,
                 for example, contains just one length scale for each objective,
                 representing the length scale for all input dimensions).
            model_state_dict: The state_dict of the GP model, containing the
                hyper-parameters of the model. (The hyper-parameters are saved in two
                ways - this argument should be the model state_dict which, for example,
                contains a separate length scale entry for each pair of input dimension
                and objective).
            train_x: The input locations for the training data.
            train_y: The output values for the training data.
            ref_point: The precomputed reference point
            max_hv: The precomputed hyper-volume associated with the true Pareto front
                and reference point given by the `ref_point` argument.
            negate: A boolean indicating whether the problem should be negated.
        """
        shared_gp_dpath = os.path.join(DATA_DIR, SHARED_DNAME, GP_PROBLEM_DNAME)

        # The problem 'name' may contain '/' characters to support repeated instances of
        # a problem using the same hyper-parameters. Therefore, we pull out the full
        # directory path without the file name and pass this to _create_dir. Otherwise,
        # we get a FileNotFoundError when trying to save a problem containing '/'
        # characters.
        subpath = GP_PROBLEM_FNAME_FMT.format(name=name)
        dpath = os.path.join(shared_gp_dpath, os.path.dirname(subpath))
        _create_dir(shared_gp_dpath)
        _create_dir(dpath, root_dir=shared_gp_dpath)

        fpath = os.path.join(shared_gp_dpath, subpath)

        torch.save(
            {
                "bounds": bounds,
                "fixed_hyperparams": fixed_hyperparams,
                "model_state_dict": model_state_dict,
                "train_x": train_x,
                "train_y": train_y,
                "ref_point": ref_point,
                "max_hv": max_hv,
                "negate": negate,
            },
            fpath,
        )

    @staticmethod
    def load_shared_gp_test_problem_data(name: str, device=None) -> Dict[str, Any]:
        fpath = os.path.join(
            DATA_DIR,
            SHARED_DNAME,
            GP_PROBLEM_DNAME,
            GP_PROBLEM_FNAME_FMT.format(name=name),
        )
        return torch.load(fpath, map_location=device)

    def __init__(self, namespace=None):
        if not namespace:
            namespace = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        if namespace.split("/", maxsplit=1)[0] == SHARED_DNAME:
            raise ValueError(
                f"The namespace {SHARED_DNAME!r} is reserved for shared data."
            )

        self.namespace = namespace

    def get_new_log_file_path(self):
        """Return the next unused log-file name"""
        dpath = self._get_path(LOGS_DNAME)
        _create_dir(dpath)
        matches = {f: re.match(r"^run_(\d+).log$", f) for f in os.listdir(dpath)}
        log_files_by_idx = {int(m.group(1)): f for f, m in matches.items() if m}
        new_index = max(log_files_by_idx, default=-1) + 1
        return os.path.join(dpath, f"run_{new_index:02}.log")

    def save_config(self, config):
        """Save the final configuration file used for the run.

        This can have had values overridden by commandline arguments.
        """
        dpath = self._get_path()
        _create_dir(dpath)
        fpath = os.path.join(dpath, CONFIG_FNAME)
        with open(fpath, "w") as f:
            yaml.dump(config, f, indent=2, default_flow_style=None)

    def save_commandline_args(self, commandline_args):
        """Save the commandline arguments as json for data lineage purposes"""
        dpath = self._get_path()
        _create_dir(dpath)
        fpath = os.path.join(dpath, COMMANDLINE_ARGS_FNAME)
        with open(fpath, "w") as f:
            json.dump(vars(commandline_args), f, indent=2)

    def save_true_pareto(self, pareto_set, pareto_front):
        dpath = self._get_path()
        _create_dir(dpath)
        fpath = os.path.join(dpath, TRUE_PARETO_FNAME)
        np.savez(fpath, pareto_set=pareto_set, pareto_front=pareto_front)

    def load_true_pareto(self) -> Tuple[np.ndarray, np.ndarray]:
        fpath = self._get_path(TRUE_PARETO_FNAME)
        loaded = np.load(fpath)
        return loaded["pareto_set"], loaded["pareto_front"]

    def save_problem_max_possible_expected_scalarisation(self, expected_best: float):
        dpath = self._get_path()
        _create_dir(dpath)
        fpath = os.path.join(dpath, MAX_SCALARISED_PERFORMANCE_FNAME)
        np.save(fpath, expected_best)

    def load_problem_max_possible_expected_scalarisation(self):
        fpath = self._get_path(MAX_SCALARISED_PERFORMANCE_FNAME)
        return np.load(fpath).item()

    def save_initial_data(self, train_x, train_obj, train_obj_true):
        """Save initial data to be used to compare several BO runs"""
        dpath = self._get_path()
        _create_dir(dpath)
        fpath = os.path.join(dpath, INITIAL_DATA_FNAME)
        torch.save(
            {
                "train_x": train_x,
                "train_obj": train_obj,
                "train_obj_true": train_obj_true,
            },
            fpath,
        )

    def load_initial_data(self, device=None):
        """Load the initial data used to compare several BO runs"""
        fpath = self._get_path(INITIAL_DATA_FNAME)
        return torch.load(fpath, map_location=device)

    def save_model_hyperparameters(self, model_state_dict):
        """Save the set of model hyper-parameters to be used throughout each BO run"""
        dpath = self._get_path()
        _create_dir(dpath)
        fpath = os.path.join(dpath, HYPERPARAMETERS_FNAME)
        torch.save(model_state_dict, fpath)

    def load_model_hyperparameters(self, device=None) -> Dict[str, Any]:
        """Load the set of model hyper-parameters to be used throughout each BO run"""
        fpath = self._get_path(HYPERPARAMETERS_FNAME)
        return torch.load(fpath, map_location=device)

    def delete_model_hyperparameters(self):
        """Delete cached model hyper-parameters"""
        fpath = self._get_path(HYPERPARAMETERS_FNAME)
        if os.path.exists(fpath):
            os.remove(fpath)

    def save_scalarisations(self, weights: Tensor):
        dpath = self._get_path()
        _create_dir(dpath)
        fpath = os.path.join(dpath, SCALARISATIONS_FNAME)
        torch.save(weights, fpath)

    def load_scalarisations(self, device=None):
        fpath = self._get_path(SCALARISATIONS_FNAME)
        return torch.load(fpath, map_location=device)

    def save_bo_run(self, run_key: str, query_history_df):
        dpath = self._get_path(BO_RUN_DNAME)
        _create_dir(dpath)
        fname = BO_RUN_FNAME_FMT.format(run_key=run_key)
        fpath = os.path.join(dpath, fname)
        query_history_df.to_parquet(fpath)

    def load_bo_run(self, run_key: str) -> pd.DataFrame:
        fname = BO_RUN_FNAME_FMT.format(run_key=run_key)
        fpath = self._get_path(BO_RUN_DNAME, fname)
        return pd.read_parquet(fpath)

    def save_posterior_pareto(
        self,
        run_key: str,
        iteration: int,
        pareto_set: np.ndarray,
        pareto_front: np.ndarray,
    ):
        """Save the posterior Pareto front & set during the BO run

        Save the pareto set and pareto front associated with the posterior mean at a
        given iteration in the BO run.

        Args:
            run_key: An identifier for the BO run
            iteration: The iteration number within the BO run
            pareto_set: An npoints-by-d matrix of points in the pareto set
            pareto_front: An npoints-by-m matrix of points in the pareto front
        """
        dpath = self._get_path(POSTERIOR_PARETO_DNAME, run_key)
        _create_dir(dpath)
        fpath = os.path.join(dpath, POSTERIOR_PARETO_FNAME_FMT.format(iteration))
        np.savez(fpath, pareto_set=pareto_set, pareto_front=pareto_front)

    def load_posterior_pareto(
        self, run_key: str, iteration: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load the posterior Pareto front & set calculated during a BO run

        Load the pareto set and pareto front associated with the posterior mean at a
        given iteration in a BO run.

        Args:
            run_key: An identifier for the BO run
            iteration: The iteration number within the BO run

        Returns:
            tuple: `(pareto_set, pareto_front)`
                pareto_set - An npoints-by-d matrix of points in the pareto set
                pareto_front - An npoints-by-m matrix of points in the pareto front
        """
        if iteration < 0:
            # Align -1 to last element
            iteration += self.num_posterior_pareto_iterations(run_key)

        fname = POSTERIOR_PARETO_FNAME_FMT.format(iteration)
        fpath = self._get_path(POSTERIOR_PARETO_DNAME, run_key, fname)
        with np.load(fpath) as loaded:
            return loaded["pareto_set"], loaded["pareto_front"]

    def delete_all_posterior_pareto(self):
        """Delete all data for posterior Pareto front

        Useful for clearing up cluster storage.
        """
        dpath = self._get_path(POSTERIOR_PARETO_DNAME)
        if os.path.isdir(dpath):
            shutil.rmtree(dpath)

    def num_posterior_pareto_iterations(self, run_key: str):
        """Return the number of posterior_pareto files"""
        dpath = self._get_path(POSTERIOR_PARETO_DNAME, run_key)
        if not os.path.isdir(dpath):
            return 0
        else:
            fnames = os.listdir(dpath)
            expected_fnames = [
                POSTERIOR_PARETO_FNAME_FMT.format(i) for i in range(len(fnames))
            ]
            if sorted(fnames) != sorted(expected_fnames):
                raise ValueError(
                    f"Found unexpected file names in {POSTERIOR_PARETO_DNAME!r} "
                    f"directory"
                )
            return len(fnames)

    def save_checkpoint(
        self,
        run_key: str,
        iteration: int,
        model_state_dict: Dict[str, Any],
        model_config: Dict[str, Any],
        train_x: List[Union[Tensor, np.ndarray]],
        train_obj: List[Union[Tensor, np.ndarray]],
        train_obj_true: List[Union[Tensor, np.ndarray]],
        problem_bounds: Tensor,
    ):
        """Save a checkpoint during the BO run.

        Save enough information to resume the run later. This includes model
        hyper-parameters as well as observations collected so far.
        """
        dpath = self._get_path(CHECKPOINTS_DNAME, run_key)
        _create_dir(dpath)
        fpath = os.path.join(dpath, CHECKPOINT_FNAME_FMT.format(iteration))
        torch.save(
            {
                "run_key": run_key,
                "iteration": iteration,
                "model_state_dict": model_state_dict,
                "model_config": model_config,
                "train_x": train_x,
                "train_obj": train_obj,
                "train_obj_true": train_obj_true,
                "problem_bounds": problem_bounds,
            },
            fpath,
        )

    def load_checkpoint(self, run_key: str, iteration: int, device=None):
        """Load a previously saved checkpoint.

        Note that it is up to the caller to re-instantiate the model using the state
        dict and training data.
        """
        if self.num_checkpoints(run_key) == 0:
            raise RuntimeError("No checkpoints! Did you forget to uncompress them?")

        if iteration < 0:
            # Align -1 to last element
            iteration += self.num_checkpoints(run_key)

        if iteration < 0:
            raise IndexError("checkpoint index out of range")

        dpath = self._get_path(CHECKPOINTS_DNAME, run_key)
        fpath = os.path.join(dpath, CHECKPOINT_FNAME_FMT.format(iteration))
        loaded = torch.load(fpath, map_location=device)

        return loaded

    def num_checkpoints(self, run_key: str):
        """Return the number of (uncompressed) checkpoint files"""
        dpath = self._get_path(CHECKPOINTS_DNAME, run_key)
        if not os.path.isdir(dpath):
            return 0
        else:
            fnames = os.listdir(dpath)
            expected_fnames = [
                CHECKPOINT_FNAME_FMT.format(i) for i in range(len(fnames))
            ]
            if sorted(fnames) != sorted(expected_fnames):
                raise ValueError("Found unexpected file names in checkpoints directory")
            return len(fnames)

    def compress_checkpoints(self, run_key: str):
        """Compress the checkpoints directory

        This is useful since otherwise the large number of checkpoint files can cause
        us to hit the file limit on Avon.
        """
        checkpoints_dpath = self._get_path(CHECKPOINTS_DNAME, run_key)
        compressed_fname = CHECKPOINTS_COMPRESSED_FNAME_FMT.format(run_key=run_key)
        compressed_fpath = self._get_path(CHECKPOINTS_DNAME, compressed_fname)
        with tarfile.open(compressed_fpath, "w:gz") as f:
            f.add(checkpoints_dpath, arcname="")
        shutil.rmtree(checkpoints_dpath)

    def uncompress_checkpoints(self, run_key: str):
        """Uncompress the checkpoints directory"""
        checkpoints_dpath = self._get_path(CHECKPOINTS_DNAME, run_key)
        compressed_fname = CHECKPOINTS_COMPRESSED_FNAME_FMT.format(run_key=run_key)
        compressed_fpath = self._get_path(CHECKPOINTS_DNAME, compressed_fname)

        if os.path.exists(checkpoints_dpath):
            raise FileExistsError(checkpoints_dpath)

        with tarfile.open(compressed_fpath, "r:gz") as f:
            f.extractall(checkpoints_dpath)

        os.remove(compressed_fpath)

    def delete_all_checkpoints(self):
        """Delete all checkpoints

        Useful for clearing up cluster storage.
        """
        dpath = self._get_path(CHECKPOINTS_DNAME)
        if os.path.isdir(dpath):
            shutil.rmtree(dpath)

    def save_metrics(self, run_key: str, metrics_df):
        dpath = self._get_path(METRICS_DNAME)
        _create_dir(dpath)
        fname = METRICS_FNAME_FMT.format(run_key=run_key)
        fpath = os.path.join(dpath, fname)
        metrics_df.to_parquet(fpath)

    def load_metrics(self, run_key: str):
        fname = METRICS_FNAME_FMT.format(run_key=run_key)
        fpath = self._get_path(METRICS_DNAME, fname)
        return pd.read_parquet(fpath)

    def save_timings(self, run_key: str, timings_history_df):
        dpath = self._get_path(TIMINGS_DNAME)
        _create_dir(dpath)
        fname = TIMINGS_FNAME_FMT.format(run_key=run_key)
        fpath = os.path.join(dpath, fname)
        timings_history_df.to_parquet(fpath)

    def load_timings(self, run_key: str):
        fname = TIMINGS_FNAME_FMT.format(run_key=run_key)
        fpath = self._get_path(TIMINGS_DNAME, fname)
        return pd.read_parquet(fpath)

    def _get_path(self, *subpath):
        return os.path.join(DATA_DIR, self.namespace, *subpath)


def _create_dir(dirpath, root_dir=DATA_DIR):
    if not os.path.exists(root_dir):
        # Sanity check in case the root_dir is wrong
        raise FileNotFoundError(f"Root data directory does not exist: {root_dir}")

    if not os.path.isdir(root_dir):
        raise NotADirectoryError(root_dir)

    if ".." in os.path.relpath(dirpath, root_dir):
        raise ValueError(
            f"Cannot create a directory outside the root data directory: {dirpath}"
        )

    os.makedirs(dirpath, exist_ok=True)


def _to_numpy(tensor: Union[Tensor, np.ndarray]):
    if isinstance(tensor, np.ndarray):
        return tensor
    else:
        return tensor.detach().cpu().numpy()
