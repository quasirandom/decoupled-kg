import logging
from typing import Callable

from decoupledbo.modules.performance_after_scalarisation import (
    estimate_best_possible_expected_performance_after_scalarisation as _estimate_best_possible_expected_performance_after_scalarisation,
)
from decoupledbo.modules.utils import log_node
from decoupledbo.pipeline.constants import TKWARGS
from decoupledbo.pipeline.data_catalog import DataCatalog

logger = logging.getLogger(__name__)


@log_node
def estimate_best_possible_expected_performance_after_scalarisation(
    scalarise: Callable, catalog: DataCatalog
):
    """Calculate the expectation over scalarisations of best scalarised objective"""
    _, pfront = catalog.load_true_pareto()
    expected_best = _estimate_best_possible_expected_performance_after_scalarisation(
        pfront, scalarise, tkwargs=TKWARGS
    )
    catalog.save_problem_max_possible_expected_scalarisation(expected_best)
