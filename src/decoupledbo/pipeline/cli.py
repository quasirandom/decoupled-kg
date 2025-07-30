import argparse
from functools import partial
from typing import Type

import yaml

from decoupledbo.pipeline.constants import SMOKE_TEST
from decoupledbo.pipeline.nodes.bo_loop import preset_optimisation_strategy_factories


def parse_commandline_arguments_and_read_config():
    commandline_args = parse_commandline_arguments_raw()
    namespace = extract_namespace(commandline_args)
    config = load_config(commandline_args.config)

    # Merge config with commandline args
    # While a less custom merging process would be nice, it would take too long given
    # the current time constraints!

    config["namespace"] = namespace

    config["model"]["fit_hyperparams"] = commandline_args.fit_hyperparams
    for output_config in config["model"]["outputs"]:
        if commandline_args.fix_zero_noise is not None:
            output_config["fix_zero_noise"] = commandline_args.fix_zero_noise

        if commandline_args.standardize_outputs is not None:
            output_config["standardize_output"] = commandline_args.standardize_outputs
        elif commandline_args.fit_hyperparams in ["once", "always"]:
            output_config["standardize_output"] = True
        elif commandline_args.fit_hyperparams == "never":
            output_config["standardize_output"] = False
        else:
            raise ValueError(
                f"Unsupported value for --fit-hyperparams. "
                f"Got {commandline_args.fit_hyperparams!r}."
            )

    if commandline_args.test_problem:
        new_problem_config = convert_test_problem_arg_to_config(
            commandline_args.test_problem,
            commandline_args.regenerate_gp_test_problem,
        )
        # Merge the config in, only overwriting the new fields ("type" & "args")
        config["problem"] = {**config["problem"], **new_problem_config}

    # Delete args which are copied to config so that there is just one source of truth
    # This has the unfortunate side effect that the commandline_args.json I save for
    # debugging does not contain the raw commandline arguments.
    del (
        commandline_args.fit_hyperparams,
        commandline_args.fix_zero_noise,
        commandline_args.standardize_outputs,
        commandline_args.test_problem,
        commandline_args.regenerate_gp_test_problem,
    )

    return config, commandline_args


def parse_commandline_arguments_raw():
    parser = argparse.ArgumentParser(description="Run the main pipeline")

    parser.add_argument("--config", required=True, help="An experiment config file")

    if not SMOKE_TEST:
        parser.add_argument(
            "--namespace",
            help=(
                "The namespace to use when saving data. This is required when "
                "SMOKE_TEST is not enabled via the environment variable."
            ),
            required=True,
        )
    else:
        parser.add_argument(
            "--namespace",
            help=(
                "The namespace to use when saving data. When SMOKE_TEST is enabled via "
                "the environment variable, the namespace will be prepended by "
                "'smoke-test-'."
            ),
        )

    parser.add_argument("--test-problem", default=None)
    parser.add_argument(
        "--fit-hyperparams",
        choices=["never", "once", "always"],
        required=True,
        help=(
            "If 'never' then hyper-parameters will be be fixed to the values used to "
            "generate the GP test problem (not available for other test problems). "
            "The observation noise is included in the definition of hyper-parameters "
            "here, although that will be overrided using the --fix-zero-noise flag."
        ),
    )
    parser.add_argument(
        "--fix-zero-noise",
        action=argparse.BooleanOptionalAction,
        help=(
            "If set, then noise will be fixed at zero in the surrogate model "
            "(regardless of the value of --fit-hyperparams)."
        ),
    )
    parser.add_argument(
        "--standardize-outputs",
        action=argparse.BooleanOptionalAction,
        help=(
            "Determines whether the observations will be standardized to have zero "
            "mean and unit variance before the GP surrogate model is fitted. By "
            "default, observations will be standardized if --fit-hyperparams is 'once' "
            "or 'always' and will not be standardized if --fit-hyperparams is 'neve'."
        ),
    )

    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="The global random seed for the pipeline run (optional).",
    )
    parser.add_argument(
        "--acq-strategy",
        choices=list(preset_optimisation_strategy_factories),
        default="discrete_kg",
    )
    parser.add_argument(
        "--regenerate-gp-test-problem",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If set and if using a GP test problem, then regenerate that GP test "
            "problem."
        ),
    )
    parser.add_argument(
        "--scalarisations-per-step",
        type=int,
        default=None,
        help=(
            "The number of scalarisations to use per step when estimating the "
            "expectation over scalarisations in the acquisition function. If left "
            "unset, then use random scalarisations taken from a Sobol' sample (i.e. in "
            "this case, scalarisations will not be independent between iterations)."
        ),
    )

    return parser.parse_args()


def _list_of(elt_type: Type):
    return partial(_split_list, elt_type=elt_type)


def _split_list(arg: str, elt_type: Type):
    if not isinstance(arg, str):
        raise TypeError(f"Expected arg to be a str. Got {type(arg)=}")

    return [elt_type(i) for i in arg.split(",")]


def extract_namespace(args):
    if SMOKE_TEST:
        if args.namespace:
            return "smoke-test-" + args.namespace
        else:
            return "smoke-test"
    else:
        return args.namespace


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def convert_test_problem_arg_to_config(test_problem_arg, regenerate_gp_test_problem):
    """Map the commandline argument for a test problem to corresponding config dict"""
    if test_problem_arg.startswith("gp-sample:"):
        problem_name = test_problem_arg.split(":", maxsplit=1)[1]
        return {
            "type": "gp-sample",
            "args": {
                "id": problem_name,
                "regenerate": regenerate_gp_test_problem,
            },
        }
    else:
        raise ValueError(f"Unrecognised '--test-problem': {test_problem_arg}")
