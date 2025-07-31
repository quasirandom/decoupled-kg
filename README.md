# Knowledge Gradient for Multi-Objective Bayesian Optimization with Decoupled Evaluations
[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Unit tests](https://github.com/JackBuck/decoupled-kg/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/JackBuck/decoupled-kg/actions/workflows/test.yaml)

This repository contains the code to run the experiments in the paper (Buckingham et al., 2025):
> Buckingham, J. M., Rojas Gonzalez, S., & Branke, J. (2025). Knowledge Gradient for Multi-Objective Bayesian Optimization with Decoupled Evaluations. In _International Conference on Evolutionary Multi-objective Optimization_ (pp. 117-132). [DOI: 10.1007/978-981-96-3538-2_9](https://www.doi.org/10.1007/978-981-96-3538-2_9) [[arXiv](https://arxiv.org/abs/2302.01310)]

The paper introduces an acquisition function for finding the Pareto front of a multi-objective optimization problem when the objectives are given by separate, time-consuming experiments (the so-called 'decoupled' setting).

![Example of convergence of the Pareto front and Pareto set](/assets/pareto-fronts-lengthscales-run1.png)

## Repository structure
The repository consists of:
  - a BoTorch-compatible implementation of the acquisition functions proposed;
  - a data pipeline to run the experiments (these were run on a SLURM cluster for this paper);
  - jupyter notebooks to generate the plots in the paper.

### Notebooks
Before the pipeline can be run, the notebook `gp-test-problem-generation.ipynb` in the `notebooks/` directory must be used to generate all instances of the GP test problems. These are saved in the `data/shared/gp-problem/` directory. The values used to run the experiments in the paper have been committed to this repository. Suitable hyperparameters for the two families of GP test problem were chosen using the notebook `gp-test-problem.ipynb`.

Once the pipeline has been used to generate results of all experiments (including repeats) and results have been copied back from the cluster, the notebooks in the `notebooks/` directory can be used to generate the figures from the paper:
  - `process-results.ipynb` processes experiment results and generate figures 1-4 in the paper;
  - `gp-test-problem-exhibit.ipynb` generates figures 5 and 6 (in the supplementary material);
  - `lengthscale-priors.ipynb` generates figures 7 and 8 (in the supplementary material).

### Pipeline
The project is split into three main packages:
  - `modules` contains reusable code for running the experiments;
  - `pipeline` contains wrappers around code in `modules` which are stateful (they load and save data via the `DataCatalog`);
  - `postprocessing` contains functions used by the `process-results.ipynb` notebook to process experiment results and generate plots.

The entrypoint for the experiment pipeline is `src/decoupledbo/pipeline/main.py`.

The implementation of the multi-objective knowledge gradient acquisition function (MOKG) can be found in the `DiscreteKnowledgeGradient` class in `src/decoupledbo/modules/knowledge_gradient.py`. Costs are taken into account in the accompanying "acquisition strategy" `DiscreteKgOptimisationSpec` in `acquisition_optimisation_strategy.py` which optimises the acquisition function in order to generate a recommendation. Strategies for HVKG and JES-LB are also found in this file.

## Installation
This repository is designed to be run out-of-the-box and is not formally packaged.
This section of the readme handles installation of dependencies.

Installation via anaconda is very slow, so I recommend installing python and PyGMO via conda, and using pip for the remaining libraries.
```bash
conda create --name decoupled-kg --channel conda-forge python=3.9.5
conda activate decoupled-kg
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install pygmo=2.18.0
pip install -r requirements.txt  # Or requirements-full.txt (see below)
```

##### Requirements files
Minimal requirements are specified by `requirements.txt` but `requirements-full.txt` contains the full set of package versions used to run the experiments for the paper. Note that not all the packages listed in `requirements-full.txt` are used by the project, since some are simply present as standard on the cluster where experiments were run.

## Running the pipeline
To run a BO experiment, use the following command. The test problem and other flags can be changed according to the experiment you wish to run.
```bash
PYTHONPATH="src/:$PYTHONPATH" python src/decoupledbo/pipeline/main.py \
  --config=config/experiment-lengthscales.yaml \  # Specifies the config for the GP model
  --test-problem=gp-sample:lengthscales/0 \  # Override the GP test problem in the config file
  --fit-hyperparams=always \  # Refit model hyperparameters after every sample
  --acq-strategy=discrete_kg \  # Use the discrete KG acquisition function
  --scalarisations-per-step=16 \  # Approximate the expectation over scalarisations with a qMC sample of 16 points
  --namespace=gp-lengthscales-discretekg-refitalways/0  # Directory under 'data/' in which to save the data
```
Help for the CLI can be found by running
```bash
PYTHONPATH="src/:$PYTHONPATH" python src/decoupledbo/pipeline/main.py --help
```

A smoke test version of the pipeline with very small parameter values can be enabled by exporting the environment variable `SMOKE_TEST=1`. This can be used to quickly test the pipeline but with very poor numerical results.

Results of the pipeline are saved in a namespace under the `data/` top-level directory. Checkpoints are compressed at the end of the pipeline to avoid exceeding the file limit (inode limit) on many systems.

### Experiments in the paper
For the experiments in the paper, the following CLI arguments were passed:
  - Values which depend on the test problem family:
    - The experiments with different length scales use `--config=config/experiment-lengthscales.yaml` and `--test-problem=gp-sample:lengthscales/#`
    - The experiments with and without observation noise use `-config=config/experiment-observationnoise.yaml` and `--test-problem=gp-sample:observationnoise/#`
  - Values which depend on the algorithm:
    - The C-MOKG algorithm uses `--acq-strategy=discrete_kg` and `--scalarisations-per-step=16`
    - The variant of the C-MOKG algorithm with random scalarisations is achieved by omitting the `--scalarisations-per-step` flag completely
    - The HVKG algorithm uses `--acq-strategy=hvkg`
    - The JES-LB algorithm uses `--acq-strategy=jes_lb`
  - All experiments use `--fit-hyperparams=always`
  - The seeds passed for the 100 problem instances are 1111 to 1210 and are passed using `--seed` (the problems themselves are pregenerated by the `gp-test-problem-generation.ipynb` notebook, but this seed controls the initial design, for example)

## Tests
The repository comes with a small number of unit tests, specifically on the implementation of the knowledge gradient acquisition function. To run the tests, change directory to `src/` then run:
```bash
python -m pytest ../tests
```

## BibTeX citation
If you would like to cite the work (which would be much appreciated!) then you may find the following bibtex entry useful.

```
@inproceedings{buckingham2025decoupledkg,
  author={Buckingham, Jack M. and {Rojas Gonzalez}, Sebastian and Branke, Juergen},
  editor={Singh, Hemant and Ray, Tapabrata and Knowles, Joshua and Li, Xiaodong and Branke, Juergen and Wang, Bing and Oyama, Akira},
  title={Knowledge Gradient for Multi-objective {B}ayesian Optimization with Decoupled Evaluations},
  booktitle={Evolutionary Multi-Criterion Optimization},
  series={Lecture Notes in Computer Science},
  volume={15513},
  year={2025},
  publisher={Springer Nature Singapore},
  address={Singapore},
  pages={117--132},
  isbn={978-981-96-3538-2},
  doi={10.1007/978-981-96-3538-2_9},
}
```
