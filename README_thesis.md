# Bachelor Thesis
This repository was extended as part of my bachelor thesis at the DSME (RWTH-Aachen), titled 'Exploring Last-Layer Surrogate Models for Model-based Reinforcement Learning'.

## Extended Functionality
The primary extensions introduced in this work are listed below. It should be noted that additional modifications were made to various utility and helper functions, as well as core files, which are not explicitly listed here.

- Adaptation of the PETS algorithm to support VBLL models, enabling techniques such as Thompson sampling and recursive updates: [link](mbrl/algorithms/pets.py)
- Adaptation of the MBPO algorithm to support VBLL models, enabling techniques such as Thompson sampling and recursive updates: [link](mbrl/algorithms/mbpo.py)

- Integration of a VBLL model into the library: [link](mbrl/models/vbll_mlp.py)
- Integration of an ensemble of VBLL models which allows optimized Thompson sampling: [link](mbrl/models/vbll_ensemble.py)

- Implementation of a Weights & Biases logger, more advanced logging of runtime and more: [link](mbrl/util/wandb_logger.py)
- Development of advanced visualization tools for rollouts: [link](mbrl/diagnostics/vis_rollout.py)
- Creation of a generalized noise environment that augments any given environment with measurement noise: [link](mbrl/env/env_with_measurement_noise.py)

### Usage
An example configuration for using the VBLL ensemble is available [here](mbrl/examples/conf/dynamics_model/vbll_integrated_ensemble.yaml).

An example configuration for the overrides (environment specific) file is available [here](mbrl/examples/conf/overrides/vbllts_cartpole.yaml).