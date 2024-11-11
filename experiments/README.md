### Usage:
Run visualize_model_preds (as a module) and give the args: experiments_dir,num_steps, num_model_samples, lookahead, model_subdir.
An example is given in this vs code launch config:

Valid configuration file for vs code:
{
    "name": "Viz Debug",
    "type": "debugpy",
    "request": "launch",
    "module": "mbrl.diagnostics.visualize_model_preds",
    "console": "integratedTerminal",
    "args": [
        "--experiments_dir=experiments",
        "--num_steps=100",
        "--num_model_samples=1000",
        "--lookahead=15",
        "--cfg_file=experiments/.hydra/Hopper_VBLL.yaml",
        // "--model_subdir=models/VBLL",
        // "--agent_dir=None",
        // "${command:pickArgs}"
    ],
}

There should be this project structure:
experiments
    .hydra
        configs
    diagnostics (will be created when running)
        different runs and vis
    models
        PE
            env_folders
                model.pth
        VBLL
            env_folders
                model.pth
    README (this)