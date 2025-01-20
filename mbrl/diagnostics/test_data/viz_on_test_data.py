# Description: This script compares the runtime of different models on different dataset sizes.
import hydra
import omegaconf
import torch
from torch.utils.data import DataLoader
import time
from torch import optim

from mbrl.diagnostics.test_data.dataset import SimpleFnDataset
from mbrl.diagnostics.test_data.utils.seed import set_seed
from mbrl.diagnostics.test_data.utils.general import path_to_save_plots

import os

from mbrl.diagnostics.test_data.viz.viz_thompson_ensemble import viz_ensemble
from mbrl.diagnostics.test_data.viz.viz_model_with_mean_var import viz_model
from mbrl.util.checks import is_VBLL_dynamics_model

class train_cfg_vbll:
        NUM_EPOCHS = 1000
        BATCH_SIZE = 32
        LR = 1e-3
        WD = 1e-4
        OPT = torch.optim.AdamW
        CLIP_VAL = 1
        VAL_FREQ = 100

@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    set_seed(cfg.seed)
    dataset_size = cfg.get("dataset_size", 512)
    dataset = SimpleFnDataset(num_samples=dataset_size)

    if not os.path.exists(path_to_save_plots):
        os.makedirs(path_to_save_plots)

    train_cfg = train_cfg_vbll()
    # ---- end init

    model = hydra.utils.instantiate(cfg.dynamics_model)
    if is_VBLL_dynamics_model:
        for member in model.members:
            member.update_regularization_weight_from_dataset_length(dataset_size)

    param_list = [
        # All parameters except the VBLL.Regression layer get the specified weight decay
        {'params': [param for name, param in model.named_parameters() if 'out_layer' not in name], 'weight_decay': train_cfg.WD},
        
        # The VBLL.Regression layer (out_layer) has weight decay set to zero
        {'params': [param for name, param in model.named_parameters() if 'out_layer' in name], 'weight_decay': 0.},
    ]
    optimizer = optim.Adam(
        param_list,
        lr=train_cfg.LR,
        weight_decay=train_cfg.WD
    )
    dataloader = DataLoader(dataset, batch_size=train_cfg.BATCH_SIZE, shuffle=True)

    # main train, measure runtime and viz section
    model_name = cfg.dynamics_model._target_
    print(f"Training {model_name} model")

    # Train model
    start = time.perf_counter()
    for epoch in range(train_cfg.NUM_EPOCHS + 1):
        model.train()

        for _, (x, y) in enumerate(dataloader):
            x_sequence = [x.clone() for _ in range(len(model))]
            y_sequence = [y.clone() for _ in range(len(model))]
            model.update(x_sequence, optimizer, target = y_sequence)

    end = time.perf_counter()
    print(f"Training time: {end-start:.2f} s")
    # End training
        
    
    viz_model(model, dataloader, title=model_name, save_path=path_to_save_plots + model_name + '.png')
    if cfg.get("viz_members", False):
        viz_ensemble(model, dataloader,title=model_name + ' members', save_path=path_to_save_plots + model_name + ' members.png')

if __name__ == "__main__":
    run()