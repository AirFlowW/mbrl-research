
def is_VBLL_dynamics_model(cfg, model_targets = ["mbrl.models.BasicEnsemble","mbrl.models.VBLLMLP"]):
    if cfg.dynamics_model.get("_target_","") in model_targets \
            and cfg.dynamics_model.member_cfg.get("_target_","") in model_targets:
        return True
    return False