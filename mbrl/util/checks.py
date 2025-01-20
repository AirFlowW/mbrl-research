
def is_VBLL_dynamics_model(cfg, model_targets = ["mbrl.models.BasicEnsemble","mbrl.models.VBLLMLP", "mbrl.models.VBLLEnsemble"]):
    if cfg.dynamics_model.get("_target_","") in model_targets or (cfg.dynamics_model.get("_target_","") in model_targets \
            and cfg.dynamics_model.member_cfg.get("_target_","") in model_targets):
        return True
    return False

def is_thompson_sampling_active(cfg, model_targets = ["mbrl.models.VBLLEnsemble"]):
    if cfg.dynamics_model.get("_target_","") in model_targets:
        if cfg.dynamics_model.get("no_thompson_heads", 0) > 0:
            return True
    return False