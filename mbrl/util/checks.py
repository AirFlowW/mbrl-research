import omegaconf
from mbrl.models.basic_ensemble import BasicEnsemble
from mbrl.models.one_dim_tr_model import OneDTransitionRewardModel
from mbrl.models.vbll_ensemble import VBLLEnsemble
from mbrl.models.vbll_mlp import VBLLMLP

def is_VBLL_dynamics_model(cfg, model_targets = ["mbrl.models.BasicEnsemble","mbrl.models.VBLLMLP", "mbrl.models.VBLLEnsemble"]):
    if isinstance(cfg, OneDTransitionRewardModel):
        cfg = cfg.model
    if (isinstance(cfg, VBLLEnsemble) or (isinstance(cfg, BasicEnsemble) and isinstance(cfg.members[0], VBLLMLP))): 
        return True
    is_cfg = isinstance(cfg, omegaconf.DictConfig) or isinstance(cfg, omegaconf.OmegaConf)
    if  is_cfg \
        and (cfg.dynamics_model.get("_target_","") in model_targets or (cfg.dynamics_model.get("_target_","") in model_targets \
        and cfg.dynamics_model.member_cfg.get("_target_","") in model_targets)):
            return True
    return False

def is_thompson_sampling_active(cfg, model_targets = ["mbrl.models.VBLLEnsemble"]):
    if cfg.dynamics_model.get("_target_","") in model_targets:
        if cfg.dynamics_model.get("no_thompson_heads", 0) > 0:
            return True
    return False