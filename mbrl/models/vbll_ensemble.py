from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import hydra
import omegaconf
import torch
from torch import nn
from mbrl.models.basic_ensemble import BasicEnsemble
from .util import EnsembleLinearLayer

class Mode(Enum):
    NORMAL = 0
    THOMPSON_FORWARD = 1

class VBLLEnsemble(BasicEnsemble):
    def __init__(
        self,
        ensemble_size: int,
        device: Union[str, torch.device],
        member_cfg: omegaconf.DictConfig,
        propagation_method: Optional[str] = None,
        clip_val: Optional[float] = None,
        no_thompson_heads: int = 25
    ):
        super().__init__(
            ensemble_size=ensemble_size,
            device=device,
            member_cfg=member_cfg,
            propagation_method=propagation_method,
            clip_val=clip_val,
        )
        self.thompson_heads = None
        self.mlp_feature_extractors = None
        self.activation_fn_cfg =  member_cfg.get("activation_fn_cfg", None)
        self.no_thompson_heads = no_thompson_heads
        self.mode = Mode.NORMAL

    def _create_activation_fn(self):
        if self.activation_fn_cfg is None:
            return nn.ELU()
        else:
            cfg = omegaconf.OmegaConf.create(self.activation_fn_cfg)
            activation_func = hydra.utils.instantiate(cfg)
            return activation_func

    def reset_thompson_mlps(self):
        self.mlp_feature_extractors = None

    @property
    def ensemble_size(self) -> int:
        return len(self)
    
    def set_mode(self, mode: Mode):
        self.mode = mode

    def set_mode_from_str(self, mode: str):
        if mode == "normal":
            self.set_mode(Mode.NORMAL)
        elif mode == "thompson_forward":
            self.set_mode(Mode.THOMPSON_FORWARD)
        else:
            raise ValueError(f"Mode {mode} not recognized.")
        
    def set_thompson_sampling_active(self):
        self.deterministic = True
        self.mode = Mode.THOMPSON_FORWARD

    def set_thompson_sampling_inactive(self):
        self.deterministic = False
        self.mode = Mode.NORMAL

    def _extract_linear_layers(self):
        """Extracts the linear layers from the ensemble members.
        Args:
            
        Returns:
            (list of list of nn.Linear): list of linear layers for each ensemble member.
            Dim: ``E x L``, where ``E`` is the ensemble size, and ``L`` is the number of layers per member.
        """
        linear_layers = []
        for member in self.members:
            member_layers = []
            for layer in member.feature_extractor:
                if isinstance(layer, nn.Linear):
                    member_layers.append(layer)
                elif isinstance(layer, nn.Sequential):
                    for sub_layer in layer:
                        if isinstance(sub_layer, nn.Linear):
                            member_layers.append(sub_layer)
            linear_layers.append(member_layers)
        return linear_layers
    
    def _extract_weights_and_biases(self):
        """Extracts the weights and biases from the ensemble members.
        Args:

        Returns:
            (tuple of list of list of weights/biases): list of tuples for each ensemble member.
            Dim: ``2 x E x L x w/b``, where ``E`` is the ensemble size, ``L`` is the number of layers per member,
            this for the weight as well as biases of the layers.
        """
        linear_layers = self._extract_linear_layers()
        weights_of_layers = []
        biases_of_layers = []
        for member_layers in linear_layers:
            member_weights = []
            member_biases = []
            for layer in member_layers:
                member_weights.append(layer.weight)
                member_biases.append(layer.bias)
            weights_of_layers.append(member_weights)
            biases_of_layers.append(member_biases)

        return weights_of_layers, biases_of_layers
    
    def _create_ensemble_mlp_feature_extractors(self):
        extracted_linear_layer_weights, extracted_linear_layer_biases = self._extract_weights_and_biases()
        linear_layers = []
        
        for no_layer in range(len(extracted_linear_layer_weights[0])):
            weight_list = []
            bias_list = []
            for weights in extracted_linear_layer_weights:
                weight_list.append(weights[no_layer])
            
            for biases in extracted_linear_layer_biases:
                bias_list.append(biases[no_layer])

            weight_of_complete_layer = torch.stack(weight_list, dim=0).permute(0, 2, 1)
            bias_of_complete_layer = torch.stack(bias_list, dim=0)
            bias_of_complete_layer = bias_of_complete_layer.unsqueeze(1)
            linear_layers.append(EnsembleLinearLayer(self.ensemble_size, weight_of_complete_layer.shape[1],
                            weight_of_complete_layer.shape[2], weights=weight_of_complete_layer, biases=bias_of_complete_layer))

        feature_extractor = nn.Sequential(*[nn.Sequential(layer, self._create_activation_fn()) for layer in linear_layers])

        return feature_extractor
    
    def _create_thompson_heads(self, no_thompson_heads=10):
        thompson_head_weights = []
        for member in self.members:
            if no_thompson_heads == 1:
                thompson_head_weights.append(member.out_layer.create_mean_thompson_head())
            else:
                thompson_head_weights.append(member.out_layer.create_thompson_heads(no_thompson_heads))
        
        flattened_tensors = [torch.transpose(tensor, 0, 1) for sublist in thompson_head_weights for tensor in sublist]
        thompson_heads = torch.stack(flattened_tensors, dim=0)
        thompson_heads = EnsembleLinearLayer(self.ensemble_size*no_thompson_heads, thompson_heads.shape[1], thompson_heads.shape[2], bias=False, weights=thompson_heads)
        return thompson_heads

    def _fast_thompson_forward(  # type: ignore
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the output of the ensemble.

        The forward pass for the ensemble computes forward passes of its models, and
        aggregates the prediction in different ways, according to the desired
        epistemic uncertainty ``propagation`` method.

        If no propagation is desired (i.e., ``self.propagation_method is None``),
        then the outputs of the model are stacked into single tensors
        (one for mean, one for logvar). The shape
        of each output tensor will then be ``E x B x D``, where ``E``, ``B`` and ``D``
        represent ensemble size, batch size, and output dimension, respectively.


        For all other propagation options, the output is of size ``B x D``.

        Args:
            x (tensor): the input to the models (shape ``B x D``). The input will be
                evaluated over all models, then aggregated according to
                ``propagation``, as explained above.
            rng (torch.Generator, optional): random number generator to use for
                "random_model" propagation.
            propagation_indices (tensor, optional): propagation indices to use for
                "fixed_model" propagation method.

        Returns:
            (tuple of two tensors): one for aggregated mean predictions, and one for
            aggregated log variance prediction (or ``None`` if the ensemble members
            don't predict variance).

        """
        if self.mlp_feature_extractors is None:
            self.mlp_feature_extractors = self._create_ensemble_mlp_feature_extractors()
            self.thompson_heads = self._create_thompson_heads(no_thompson_heads=self.no_thompson_heads)

        if self.propagation_method == "random_model":

            x = x.unsqueeze(0)
            model_indices_features = torch.randperm(x.shape[1], device=self.device)
            out = self._forward_from_indices_ensemble_linear_layer(
                x, model_indices_features, self.mlp_feature_extractors, False
            )
            out = out.unsqueeze(0)
            model_indices = torch.arange(out.shape[1], device=self.device)
            mean = self._forward_from_indices_ensemble_linear_layer(
                out, model_indices, self.thompson_heads, True
            )
            mean = mean.view(x.shape[1], -1)
            # invert shuffle
            mean[model_indices_features] = mean.clone()
            return mean, None

        elif self.propagation_method == "all_thompson_heads":
            x = x.unsqueeze(0)
            model_indices_features = torch.arange(x.shape[1], device=self.device)
            out = self._forward_from_indices_ensemble_linear_layer(
                x, model_indices_features, self.mlp_feature_extractors, False
            )
            out = out.unsqueeze(0)
            out = out.repeat(self.no_thompson_heads, 1, 1)
            mean = self.thompson_heads(out)
            return mean, None

        else:
            raise NotImplementedError(
                f"Propagation method {self.propagation_method} not implemented in Thompson forward."
            )

    def _forward_from_indices_ensemble_linear_layer(
        self, x: torch.Tensor, model_shuffle_indices: torch.Tensor, model: Union[EnsembleLinearLayer,nn.Sequential],
        invert_shuffle: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for a model being an Ensemble Linear layer or nn.Sequential with Ensemble Linear Layers."""

        _, batch_size, _ = x.shape

        ensemble_layer = model
        while not isinstance(ensemble_layer, EnsembleLinearLayer):
            ensemble_layer = ensemble_layer[0]
        num_models = ensemble_layer.num_members

        # extend x to be a multiple of num models
        not_aligns = bool(batch_size % num_models)
        num_fill_tensors = num_models - (batch_size % num_models) if not_aligns else 0
        reference_tensor = x[:,0,:].unsqueeze(dim=1)
        zero_tensors = torch.cat([torch.zeros_like(reference_tensor, device=reference_tensor.device) for _ in range(num_fill_tensors)], dim=1)

        x_extended = torch.cat([x, zero_tensors], dim=1)
        start_value_for_extending = len(model_shuffle_indices)
        new_values = torch.arange(start_value_for_extending, start_value_for_extending + num_fill_tensors, device=reference_tensor.device)
        model_shuffle_indices_extended = torch.cat([model_shuffle_indices,new_values])

        shuffled_x = x_extended[:, model_shuffle_indices_extended, ...].view(
            num_models, (batch_size // num_models) + not_aligns, -1
        )

        out = model(shuffled_x)
        out = out.view(batch_size + num_fill_tensors, -1)
        out = out[:-num_fill_tensors,:]

        # note that out is shuffled and in view of EnsembleLinearLayer
        if invert_shuffle:
            out[model_shuffle_indices] = out.clone()  # invert the shuffle

        return out

    def forward(
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == Mode.THOMPSON_FORWARD:
            return self._fast_thompson_forward(x, rng, propagation_indices)
        elif self.mode == Mode.NORMAL:
            return super().forward(x, rng, propagation_indices)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented.")

    def update(
        self,
        model_in: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        self.reset_thompson_mlps()
        return super().update(model_in, optimizer, target)

    def eval_score(  # type: ignore
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None, uncertainty = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the average score over all members given input/target.

        The input and target tensors are replicated once for each model in the ensemble.

        Args:
            model_in (tensor): the inputs to the models.
            target (tensor): the expected output for the given inputs.

        Returns:
            (tensor): the average score over all models.
        """
        if not uncertainty:
            return super().eval_score(model_in, target, uncertainty)
        
        loss, meta = super().eval_score(model_in, target, uncertainty)

        # extract the uncertainty of each member and put them together
        uncertainty = [meta[key]["uncertainty"] for key in meta.keys()]
        uncertainty = {key: torch.stack([item[key] for item in uncertainty]) for key in uncertainty[0].keys()}

        # ensemble var of a combined mixture of gaussians
        var = uncertainty["mean_variance"]
        ensemble_mean = torch.mean(uncertainty["predicted_mean"], dim=0)
        inner = var + torch.pow(uncertainty["predicted_mean"], 2)
        uncertainty["ensemble_uncertainty"] = torch.mean(inner, dim=0) - torch.pow(ensemble_mean, 2)

        for key, item in uncertainty.items():
            if not (item.shape[0] == self.num_members and item.dim() > 1):
                continue
            if key.startswith("max"):
                uncertainty[key] = torch.max(item, dim=0)[0]
            elif key.startswith("min"):
                uncertainty[key] = torch.min(item, dim=0)[0]
            else:
                uncertainty[key] = torch.mean(item, dim=0)

        for _, item in meta.items():
            del item["uncertainty"]
        meta["uncertainty"] = uncertainty

        return loss, meta