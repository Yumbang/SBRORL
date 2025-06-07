import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict


class HybridActorCritic(TorchModelV2, nn.Module):
    """
    Outputs:
        2  means          (continuous)
        2  log-stds       (continuous)
        2  logits         (binary discrete)
      ------------------------------------------------
        6 neurons total   (MultiActionDistribution will split them)
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config: ModelConfigDict, name
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.backbone = nn.Sequential(
            nn.Linear(obs_space.shape[0], 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.mean_layer = nn.Linear(128, 2)
        self.log_std_param = nn.Parameter(torch.zeros(2))  # global log-std
        self.discrete_layer = nn.Linear(128, 2)  # logits
        self.value_layer = nn.Linear(128, 1)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = self.backbone(input_dict["obs"].float())
        cont_mean = self.mean_layer(x)
        cont_logstd = self.log_std_param.expand_as(cont_mean)
        disc_logits = self.discrete_layer(x)
        self._value_out = self.value_layer(x).squeeze(-1)
        # concat: [mean(2), logstd(2), logits(2)] => 6
        return torch.cat([cont_mean, cont_logstd, disc_logits], dim=1), state

    def value_function(self):
        return self._value_out
