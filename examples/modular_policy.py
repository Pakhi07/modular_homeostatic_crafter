import torch
import torch.nn as nn
from torch.distributions import Categorical

from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from typing import Callable, Dict, List, Tuple

class ModularMultiInputActorCriticPolicy(MultiInputActorCriticPolicy):
    """
    A modular Actor-Critic Policy for environments with dictionary observations.
    
    This version correctly integrates with the SB3 policy architecture.
    It lets the parent class build the standard feature extractor and MLP extractor,
    then attaches the custom modular heads for action selection.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule: Callable[[float], float],
        num_selves: int = 4,
        *args,
        **kwargs,
    ):
        self.num_selves = num_selves
        # First, call the parent constructor. This will build the default
        # features_extractor, mlp_extractor, action_net, and value_net.
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
        
        # Now, we override the parts we need with our custom modular heads.
        # The `mlp_extractor` created by the parent class has a `policy_net` and a `value_net`.
        # We will use the `policy_net` as the shared body for our manager and selves.
        
        # The latent_dim_pi is the output size of the mlp_extractor's policy network.
        latent_dim_pi = self.mlp_extractor.latent_dim_pi 

        # --- Define the Modular Heads ---
        # We replace the default single `action_net` with our modular structure.
        self.manager_net = nn.Linear(latent_dim_pi, self.num_selves)
        self.selves_nets = nn.ModuleList([
            nn.Linear(latent_dim_pi, self.action_space.n) for _ in range(self.num_selves)
        ])

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the entire modular network.
        """
        # 1. Feature Extraction and Shared MLP
        # The standard mlp_extractor returns separate latent features for policy and value.
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # 2. Value Estimation (using the standard value head)
        value = self.value_net(latent_vf)

        # 3. Modular Action Logic (using the policy latent features)
        manager_logits = self.manager_net(latent_pi)
        manager_dist = Categorical(logits=manager_logits)

        all_selves_logits = torch.stack([self_net(latent_pi) for self_net in self.selves_nets], dim=1)
        
        if deterministic:
            chosen_self_idx = torch.argmax(manager_dist.probs, dim=1)
            batch_indices = torch.arange(latent_pi.size(0), device=self.device)
            best_selves_logits = all_selves_logits[batch_indices, chosen_self_idx]
            
            action_distribution = self.action_dist.proba_distribution(action_logits=best_selves_logits)
            action = action_distribution.get_actions(deterministic=True)
            log_prob = action_distribution.log_prob(action)
        else:
            selves_probs = torch.softmax(all_selves_logits, dim=-1)
            manager_probs = manager_dist.probs
            mixed_action_probs = torch.sum(manager_probs.unsqueeze(-1) * selves_probs, dim=1)
            final_action_dist = Categorical(probs=mixed_action_probs)
            action = final_action_dist.sample()
            log_prob = final_action_dist.log_prob(action)
            
        return action, value, log_prob

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for the PPO update.
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        value = self.value_net(latent_vf)

        manager_logits = self.manager_net(latent_pi)
        manager_dist = Categorical(logits=manager_logits)

        all_selves_logits = torch.stack([self_net(latent_pi) for self_net in self.selves_nets], dim=1)
        selves_probs = torch.softmax(all_selves_logits, dim=-1)
        manager_probs = manager_dist.probs
        
        mixed_action_probs = torch.sum(manager_probs.unsqueeze(-1) * selves_probs, dim=1)
        final_action_dist = Categorical(probs=mixed_action_probs)
        
        log_prob = final_action_dist.log_prob(actions)
        entropy = final_action_dist.entropy()
        
        return value, log_prob, entropy

    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict the value of a state.
        """
        features = self.extract_features(obs)
        # We only need the value latent features
        _ , latent_vf = self.mlp_extractor(features)
        return self.value_net(latent_vf)