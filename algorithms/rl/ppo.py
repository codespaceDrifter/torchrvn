import torch
import torch.nn as nn

class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=64):
        super().__init__()
        
        # Shared base network (can remove if you want separate nets)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
        )
        
        # Policy (actor) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()  # For continuous actions in [-1, 1]
        )

        # Value (critic) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, obs):
        x = self.encoder(obs)
        action_logits = self.actor(x)   # shape: [batch, action_dim]
        state_value = self.critic(x)    # shape: [batch, 1]
        return action_logits, state_value.squeeze(-1)