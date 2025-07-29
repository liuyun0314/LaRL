import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class Reward_Model(nn.Module):
    def __init__(self, input_dim, output_dim=1, n_layers=3, hidden_dim=256, device='cuda'):
        super().__init__()
        if n_layers == 1:
            self.model = nn.Linear(input_dim, output_dim)
        else:
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(n_layers - 2)],
                nn.Linear(hidden_dim, output_dim)
            )
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.model(x)

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)


class RRDRewardDecomposer(object):
    def __init__(self, args):
        super(RRDRewardDecomposer, self).__init__()
        self.args = args
        self.K = self.args.rrd_k

        self.reward_model = Reward_Model(input_dim=self.args.share_obs_dim * 2 + self.args.num_machines)
        self.reward_model.to('cuda')

        self.optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=3e-4)
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, states, actions, next_states):
        states = torch.cat([states, actions, states - next_states], dim=-1)

        rewards = self.reward_model(states)  # bs,t,-1 -> bs,t,1

        return rewards

    def update(self, states, actions, next_states, episode_return, episode_length):
        self.optimizer.zero_grad()

        states = torch.tensor(states.cpu().numpy(), device='cuda')
        actions = torch.tensor(actions.cpu().numpy(), device='cuda')
        next_states = torch.tensor(next_states.cpu().numpy(), device='cuda')
        rewards = self.forward(states, actions, next_states)    # predicte rewards for each trajectory
        sampled_rewards = []
        var_coef = []
        all_episode_lengths = [int(episode_length[i].item()) for i in range(episode_length.shape[0])]
        for i in range(episode_length.shape[0]):    # sample k rewards and calculate variance coefficient
            local_episode_length = int(episode_length[i].item())
            sampled_steps = np.random.choice(local_episode_length, self.K, replace=self.K > local_episode_length)
            begin_index = sum(all_episode_lengths[:i])
            end_index = begin_index + int(all_episode_lengths[i])
            batch_rewards = rewards[begin_index:end_index, :]
            sampled_rewards.append(batch_rewards[sampled_steps,:])
            var_coef.append(1.0 - self.K / local_episode_length)

        # Combine the rewards of all batches into a tensor of shape (bs,k,1)
        sampled_rewards = torch.stack(sampled_rewards, dim=0)  # bs,k,1

        # Calculate the reward variance and process it by weighting
        sampled_rewards_var = torch.sum(torch.square(sampled_rewards - torch.mean(sampled_rewards, dim=1, keepdim=True)), dim=1) / (self.K - 1)  # bs,1
        sampled_rewards_var = torch.mean(sampled_rewards_var.squeeze() * torch.tensor(var_coef).to(sampled_rewards_var.device) / self.K)

        # Calculate the loss between the predicted return and the true return
        pred_returns = sampled_rewards.mean(dim=1).reshape(-1)
        episode_return = episode_return.sum(dim=1) / episode_length.reshape(-1)
        # episode_return = episode_return.reshape(-1) / episode_length.reshape(-1)
        loss = self.loss_fn(pred_returns, episode_return)
        if self.args.rrd_unbiased:
            loss = loss - sampled_rewards_var
        loss.backward()
        self.optimizer.step()
        return loss.item()

class RDRewardDecomposer(nn.Module):
    def __init__(self, args):
        super(RDRewardDecomposer, self).__init__()
        self.args = args

        self.reward_model = Reward_Model(input_dim=self.args.obs_dim * 2 + self.args.action_dim)
        self.reward_model.to('cuda')
        self.optimizer = torch.optim.Adam(list(self.reward_model.parameters()), lr=3e-4)
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, states, actions, next_states):
        states = torch.cat([states, actions, states - next_states], dim=-1)
        rewards = self.reward_model(states)  # bs,t,-1 -> bs,t,1
        return rewards

    def update(self, states, actions, next_states, episode_return, episode_length):
        self.optimizer.zero_grad()
        rewards = self.forward(states, actions, next_states)
        for i in range(rewards.shape[0]):
            rewards[i, int(episode_length[i].item()):] = 0
        pred_returns = rewards.sum(dim=1).reshape(-1) / episode_length.reshape(-1)
        episode_return = episode_return.reshape(-1) / episode_length.reshape(-1)
        loss = self.loss_fn(pred_returns, episode_return)
        loss.backward()
        self.optimizer.step()
        return loss.item()

