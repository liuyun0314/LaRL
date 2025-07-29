import os
import torch
import datetime
import numpy as np
from model.util import check
from model.Actor3 import MultiAgentTransformer as policy
from model.Actor3 import Critic
from model.util import update_linear_schedule
from model.util import get_shape_from_obs_space, get_shape_from_act_space

class TransformerPolicy:
    """
    MAT Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, num_agents, device=torch.device("cpu")):
        self.device = device
        self.algorithm_name = "mat"
        self.lr = args.lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self._use_policy_active_masks = args.use_policy_active_masks
        self.action_type = 'Discrete'
        self.lr_decay_rate = args.lr_decay_rate

        self.obs_dim = args.input_obs_dim
        share_obs_dim = args.input_obs_dim * num_agents
        self.share_obs_dim = share_obs_dim
        self.reasoning_dim = args.reasoning_dim
        self.act_dim = args.act_dim     
        self.act_num = 1

        self.num_agents = num_agents
        self.num_all_agents = args.num_all_agents
        self.tpdv = dict(dtype=torch.float32, device=device)

        share_obs_dim = args.obs_dim * self.num_agents
        self.transformer = policy(share_obs_dim, self.obs_dim, self.reasoning_dim, self.act_dim, num_agents,
                               n_block=args.n_block, n_embd=args.n_embd, n_head=args.n_head,
                               encode_state=args.encode_state, device=device, dropout=args.dropout,
                               action_type=self.action_type, dec_actor=args.dec_actor,
                               share_actor=args.share_actor)
        self.optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

        self.Critic = Critic(share_obs_dim, self.obs_dim, args.num_all_agents,
                               n_block=args.n_block, n_embd=args.n_embd, n_head=args.n_head,
                               encode_state=args.encode_state, dropout=args.dropout, device=device)
        self.Critic.to(device)
        self.critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)


    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """

        lr = self.lr - (self.lr * self.lr_decay_rate)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_actions(self, cent_obs, obs, available_actions, deterministic):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        """

        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.act_dim)

        actions, action_log_probs, values,_ = self.transformer.get_actions(cent_obs, obs, available_actions, deterministic)

        actions = actions.view(-1, self.act_num)   # (batch_size, num_agent, act_dim) -> (batch_size*num_agent, act_dim)
        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, 1)
        return values, actions, action_log_probs

    def get_values(self, cent_obs, obs, masks, available_actions):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """

        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        values = self.transformer.get_values(cent_obs, obs, available_actions)
        values = values.detach().cpu().numpy()
        values = np.mean(values, axis=1)
        return values

    def compute_values(self, cent_obs, obs):
        cent_obs = cent_obs.reshape(-1, self.num_all_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_all_agents, self.obs_dim)
        values = self.Critic(cent_obs, obs)
        values = values.view(-1, 1)
        return values


    def evaluate_actions(self, cent_obs, obs, actions, available_actions, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param actions: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """

        action_log_probs, values, entropy = self.transformer(cent_obs, obs, actions, available_actions)
        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.mean(dim=1)
        action_log_probs = action_log_probs.reshape(active_masks.shape[0], active_masks.shape[1], active_masks.shape[1])
        entropy = entropy.reshape(active_masks.shape[0], active_masks.shape[1], active_masks.shape[1])

        if self._use_policy_active_masks and active_masks is not None:
            entropy = (entropy*active_masks).sum()/active_masks.sum()
        else:
            entropy = entropy.mean()

        return action_log_probs, values, entropy

    def act(self, cent_obs, obs, available_actions, deterministic):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """

        # this function is just a wrapper for compatibility
        _, actions, _ = self.get_actions(cent_obs, obs, available_actions, deterministic)

        return actions

    def save(self, save_dir, episode):
        torch.save(self.transformer.state_dict(), str(save_dir) + "/transformer_" + str(episode) + ".pt")

    def restore(self, model_dir):
        transformer_state_dict = torch.load(model_dir)
        self.transformer.load_state_dict(transformer_state_dict)

    def train(self):
        self.transformer.train()

    def eval(self):
        self.transformer.eval()