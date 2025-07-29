import torch
import numpy as np
from model.Actor3 import Critic
from used_trains.utils import check
from used_trains.valuenorm import ValueNorm
from used_trains.utils import get_gard_norm, huber_loss, mse_loss


class CriticTrianer:
    def __init__(self, args, device=torch.device("cpu")):
        self.args = args
        self.lr = args.critic_lr
        self.device = args.device
        self.opti_eps = args.opti_eps
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.weight_decay = args.weight_decay
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.obs_dim = args.input_obs_dim
        self.num_all_agents = args.num_all_agents
        self.num_mini_batch = args.critic_num_mini_batch

        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_value_active_masks = args.use_value_active_masks
        self._use_huber_loss = args.use_huber_loss
        self._use_valuenorm = args.use_valuenorm
        self.huber_delta = args.huber_delta
        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

        self.share_obs_dim = args.input_obs_dim * args.num_machines
        self.Critic = Critic(self.share_obs_dim, self.obs_dim, args.num_all_agents,
                             n_block=args.n_block, n_embd=args.n_embd, n_head=args.n_head,
                             encode_state=args.encode_state, dropout=args.dropout, device=device)
        self.Critic.to(device)
        self.critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=self.lr, eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def compute_values(self, cent_obs, obs, num_factories):
        obs = obs.reshape(-1, obs.shape[0], obs.shape[1])
        values = self.Critic(cent_obs, obs, num_factories)
        values = values.view(-1, 1)
        return values

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.
        :return value_loss: (torch.Tensor) value function loss.
        """

        return_batch = check(return_batch)
        value_preds_batch = check(value_preds_batch)
        active_masks_batch = check(active_masks_batch)
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)

        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original
        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def update_parameters(self, sample, num_factories):
        current_share_obs_batch, current_obs_batch, share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, episode_batch = sample
        values = self.Critic(current_share_obs_batch, current_obs_batch, num_factories)
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        critic_grad_norm = get_gard_norm(self.Critic.encoder.parameters())
        self.critic_optimizer.step()
        return value_loss.item(), critic_grad_norm

    def train(self, buffer):
        advantages_copy = buffer.advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (buffer.advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}
        train_info['value_loss'] = 0
        train_info['critic_grad_norm'] = 0

        for _ in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator_transformer(advantages, self.num_mini_batch)
            for sample in data_generator:
                value_loss, critic_grad_norm = self.update_parameters(sample, buffer.num_factories)
                train_info['value_loss'] += value_loss
                train_info['critic_grad_norm'] += critic_grad_norm

        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info


