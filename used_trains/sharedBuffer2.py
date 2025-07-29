import torch
import random
import numpy as np
from torch.multiprocessing import Manager


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols


class SharedReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, num_agents, action_num, act_dim):
        # self.device = args.device
        self.args = args
        self.max_traj_len = args.batch_size
        self.device = args.device if torch.cuda.is_available() else 'cpu'
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        # self.algo = args.algorithm_name
        self.num_agents = num_agents
        self.num_factories = action_num
        self.counter = 0
        self.manager = Manager()
        self.tuples = self.manager.list()  # Temporary shared buffer to get experiences from processes

        obs_shape = args.input_obs_dim  
        share_obs_shape = obs_shape * self.num_agents  

        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, share_obs_shape),
                                  dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, obs_shape), dtype=np.float32)

        self.current_share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, share_obs_shape),
                                  dtype=np.float32)
        self.current_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, obs_shape), dtype=np.float32)
        act_shape = act_dim 
        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.advantages = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1, 1), dtype=np.float32)

        self.available_actions = np.zeros((self.episode_length + 1, self.n_rollout_threads, action_num, act_shape),
                                        dtype=np.float32)
        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, action_num, 1), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, action_num, 1), dtype=np.float32)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, 1, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)
        self.episodes = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1, 1), dtype=np.float32)
        self.not_dones = np.zeros_like(self.episodes)
        self.step = 0
        self.current_share_obs_batch = []
        self.current_obs_batch = []
        self.share_obs_batch = []
        self.obs_batch = []
        self.value_preds_batch = []
        self.returns_batch = []
        self.advantages_batch = []
        self.available_actions_batch = []
        self.actions_batch = []
        self.action_log_probs_batch = []
        self.rewards_batch = []
        self.masks_batch = []
        self.bad_masks_batch = []
        self.active_masks_batch = []
        self.traj_head_id = []
        self.traj_end_id = []
        self.episode_return = []

    def insert(self, current_state, current_obs, share_obs, obs, actions, action_log_probs, value_preds, rewards, mask, available_actions, episode):
        """
        Insert data into the buffer.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) action space for agents.
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.counter += 1
        self.current_share_obs[self.step] = current_state.copy()
        self.current_obs[self.step] = current_obs.copy()
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.available_actions[self.step + 1] = available_actions.copy()
        self.step = (self.step + 1) % self.episode_length
        self.episodes[self.step] = episode
        self.masks[self.step] = mask.copy()

    def referesh(self):
        """Housekeeping
            Parameters:
                None
            Returns:
                None
        """

        # Add ALL EXPERIENCE COLLECTED TO MEMORY concurrently
        for _ in range(len(self.tuples)):
            exp = self.tuples.pop()
            current_share_obs, current_obs, share_obs, obs, actions, action_log_probs, value_preds, rewards, available_actions, episode = exp
            self.insert(current_share_obs, current_obs, share_obs, obs, actions, action_log_probs, value_preds, rewards, available_actions, episode)

        # Trim to make the buffer size < capacity
        while self.step >= self.episode_length:
            self.after_update()
            self.step = 0

    def tensorify(self):
        """Method to save experiences to drive
               Parameters:
                   None
               Returns:
                   None
           """
        self.referesh()  # Referesh first
        if self.counter > 1:
            self.current_share_obs_batch = torch.tensor(self.current_share_obs[:self.counter], dtype=torch.float32).to(self.device)
            self.current_obs_batch = torch.tensor(self.current_obs[:self.counter], dtype=torch.float32).to(self.device)
            self.share_obs_batch = torch.tensor(self.share_obs[:self.counter], dtype=torch.float32).to(self.device)
            self.obs_batch = torch.tensor(self.obs[:self.counter], dtype=torch.float32).to(self.device)
            self.value_preds_batch = torch.tensor(self.value_preds[:self.counter], dtype=torch.float32).to(self.device)
            self.returns_batch = torch.tensor(self.returns[:self.counter], dtype=torch.float32).to(self.device)
            self.advantages_batch = torch.tensor(self.advantages[:self.counter], dtype=torch.float32).to(self.device)
            self.available_actions_batch = torch.tensor(self.available_actions[:self.counter], dtype=torch.float32).to(self.device)
            self.actions_batch = torch.tensor(self.actions[:self.counter], dtype=torch.float32).to(self.device)
            self.action_log_probs_batch = torch.tensor(self.action_log_probs[:self.counter], dtype=torch.float32).to(self.device)
            self.rewards_batch = torch.tensor(self.rewards[:self.counter], dtype=torch.float32).to(self.device)
            self.masks_batch = torch.tensor(self.masks[:self.counter], dtype=torch.float32).to(self.device)
            self.bad_masks_batch = torch.tensor(self.bad_masks[:self.counter], dtype=torch.float32).to(self.device)
            self.active_masks_batch = torch.tensor(self.active_masks[:self.counter], dtype=torch.float32).to(self.device)
            self.episode_batch = torch.tensor(self.episodes[:self.counter], dtype=torch.float32).to(self.device)

    def chooseinsert(self, current_share_obs, current_obs, share_obs, obs, actions, action_log_probs,
                     value_preds, rewards, masks, episode, bad_masks=None, active_masks=None, available_actions=None):
        """
        Insert data into the buffer. This insert function is used specifically for Hanabi, which is turn based.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) denotes indicate whether whether true terminal state or due to episode limit
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.current_share_obs[self.step] = current_share_obs.copy()
        self.current_obs[self.step] = current_obs.copy()
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.episodes[self.step] = episode.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()
        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.counter = 1
        self.current_share_obs[0] = self.current_share_obs[-1].copy()
        self.current_obs[0] = self.current_obs[-1].copy()
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.episodes[0] = self.episodes[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def chooseafter_update(self):
        """Copy last timestep data to first index. This method is used for Hanabi."""
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            if self._use_popart or self._use_valuenorm:
                delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                self.advantages[step] = gae
                self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
            else:
                delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                self.advantages[step] = gae
                self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator_transformer(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """

        num_agents = self.obs.shape[2]
        episode_length, n_rollout_threads, num_factory = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (   
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length,
                          n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()    # generate a random permutation of 0 to batch_size-1 and convert to numpy array
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        rows, cols = _shuffle_agent_grid(batch_size, num_agents)
        rows2, cols2 = _shuffle_agent_grid(batch_size, num_factory)

        # keep (num_agent, dim)
        current_share_obs = self.current_share_obs.reshape(-1, *self.current_share_obs.shape[2:])
        current_share_obs = current_share_obs[rows, cols]
        current_obs = self.current_obs.reshape(-1, *self.current_obs.shape[2:])
        current_obs = current_obs[rows, cols]
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        share_obs = share_obs[rows, cols]
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        obs = obs[rows, cols]
        episodes = self.episodes[:-1].reshape(-1, *self.episodes.shape[2:])
        episodes = episodes[rows[:,0], cols[:,0]]
        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        actions = actions[rows2, cols2]
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[2:])
            available_actions = available_actions[rows2, cols2]
        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
        value_preds = value_preds[rows2, cols2]
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
        returns = returns[rows2, cols2]
        masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
        masks = masks[rows2, cols2]
        active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])
        active_masks = active_masks[rows2, cols2]
        action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[2:])
        action_log_probs = action_log_probs[rows2, cols2]
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        advantages = advantages[rows2, cols2]

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            current_share_obs_batch = current_share_obs[indices]
            current_obs_batch = current_obs[indices]
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            episode_batch = episodes[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield current_share_obs_batch, current_obs_batch, share_obs_batch, obs_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                adv_targ, available_actions_batch, episode_batch

    def compute_returns2(self, value_preds, next_value, masks, rewards, returns, advantages, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(rewards.shape[0])):
            if self._use_popart or self._use_valuenorm:
                delta = rewards[step] + self.gamma * value_normalizer.denormalize(value_preds[step + 1]) * masks[step + 1] - value_normalizer.denormalize(value_preds[step])
                gae = delta + self.gamma * self.gae_lambda * masks[step + 1] * gae
                advantages[step] = gae
                returns[step] = gae + value_normalizer.denormalize(value_preds[step])
            else:
                delta = rewards[step] + self.gamma * value_preds[step + 1] * masks[step + 1] - value_preds[step]
                gae = delta + self.gamma * self.gae_lambda * masks[step + 1] * gae

                advantages[step] = gae
                returns[step] = gae + value_preds[step]

    def feed_forward_generator_transformer2(self, share_obs, obs, rewards, returns, advantages, actions, action_log_probs, available_actions, value_preds, masks, active_masks, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (   
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length,
                          n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()    # generate a random permutation of 0 to batch_size-1 and convert to numpy array
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        rows, cols = _shuffle_agent_grid(batch_size, num_agents)

        # keep (num_agent, dim)
        tmp_share_obs = share_obs[:-1].reshape(-1, *share_obs.shape[2:])
        tmp_share_obs = tmp_share_obs[rows, cols]
        tmp_obs = obs[:-1].reshape(-1, *obs.shape[2:])
        tmp_obs = tmp_obs[rows, cols]
        tmp_actions = actions.reshape(-1, *actions.shape[2:])
        tmp_actions = tmp_actions[rows, cols]
        if available_actions is not None:
            tmp_available_actions = available_actions[:-1].reshape(-1, *available_actions.shape[2:])
            tmp_available_actions = tmp_available_actions[rows, cols]
        tmp_value_preds = value_preds[:-1].reshape(-1, *value_preds.shape[2:])
        tmp_value_preds = tmp_value_preds[rows, cols]
        tmp_returns = returns[:-1].reshape(-1, *returns.shape[2:])
        tmp_returns = tmp_returns[rows, cols]
        tmp_masks = masks[:-1].reshape(-1, *masks.shape[2:])
        tmp_masks = tmp_masks[rows, cols]
        tmp_active_masks = active_masks[:-1].reshape(-1, *active_masks.shape[2:])
        tmp_active_masks = tmp_active_masks[rows, cols]
        tmp_action_log_probs = action_log_probs.reshape(-1, *action_log_probs.shape[2:])
        tmp_action_log_probs = tmp_action_log_probs[rows, cols]
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        advantages = advantages[rows, cols]

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            share_obs_batch = tmp_share_obs[indices].reshape(-1, *tmp_share_obs.shape[2:])
            obs_batch = tmp_obs[indices].reshape(-1, *tmp_obs.shape[2:])
            actions_batch = tmp_actions[indices].reshape(-1, *tmp_actions.shape[2:])
            if available_actions is not None:
                available_actions_batch = tmp_available_actions[indices].reshape(-1, *tmp_available_actions.shape[2:])
            else:
                available_actions_batch = None
            value_preds_batch = tmp_value_preds[indices].reshape(-1, *tmp_value_preds.shape[2:])
            return_batch = tmp_returns[indices].reshape(-1, *tmp_returns.shape[2:])
            masks_batch = tmp_masks[indices].reshape(-1, *tmp_masks.shape[2:])
            active_masks_batch = tmp_active_masks[indices].reshape(-1, *tmp_active_masks.shape[2:])
            old_action_log_probs_batch = tmp_action_log_probs[indices].reshape(-1, *tmp_action_log_probs.shape[2:])
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])

            yield share_obs_batch, obs_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                adv_targ, available_actions_batch

    def sample_traj(self, traj_num, length_priority=False):
        """
        """
        if not hasattr(self, 'traj_head_id') or len(self.traj_head_id) == 0:
            raise ValueError("No trajectory data available for sampling.")

        # Step 1: Select the episode index to be sampled
        if not length_priority:
            ind = np.random.randint(0, len(self.traj_head_id), size=traj_num)
        else:
            weights = np.array(self.episode_length) / np.sum(self.episode_length)
            ind = np.random.choice(np.arange(len(self.traj_head_id)), p=weights, size=traj_num)

        # Step 2: Initialize the batch container
        max_traj_len = self.max_traj_len
        sampled_states = np.zeros((traj_num, max_traj_len, self.args.share_obs_dim))
        sampled_actions = np.zeros((traj_num, max_traj_len, self.num_agents))
        sampled_next_states = np.zeros((traj_num, max_traj_len, self.args.share_obs_dim))
        sampled_rewards = np.zeros((traj_num, max_traj_len, 1))
        sampled_dense_rewards = np.zeros((traj_num, max_traj_len, 1))
        sampled_not_dones = np.zeros((traj_num, max_traj_len, 1))
        sampled_a_log_probs = np.zeros((traj_num, max_traj_len, self.num_agents)) if self.num_agents else None
        sampled_episode_returns = np.zeros((traj_num, 1))
        sampled_episode_lengths = np.zeros((traj_num, 1))

        # Step 3: Fill in each episode trajectory
        for i, idx in enumerate(ind):
            traj_head = self.traj_head_id[idx]
            traj_end = self.traj_end_id[idx]
            episode_return = self.episode_return[idx]
            episode_length = self.episode_length[idx]

            sampled_episode_returns[i] = episode_return
            sampled_episode_lengths[i] = episode_length

            if traj_head < traj_end:
                assert traj_end - traj_head == episode_length
                sampled_states[i, :episode_length] = self.states[traj_head:traj_end]
                sampled_actions[i, :episode_length] = self.actions[traj_head:traj_end]
                sampled_next_states[i, :episode_length] = self.next_states[traj_head:traj_end]
                sampled_rewards[i, :episode_length] = self.rewards[traj_head:traj_end]
                sampled_dense_rewards[i, :episode_length] = self.dense_rewards[traj_head:traj_end]
                sampled_not_dones[i, :episode_length] = self.not_dones[traj_head:traj_end]
                if self.log_prob:
                    sampled_a_log_probs[i, :episode_length] = self.a_log_probs[traj_head:traj_end]
            else:
                upper_part = self.max_size - traj_head
                lower_part = episode_length - upper_part

                sampled_states[i, :upper_part] = self.states[traj_head:]
                sampled_states[i, upper_part:upper_part + lower_part] = self.states[:lower_part]

                sampled_actions[i, :upper_part] = self.actions[traj_head:]
                sampled_actions[i, upper_part:upper_part + lower_part] = self.actions[:lower_part]

                sampled_next_states[i, :upper_part] = self.next_states[traj_head:]
                sampled_next_states[i, upper_part:upper_part + lower_part] = self.next_states[:lower_part]

                sampled_rewards[i, :upper_part] = self.rewards[traj_head:]
                sampled_rewards[i, upper_part:upper_part + lower_part] = self.rewards[:lower_part]

                sampled_dense_rewards[i, :upper_part] = self.dense_rewards[traj_head:]
                sampled_dense_rewards[i, upper_part:upper_part + lower_part] = self.dense_rewards[:lower_part]

                sampled_not_dones[i, :upper_part] = self.not_dones[traj_head:]
                sampled_not_dones[i, upper_part:upper_part + lower_part] = self.not_dones[:lower_part]

                if self.log_prob:
                    sampled_a_log_probs[i, :upper_part] = self.a_log_probs[traj_head:]
                    sampled_a_log_probs[i, upper_part:upper_part + lower_part] = self.a_log_probs[:lower_part]

        # Step 4: Construct and return the batch data
        if self.log_prob is not None:
            return (
                torch.FloatTensor(sampled_states).to(self.device),
                torch.FloatTensor(sampled_actions).to(self.device),
                torch.FloatTensor(sampled_next_states).to(self.device),
                torch.FloatTensor(sampled_rewards).to(self.device),
                torch.FloatTensor(sampled_dense_rewards).to(self.device),
                torch.FloatTensor(sampled_not_dones).to(self.device),
                torch.FloatTensor(sampled_episode_returns).to(self.device),
                torch.FloatTensor(sampled_episode_lengths).to(self.device),
                torch.FloatTensor(sampled_a_log_probs).to(self.device)
            )
        else:
            return (
                torch.FloatTensor(sampled_states).to(self.device),
                torch.FloatTensor(sampled_actions).to(self.device),
                torch.FloatTensor(sampled_next_states).to(self.device),
                torch.FloatTensor(sampled_rewards).to(self.device),
                torch.FloatTensor(sampled_dense_rewards).to(self.device),
                torch.FloatTensor(sampled_not_dones).to(self.device),
                torch.FloatTensor(sampled_episode_returns).to(self.device),
                torch.FloatTensor(sampled_episode_lengths).to(self.device)
            )