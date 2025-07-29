import torch
import copy
from used_trains.sharedBuffer import SharedReplayBuffer
from Runner.utils import *
from Runner.utils import ReplayBuffer


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class fatory_Runner:
    def __init__(self, args, agent):
        self.act_num = 1
        self.args = args
        self.agent = agent
        self.buffer = agent.buffer
        self.num_agents = args.num_machines
        self.device = args.device if torch.cuda.is_available() else 'cpu'
        self.obs_dim = args.obs_dim
        self.share_obs_dim = args.share_obs_dim
        self.use_centralized_V = args.use_centralized_V
        self.n_rollout_threads = args.n_rollout_threads
        self.max_reDecision_steps = args.max_reDecision_steps
        self.act_dim = args.act_dim
        self.replay_buffer = ReplayBuffer(args.share_obs_dim, self.num_agents)
        self.reward_scaling = RewardScaling(shape=1, gamma=0.99)
        self.reward_scaling.reset()
        self.state_norm = Normalization(shape=args.obs_dim)  # Trick 2:state normalization

    def act(self, envs, factory_id, agent, obs, state, ava, episode_lengths):
        idle_machines = []
        actor_policy = agent.trainer.policy.transformer
        actor_policy.to(self.device)
        jobs = envs.factories[factory_id].assigned_jobs
        machines = envs.factories[factory_id].machines
        for m in machines:
            if m.currentTime <= envs.env.now:
                idle_machines.append(m)

        values, action, action_log_probs, ava_actions = self.collect(actor_policy, state, obs, ava, self.args.epsilon)
        episode_lengths += 1
        return action, action_log_probs, obs, state, episode_lengths, ava_actions, values

    # def collect(self, actor_policy, state, obs, ava, epsilon, Critic, global_state, global_obs):
    def collect(self, actor_policy, state, obs, ava, epsilon):
        # def collect(self, step, actor_policy, ava):
        self.agent.trainer.prep_rollout()
        cent_obs = np.broadcast_to(state, (self.num_agents, state.shape[0]))

        cent_obs = cent_obs.reshape(-1, self.num_agents, state.shape[0])
        obs = obs.reshape(-1, self.num_agents, obs.shape[1])
        if ava is not None:
            available_actions = ava.reshape(-1, self.act_dim)

        actions, action_log_probs, values, ava_actions = actor_policy.get_actions(cent_obs, obs, available_actions, epsilon)
        actions = _t2n(actions)
        action_log_probs = _t2n(action_log_probs)
        ava_actions = np.array(_t2n(ava_actions))
        values = _t2n(values)
        value = np.mean(values, axis=1)
        return value, actions, action_log_probs, ava_actions


