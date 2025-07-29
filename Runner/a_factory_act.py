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
        self.buffer = SharedReplayBuffer(args, self.num_agents, self.args.episode_length, args.act_dim)
        self.replay_buffer = ReplayBuffer(args.share_obs_dim, self.num_agents)
        self.reward_scaling = RewardScaling(shape=1, gamma=0.99)
        self.reward_scaling.reset()
        self.state_norm = Normalization(shape=args.obs_dim)  # Trick 2:state normalization

    def act(self, envs, factory_id, agent, ready_operas, ava, episode_lengths, llm_flag):
        idle_machines = []

        actor_policy = agent.trainer.policy.transformer
        actor_policy.to(self.device)

        jobs = envs.factories[factory_id].assigned_jobs
        machines = envs.factories[factory_id].machines
        for m in machines:
            if m.currentTime <= envs.env.now:
                idle_machines.append(m)

        obs, state = envs.agent_obervation(factory_id, ready_operas, llm_flag)
        if len(jobs) == 0 or len(idle_machines) == 0:
            actions = np.zeros((1, self.num_agents, 1))
            return actions, None, obs, state, None, episode_lengths
        values, action, action_log_probs = self.collect(actor_policy, state, obs, ava, self.args.epsilon)
        episode_lengths += 1
        return action, action_log_probs, obs, state, values, episode_lengths

    def negotiation_mechanism(self, actions, ready_operas, ava):
        # with considering the machine can be not selected
        ava_copy = copy.deepcopy(ava)
        ready_operas_copy = copy.deepcopy(ready_operas)
        # check the decision conflict
        flat_actions = actions.flatten()
        unique_values, inverse_indices = np.unique(flat_actions, return_inverse=True)
        value_indices = {value: [] for value in unique_values}
        for idx, value in enumerate(inverse_indices):
            value_indices[unique_values[value]].append(idx)

        # modify the decisions of other agents except select_mac, i.e., Select new actions for conflicting agents
        conflict_indices = []
        for key, value in value_indices.items():
            if key != 0:  # when key is 0, it means the machines is busy or unavailable. So we only need to modify the decisions of other agents.
                opera = ready_operas_copy[key - 1]
                ready_operas_copy[key - 1] = None  # the operation has been selected, and it cannot be selected again
                ava_copy[:, key] = 0  # the decision of this operation is not available
                # select the machine with the shortest processing time
                cadidate_macs = opera.cMachines
                macs_name = [f'M{i}' for i in value]
                min_value = float('inf')
                min_key = None
                for m in macs_name:
                    if m in cadidate_macs:  
                        if cadidate_macs[m] < min_value:  
                            min_value = cadidate_macs[m]
                            min_key = m
                selected_mac_id = int(min_key[1:])
                other_values = [val for val in value if val != selected_mac_id]
                conflict_indices.extend(other_values)  # record the indices of the conflicting agents

        # re-select the actions of the conflicting agents acording to SPT
        non_none_values_with_indices = [(value, index + 1) for index, value in enumerate(ready_operas_copy) if
                                        value is not None]
        has_non_none = len(non_none_values_with_indices)
        if has_non_none > 0:
            queues = {}
            index_queue = {}
            for i in conflict_indices:
                queues[i] = []
                index_queue[i] = []
            for opera, index in non_none_values_with_indices:
                selected_mac = sorted(opera.cMachines, key=opera.cMachines.get)[0]
                selected_mac_id = int(selected_mac[1:])
                if selected_mac_id in conflict_indices:
                    queues[selected_mac_id].append(opera)
                    index_queue[selected_mac_id].append(index)
            for key, value in queues.items():
                if len(value) == 0:
                    actions[0][key][0] = 0
                else:
                    if len(value) == 1:
                        actions[0][key][0] = index_queue[key][0]
                    else:
                        mac_name = "M" + str(key)
                        min_pt = value[0].cMachines[mac_name]
                        min_index = 0
                        for i in range(1, len(value)):
                            opera = value[i]
                            pt = opera.cMachines[mac_name]
                            if pt < min_pt:
                                min_pt = pt
                                min_index = i
                        actions[0][key][0] = index_queue[key][min_index]
        else:  # no ready operations
            for m in conflict_indices:
                actions[0][m][0] = 0
        return actions

    def collect(self, actor_policy, state, obs, ava, epsilon):
        self.agent.trainer.prep_rollout()
        cent_obs = np.broadcast_to(state, (self.num_agents, state.shape[0]))

        cent_obs = cent_obs.reshape(-1, self.num_agents, state.shape[0])
        obs = obs.reshape(-1, self.num_agents, obs.shape[1])
        if ava is not None:
            available_actions = ava.reshape(-1, ava.shape[0], ava.shape[1])
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

        actions, action_log_probs, values = actor_policy.get_actions(cent_obs, obs, available_actions, epsilon)

        actions = actions.view(-1, self.act_num)  # (batch_size, num_agent, act_dim) -> (batch_size*num_agent, act_dim)
        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, 1)

        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        return values, actions, action_log_probs
