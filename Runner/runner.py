import torch
import copy
import simpy
import logging
import os
import psutil
import numpy as np
from LLMforReward.jspEnvironments.workshopllm import JobShop
# from Environments.JobShop import JobShop
from Environments.Reward import *
from used_trains.utils import check
import random
from numba import cuda
import wandb
from used_trains.sharedBuffer import SharedReplayBuffer
from utils import *
from Runner.utils import ReplayBuffer


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class rollout_Runner:
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
        self.buffer = SharedReplayBuffer(args, self.num_agents, self.args.episode_length)
        self.replay_buffer = ReplayBuffer(args.share_obs_dim, self.num_agents)
        self.reward_scaling = RewardScaling(shape=1, gamma=0.99)
        self.reward_scaling.reset()
        self.state_norm = Normalization(shape=args.obs_dim)  # Trick 2:state normalization

    def run(self, dataSet, type, agent, store_transitions, teams_blueprint, epsilon):
        '''
        :param envs:
        :param task_pope (pipe): Receiver end of the task pipe used to receive signal to start on a task
        :param result_pipe(pipe): Sender end of the pipe used to report back results
        :param data_ducket: A shared list object managed by a manager that is used to store experience tuples
        :param model_ducket: A shared list object managed by a manager used to store all the models (actors)
        :param store_transitions: Log experiences to exp_list?
        :param reward_func: The reward function used to calculate the reward of each step
        :return:
        '''
        last_train = 0
        all_steps = 0
        log = []
        actor_policy = agent.trainer.policy.transformer
        actor_policy.to(self.device)
        all_objectives = np.zeros((args.max_train_steps, 3))
        for t in range(args.max_train_steps):
            env = simpy.Environment()
            envs = JobShop(env, self.args)
            rollout_returns = 0
            episode_reward = 0
            raw_states, raw_next_states = [], []
            states, actions, next_states, rewards, dense_rewards, dones, a_log_probs = [], [], [], [], [], [], []
            obs, state, ava, ready_operas = envs.reset(dataSet)
            if args.state_norm:
                obs = self.state_norm(obs)
                state = self.state_norm(state)
            self.buffer.share_obs[0] = np.concatenate(state).copy()
            self.buffer.obs[0] = obs.copy()
            self.buffer.available_actions[0] = ava.copy()

            episode_lengths = 0
            episode_rewards = []
            episode_reward_components = {}
            training_counter = -1
            while not envs.done:
                envs.new_job_arrival()
                if envs.env.now == envs.decision_points[0]:
                    all_steps += 1
                    total_loss_log = {}
                    if self.args.use_linear_lr_decay:
                        agent.trainer.policy.lr_decay(episode_lengths, self.args.max_train_steps)
                    training_counter += 1
                    if self.replay_buffer.size >= 10e2:   # 10e3
                        reward_loss = agent.trainer.train_reward(self.replay_buffer)
                        total_loss_log['reward_model_loss'] = reward_loss
                    if training_counter == self.args.episode_length - 1:
                        training_counter = -1
                        train_infos = agent.update_parameters()
                        for key in train_infos:
                            if key not in total_loss_log:
                                total_loss_log[key] = 0
                            total_loss_log[key] = total_loss_log[key] + train_infos[key]
                        if all_steps - last_train >= 100:
                            log.append(total_loss_log)
                            last_train = all_steps
                    envs.decision_points.remove(envs.decision_points[0])
                    idle_machines = []
                    for m in envs.machines:
                        if m.currentTime <= envs.env.now:
                            idle_machines.append(m)
                    if len(envs.jobs) == 0 or len(idle_machines) == 0:
                        if len(envs.decision_points) == 0:
                            envs.env.timeout(1)
                            envs.env.run()
                        else:
                            envs.env.timeout(envs.decision_points[0] - envs.env.now)
                            envs.env.run()
                        continue

                    obs, ava, ready_operas = envs.get_observation()
                    if args.state_norm:
                        obs = self.state_norm(obs)
                    state = obs.copy()
                    values, action, action_log_probs = self.collect(actor_policy, state, obs, ava, epsilon)
                    current_obs = obs.copy()
                    current_state = state.copy()
                    # determine the actions of all agents through the actor policy
                    has_duplicates = True
                    redecision_counter = 0
                    while has_duplicates:
                        redecision_counter += 1
                        # check the decision conflict
                        non_zero_actions = action[action != 0]
                        if len(non_zero_actions) == 0:
                            break
                        has_duplicates = non_zero_actions.size != len(np.unique(non_zero_actions))
                        if redecision_counter > self.max_reDecision_steps:
                            # _, ava, ready_operas = envs.get_observation()
                            action = self.negotiation_mechanism(action, ready_operas, ava)
                            break
                    is_all_zero = np.mean(action)
                    if not is_all_zero:  # all machines do not select any operation
                        if len(envs.decision_points):
                            envs.env.timeout(envs.decision_points[0] - envs.env.now)
                            envs.env.run()
                        else:
                            envs.env.timeout(1)
                            envs.env.run()
                        if envs.env.now not in envs.decision_points:
                            envs.decision_points.append(envs.env.now)
                            envs.decision_points = sorted(envs.decision_points)
                        continue

                    obs, state, ava, ready_operas = envs.step(action, ready_operas)
                    raw_next_state = np.concatenate(state)
                    raw_next_state = np.broadcast_to(raw_next_state, (self.num_agents, self.share_obs_dim))
                    if args.state_norm:
                        obs = self.state_norm(obs)
                        state = self.state_norm(state)
                    reward = agent.trainer.reward_model.forward(np.concatenate(current_obs), action, np.concatenate(obs))
                    raw_reward = reward
                    if args.reward_scale:
                        reward = self.reward_scaling(reward)[0]
                    episode_rewards.append(reward)
                    rollout_returns += reward

                    episode_lengths += 1
                    if store_transitions:
                        current_state = np.concatenate(current_state)
                        current_state = np.broadcast_to(current_state, (self.num_agents, self.share_obs_dim))
                        np.broadcast_to(current_obs, (self.num_agents, self.obs_dim))
                        shared_obs = np.concatenate(state)
                        shared_obs = np.broadcast_to(shared_obs, (self.num_agents, self.share_obs_dim))
                        np.broadcast_to(obs, (self.num_agents, self.obs_dim))
                        agent_rewards = np.broadcast_to(reward, (self.num_agents, 1))

                        self.buffer.insert(current_state, current_obs, shared_obs, obs, action, action_log_probs, values, agent_rewards, ava, episode_lengths)
                        states.append(current_state[0, :])
                        raw_states.append(current_state[0, :])
                        actions.append(action.squeeze())
                        next_states.append(shared_obs[0, :])
                        raw_next_states.append(raw_next_state[0, :])
                        rewards.append([0 if envs.done else episode_reward])
                        dense_rewards.append([reward])
                        dones.append([float(envs.done)])
                        a_log_probs.append(action_log_probs.squeeze())

                    episode_reward = episode_reward + raw_reward

                    if len(envs.decision_points) == 0:
                        envs.env.timeout(1)
                        envs.env.run()
                    else:
                        envs.env.timeout(envs.decision_points[0] - envs.env.now)
                        envs.env.run()
                if len(envs.decision_points) == 0:
                    envs.decision_points.append(envs.env.now)
                envs.decision_points = sorted(envs.decision_points)

            if envs.done:
                self.replay_buffer.add_traj(np.array(raw_states), np.array(actions), np.array(raw_next_states), np.array(rewards),
                                            np.array(dense_rewards), np.array(dones), episode_reward, episode_lengths, np.array(a_log_probs))

            #  calculate the average episode length and the standard deviation of the episode length
            episode_rewards_avg = np.mean(episode_rewards)
            episode_lengths_std = np.std(episode_rewards)

            # calculate the objectives of each task
            objectives = [-1 for _ in range(envs.num_tasks)]
            for i in range(envs.num_tasks):
                obj = envs.tasks_list[i].objective
                if obj == 'WTmean':
                    objectives[i] = WT_mean_func(envs.completed_jobs[i])
                elif obj == 'WFmean':
                    objectives[i] = WF_mean_func(envs.completed_jobs[i])
                else:
                    objectives[i] = WT_max_func(envs.completed_jobs[i])
                all_objectives[t, :] = objectives

        if type == 'evo':
            return [teams_blueprint, rollout_returns, episode_lengths, objectives]
        elif type == 'pg':
            return [rollout_returns, episode_lengths, objectives, episode_rewards_avg, episode_lengths_std]
        else:
            return [teams_blueprint, rollout_returns]

    def negotiation_mechanism(self, actions, ready_operas, ava):
        # with considering the machine can be not selected
        ava_copy = copy.deepcopy(ava)
        ready_operas_copy = copy.deepcopy(ready_operas)
        # check the decision conflict
        flat_actions = actions.flatten()
        unique_values, inverse_indices = np.unique(flat_actions, return_inverse=True)
        # unique_values, inverse_indices = torch.unique(flat_actions, return_inverse=True)
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
        # def collect(self, step, actor_policy, ava):
        self.agent.trainer.prep_rollout()
        shared_obs = np.concatenate(state)

        cent_obs = np.broadcast_to(shared_obs, (self.num_agents, self.share_obs_dim))
        obs = np.broadcast_to(obs, (self.num_agents, self.obs_dim))

        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
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



if __name__ == '__main__':
    from tqdm import tqdm
    from params import args
    from recordData import recordData
    from used_trains.Agent import Agents

    with open(args.root_dir, 'r') as f:
        datasets = f.readlines()
    agents = Agents(args)
    runner = rollout_Runner(args, agents)
    entry = runner.run(datasets, 'pg', agents, args.rollout_size>0, 0, args.epsilon)

