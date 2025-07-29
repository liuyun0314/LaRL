import os
import copy
import simpy
import time
import datetime
import numpy as np
import pandas as pd
from params import args
from openai import OpenAI
from Runner.utils import *
from used_trains.Agent3 import Agents
from Environments.DJobShop3 import DJobShop
from LLMforPlanning.prompt_template import *
from Runner.factory_act4 import fatory_Runner
from LLMforReward.LLMR import LLMRewardDecomposer
from used_trains.sharedBuffer2 import SharedReplayBuffer
from used_trains.PPO_mian2 import CriticTrianer
from Environments.Reward import WT_mean_func, WF_mean_func, WT_max_func, makespan, estimated_reward
from LLMforPlanning.task_planning import LLM_task_planning as order_planning
from concurrent.futures import ThreadPoolExecutor
from Runner.utils import ReplayBuffer2 as ReplayBuffer

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

def AR(factories):
    all_available_time = []
    for i, factory in enumerate(factories):
        available_time = factory.available_mac_time()
        min_pt = float('inf')
        for item in available_time:
            pt = list(item.values())[0]
            if pt < min_pt:
                min_pt = pt
        all_available_time.append(min_pt)
    selected_factory = np.argmin(all_available_time)
    return selected_factory

def save(policy, episode, path):
    """Save policy's actor and critic networks."""
    path = f"{path}/model_{episode}"
    if not os.path.exists(path):
        os.makedirs(path)
    policy.save(path, episode)

def assigned_factory(envs):
    llm_flag = [-1 for _ in range(envs.num_factories)] 
    selected_factory = -1
    if envs.env.now in envs.in_system_job_dic:
        new_jobs = envs.in_system_job_dic[envs.env.now]
        if len(new_jobs):
            llm_flag = [0 for _ in range(envs.num_factories)]
            for job in new_jobs:
                selected_factory = order_planning(envs.env.now, envs.factories, job, envs.num_factories)
                # selected_factory = AR(envs.factories)  # select factory with minimum available time
                job.assigned_factory = selected_factory
                envs.factories[selected_factory].assigned_jobs.append(job)
                llm_flag[selected_factory] = 1
    return llm_flag, selected_factory

def main():
    now = datetime.datetime.now()
    now_time = str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute)
    models_save_path = f"./trainedModel/{now_time}/"
    if not os.path.exists(models_save_path):
        os.makedirs(models_save_path)
    args.root_dir = /GenertedInstances/trainingData/85/60570.txt'
    with open(args.root_dir, 'r') as f:
        dataSets = f.readlines()
    env = simpy.Environment()
    envs = DJobShop(env, args)
    envs.reset(dataSets)
    agents = [Agents(args, envs.factories[i].num_machines, args.reward_id) for i in range(envs.num_factories)]
    factory_actors = [fatory_Runner(args, agents[i]) for i in range(envs.num_factories)]
    Critic= CriticTrianer(args)

    reward_model = LLMRewardDecomposer(args, args.reward_id)
    num_all_machines = 0
    for i in range(envs.num_factories):
        num_all_machines += envs.factories[i].num_machines
    global_state_dim = args.input_obs_dim * num_all_machines
    action_dim = num_all_machines
    global_action_dim = args.act_dim * envs.num_factories
    global_buffer = SharedReplayBuffer(args, num_all_machines, envs.num_factories, args.act_dim)
    replay_buffer = ReplayBuffer(envs.num_factories, envs.factories[0].num_machines, args.input_obs_dim, envs.num_factories)
    reward_scaling = RewardScaling(shape=1, gamma=0.99)
    reward_scaling.reset()

    all_steps = 0
    last_train = 0
    all_episodes_rewards = []
    scale_rewards = []
    update_times, rewardModel_update_times = 0, 0
    logs = {}
    updat_times = [0 for _ in range(envs.num_factories)]
    for i in range(envs.num_factories):
        k = 'factory_actor_' + str(i)
        logs[k] = []
    critic_value_loss = []
    rewardModel_loss_log = []
    all_objectives = np.zeros((args.max_train_steps, 5))
    svae_time = 0
    max_makespan = float('inf')
    reward_update_frequency = 200
    episode_return  = []

    training_data_path = '/GenertedInstances/trainingData/100/'
    txt_files = [f for f in os.listdir(training_data_path) if f.endswith('.txt')]

    for t in range(args.max_train_steps):

        env = simpy.Environment()
        envs = DJobShop(env, args)
        envs.reset(dataSets)

        episode_steps = 0
        episode_reward = 0
        episode_lengths = [0 for i in range(envs.num_factories)]
        raw_states, raw_next_states, all_rewards, all_dense_rewards, all_agents_actions, all_agents_actions_log_probs, all_ava_actions, dones = [], [], [], [], [], [], [], []
        if args.use_linear_lr_decay:
            for i in range(envs.num_factories):
                agents[i].trainer.policy.lr_decay(t, args.max_train_steps)

        while not envs.done:
            if envs.env.now in envs.arrival_times:
                envs.new_job_arrival()
                # new order assignment
                llm_flag, selected_factory = envs.assigned_factory()

            if envs.env.now == envs.decision_points[0]:
                envs.decision_points.remove(envs.env.now)
                ready_opera = envs.get_ready_opera()
                if all(x is None for x in ready_opera):
                    if len(envs.decision_points) == 0:
                        envs.env.timeout(1)
                        envs.env.run()
                    else:
                        envs.env.timeout(envs.decision_points[0] - envs.env.now)
                        envs.env.run()
                    if envs.env.now not in envs.decision_points:
                        envs.decision_points.append(envs.env.now)
                    envs.decision_points = sorted(envs.decision_points)
                    continue

                factory_ready_operations = [[] for _ in range(envs.num_factories)]
                for opera in ready_opera:
                    if opera is not None:
                        job = envs.jobs[opera.global_jobID]
                        factory_ready_operations[job.assigned_factory].append(opera)
                max_len = 0
                for item in factory_ready_operations:
                    if len(item) > max_len:
                        max_len = len(item)

                for _ in range(max_len):
                    all_actions, all_action_log_probs, all_obs, all_state, all_ready_opera, all_values, avas = [], [], [], [], [], [], []
                    for i in range(envs.num_factories):

                        obs, state = envs.agent_state(i, factory_ready_operations[i], llm_flag)
                        all_obs.append(obs)
                        all_state.append(state)
                        ava = envs.get_ava(envs.factories[i].machines, factory_ready_operations[i])
                        action, action_log_probs, obs, state, episode_length, ava_actions, value = factory_actors[i].act(envs, i, agents[i], all_obs[i], all_state[i], ava, args.epsilon)
                        all_actions.append(action)
                        all_action_log_probs.append(action_log_probs)
                        avas.append(ava)
                        all_values.append(value)

                    if len(envs.decision_points):
                        next_time = envs.decision_points[0]
                    else:
                        next_time = envs.env.now + 1
                    next_obs, next_state, factory_ready_operations = envs.act(all_actions, factory_ready_operations, llm_flag, next_time)
                    global_state = np.concatenate(all_obs)
                    global_state = np.expand_dims(global_state, 0)
                    next_global_state = np.concatenate(next_obs)
                    next_global_state = np.expand_dims(next_global_state, 0)
                    global_actions = np.array(all_actions)
                    global_actions = np.expand_dims(global_actions, 0)
                    reward = reward_model.forward(torch.tensor(global_state), torch.tensor(global_actions), torch.tensor(next_global_state))
                    raw_reward = reward
                    if args.reward_scale:
                        reward = reward_scaling(reward)[0]
                    episode_reward = episode_reward + raw_reward
                    all_episodes_rewards.append(raw_reward)
                    scale_rewards.append(reward)
                    masks = [float(envs.done) for _ in range(envs.num_factories)]

                    mask = np.ones((args.n_rollout_threads, 1, 1), dtype=np.float32)
                    if envs.done:
                        mask[:] = 1.0
                    else:
                        mask[:] = 0


                    values = Critic.compute_values(np.concatenate(all_state), np.concatenate(all_obs), envs.num_factories)
                    values = np.array(np.split(_t2n(values), args.n_rollout_threads))
                    global_buffer.insert(np.concatenate(all_state), np.concatenate(all_obs), np.concatenate(next_state), np.concatenate(next_obs), np.array(all_actions).reshape(1, len(all_actions), 1),
                                         np.array(all_action_log_probs).reshape(1, len(all_actions), 1), values, reward, mask, np.stack(avas), episode_steps)
                    if global_buffer.step == args.episode_length - 1:
                        critic_train_info = Critic.train(global_buffer)
                        critic_value_loss.append(critic_train_info['value_loss'])
                        global_buffer.after_update()

                    total_loss_log = {}
                    for i in range(envs.num_factories):
                        k = f"factory_actor_{i}"
                        total_loss_log[k] = []
                        loss_log = {}

                        agents[i].buffer.insert(all_state[i], all_obs[i], next_state[i], next_obs[i], all_actions[i], all_action_log_probs[i], all_values[i], reward, mask, avas[i], episode_lengths[i])

                        if agents[i].buffer.step == args.episode_length - 1:
                            updat_times[i] += 1
                            train_infos = agents[i].update_paras(i, Critic.Critic, global_buffer.share_obs[-1], global_buffer.obs[-1], global_buffer.num_factories)
                            for key in train_infos:
                                if key not in loss_log:
                                    loss_log[key] = 0
                                loss_log[key] = loss_log[key] + train_infos[key]
                            total_loss_log[k].append(loss_log)
                            if updat_times[i] % args.save_interval == 0:
                                path = f"{models_save_path}/Agents_factory_{i}/"
                                save(agents[i].trainer.policy, updat_times[i], path)
                            update_times += 1

                            if all_steps - last_train >= 100:
                                if len(total_loss_log[k]) > 0:
                                    logs[k].append(total_loss_log[k])
                            if (i == envs.num_factories - 1) and (all_steps - last_train >= 100):
                                last_train = all_steps

                    all_steps += 1
                    episode_steps += 1
                    raw_states.append(np.concatenate(all_obs))
                    raw_next_states.append(np.concatenate(next_obs))
                    all_agents_actions.append(np.array(all_actions))
                    all_rewards.append([0 if envs.done else episode_reward])
                    all_dense_rewards.append([reward])
                    dones.append([float(envs.done)])
                    if replay_buffer.size >= 10e3:  # 10e3
                        if all_steps % reward_update_frequency == 0:
                            reward_loss = reward_model.train_reward(replay_buffer)
                            rewardModel_loss_log.append(reward_loss)

            if envs.done:
                replay_buffer.add_traj(np.array(raw_states), np.array(all_agents_actions), np.array(raw_next_states),
                                       np.array(all_rewards), np.array(all_dense_rewards), np.array(dones), episode_reward, episode_steps)
                reward_scaling.reset()
        # calculate the objectives of each task
        objectives = [-1 for _ in range(envs.num_tasks+2)]
        for i in range(envs.num_tasks):
            obj = envs.tasks_list[i].objective
            if obj == 'WTmean':
                objectives[i] = WT_mean_func(envs.completed_jobs[i])
            elif obj == 'WFmean':
                objectives[i] = WF_mean_func(envs.completed_jobs[i])
            else:
                objectives[i] = WT_max_func(envs.completed_jobs[i])
        objectives[-2] = makespan(envs.jobs)
        objectives[-1] = WT_mean_func(envs.jobs)
        all_objectives[t, :] = objectives
        if objectives[-2] < max_makespan:
            max_makespan = objectives[-2]
            optimal_path = f"{models_save_path}/optimal_model/"
            if not os.path.exists(optimal_path):
                os.makedirs(optimal_path)
            for i in range(envs.num_factories):
                path = f"{optimal_path}/Agents_factory_{i}/"
                save(agents[i].trainer.policy, svae_time, path)
            svae_time += 1
        episode_return.append(episode_reward)
        print("episode {:.2f} : episode length: {:.2f} - > episode return: {:.2f} - > average episode reward: {:.2f} - > WTmean: {:.2f}, WFmean: {:.2f}, WTmax: {:.2f}, makespan: {:.2f}.".format(
                round(t, 2), round(episode_steps, 2), round(episode_reward, 2), round(episode_reward/episode_steps, 2), round(objectives[0], 2), round(objectives[1], 2), round(objectives[2], 2), round(objectives[3], 2)))
    data_path = f"{models_save_path}training_logs.xlsx"
    record_logs(data_path, episode_return, scale_rewards, all_episodes_rewards, all_objectives, logs, episode_return)
    return episode_return, scale_rewards, all_episodes_rewards, all_objectives, logs, rewardModel_loss_log, episode_return

def record_logs(path, episode_return, scale_rewards, all_episodes_rewards, all_objectives, logs, episodic_return):
    # Step 1: all_objectives
    df_objectives = pd.DataFrame(all_objectives, columns=["Objective_1", "Objective_2", "Objective_3", "Objective_4", "Objective_5"])

    # Step 2: all_episodes_rewards
    df_all_episodes_rewards = pd.DataFrame(all_episodes_rewards)
    df_scale_rewards = pd.DataFrame(scale_rewards)
    df_episode_return = pd.DataFrame(episode_return)
    df_episodic_return = pd.DataFrame(episodic_return)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_objectives.to_excel(writer, sheet_name="Objectives", index=False)
        for sheet_name, data in logs.items():
            df_logs = pd.DataFrame(data)
            df_logs.to_excel(writer, sheet_name=sheet_name, index=False)
        df_all_episodes_rewards.to_excel(writer, sheet_name="rewards", index=False)
        df_scale_rewards.to_excel(writer, sheet_name="scaledRewards", index=False)
        df_episode_return.to_excel(writer, sheet_name="episode_return", index=False)
        df_episodic_return.to_excel(writer, sheet_name="episodic_return", index=False)

if __name__ == '__main__':
    episode_return, scale_rewards, all_episodes_rewards, all_objectives, logs, rewardModel_loss_log, episode_return = main()
    # Step 1: all_objectives
    df_objectives = pd.DataFrame(all_objectives, columns=["Objective_1", "Objective_2", "Objective_3", "Objective_4", "Objective_5"])

    # Step 2: all_episodes_rewards
    df_all_episodes_rewards = pd.DataFrame(all_episodes_rewards)
    df_scale_rewards = pd.DataFrame(scale_rewards)

    # Step 3: rewardModel_loss_log
    if isinstance(rewardModel_loss_log[0], dict):
        df_reward_loss = pd.DataFrame(rewardModel_loss_log)
    else:
        df_reward_loss = pd.DataFrame({"Reward_Model_Loss": rewardModel_loss_log})

    dfepisode_return = pd.DataFrame(episode_return)

    # Step 4: record Excel
    path = '/used_trains/training_log/training_logs.xlsx'
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_objectives.to_excel(writer, sheet_name="Objectives", index=False)
        df_reward_loss.to_excel(writer, sheet_name="Reward_Model_Loss", index=False)
        for sheet_name, data in logs.items():
            df_logs = pd.DataFrame(data)
            df_logs.to_excel(writer, sheet_name=sheet_name, index=False)
        df_all_episodes_rewards.to_excel(writer, sheet_name="rewards", index=False)
        df_scale_rewards.to_excel(writer, sheet_name="scaledRewards", index=False)
        dfepisode_return.to_excel(writer, sheet_name="episode_return", index=False)

