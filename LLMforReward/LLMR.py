import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import wandb
from torch.distributions import Beta, Normal
from LLMforReward.reward_model import Reward_Model, RRDRewardDecomposer, RDRewardDecomposer
from LLMforReward.LLMreward import LLM_generate_reward
from LLMforReward.prompt.prompt_template import jsp_prompt
import os
import json

ROOT_DIR = os.getcwd()
class LLMRewardDecomposer(RRDRewardDecomposer):
    def __init__(self, args, id):
        super(LLMRewardDecomposer, self).__init__(args)
        self.args = args
        self.K = self.args.rrd_k
        root_dir = os.path.dirname(os.path.abspath(__file__))
        self.rd_save_dir = f"{root_dir}/jspEnvironments/"
        self.id = id
        with open(self.args.root_dir, 'r') as f:
            dataSet = f.readlines()
        self.prompt = jsp_prompt(self.args, dataSet, 'DJobShop')
        self.load_rd_functions()
        self.get_factor_num()

        self.reward_model = Reward_Model(input_dim=self.factor_num)
        self.reward_model.to('cuda')

        self.optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=3e-4)
        self.loss_fn = nn.MSELoss(reduction='mean')

    def save(self, save_dir, episode):
        torch.save(self.reward_model.state_dict(), str(save_dir) + "/reward_model_" + str(episode) + ".pt")

    def load_rd_functions(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.load('/LLMforReward/cfg/config.yaml')

        response_dir = self.rd_save_dir + 'response_code/'
        if not os.path.exists(response_dir):
            os.makedirs(response_dir)
        while not os.path.exists(os.path.join(response_dir, f'response_{self.id}.npy')):
            LLM_generate_reward(cfg, factor=(not self.args.direct_generate))

        rd_responses = np.load(os.path.join(response_dir, f'response_{self.id}.npy'), allow_pickle=True)
        rd_functions = []
        for i in range(len(rd_responses)):
            func = json.loads(rd_responses[i])['Functions']
            rd_functions.append(func)
        self.rd_functions = rd_functions

    def get_factor_num(self):
        factor_dir = self.rd_save_dir + 'factor_num/'
        factor_num = np.load(os.path.join(factor_dir, f'factor_num_{self.id}.npy'), allow_pickle=True)
        self.factor_num = factor_num[0]

    def func_forward(self, states, actions, next_states):
        func = self.rd_functions[0]
        func = func.encode().decode('unicode_escape')
        device = next(self.reward_model.parameters()).device
        raw_shape = states.shape

        namespace = {'np': np, 'torch': torch}
        if "@torch.jit.script\n" in func:
            func = func.replace("@torch.jit.script\n", "")
        exec(func, namespace)
        evaluation_func = namespace['evaluation_func']
        factor_scores = evaluation_func(states, actions)  # list of numpy array, shape: batch,1
        cat_factor_scores = np.concatenate(factor_scores, axis=-1)  # bs,nfactors
        tensor_scores = torch.FloatTensor(cat_factor_scores).to(device)
        return tensor_scores

    def forward(self, states, actions, next_states):
        states = self.func_forward(states, actions, next_states)
        if self.args.direct_generate:
            rewards = states
        else:
            rewards = self.reward_model(states)  # bs,t,-1 -> bs,t,1
        if rewards.shape[0] == 1:
            return rewards.cpu().item()
        else:
            return rewards

    def train_reward(self, replay_buffer, batch_size=256):
        train_info = {}
        reward_model_loss = 0
        if self.reward_model is not None:
            traj_state, traj_action, traj_next_state, traj_reward, traj_dense_reward, traj_not_done, traj_episode_return, traj_episode_length, _ = replay_buffer.sample_traj(
                int(batch_size // self.args.rrd_k))
            B, T, _, _ = traj_state.shape
            traj_state = traj_state.view(B * T, 20, -1)
            traj_action = traj_action.view(B * T, -1)  
            traj_next_state = traj_next_state.view(B * T, 20, -1)
            reward_model_loss = self.update(traj_state, traj_action, traj_next_state, traj_episode_return, traj_episode_length)
            train_info['reward_model_loss'] = reward_model_loss
        return reward_model_loss
class LLMRDRewardDecomposer(RDRewardDecomposer):
    def __init__(self, args):
        super(LLMRDRewardDecomposer, self).__init__(args)
        self.args = args
        self.K = self.args.rrd_k
        self.rd_save_dir = f"{ROOT_DIR}/jspEnvironments/response_code/"
        self.id = args.reward_id
        with open(self.args.root_dir, 'r') as f:
            dataSet = f.readlines()
        self.prompt = jsp_prompt(self.args, dataSet, 'DJobShop')
        self.load_rd_functions()
        self.get_factor_num()

        self.reward_model = Reward_Model(input_dim=self.factor_num)
        self.reward_model.to('cuda')

        self.optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=3e-4)
        self.loss_fn = nn.MSELoss(reduction='mean')

    def load_rd_functions(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.load("cfg/config.yaml")
        reposnse_dir = self.rd_save_dir + 'response_code/'
        if not os.path.exists(reposnse_dir):
            os.makedirs(reposnse_dir)
        while not os.path.exists(os.path.join(reposnse_dir, f'response_{self.id}.npy')):
            LLM_generate_reward(cfg, factor=(not self.args.direct_generate))

        rd_responses = np.load(os.path.join(reposnse_dir, f'response_{self.id}.npy'), allow_pickle=True)
        rd_functions = []
        for i in range(len(rd_responses)):
            func = json.loads(rd_responses[i])['Functions']
            rd_functions.append(func)
        self.rd_functions = rd_functions

    def get_factor_num(self):
        factor_dir = self.rd_save_dir + 'factor_num/'
        self.factor_num = np.load(os.path.join(factor_dir, f'factor_num_{self.id}.npy'), allow_pickle=True)

    def func_forward(self, states, actions, next_states):
        func = self.rd_functions[0]
        # device = states.device
        device = next(self.reward_model.parameters()).device
        raw_shape = states.shape
        states = states.cpu().numpy().reshape(-1, states.shape[-1])
        actions = actions.cpu().numpy().reshape(-1, actions.shape[-1])
        next_states = next_states.cpu().numpy().reshape(-1, states.shape[-1])

        namespace = {}
        exec(func, namespace)
        evaluation_func = namespace['evaluation_func']
        factor_scores = evaluation_func(states, actions)  # , next_states)
        cat_factor_scores = np.concatenate(factor_scores, axis=-1)  # bs,nfactors
        if len(raw_shape) == 3:    # transform to tensor
            tensor_scores = torch.FloatTensor(cat_factor_scores).to(device).reshape(raw_shape[0], raw_shape[1], -1)
        else:
            tensor_scores = torch.FloatTensor(cat_factor_scores).to(device)
        return tensor_scores

    def forward(self, states, actions, next_states):
        states = self.func_forward(states, actions, next_states)
        rewards = self.reward_model(states)  # bs,t,-1 -> bs,t,1
        return rewards.cpu().item()