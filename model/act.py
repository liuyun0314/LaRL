import copy

import torch
import numpy as np
from torch.distributions import Categorical, Normal
from torch.nn import functional as F

def discrete_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                available_actions, deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)  
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)
    available_actions = available_actions.reshape(batch_size, available_actions.shape[1], available_actions.shape[-1])
    ava = available_actions.clone().detach()

    for i in range(n_agent):
        logit = decoder(shifted_action, obs_rep, obs)[:, i, :]
        logit[ava[:, i, :] == 0] = -1e10
        distri = Categorical(logits=logit)
        if deterministic:
            action = distri.sample()
        else:
            action = distri.probs.argmax(dim=-1)
        action_log = distri.log_prob(action)
        output_action[:, i, :] = action.unsqueeze(-1)
        output_action_log[:, i, :] = action_log.unsqueeze(-1)
        if action.item() != 0:
            ava[:, :, action] = 0
        if i + 1 < n_agent:
            shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)
    return output_action, output_action_log

def discrete_semi_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                available_actions, deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)  
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)
    ava = available_actions.clone().detach()
    ava_actions = torch.zeros((batch_size, n_agent, action_dim))

    for i in range(n_agent):
        ava_actions[:, i, :] = ava
        logit = decoder(shifted_action, obs_rep, obs)[:, i, :]
        logit[ava == 0] = -1e10
        distri = Categorical(logits=logit)
        if np.random.random() < deterministic:
            action = distri.sample()
        else:
            action = distri.probs.argmax(dim=-1)
        action_log = distri.log_prob(action)
        output_action[:, i, :] = action.unsqueeze(-1)
        output_action_log[:, i, :] = action_log.unsqueeze(-1)
        if action.item() != 0:
            index = int((action.item()-1)/n_agent)
            mac_index = (action.item()-1)%n_agent+1
            begin_index = index*n_agent+1
            end_index = begin_index + n_agent
            ava[:, begin_index:end_index] = 0
            indices = torch.arange(mac_index, ava.size(1), n_agent)
            ava[:, indices] = 0
        if torch.all(ava == 0):
            ava[:, 0] = 1
        if i + 1 < n_agent:
            shifted_action[:, i + 1, :] = F.one_hot(action, num_classes=action_dim)
    return output_action, output_action_log, ava_actions

def discrete_single_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                available_actions, deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)   
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)
    ava = available_actions.clone().detach()
    ava_actions = torch.zeros((batch_size, n_agent, action_dim))

    for i in range(n_agent):
        ava_actions[:, i, :] = ava
        logit = decoder(shifted_action, obs_rep, obs)[:, i, :]
        logit[ava == 0] = -1e10
        distri = Categorical(logits=logit)
        if np.random.random() < deterministic:
            action = distri.sample()
        else:
            action = distri.probs.argmax(dim=-1)
        action_log = distri.log_prob(action)
        output_action[:, i, :] = action.unsqueeze(-1)
        output_action_log[:, i, :] = action_log.unsqueeze(-1)
        if action.item() != 0:
            index = int((action.item()-1)/n_agent)
            mac_index = (action.item()-1)%n_agent+1
            begin_index = index*n_agent+1
            end_index = begin_index + n_agent
            ava[:, begin_index:end_index] = 0
            indices = torch.arange(mac_index, ava.size(1), n_agent)
            ava[:, indices] = 0
        if torch.all(ava == 0):
            ava[:, 0] = 1
        if i + 1 < n_agent:
            shifted_action[:, i + 1, :] = F.one_hot(action, num_classes=action_dim)
    return output_action, output_action_log, ava_actions

def discrete_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv,
                          available_actions):

    one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim)  
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    shifted_action[:, 1:, :] = one_hot_action[:, :-1, :]
    available_actions = available_actions.reshape(batch_size, n_agent, action_dim)

    ava = available_actions.clone().detach()

    logit = decoder(shifted_action, obs_rep, obs)
    logit[ava == 0] = -1e10
    distri = Categorical(logits=logit)
    action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
    entropy = distri.entropy().unsqueeze(-1)
    return action_log, entropy
#
def discrete_simultaneous_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                available_actions, deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)  
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)
    available_actions = available_actions.reshape(batch_size, available_actions.shape[1], available_actions.shape[-1])
    ava = available_actions.clone().detach()

    logits = decoder(shifted_action, obs_rep, obs)
    for i in range(n_agent):
        logit = logits[:, i, :]
        logit[ava[:, i, :] == 0] = -1e10
        distri = Categorical(logits=logit)
        if np.random.random() < deterministic:
            action = distri.sample()
        else:
            action = distri.probs.argmax(dim=-1)
        action_log = distri.log_prob(action)
        output_action[:, i, :] = action.unsqueeze(-1)
        output_action_log[:, i, :] = action_log.unsqueeze(-1)
        if action.item() != 0:
            ava[:, :, action] = 0
        if i + 1 < n_agent:
            shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)
    return output_action, output_action_log
