import numpy as np
import torch
# from off_trainer import MultiTD3, MATD3
from torch.multiprocessing import Manager
from used_trains.buffer import Buffer
from used_trains.Neuroevolution import SSNE
import used_trains.utils as mod
from model.Actor3 import MultiAgentTransformer as Actor
# from core.policy_trainer import MATTrainer as TrainAlgo
from used_trains.MATTrainer3 import MATTrainer as TrainAlgo
from used_trains.sharedBuffer2 import SharedReplayBuffer

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Agents:
    def __init__(self, args, num_agents, id):
        self.args = args
        self.device = args.device if torch.cuda.is_available() else 'cpu'
        self.num_agents = num_agents
        self.training_loss = []
        self.n_rollout_threads = args.n_rollout_threads
        self.action_type = 'Discrete'
        act_dim = args.num_new_jobs + args.num_tasks * args.num_warmup_jobs + 1 
        #### INITIALIZE training ALGO #####
        self.trainer = TrainAlgo(args, self.num_agents, id, device=self.device)

        #### Rollout Actor is a template used for MP #####
        if args.use_gpu:
            self.trainer.policy.transformer.to(self.device)
        # Initalize buffer
        self.buffer = SharedReplayBuffer(args, self.num_agents, 1, args.act_dim)
        ###Best Policy HOF####
        self.champ_ind = 0

    def update_parameters(self):
        self.trainer.prep_rollout()
        if self.buffer.available_actions is None:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                         np.concatenate(self.buffer.obs[-1]),
                                                         np.concatenate(self.buffer.masks[-1]))
        else:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                         np.concatenate(self.buffer.obs[-1]),
                                                         np.concatenate(self.buffer.masks[-1]),
                                                         np.concatenate(self.buffer.available_actions[-1]))
        next_values = np.array(np.split(next_values, self.n_rollout_threads))

        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.training_loss.append(train_infos['policy_loss'])
        self.buffer.after_update()
        return train_infos

    def update_paras(self, i, critic, next_share_obs, next_obs, num_factories):
        self.trainer.prep_rollout()
        next_values = critic(next_share_obs, next_obs, num_factories)
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
        self.trainer.prep_training()
        train_infos = self.trainer.train1(self.buffer)
        self.training_loss.append(train_infos['policy_loss'])
        self.buffer.after_update()
        return train_infos

class TestAgent:
    """Learner object encapsulating a local learner
    		Parameters:
    		algo_name (str): Algorithm Identifier
    		state_dim (int): State size
    		action_dim (int): Action size
    		actor_lr (float): Actor learning rate
    		critic_lr (float): Critic learning rate
    		gamma (float): DIscount rate
    		tau (float): Target network sync generate
    		init_w (bool): Use kaimling normal to initialize?
    		**td3args (**kwargs): arguments for TD3 algo
    	"""

    def __init__(self, args):
        self.args = args
        self.num_agents = args.num_machines
        self.device = args.device if torch.cuda.is_available() else 'cpu'
        self.action_type = 'Discrete'
        act_dim = args.num_new_jobs + args.num_tasks * args.num_warmup_jobs + 1
        #### Rollout Actor is a template used for MP #####
        self.manager = Manager()  # provide a manager for shared objects between processes
        self.rollout_actor = self.manager.list()
        self.rollout_actor.append(Actor(args.share_obs_dim, args.obs_dim, args.reasoning_dim, act_dim, args.num_machines,
                                      n_block=args.n_block, n_embd=args.n_embd, n_head=args.n_head,
                                      encode_state=args.encode_state, device=self.device,
                                      action_type=self.action_type, dec_actor=args.dec_actor,
                                      share_actor=args.share_actor))

    def make_champ_team(self, agent):
        if self.args.popn_size <= 1:  
            agent.update_rollout_actor()
            mod.hard_update(self.rollout_actor[0], agent.rollout_actor[0])
        else:
            mod.hard_update(self.rollout_actor[0], agent.popn[agent.champ_ind])


