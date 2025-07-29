import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from model.util import check, init
from torch.nn import Identity
from torch.multiprocessing import Manager
from model.act import discrete_autoregreesive_act, discrete_parallel_act, discrete_simultaneous_act

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

class MLPs(nn.Module):
    '''
    MLPs in operation node embedding
    '''
    def __init__(self, W_sizes_ope, hidden_size_ope, out_size_ope, num_head, dropout):
        '''
        The multi-head and dropout mechanisms are not actually used in the final experiment.
        :param W_sizes_ope: A list of the dimension of input vector for each type,
        including [machine, operation (pre), operation (sub), operation (self)]
        :param hidden_size_ope: hidden dimensions of the MLPs
        :param out_size_ope: dimension of the embedding of operation nodes
        '''
        super(MLPs, self).__init__()
        self.in_sizes_ope = W_sizes_ope
        self.hidden_size_ope = hidden_size_ope
        self.out_size_ope = out_size_ope
        self.num_head = num_head
        self.dropout = dropout
        self.gnn_layers = nn.ModuleList()

        # A total of five MLPs and MLP_0 (self.project) aggregates information from other MLPs
        for i in range(len(self.in_sizes_ope)):
            self.gnn_layers.append(MLPsim(self.in_sizes_ope[i], self.out_size_ope, self.hidden_size_ope, self.num_head,
                                          self.dropout, self.dropout))
        self.project = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.out_size_ope * len(self.in_sizes_ope), self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.out_size_ope),
        )

    def forward(self, ope_ma_adj_batch, feats):
        '''
        :param ope_ma_adj_batch: Adjacency matrix of operation and machine nodes
        :param ope_pre_adj_batch: Adjacency matrix of operation and pre-operation nodes
        :param ope_sub_adj_batch: Adjacency matrix of operation and sub-operation nodes
        :param batch_idxes: Uncompleted instances
        :param feats: Contains operation, machine and edge features
        '''
        h = (feats[1], feats[0], feats[0], feats[0])
        B, N_op, _ = ope_ma_adj_batch.shape
        device = ope_ma_adj_batch.device
        self_loop = torch.eye(N_op, dtype=torch.int64, device=device).unsqueeze(0).expand(B, N_op, N_op)
        adj = (ope_ma_adj_batch, self_loop)
        MLP_embeddings = []
        for i in range(len(adj)):
            MLP_embeddings.append(self.gnn_layers[i](h[i], adj[i]))
        MLP_embedding_in = torch.cat(MLP_embeddings, dim=-1)
        mu_ij_prime = self.project(MLP_embedding_in)
        return mu_ij_prime

class MLPsim(nn.Module):
    '''
    Part of operation node embedding
    '''
    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_dim,
                 num_head,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False):
        '''
        :param in_feats: Dimension of the input vectors of the MLPs
        :param out_feats: Dimension of the output (operation embedding) of the MLPs
        :param hidden_dim: Hidden dimensions of the MLPs
        :param num_head: Number of heads
        '''
        super(MLPsim, self).__init__()
        self._num_heads = num_head
        self._in_feats = in_feats
        self._out_feats = out_feats

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.project = nn.Sequential(
            nn.Linear(self._in_feats, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self._out_feats),
        )

        # Deprecated in final experiment
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, self._num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

    def forward(self, feat, adj):
        a = adj.unsqueeze(-1) * feat.unsqueeze(-3)
        b = torch.sum(a, dim=-2)
        c = self.project(b)
        return c


class GATedge(nn.Module):
    '''
    Machine node embedding
    '''
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_head,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        '''
        :param in_feats: tuple, input dimension of (operation node, machine node)
        :param out_feats: Dimension of the output (machine embedding)
        :param num_head: Number of heads
        '''
        super(GATedge, self).__init__()
        self._num_heads = num_head  # single head is used in the actual experiment
        self._in_src_feats = in_feats[0]
        self._in_dst_feats = in_feats[1]
        self._out_feats = out_feats

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_head, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_head, bias=False)
            self.fc_edge = nn.Linear(
                1, out_feats * num_head, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_head, bias=False)
        self.attn_l = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.attn_r = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.attn_e = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        # Deprecated in final experiment
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_head * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)

    def forward(self, ope_ma_adj_batch, feat):
        # Two linear transformations are used for the machine nodes and operation nodes, respective
        # In linear transformation, an W^O (\in R^{d \times 7}) for \mu_{ijk} is equivalent to W^{O'} (\in R^{d \times 6}) and W^E (\in R^{d \times 1}) for the nodes and edges respectively
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            if not hasattr(self, 'fc_src'):
                self.fc_src, self.fc_dst = self.fc, self.fc
            feat_src = self.fc_src(h_src)
            feat_dst = self.fc_dst(h_dst)
        else:
            # Deprecated in final experiment
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self._num_heads, self._out_feats)
        feat_edge = self.fc_edge(feat[2].unsqueeze(-1))

        # Calculate attention coefficients
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        ee = (feat_edge * self.attn_l).sum(dim=-1).unsqueeze(-1)

        adj = ope_ma_adj_batch.unsqueeze(-1)  # (B, N_op, N_ma, 1)
        el_add_ee = adj * el.unsqueeze(-2) + ee
        a = el_add_ee + adj * er.unsqueeze(-3)

        eijk = self.leaky_relu(a)
        ekk = self.leaky_relu(er + er)

        B, N_op, N_ma = ope_ma_adj_batch.shape
        device = ope_ma_adj_batch.device

        mask1 = (ope_ma_adj_batch.unsqueeze(-1) == 1)  # (B, N_op, N_ma, 1)
        mask2 = torch.ones((B, 1, N_ma, 1), dtype=torch.bool, device=device)  # self-loop for machines
        mask = torch.cat((mask1, mask2), dim=1)  # (B, N_op+1, N_ma, 1)

        e = torch.cat((eijk, ekk.unsqueeze(-3)), dim=-3)
        e[~mask] = float('-inf')
        alpha = F.softmax(e.squeeze(-1), dim=-2)
        alpha_ijk = alpha[..., :-1, :]
        alpha_kk = alpha[..., -1, :].unsqueeze(-2)

        # Calculate an return machine embedding
        Wmu_ijk = feat_edge + feat_src.unsqueeze(-2)
        a = Wmu_ijk * alpha_ijk.unsqueeze(-1)
        b = torch.sum(a, dim=-3)
        c = feat_dst * alpha_kk.squeeze().unsqueeze(-1)
        nu_k_prime = torch.sigmoid(b+c)
        return nu_k_prime


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1))
        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y

class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # self.attn = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x

class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * 512), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * 512, n_embd))
        )

    def forward(self, x, rep_enc):
        # Masked Multi-Head Attention + Add & Norm
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x

class Encoder(nn.Module):

    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state, dropout):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state
        self.dropout = dropout
        self.out_size_ma = 8
        self.in_size_ope = 5
        self.hidden_size_ope = 128
        self.out_size_ope = 8
        self.num_heads = 1
        self.in_size_ma = 5
        self.s_embd = 16
        self.num_layers = 2
        self.activation = nn.Tanh()

        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), self.activation)
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), self.activation)

        self.layer_norm = nn.LayerNorm(self.s_embd)
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(self.s_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), self.activation, nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1)))

        self.operation_embedding = MLPs([self.out_size_ma, self.in_size_ope],
                                self.hidden_size_ope, self.out_size_ope, self.num_heads, self.dropout)
        self.edg_embedding = GATedge((self.in_size_ope, self.in_size_ma), self.out_size_ma, self.num_heads, self.dropout, self.dropout, activation=F.elu)
        self.linears = torch.nn.ModuleList()
        self.linears.append(nn.Linear(self.s_embd, self.hidden_size_ope))
        for layer in range(self.num_layers - 1):
            self.linears.append(nn.Linear(self.hidden_size_ope, self.hidden_size_ope))

    def forward(self, state, obs, indices):
        # indices = (10, 15, 20)
        # state: (batch, num_machines, in_src_feats)
        # obs: (batch, num_machines, in_src_feats)
        batch_index = 0
        raw_PT = obs[:, :, :indices[0]]
        raw_opes = obs[:, :, indices[0]:indices[1]]
        raw_machines = obs[:, :, indices[1]:indices[2]]
        features = (raw_opes, raw_machines, raw_PT)

        h_mas = self.edg_embedding(raw_PT, features)
        features = (features[0], h_mas, features[2])
        h_opes = self.operation_embedding(raw_PT, features)

        x = torch.cat((h_opes, h_mas), dim=-1)

        rep = self.blocks(self.layer_norm(x))
        for layer in range(self.num_layers):
            rep = torch.tanh(self.linears[layer](rep))
        v_loc = self.head(rep)

        return v_loc, rep

class Decoder(nn.Module):

    def __init__(self, obs_dim, reasoning_dim, action_dim, n_block, n_embd, n_head, n_agent,
                 action_type='Discrete', dec_actor=True, share_actor=False):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.reasoning_dim = reasoning_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type
        self.activation = nn.Tanh()

        if self.dec_actor:
            if self.share_actor:
                self.mlp = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), self.activation,
                                         nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, n_embd), activate=True), self.activation,
                                         nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, action_dim)))

            else:
                self.mlp = nn.ModuleList()
                for n in range(n_agent):
                    actor = nn.Sequential(nn.LayerNorm(obs_dim),
                                          init_(nn.Linear(obs_dim, n_embd), activate=True), self.activation,
                                          nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, n_embd), activate=True), self.activation,
                                          nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, action_dim)))
                    self.mlp.append(actor)
        else:
            if action_type == 'Discrete':
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
                                                    self.activation)  # include a linnear layer and a GELU activation function
            else:
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True), self.activation)
            self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                             init_(nn.Linear(obs_dim, n_embd), activate=True), self.activation)
            self.ln = nn.LayerNorm(n_embd)
            self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
            self.head = nn.Sequential(init_(nn.Linear(n_embd, 2048), activate=True), self.activation, nn.LayerNorm(2048),
                                      init_(nn.Linear(2048, self.action_dim)))

            self.output_mlp = ResidualMLP(input_dim=128, output_dim=5001, hidden_dims=[512, 1024])
            self.mlp = nn.Sequential(nn.LayerNorm(n_embd),
                                     init_(nn.Linear(n_embd, n_embd), activate=True), self.activation, nn.LayerNorm(n_embd),
                                     init_(nn.Linear(n_embd, n_embd), activate=True), self.activation, nn.LayerNorm(n_embd),
                                     init_(nn.Linear(n_embd, action_dim)))

    # state, action, and return
    def forward(self, obs_rep):
        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)
        logit = self.mlp(obs_rep)
        logit = logit.mean(dim=1, keepdim=True)
        logits = logit.reshape(logit.shape[0], logit.shape[-1])
        return logits

class ResidualMLP(nn.Module):
    def __init__(self, input_dim=128, output_dim=5001, hidden_dims=[512, 1024], activation=nn.GELU):
        """
        Args:
            input_dim:     
            output_dim:    
            hidden_dims:   
            activation:     
        """
        super(ResidualMLP, self).__init__()

        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(dims[i], dims[i + 1]),
                    activation(),
                    nn.LayerNorm(dims[i + 1])
                )
            )

        self.mlp = nn.Sequential(*layers)
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None
        self.output_layer = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        """
        Args:
            x: shape [batch_size, n_agent, input_dim]
        Returns:
            out: shape [batch_size, n_agent, output_dim]
        """
        residual = x if self.residual_proj is None else self.residual_proj(x)

        x = self.mlp(x)
        x = self.output_layer(x)
        x = x + residual

        return x

class MultiAgentTransformer(nn.Module):

    def __init__(self, state_dim, obs_dim, reasoning_dim, action_dim, n_agent, n_block, n_embd, n_head, encode_state, dropout, device, action_type='Discrete', dec_actor=False, share_actor=False):
        super(MultiAgentTransformer, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device if torch.cuda.is_available() else 'cpu'
        ### add by ly
        self.state_dim = state_dim
        self.dropout = dropout

        self.encoder = Encoder(state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state, dropout)
        self.decoder = Decoder(obs_dim, reasoning_dim, action_dim, n_block, n_embd, n_head, n_agent,
                               action_type, dec_actor=dec_actor, share_actor=share_actor)
        ####
        # self.to(device)

    def forward(self, state, obs, action, available_actions):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        available_actions = np.squeeze(available_actions, axis=1)
        available_actions = torch.tensor(available_actions)

        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], self.state_dim), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(state)[0]
        indices = (10, 15, 20)
        v_loc, obs_rep = self.encoder(state, obs, indices)
        logits = self.decoder(obs_rep)


        if available_actions is not None:
            logits[available_actions == 0] = -1e10

        dist = Categorical(logits=logits)

        action = action.long()
        action_log = dist.log_prob(action.squeeze())
        entropy = dist.entropy()
        return action_log, v_loc, entropy

    # def get_actions(self, state, obs, available_actions, deterministic=False):
    def get_actions(self, state, obs, available_actions, deterministic):
        # state unused

        ########## With GPU ########
        ori_shape = np.shape(obs)
        state = check(state).to(self.device)
        obs = check(obs).to(self.device)
        available_actions = check(available_actions).to(self.device)

        batch_size = np.shape(obs)[0]
        indices = (10, 15, 20)
        v_loc, obs_rep = self.encoder(state, obs, indices)
        global_logits = self.decoder(obs_rep)

        if available_actions is not None:
            global_logits[available_actions == 0] = -1e10

        dist = Categorical(logits=global_logits)
        if deterministic:
            actions = dist.probs.argmax(dim=-1)
        else:
            actions = dist.sample()

        action_log_probs = dist.log_prob(actions)

        return actions.squeeze(-1), action_log_probs.squeeze(-1), v_loc, available_actions

    def get_values(self, state, obs, available_actions=None):
        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], self.state_dim))

        state = check(state)
        obs = check(obs)
        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        indices = (10, 15, 20)
        v_tot, obs_rep = self.encoder(state, obs, indices)
        return v_tot

class Critic(nn.Module):
    def __init__(self, state_dim, obs_dim, n_agent, n_block, n_embd, n_head, encode_state, dropout, device):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device if torch.cuda.is_available() else 'cpu'
        ### add by ly
        self.state_dim = state_dim
        self.encoder = Encoder(state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state, dropout)

    def forward(self, state, obs, num_factories):

        indices = (10, 15, 20)
        obs = check(obs).to(**self.tpdv)
        state = check(state).to(**self.tpdv)
        v_tots = []
        per_agent = np.shape(obs)[1]//num_factories
        for i in range(num_factories):
            v_tot, _ = self.encoder(state, obs[:,i*per_agent:(i+1)*per_agent, :], indices)
            v_tots.append(v_tot)
        v_cat = torch.cat(v_tots, dim=1)
        value = v_cat.mean(dim=1, keepdim=True)
        return value