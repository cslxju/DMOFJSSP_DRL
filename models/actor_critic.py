import torch.nn as nn
from models.mlp import MLP_Fea, MLPActor, MLPCritic
import torch.nn.functional as F
import torch
from Params import configs


class Job_Mch_Actor(nn.Module):
    def __init__(self,
                 n_j,
                 n_m,
                 n_layers_fea,
                 input_dim_fea,
                 hidden_dim_fea,
                 out_dim_fea,
                 n_layers_actor,
                 input_dim_actor,
                 hidden_dim_actor,
                 out_dim_actor,
                 device):
        super(Job_Mch_Actor,self).__init__()
        self.n_j = n_j
        self.n_m = n_m
        self.device=device
        self.hidden_size=out_dim_fea
        self.bn = torch.nn.BatchNorm1d(out_dim_fea).to(device)

        self.fc = MLP_Fea(n_layers_fea, input_dim_fea, hidden_dim_fea, out_dim_fea).to(device)
        self.actor = MLPActor(n_layers_actor, input_dim_actor, hidden_dim_actor, out_dim_actor).to(device)

    def forward(self,
                fea1,    
                fea2,
                fea3,                                
                mask,                
                mch_time,
                job_time,
                ci               
                ):
        fea1 = fea1 / configs.et_normalize_coef
        fea2 = fea2 / configs.et_normalize_coef
        fea3 = fea3 / configs.et_normalize_coef      
        mch_time = mch_time / configs.et_normalize_coef
        job_time = job_time / configs.et_normalize_coef
        ci = ci / configs.et_normalize_coef

        # print('fea1:\n',fea1)
        # print('fea2:\n',fea2)    
        # print('fea3:\n',fea3) 
        # print('mch_time:\n',mch_time) 
        # print('job_time:\n',job_time)
        # print('ci:\n',ci)
                
        feature = torch.cat([fea1.unsqueeze(-1), fea2.unsqueeze(-1), fea3.unsqueeze(-1), mch_time.unsqueeze(-1), job_time.unsqueeze(-1), ci.unsqueeze(-1)], -1)
        feature_job_mch = self.bn(self.fc(feature).reshape(-1, self.hidden_size)).reshape(-1,self.n_j*self.n_m,self.hidden_size)
        job_mch_pool = feature_job_mch.mean(dim=1)
        job_mch_pool_repeated = job_mch_pool.unsqueeze(1).expand_as(feature_job_mch)        
        concateFea = torch.cat((feature_job_mch, job_mch_pool_repeated), dim=-1)
        job_mch_scores = self.actor(concateFea)
        job_mch_scores = job_mch_scores.squeeze(-1) * 10           
        job_mch_scores = job_mch_scores.masked_fill(mask.bool(), float("-inf"))             
        pi_job_mch = F.softmax(job_mch_scores, dim=1)        

        return pi_job_mch, job_mch_pool

class Job_Mch_Critic(nn.Module):
    def __init__(self,
                 n_layers_critic,
                 input_dim_critic,
                 hidden_dim_critic,
                 out_dim_critic,
                 device
                 ):
        super(Job_Mch_Critic, self).__init__()
        self.critic = MLPCritic(n_layers_critic, input_dim_critic, hidden_dim_critic, out_dim_critic).to(device)
    def forward(self,
                h_pooled,
                ):
        concateFea = h_pooled           
        val = self.critic(concateFea)

        return  val
