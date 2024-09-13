from torch.distributions.categorical import Categorical
from Params import configs
import torch
device = torch.device(configs.device)

def select_action_mch(p, candidate, memory):

    dist = Categorical(p.squeeze())
    s = dist.sample()
    if memory is not None: memory.job_mch_logprobs.append(dist.log_prob(s))
    row = s // configs.n_m
    col = s % configs.n_m
    action = candidate[row].item()
    mch = col.item()

    
  
    return action, mch, s

# evaluate the actions
def eval_actions_mchs(p, actions):
    dist = Categorical(p)
    log_a = dist.log_prob(actions).reshape(-1)
    entropy_a = dist.entropy().mean()
    return log_a, entropy_a


# select action method for test
def sample_select_action_mch(p, candidate, n_m):

    dist = Categorical(p.squeeze())
    s = dist.sample()
    row = s // n_m
    col = s % n_m
    action = candidate[row].item()
    mch = col.item()

    return action, mch

# select action method for test
def greedy_select_action_mch(p, candidate, n_m):
    _, index = p.squeeze().max(0)
    row = index // n_m
    col = index % n_m
    action = candidate[row].item()
    mch = col.item()

    return action, mch

