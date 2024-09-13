from Params import configs
from poisson_instance import Instance_Generator
import numpy as np
from mb_agg import *
from models.actor_critic import Job_Mch_Actor, Job_Mch_Critic
from copy import deepcopy
from agent_utils import select_action_mch, eval_actions_mchs
import torch.nn as nn
from validation import validate
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device(configs.device)
class Memory:
    def __init__(self):
        self.fea1_mb = []
        self.fea2_mb = []
        self.fea3_mb = []        
        self.mask_mb = []
        self.mch_time_mb = []
        self.job_time_mb = []
        self.ci_mb = []                
        self.a_m_mb = []        
        self.r_mb = []
        self.done_mb = []
        self.job_mch_logprobs = []

    def clear_memory(self):
        del self.fea1_mb[:]
        del self.fea2_mb[:]
        del self.fea3_mb[:]        
        del self.mask_mb[:]
        del self.mch_time_mb[:]        
        del self.job_time_mb[:]
        del self.ci_mb[:]                
        del self.a_m_mb[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.job_mch_logprobs[:]                


def adv_normalize(adv):
    std = adv.std()
    assert std != 0. and not torch.isnan(std), 'Need nonzero std'
    n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
    return n_advs

class PPO:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 n_j,
                 n_m,
                 ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy_job_mch = Job_Mch_Actor(n_j=n_j,
                                            n_m=n_m,
                                            n_layers_fea = configs.num_mlp_layers_fea,
                                            input_dim_fea = configs.input_dim_fea,
                                            hidden_dim_fea = configs.hidden_dim_fea,
                                            out_dim_fea = configs.out_dim_fea,
                                            n_layers_actor = configs.num_mlp_layers_actor,
                                            input_dim_actor = configs.input_dim_actor,
                                            hidden_dim_actor = configs.hidden_dim_actor,
                                            out_dim_actor = configs.out_dim_actor,
                                            device=device)

        self.critic_job_mch = Job_Mch_Critic(
                                            n_layers_critic = configs.num_mlp_layers_critic,
                                            input_dim_critic = configs.input_dim_critic,
                                            hidden_dim_critic= configs.hidden_dim_critic,
                                            out_dim_critic = configs.out_dim_critic,
                                            device=device)

        self.policy_old_job_mch = deepcopy(self.policy_job_mch)
        self.policy_old_job_mch.load_state_dict(self.policy_job_mch.state_dict())
        self.job_mch_optimizer = torch.optim.Adam(self.policy_job_mch.parameters(), lr=lr)
        self.job_mch_scheduler = torch.optim.lr_scheduler.StepLR(self.job_mch_optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)
        self.MSE = nn.MSELoss()

    def update(self,  memory):
        '''self.policy_job.train()
        self.policy_mch.train()'''
        vloss_coef = configs.vloss_coef
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.r_mb), reversed(memory.done_mb)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)            
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # process each env data
        fea1_mb_t = torch.stack(memory.fea1_mb).to(device)
        fea1_mb_t = fea1_mb_t.reshape(-1, fea1_mb_t.size(-1))

        fea2_mb_t = torch.stack(memory.fea2_mb).to(device)
        fea2_mb_t = fea2_mb_t.reshape(-1, fea2_mb_t.size(-1))

        fea3_mb_t = torch.stack(memory.fea3_mb).to(device)
        fea3_mb_t = fea3_mb_t.reshape(-1, fea3_mb_t.size(-1))

        mask_mb_t=torch.stack(memory.mask_mb).to(device).squeeze()
        a_m_mb_t=torch.stack(memory.a_m_mb).to(device).squeeze()          
        mch_time_mb_t=torch.stack(memory.mch_time_mb).reshape(-1,configs.n_m).to(device).squeeze()        
        job_time_mb_t=torch.stack(memory.job_time_mb).reshape(-1,configs.n_m).to(device).squeeze()        
        ci_mb_t=torch.stack(memory.ci_mb).reshape(-1,configs.n_m).to(device).squeeze()        
        old_job_mch_logprobs_mb_t=torch.stack(memory.job_mch_logprobs).to(device).squeeze()        

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            job_mch_loss_sum = 0
            v_loss_sum = 0
            pi_a_mch, a_mch_pool = self.policy_job_mch(fea1=fea1_mb_t,
                                                        fea2=fea2_mb_t,
                                                        fea3=fea3_mb_t,
                                                        mask=mask_mb_t,
                                                        mch_time=mch_time_mb_t,                                                            
                                                        job_time=job_time_mb_t,
                                                        ci=ci_mb_t)

            vals = self.critic_job_mch(a_mch_pool)
            job_mch_v_loss = self.MSE(vals.squeeze(), rewards)                
            advantages = rewards - vals.view(-1).detach()
            advantages = adv_normalize(advantages)

            job_mch_logprobs, job_mch_ent_loss = eval_actions_mchs(pi_a_mch.squeeze(), a_m_mb_t)
            job_mch_ratios = torch.exp(job_mch_logprobs - old_job_mch_logprobs_mb_t.detach())
            job_mch_surr1 = job_mch_ratios * advantages
            job_mch_surr2 = torch.clamp(job_mch_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            job_mch_p_loss = -1*torch.min(job_mch_surr1, job_mch_surr2).mean()
            job_mch_ent_loss = - job_mch_ent_loss.clone()
            job_mch_loss = ploss_coef * job_mch_p_loss + entloss_coef * job_mch_ent_loss + vloss_coef * job_mch_v_loss
            job_mch_loss_sum += job_mch_loss
            v_loss_sum += job_mch_v_loss

            self.job_mch_optimizer.zero_grad()
            job_mch_loss_sum.mean().backward(retain_graph=True)
            
            self.policy_old_job_mch.load_state_dict(self.policy_job_mch.state_dict())
            if configs.decayflag:
                self.job_mch_scheduler.step()   
            self.job_mch_optimizer.step()

            return job_mch_loss_sum.mean().item(), v_loss_sum.mean().item()


def main():
    import random
    from Configuration_Env2.MODFJSSP_Env_2 import MODFJSSP
    memory = Memory()
    torch.manual_seed(configs.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)        
    np.random.seed(configs.np_seed_train)
    random.seed(configs.np_seed_train)                      
    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
                n_j=configs.n_j,
                n_m=configs.n_m)

    vali_durs = np.load('./DataGen/durs' + str(configs.n_j) + '_' + str(configs.n_m) + '_Seed' + str(configs.np_seed_validation) + '.npy')
    vali_ords = np.load('./DataGen/ords' + str(configs.n_j) + '_' + str(configs.n_m) + '_Seed' + str(configs.np_seed_validation) + '.npy')

   # training loop
    log = []
    vali_makespan = []
    vali_U_ave_1 = []     
    vali_lateness_ave = []

    e_ave=20
    New_insert=5
    machine=5
    DDT=1.2
    max_updates=500   # No. of episodes for training
    V=20   # No. of episodes per validation
    for i_update in range(max_updates):
        Processing_time, A, D, M_num, Op_num, J, O_num, J_num = Instance_Generator(machine, e_ave, New_insert, DDT)
        env = MODFJSSP(J_num, M_num, max(Op_num))                   
        fea1, fea2, fea3, candidate, mask, mch_time, job_time, ci, Lateness_ave, U_ave = env.reset(Processing_time, A, D, M_num, Op_num, J, O_num, J_num)
        while True:
            fea1_tensor = torch.from_numpy(np.copy(fea1)).to(device).type(torch.float32)
            fea2_tensor = torch.from_numpy(np.copy(fea2)).to(device).type(torch.float32)
            fea3_tensor = torch.from_numpy(np.copy(fea3)).to(device).type(torch.float32)                        
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device).type(torch.int64)
            mask_tensor = torch.from_numpy(np.copy(mask)).to(device).reshape(1,-1)
            mch_time_tensor = torch.from_numpy(np.copy(mch_time)).to(device).type(torch.float32).expand(J_num, M_num) 
            job_time_tensor = torch.from_numpy(np.copy(job_time)).to(device).type(torch.float32).expand(J_num, M_num)

            ci_tensor = torch.from_numpy(np.copy(ci)).to(device).type(torch.float32).expand(J_num, M_num)

            with torch.no_grad():
                pi_a_mch, a_mch_pool = ppo.policy_old_job_mch(fea1=fea1_tensor,
                                                            fea2=fea2_tensor,
                                                            fea3=fea3_tensor,
                                                            mask=mask_tensor,
                                                            mch_time=mch_time_tensor,
                                                            job_time=job_time_tensor,
                                                            ci=ci_tensor                                                              
                                                            )

                action, mch, a_m_idx = select_action_mch(pi_a_mch, candidate_tensor, memory)
                # print('action:',action)
                # print('mch:',mch)
                memory.a_m_mb.append(a_m_idx)
                memory.fea1_mb.append(fea1_tensor)
                memory.fea2_mb.append(fea2_tensor)
                memory.fea3_mb.append(fea3_tensor)        
                memory.mask_mb.append(mask_tensor)
                memory.mch_time_mb.append(mch_time_tensor)
                memory.job_time_mb.append(job_time_tensor)
                memory.ci_mb.append(ci_tensor)
                fea1, fea2, fea3, reward, done, candidate, mask, mch_time, job_time, ci, Lateness_ave, U_ave = env.step(action, mch)                
                ep_reward = max(mch_time)
                memory.r_mb.append(reward)
                memory.done_mb.append(done)

            if env.done():
                Makespan = max(torch.squeeze(job_time).numpy())
                # or
                # Makespan = max(mch_time)
                U_ave_1 = 1/U_ave
                Lateness_ave=Lateness_ave
                break

        # ppo.update(memories)
        job_mch_loss, v_loss = ppo.update(memory)
        memory.clear_memory()

        log.append([i_update, Makespan, U_ave_1, Lateness_ave])
        if (i_update + 1) % V == 0:
            file_writing_obj = open('./' + 'log_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            file_writing_obj.write(str(log))

        # log results
        print('Episode {}\t Last makespan: {:.2f}\t Last U_ave_1: {:.2f}\t Last Lateness_ave: {:.2f}\t Mean_Vloss: {:.8f}'.format(
            i_update + 1, Makespan, U_ave_1, Lateness_ave, v_loss))

        if (i_update + 1) % V == 0:
            make_spans, U_ave_1, Lateness_ave = validate(vali_durs, vali_ords, configs.n_j, configs.n_m, ppo.policy_job_mch)
            vali_makespan.append(make_spans)
            vali_U_ave_1.append(U_ave_1)     
            vali_lateness_ave.append(Lateness_ave)
            torch.save(ppo.policy_job_mch.state_dict(), './{}.pth'.format('policy_job_mch'))
            print('The validation quality is:', make_spans, U_ave_1, Lateness_ave)
            file_writing_obj1 = open(
                './' + 'vali_obj1_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            file_writing_obj1.write(str(vali_makespan))

            file_writing_obj2 = open(
                './' + 'vali_obj2_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            file_writing_obj2.write(str(vali_U_ave_1))

            file_writing_obj3 = open(
                './' + 'vali_obj3_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            file_writing_obj3.write(str(vali_lateness_ave)) 

if __name__ == '__main__':
    main()
