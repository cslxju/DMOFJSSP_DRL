def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path)]

if __name__ == '__main__':

    import numpy as np
    import torch
    import time
    import os
    import argparse
    from Params import configs
    from torch.utils.data import DataLoader
    from models.actor_critic import Job_Mch_Actor
    import time
    from validation import validate
    device = torch.device(configs.device) 

    class PPO:
        def __init__(self,
                    lr,
                    gamma,
                    k_epochs,
                    eps_clip,
                    n_j,
                    n_m
                    ):

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
    

    #10_5, 20_5, 50_5, 20_10, 50_10, 100_10, 50_15, 100_15, 200_15
    n_j=10  #10, 20, 50, 100, 200
    n_m=5   #5, 10, 15
    f=1     #1, 0.8, 0.6, 0.4 
    c=1.5   #2, 1.5, 1.2
    seed_test=30

    New_durs = np.load('./Configuration_Env1/instances_test1/f_{}_c_{}_data/New_durs{}_{}_f_{}_c_{}_Seed{}'.format(f, c, n_j, n_m, f, c, seed_test) + '.npy')
    New_ords = np.load('./Configuration_Env1/instances_test1/f_{}_c_{}_data/New_ords{}_{}_f_{}_c_{}_Seed{}'.format(f, c, n_j, n_m, f, c, seed_test) + '.npy')

    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
                n_j=n_j,
                n_m=n_m)

    filepath = 'Saved_network'
    ########################## Selecting the policy model trained under the uniform distribution or under the poisson distribution

    filepath = os.path.join(filepath, 'MODFJSSP_U_f_1_c_1.5')    # The policy model trained under the uniform distribution
    # or 
    # filepath = os.path.join(filepath, 'MODFJSSP_P_e_20_DDT_1.2')    # The policy model trained under the poisson distribution
    filepaths = get_imlist(filepath)
    ppo.policy_job_mch.load_state_dict(torch.load(filepaths[0]))


    t1 = time.time()
    make_span, mean_late, mean_flow = validate(New_durs, New_ords, n_j, n_m, ppo.policy_job_mch)
    t2 = time.time()
    print('test_makespan:', round(make_span,2))
    print('test_lateness:', round(mean_late,2))    
    print('test_flow_time:', round(mean_flow,2))
    print('Times_mean:',(t2-t1)/30)