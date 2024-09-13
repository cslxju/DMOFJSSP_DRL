def validate(vali_durs, vali_ords, n_j, n_m, policy_job_mch):

    from Configuration_Env1.MODFJSSP_Env_1 import MODFJSSP
    from mb_agg import g_pool_cal
    from agent_utils import greedy_select_action_mch, sample_select_action_mch
    import numpy as np
    import torch
    from Params import configs

    device = torch.device(configs.device)
    make_spans = []
    mean_lates = []
    mean_flows = []
    # rollout using model
    for ind, dur in enumerate(vali_durs):
        ord=vali_ords[ind]
        n_op_max=n_m
        env = MODFJSSP(n_j, n_m, n_op_max)
        fea1, fea2, fea3, candidate, mask, mch_time, job_time, ci, mean_lateness, mean_flow_time = env.reset(dur,ord)
        while True:
            fea1_tensor = torch.from_numpy(np.copy(fea1)).to(device).type(torch.float32)
            fea2_tensor = torch.from_numpy(np.copy(fea2)).to(device).type(torch.float32)
            fea3_tensor = torch.from_numpy(np.copy(fea3)).to(device).type(torch.float32)                        
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device).type(torch.int64)
            mask_tensor = torch.from_numpy(np.copy(mask)).to(device).reshape(1,-1)
            mch_time_tensor = torch.from_numpy(np.copy(mch_time)).to(device).type(torch.float32).expand(n_j, n_m)
            job_time_tensor = torch.from_numpy(np.copy(job_time)).to(device).type(torch.float32).expand(n_j, n_m)
            ci_tensor = torch.from_numpy(np.copy(ci)).to(device).type(torch.float32).expand(n_j, n_m)                                     
            with torch.no_grad():
                pi_a_mch, a_mch_pool = policy_job_mch(fea1=fea1_tensor,
                                                    fea2=fea2_tensor,
                                                    fea3=fea3_tensor,
                                                    mask=mask_tensor,
                                                    mch_time=mch_time_tensor,
                                                    job_time=job_time_tensor,
                                                    ci=ci_tensor                                                     
                                                    )
                # action, mch = sample_select_action_mch(pi_a_mch, candidate_tensor, n_m)
                action, mch = greedy_select_action_mch(pi_a_mch, candidate_tensor, n_m)
                # print('action:',action)
                # print('mch:',mch)
                fea1, fea2, fea3, reward, done, candidate, mask, mch_time, job_time, ci, mean_lateness, mean_flow_time = env.step(action, mch)                   
                if done:
                    Makespan = max(torch.squeeze(job_time).numpy())
                    #or Makespan = max(mch_time)
                    Mean_lateness = mean_lateness
                    Mean_flow_time = mean_flow_time

                    break
        make_spans.append(Makespan)
        mean_lates.append(Mean_lateness)
        mean_flows.append(Mean_flow_time)
    return np.array(make_spans).mean(), np.array(mean_lates).mean(), np.array(mean_flows).mean()

