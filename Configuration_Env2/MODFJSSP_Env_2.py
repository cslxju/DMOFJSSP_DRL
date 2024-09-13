import random
import numpy as np
from copy import deepcopy
import torch
from Params import configs
from EndTime_a import EndTime_a
from updateEndTimeLB import calEndTimeLB
import statistics
class MODFJSSP():
    def __init__(self,
                 n_j,
                 n_m,
                 n_op):
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.number_of_operations_per_job = n_op
        self.number_of_tasks = n_j * n_op
        self.first_col = []
        self.last_col = []

    def done(self):
            if np.all(self.mask == True):
                return True
            return False

    def step(self, action, mch_a):
        if action not in self.partial_sol_sequeence:
            self.partial_sol_sequeence.append(action)
            # print('action:',action)
            row = action // self.number_of_operations_per_job#取整除
            col = action % self.number_of_operations_per_job#取余数
            # print('row:',row)
            # print('col:',col)
            dur_a = self.Processing_time[row][col][mch_a]
            EndTime_a(row=row, col=col, mch_a=mch_a, 
            dur_a=dur_a, mch_T=self.mch_T, mch_time=self.mch_time, job_time=self.job_time, temp1=self.temp1)
                                
            self.LBm = calEndTimeLB(self.temp1, self.input_mean)
            mask_last_col = np.full(shape=(self.number_of_machines), fill_value=1, dtype=bool)
            if action not in self.last_col:
                self.omega[action // self.number_of_operations_per_job] += 1
                row_add = (action+1) // self.number_of_operations_per_job
                col_add = (action+1) % self.number_of_operations_per_job#取余数               
                self.mask[row_add] = self.mask_mch[row_add][col_add]
            else:
                self.mask[row] = mask_last_col
             
            a_last_col = self.LBm[:, -1]
            for n in self.omega:
                row_o = n // self.number_of_operations_per_job#取整除
                col_o = n % self.number_of_operations_per_job#取余数
                self.fea1[row] = self.Processing_time[row][col]
                for m in range(self.number_of_machines):
                    self.fea2[row_o][m] = self.LBm[row_o][col_o]                              
                    self.fea3[row_o][m] = a_last_col[row_o] - self.LBm[row_o][col_o]

            C_d=self.LBm[:,-1] - self.di

            self.Lateness_ave = statistics.mean([max(C_d[i],0) for i in range(self.number_of_jobs)])
            self.UK[mch_a]=self.mch_T[mch_a]/self.mch_time[mch_a]
            self.U_ave=sum(self.UK)/self.number_of_machines

            ######reward_2
            if self.Lateness_ave<self.Lateness_ave_end or self.U_ave > 1.005*self.U_ave_end:
                rt=1
            elif self.Lateness_ave>self.Lateness_ave_end or self.U_ave < self.U_ave_end:
                rt=-1
            else:
                rt=0
                
            self.reward=rt

            self.max_endTime = self.LBm.max()
            self.Lateness_ave_end=self.Lateness_ave
            self.U_ave_end=self.U_ave            

            self.job_time_Fea = torch.from_numpy(np.copy(self.job_time)).float().unsqueeze(-1)

            self.ci = np.absolute(self.di - self.job_time)
            self.ci_Fea = torch.from_numpy(np.copy(self.ci)).float().unsqueeze(-1) 


        return self.fea1,self.fea2,self.fea3, self.reward, self.done(), self.omega, self.mask, self.mch_time, self.job_time_Fea, self.ci_Fea, self.Lateness_ave, self.U_ave


    def reset(self, Processing_time, A, D, M_num, Op_num, J, O_num, J_num):
        self.Processing_time=Processing_time
        self.partial_sol_sequeence = []
        first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
        self.first_col = np.array(first_col)
        self.last_col = np.array(last_col)

        self.mask_mch = np.full(shape=(self.number_of_jobs,self.number_of_operations_per_job, self.number_of_machines), fill_value=0,
                            dtype=bool)        

        mean = []
        for i in range(self.number_of_jobs):
            dur_mean = []
            for j in range(self.number_of_operations_per_job):
                if j< Op_num[i]:

                    dur_ij=np.array(Processing_time[i][j])
                    durmch = dur_ij[np.where(dur_ij >= 0)]
                    dur_mean.append(durmch.mean())
                    self.mask_mch[i][j] = [1 if k < 0 else 0 for k in dur_ij]
                else:
                    dur_mean.append(0)
                    self.mask_mch[i][j] = [1 for k in range(M_num)]
            mean.append(dur_mean)                                                                                        
        self.input_mean =  np.array(mean)
        self.LBm = np.cumsum(self.input_mean,-1)

        C_d=self.LBm[:,-1] - np.array(D)
        self.Lateness_ave = statistics.mean([max(C_d[i],0) for i in range(self.number_of_jobs)])

        self.UK=[0 for i in range(self.number_of_machines)]
        self.U_ave=sum(self.UK)/self.number_of_machines

        self.max_endTime = self.LBm.max()
        self.Lateness_ave_end=self.Lateness_ave
        self.U_ave_end=0

        self.fea1 = np.zeros((self.number_of_jobs,self.number_of_machines), dtype=float)
        self.fea2 = np.zeros((self.number_of_jobs,self.number_of_machines), dtype=float)
        self.fea3 = np.zeros((self.number_of_jobs,self.number_of_machines), dtype=float)

        self.omega = self.first_col.astype(np.int64)
        a_last_col = self.LBm[:, -1]

        mask = []
        for n in self.omega:
            row = n // self.number_of_operations_per_job#取整除
            col = n % self.number_of_operations_per_job#取余数
            self.fea1[row] = Processing_time[row][col]


            mask.append(self.mask_mch[row][col])
            for m in range(self.number_of_machines):
                self.fea2[row][m] = self.LBm[row][col]                            
                self.fea3[row][m] = a_last_col[row] - self.LBm[row][col]
    
        self.mask = np.array(mask)
        self.temp1 = np.zeros((self.number_of_jobs,self.number_of_operations_per_job))

        self.mch_time = np.zeros(self.number_of_machines)
        self.job_time = np.array(A)

        self.mch_T = np.zeros(self.number_of_machines)       
        self.job_time_Fea = torch.from_numpy(np.copy(self.job_time)).float().unsqueeze(-1)

        self.di = np.array(D)
        self.di_Fea = torch.from_numpy(np.copy(self.di)).float().unsqueeze(-1)

        self.ci = np.absolute(self.di - self.job_time)
        self.ci_Fea = torch.from_numpy(np.copy(self.ci)).float().unsqueeze(-1)            
        return self.fea1,self.fea2,self.fea3, self.omega, self.mask, self.mch_time, self.job_time_Fea, self.ci_Fea, self.Lateness_ave, self.U_ave

       