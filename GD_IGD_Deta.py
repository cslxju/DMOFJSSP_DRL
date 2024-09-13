import numpy as np
from metric import *

#50_20_8, 50_20_12, 50_20_16, 100_30_8, 100_30_12, 100_30_16, 200_40_8, 200_40_12, 200_40_16
e_ave=50    #50, 100, 200
New_insert=20  #20, 30, 40
machine=8  #8, 12, 16

igd_row = []
gd_row = []
deta_row=[]
popfun_array = []
file_path1='./results/ours_U/result_{}_{}_{}_save.npz'.format(e_ave, New_insert, machine)
popfun1 = np.load(file_path1)["arr_0"]
# print('popfun1:',popfun1)
popfun_array.append(popfun1)
file_path2='./results/ours_P/result_{}_{}_{}_save.npz'.format(e_ave, New_insert, machine)
popfun2 = np.load(file_path2)["arr_0"]
# print('popfun1:',popfun2)
popfun_array.append(popfun2)
file_path3='./results/nsga3/result_{}_{}_{}_save.npz'.format(e_ave, New_insert, machine)
popfun3 = np.load(file_path3)["arr_0"]
# print('popfun3:',popfun3)
popfun_array.append(popfun3)
file_path4='./results/THDQN/result_{}_{}_{}_save.npz'.format(e_ave, New_insert, machine)
popfun4 = np.load(file_path4)["arr_0"]
# print('popfun4:',popfun4)
popfun_array.append(popfun4)
file_path5='./results/DMDDQN/result_{}_{}_{}_save.npz'.format(e_ave, New_insert, machine)
popfun5 = np.load(file_path5)["arr_0"]
# print('popfun5:',popfun5)
popfun_array.append(popfun5)
file_path6='./results/DDQN/result_{}_{}_{}_save.npz'.format(e_ave, New_insert, machine)
popfun6 = np.load(file_path6)["arr_0"]
# print('popfun6:',popfun6)
popfun_array.append(popfun6)

P = np.concatenate([obj for obj in popfun_array], axis=0)
# print('P:',P)
v_max = P.max(axis=0)
v_min = P.min(axis=0)

# 归一化
P = (P - np.tile(v_min, (P.shape[0], 1))) / (np.tile(v_max, (P.shape[0], 1)) - np.tile(v_min, (P.shape[0], 1)) + 1e-6)
norm_popfun = []
begin = 0
# norm_objs记录对比方法的目标函数归一化后的目标函数值
for popfun in popfun_array:
    norm_popfun.append(P[begin:(begin + popfun.shape[0])])
    begin = begin + popfun.shape[0]
pareto = get_ps(P)  # 得到对比方法组成的pareto最优解

for i, norm_obj in enumerate(norm_popfun):
    gd = GD(norm_obj, pareto) # 计算算法的GD值
    gd='%.2e'% gd
    igd = IGD(pareto, norm_obj) # 计算算法的IGD值
    igd='%.2e'% igd
    deta=spread(norm_obj, pareto) # 计算算法的spread值
    deta='%.2e'% deta    
    gd_row.append(gd)
    igd_row.append(igd)
    deta_row.append(deta)

print('GD:', gd_row)
print('IGD:', igd_row)
print('Deta:', deta_row)


