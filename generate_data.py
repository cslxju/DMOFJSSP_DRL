import numpy as np
from uniform_instance import uni_instance_gen
from Params import configs
#10_5, 20_5, 50_5, 20_10, 50_10, 100_10, 50_15, 100_15, 200_15
# f: 1, 0.8, 0.6, 0.4
# c: 2, 1.5, 1.2
n_j=10
n_m=5
batch_size = 100
seed = 100
f=1
c=1.5
np.random.seed(seed)
New_durs = []
New_ords=[]
for _ in range(batch_size):
    New_dur, New_ord, n_op_max = uni_instance_gen(n_j, n_m, configs.O_LB, configs.O_UB, configs.low, configs.high, f, configs.c, flg=1)
    New_durs.append(New_dur)
    New_ords.append(New_ord)   
New_durs = np.array(New_durs)
New_ords = np.array(New_ords)

print('durs_shape:',New_durs.shape)
print('ords_shape:',New_ords.shape)
np.save('durs{}_{}_Seed{}.npy'.format(n_j, n_m,  seed), New_durs)
np.save('ords{}_{}_Seed{}.npy'.format(n_j, n_m, seed), New_ords)