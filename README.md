# DMOFJSSP-DRL
Dynamic scheduling for multi-objective flexible job shop via deep reinforcement learning

### Requirements
- Python = 3.7.0
- Pytorch = 1.6.0
- Torchvision = 0.7.0

### Installation 
conda create -n dmofjssp-env python=3.7.0 
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch

### Train

```
# The policy model trained under the uniform distribution
1. Training the policy model
python3 python train_U.py

2. Putting the trained model "policy_job_mch.pth" into the path "./Saved_network/MODFJSSP_U_f_1_c_1.5"


# The policy model trained under the poisson distribution
1. Training the policy model
python3 python train_P.py

2. Putting the trained model "policy_job_mch.pth" into the path "./Saved_network/MODFJSSP_P_e_20_DDT_1.2"
```

Note that there should be a validation set of the corresponding size in ```./DataGen/durs and ./DataGen/ords```.

### Test

```
# For test dataset 1, reproduce result in paper for test instances.
python3 python test_learned_on_benchmark_1.py

1. Note that the instance scales are 10*5, 20*5, 50*5, 20*10, 50*10, 100*10, 50*15, 100*15, and 200*15.
2. Taking the instance 10*5 as an example, select a policy model trained under uniform or poisson distribution(Ours_U or Ours_P).
3. Getting the F1, F2 and F3 values in paper


# For test dataset 2, reproduce result in paper for test instances.
python3 python test_learned_on_benchmark_2.py

1. Note that the instance scales are 20*8, 20*12, 20*16, 30*8, 30*12, 30*16, 40*8, 40*12, and 40*16；Besides, lambda=50, 100, 200.
2. Taking the instance 20*8 with lambda=50 as an example, select a policy model trained under uniform or poisson distribution(Ours_U or Ours_P).
3. Putting the result with the ".npz" format into the path "./results".
4. Getting the GD, IGD and Deta values in paper
python3 python GD_IGD_Deta.py

```

