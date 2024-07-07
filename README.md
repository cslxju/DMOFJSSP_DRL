# DMOFJSSP-DRL
Dynamic scheduling for multi-objective flexible job shop via deep reinforcement learning

### Installation
Installation source 
create -n dmofjssp-drl-env python=3.7.0 
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch

### train

```
# The policy model trained under the uniform distribution
training the policy model python3 python train_U.py

# The policy model trained under the poisson distribution
training the policy model python3 python train_P.py

```

Note that there should be a validation set of the corresponding size in ```./DataGen/durs and ./DataGen/ords```.

### test

```
Reproduce result in paper python3 test_learned_on_benchmark_1.py or 

test_learned_on_benchmark_2.py 

for test instances

```

