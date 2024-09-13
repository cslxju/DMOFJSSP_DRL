import argparse

parser = argparse.ArgumentParser(description='Arguments for ppo_dmofjssp')
# args for device
parser.add_argument('--device', type=str, default="cuda", help='Training device')
# args for env  10*5  20*5  50*5  20*10  50*10  100*10  50*15  100*15  200*15      
parser.add_argument('--n_j', type=int, default=10, help='Number of jobs of instance')
parser.add_argument('--n_m', type=int, default=5, help='Number of machines of instance')
parser.add_argument('--low', type=int, default=5/2, help='LB of duration')
parser.add_argument('--high', type=int, default=5*2, help='UB of duration')
parser.add_argument('--O_LB', type=int, default=1, help='LB of the number of operations')
parser.add_argument('--O_UB', type=int, default=5, help='UB of the number of operations')

parser.add_argument('--f', type=float, default=1, help='f') # Problem flexibility: 0.2(low), 0.5(moderate), 1(high)
parser.add_argument('--c', type=int, default=1.5, help='c') #Due date tightness factor: 1.2(tight), 1.5(moderate), 2(loose)


parser.add_argument('--np_seed_train', type=int, default=200, help='Seed for numpy for training')
parser.add_argument('--np_seed_validation', type=int, default=100, help='Seed for numpy for validation')
parser.add_argument('--np_seed_test', type=int, default=30, help='Seed for numpy for testing')
parser.add_argument('--torch_seed', type=int, default=600, help='Seed for torch')
parser.add_argument('--et_normalize_coef', type=int, default=1000, help='Normalizing constant for feature LBs (end time), normalization way: fea/constant')

# args for network
parser.add_argument('--num_mlp_layers_fea', type=int, default=3, help='No. of layers of MLP in fea network')
parser.add_argument('--input_dim_fea', type=int, default=6, help='input dim of MLP in fea network')
parser.add_argument('--hidden_dim_fea', type=int, default=128, help='hidden dim of MLP in fea network')
parser.add_argument('--out_dim_fea', type=int, default=64, help='out dim of MLP in fea network')

parser.add_argument('--num_mlp_layers_actor', type=int, default=2, help='No. of layers of MLP in actor network')
parser.add_argument('--input_dim_actor', type=int, default=128, help='input dim of MLP in actor network')
parser.add_argument('--hidden_dim_actor', type=int, default=64, help='hidden dim of MLP in actor network')
parser.add_argument('--out_dim_actor', type=int, default=1, help='out dim of MLP in actor network')

parser.add_argument('--num_mlp_layers_critic', type=int, default=2, help='No. of layers of MLP in critic network')
parser.add_argument('--input_dim_critic', type=int, default=64, help='input dim of MLP in critic network')
parser.add_argument('--hidden_dim_critic', type=int, default=64, help='hidden dim of MLP in critic network')
parser.add_argument('--out_dim_critic', type=int, default=1, help='out dim of MLP in critic network')

# args for PPO
parser.add_argument('--lr', type=float, default=1e-3, help='lr')
parser.add_argument('--decayflag', type=bool, default=False, help='lr decayflag')
parser.add_argument('--decay_step_size', type=int, default=2000, help='decay_step_size')
parser.add_argument('--decay_ratio', type=float, default=0.9, help='decay_ratio, e.g. 0.9, 0.95')
parser.add_argument('--gamma', type=float, default=1, help='discount factor')
parser.add_argument('--k_epochs', type=int, default=1, help='update policy for K epochs')
parser.add_argument('--eps_clip', type=float, default=0.2, help='clip parameter for PPO')
parser.add_argument('--vloss_coef', type=float, default=1, help='critic loss coefficient')
parser.add_argument('--ploss_coef', type=float, default=2, help='policy loss coefficient')
parser.add_argument('--entloss_coef', type=float, default=0.01, help='entropy loss coefficient')

configs = parser.parse_args()
