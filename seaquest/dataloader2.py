import numpy as np
import gzip
import torch

from decision_transformer_atari import GPTConfig, GPT



datasets_names = ["observation", "action", "reward", "terminal"]
datasets = {}
for dataset_name in datasets_names:
    with gzip.open("seaquest/data/"+dataset_name+".gz", 'rb') as f:
        datasets[dataset_name] = np.load(f, allow_pickle=False)

max_traj_len = 500
num_trajs = 40
trajs = []
current_traj = []
count = 0
max_count = 0
for obs, act, rew, ter in zip(datasets["observation"], datasets["action"], datasets["reward"], datasets["terminal"]):
    current_traj.append((obs, act, rew))
    count += 1
    
    if ter:
        if count <= max_traj_len:
            trajs.append(current_traj)
        count = 0
        current_traj = []
trajs = trajs[:num_trajs]

observation_traj = torch.zeros((len(trajs), max_traj_len, 4, 84, 84), dtype=torch.int8)
action_traj = torch.zeros((len(trajs), max_traj_len, 1), dtype=torch.int8)
reward_traj = torch.zeros((len(trajs), max_traj_len, 1), dtype=torch.int8)
len_traj = torch.zeros((len(trajs), 1), dtype=torch.int64)

for idx1, traj in enumerate(trajs):
    len_traj[idx1] = len(traj)
    for idx2, (obs, act, rew) in enumerate(traj):
        observation_traj[idx1, idx2] = torch.tensor(obs)
        action_traj[idx1, idx2] = torch.tensor(act)
        reward_traj[idx1, idx2] = torch.tensor(rew)
        

vocab_size = 18
block_size = 90
model_type = "reward_conditioned"
timesteps = 2719

mconf = GPTConfig(
    vocab_size,
    block_size,
    n_layer=6,
    n_head=8,
    n_embd=128,
    model_type=model_type,
    max_timestep=timesteps,
)
model = GPT(mconf)

model.load_pretrained("checkpoints/Seaquest_123.pth", cpu=True)

emb = model(observation_traj, action_traj, rtgs=reward_traj, timesteps=torch.tensor([[[timesteps]]]))
        

