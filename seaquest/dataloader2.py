import numpy as np
import gzip
import torch
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from decision_transformer_atari import GPTConfig, GPT



datasets_names = ["observation", "action", "reward", "terminal"]
datasets = {}
for dataset_name in datasets_names:
    with gzip.open("data/"+dataset_name+".gz", 'rb') as f:
        datasets[dataset_name] = np.load(f, allow_pickle=False)

max_traj_len = 30
num_trajs = 20
trajs = []
current_traj = []
num_sar = 0
total_trajs = 0
for obs, act, rew, ter in zip(datasets["observation"], datasets["action"], datasets["reward"], datasets["terminal"]):
    current_traj.append((obs, act, rew))
    num_sar += 1
    
    if num_sar == max_traj_len:
        trajs.append(current_traj)
        num_sar = 0
        current_traj = []
    elif ter:
        total_trajs += 1
        while num_sar < max_traj_len:
            current_traj.append((0,0,0))
            num_sar += 1
        trajs.append(current_traj)
        num_sar = 0
        current_traj = []
    if total_trajs == num_trajs:
        break

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


# traj_emb = []
# split = 50
# observation_traj = observation_traj.reshape((-1, split, 30, 4, 84, 84))
# action_traj = action_traj.reshape((-1, split, 30, 1))
# reward_traj = reward_traj.reshape((-1, split, 30, 1))
# len_traj = len_traj.reshape((-1, split, 1))

# for i in range(observation_traj.shape[0]):
#     _, _, traj_emb_ = model(observation_traj[i], action_traj[i], rtgs=reward_traj[i], timesteps=len_traj[i])
#     traj_emb.append(traj_emb_.detach().numpy())

_, _, traj_emb = model(observation_traj, action_traj, rtgs=reward_traj, timesteps=torch.tensor([[[timesteps]]]))

traj_emb_no_grad = traj_emb.detach().numpy().reshape((-1, 128))
print(traj_emb_no_grad.shape) # (40, 30, 128)
pca = PCA(n_components=2)
pca_traj_embeddings = pca.fit_transform(traj_emb_no_grad)
plotting_data = {'feature 1': pca_traj_embeddings[:, 0], 'feature 2': pca_traj_embeddings[:, 1]}
df = pd.DataFrame(plotting_data)

plt.figure()
sns.scatterplot(x='feature 1', y='feature 2', hue='rewards', data=df)
plt.title('PCA: Trajectory Embeddings')
plt.legend()
plt.show()