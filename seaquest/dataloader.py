# Dataloader for the SeaQuest Dataset

# run these commands to 
# pip install ale-py
# pip install git+https://github.com/takuseno/d4rl-atari
# pip install gym[accept-rom-license]

import gym
import d4rl_atari
import numpy as np

class Dataloader():
    def __init__(self) -> None:
        pass

    def load_sq_data(self, version='Seaquest-ram-v0'):
        """
        Load the seaquest dataset from the d4rl-atari repo.
        """
        print("Loading Seaquest Dataset")

        env = gym.make(version) 
        dataset = env.get_dataset()
        return dataset
    
    def dataset_info(self, dataset):
        print("Observations:")
        print(dataset['observations'].shape) # observation data in (1000000, 1, 84, 84)
        print(type(dataset['observations']))

        print("Actions:")
        print(np.unique(dataset['actions'])) # action data in (1000000,)
        print(dataset['actions'].shape)

        print("Rewards:")
        print(np.unique(dataset['rewards'])) # reward data in (1000000,
        print(dataset['rewards'].shape)

        print("Terminals:")
        print(np.unique(dataset['terminals']))  # terminal data in (1000000,)
        print(dataset['terminals'].shape)

if __name__ == "__main__":
    dl = Dataloader()
    dl.load_sq_data()   
    dl.dataset_info()

