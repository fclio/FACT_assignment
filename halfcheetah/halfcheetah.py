import gym
import d4rl # Import required to register environments, you may need to also import the submodule
import numpy as np
import d3rlpy

def main():
    dataset_d3, env = d3rlpy.datasets.get_dataset("halfcheetah-medium-v2")

    print(dataset_d3.observations.shape)
    print(dataset_d3.actions.shape)
    print(dataset_d3.rewards.shape)
        # print(dataset_d3.next_observations.shape)
    print(dataset_d3.terminals.shape)
    print(dataset_d3.terminals.sum()) # no

    env = gym.make('halfcheetah-medium-v2')
    dataset_d4 = d4rl.qlearning_dataset(env)

    print(dataset_d4['observations'].shape)
    print(dataset_d4['rewards'].shape)
    print(dataset_d4['terminals'].shape)
    print(dataset_d4['actions'].shape)

    print(dataset_d4['rewards'][1])
    print(dataset_d3.rewards[1])


    print(np.allclose(dataset_d3.actions[100], dataset_d4['actions'][100]))

    for j in range(1000):
        for i in range(999):
            if dataset_d4['rewards'][j * 999 + i] != dataset_d3.rewards[j * 1000 + i]: print("yo", i)
        # if not np.allclose(dataset_d3.observations[i], dataset_d4['observations'][i]): print('obs ongelijk')
        # if not np.allclose(dataset_d3.rewards[i], dataset_d4['rewards'][i]): print('obs ongelijk')
        # if not np.allclose(dataset_d3.actions[i], dataset_d4['actions'][i]): print('obs ongelijk')

    sac = d3rlpy.algos.SAC(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=3e-4,
        batch_size=256)

    print(sac)
    sac.fit(dataset_d3, n_steps=10000)

    actions = sac.predict(dataset_d3.observations[0])

    print(actions)


    return
    print('yo!')

    # Create the environment
    env = gym.make('halfcheetah-medium-v2')

    # d4rl abides by the OpenAI gym interface
    env.reset()
    env.step(env.action_space.sample())

    # Each task is associated with a dataset
    # dataset contains observations, actions, rewards, terminals, and infos
    # dataset = env.get_dataset()
    dataset = d4rl.qlearning_dataset(env)

    print(dataset.keys()) # An N x dim_observation Numpy array of observations
    print(dataset['rewards'].shape) # An N x dim_observation Numpy array of observations


    first_traj = []
    for i in range(50000):
        if not np.allclose(dataset['next_observations'][i],dataset['observations'][i+1]): print("yo", i, dataset['terminals'][i])
        # if dataset['terminals'][i] == True:
        #     print('traj ended at', i)
        #     break

        # first_traj.append((dataset['observations'][i],
        #                    dataset['actions'][i],
        #                    dataset['rewards'][i],
        #                    dataset['next_observations'][i]))
    # print(first_traj)


    # print(dataset['rewards'].shape) # An N x dim_observation Numpy array of observations

    # Alternatively, use d4rl.qlearning_dataset which
    # also adds next_observations.

    # import d3rlpy

    # # dataset, env = d3rlpy.datasets.get_dataset("halfcheetah-medium")

    # # prepare algorithm
    # # sac = d3rlpy.algos.SAC().create(device="cpu")

    # sac = d3rlpy.algos.SACConfig(
    #     actor_learning_rate=3e-4,
    #     critic_learning_rate=3e-4,
    #     temp_learning_rate=3e-4,
    #     batch_size=256,
    # ).create(device='cpu')


    # # train offline
    # # sac.fit(dataset, n_steps=1000)


    # # ready to control
    # actions = sac.predict(0)

if __name__ == "__main__":
    main()