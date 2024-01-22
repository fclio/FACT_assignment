from d3rlpy.algos import DiscreteSAC
import d3rlpy.algos as Algos
from d3rlpy.datasets import MDPDataset
import gzip
import numpy as np
import time
import gym
from d3rlpy.datasets import get_dataset
import numpy as np
from run import ACTION_DICT


def get_agent():
    
    # config = Config()
    # config.seed = 1
    # config.num_episodes_to_run = 450
    # # config.file_to_save_data_results = "results/data_and_graphs/Cart_Pole_Results_Data.pkl"
    # # config.file_to_save_results_graph = "results/data_and_graphs/Cart_Pole_Results_Graph.png"
    # config.show_solution_score = False
    # config.visualise_individual_results = False
    # config.visualise_overall_agent_results = True
    # config.standard_deviation_results = 1.0
    # config.runs_per_agent = 1
    # config.use_GPU = False
    # config.overwrite_existing_results_file = False
    # config.randomise_random_seed = True
    # config.save_model = True

    # datasets_names = ["observation", "action", "reward", "terminal"]
    # datasets = {}
    # for dataset_name in datasets_names:
    #     with gzip.open("data/"+dataset_name+".gz", 'rb') as f:
    #         datasets[dataset_name] = np.load(f, allow_pickle=False)
    
    # dataset, env = get_dataset('seaquest-mixed-v4')
    # # # env = gym.make('seaquest-mixed-v4')

    # # # print(datasets["observation"][0].shape)
    
    # discrete_sac = DiscreteSAC(
    #     actor_learning_rate=3e-4,
    #     critic_learning_rate=3e-4,
    #     temp_learning_rate=3e-4,
    #     batch_size=256,
    #     n_steps=100000, 
    #     use_gpu=False
    # )
    # # # # print(datasets["observation"][0][0].shape)
    # # # # print(len(datasets["observation"][0][0].shape))
    # # # # discrete_sac._create_impl((84,), 1)
    # discrete_sac.build_with_env(env)
    # # # # discrete_sac.build_with_dataset(MDPDataset(datasets["observation"], datasets["action"], datasets["reward"], datasets["terminal"]))
    # discrete_sac.load_model(fname="checkpoints/agent_c{}.pt".format(0))

    # actions = []
    # start= time.time()

    
    # for obs in dataset.observations[:1000]:
    #     pred = discrete_sac.predict([obs])
    #     actions.append(pred[0])
    # print(actions)
    # print(time.time()-start)
    # # agent_predictions = [12, 11, 10, 12, 1, 5, 16, 5]
    # # original_action = [11]
    # # indices = np.where(np.array(agent_predictions) != original_action[0])[0]
    # # print(indices)
    
    attributions = np.load("data/attributions.npy", allow_pickle=True)

    for attribution in attributions:
        if attribution["orig_act"] == attribution["new_act"]:
            print("--"*10)
            print(attribution)
            print("--"*10)
    
    # from scipy.stats import wasserstein_distance
    
    # idx= 0
    # len_clusters = 8
    
    # original_predictions = [[13]]
    # explanation_predictions = [[17],[11],[12],[13],[13],[5],[16],[5]]
        
    # original_action = original_predictions[idx][0]
    # agent_predictions = []
    # for predictions in explanation_predictions:
    #     agent_predictions.append(predictions[idx])
    
    # cluster_distance = []
    # alternative_actions = []
    # for cluster_idx in range(len(explanation_predictions)):
    #         if agent_predictions[cluster_idx] != original_action:
    #             # cluster_distance.append(wasserstein_distance(original_data_embedding, compl_dataset_embeddings[cluster_idx]))
    #             cluster_distance.append(1)
    #             # alternative_actions.append(cluster_idx)
    #         else: 
    #             cluster_distance.append(1e9)
    # print(cluster_distance)
    # responsible_cluster_id = np.argsort(cluster_distance)[0]
    # responsible_action = agent_predictions[responsible_cluster_id]
    # print('-'*10)
    # print(f'State - {idx}')
    # print(f'Distance - {cluster_distance[responsible_cluster_id]}')
    # print(f'Original Actions -{ACTION_DICT[original_action]}')
    # print(f'New Action - {ACTION_DICT[responsible_action]}')

    # print(f'Responsible data combination - data id {responsible_cluster_id}')


    
if __name__ == "__main__":
    get_agent()