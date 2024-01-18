from agents.SAC_Discrete import SAC_Discrete
from agents.utilities.Config import Config

from d3rlpy.algos import DiscreteSAC, DiscreteSACConfig

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

    disc_sac_config = DiscreteSACConfig(

    )
    
    disc_sac = DiscreteSAC(
        config=disc_sac_config,
        device="cpu"
    )

    print(disc_sac)
    # sac_discrete.fit(dataset_d3, n_steps=10000)

    # actions = sac.predict(dataset_d3.observations[0])

    # print(actions)

    
if __name__ == "__main__":
    get_agent()