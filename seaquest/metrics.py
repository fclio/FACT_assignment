import numpy as np
from scipy.stats import wasserstein_distance


def get_attr_freq(attributions, n_clusters):
    cluster_attributions = np.zeros(n_clusters)
    for attr in attributions:   
        cluster_attributions[attr["responsible_cluster"]] += 1
    cluster_attributions /= np.sum(cluster_attributions)
    
    print(f"Cluster attribution frequencies: {cluster_attributions}")
    
    return cluster_attributions


def get_initial_state_values(sv_explanation_predictions, sv_original_predictions):
    init_state_values = []
    original_v0 = np.mean(np.max(sv_original_predictions, axis=1))
    print("Initial state value estimate")
    print(f"Original policy: {original_v0}")
    
    init_state_values.append(original_v0)
    for idx, predictions in enumerate(sv_explanation_predictions):
        explanation_v0 = np.mean(np.max(predictions, axis=1))
        init_state_values.append(explanation_v0)
        print(f"Cluster {idx}: {explanation_v0}")
        
    return init_state_values


def get_action_value_difference(sv_explanation_predictions, sv_original_predictions):
    print(f"Local Mean Absolute Action-Value Differences:")
    expected_delta_q = []
    
    for cid, explanation_policy in enumerate(sv_explanation_predictions):
        difference = [] 
        for idx, predictions in enumerate(explanation_policy):
            new_action = np.argmax(predictions)
            orig_action = np.argmax(sv_original_predictions[idx])
            delta_q = np.abs(sv_original_predictions[idx][orig_action] - sv_original_predictions[idx][new_action])
            
            difference.append(delta_q)
        expected_delta_q.append(np.mean(difference))
        print(f"Cluster {cid}: {np.mean(difference)}")
    
    return expected_delta_q


def get_action_contrast(sv_explanation_predictions, sv_original_predictions):
    action_contrasts = np.zeros((len(sv_explanation_predictions)))
    
    for idx, orig_prediction in enumerate(sv_original_predictions):
        orig_action = np.argmax(orig_prediction)
        
        for cid, explanation_policy in enumerate(sv_explanation_predictions):
            new_action = np.argmax(explanation_policy[idx])
            if new_action != orig_action:
                action_contrasts[cid] += 1
                
    action_contrasts /= len(sv_original_predictions)
    print(f"Action contrasts: {action_contrasts}")
    return action_contrasts


def get_data_distance(original_data_embedding, compl_dataset_embeddings):
    distances = np.zeros((len(compl_dataset_embeddings)))
    for cid, compl_embedding in enumerate(compl_dataset_embeddings):
        distances[cid] = wasserstein_distance(original_data_embedding, compl_embedding)
    # print("distances ", distances)
    with np.printoptions(precision=5, suppress=True):
        normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())
        print(f"Normalized data distances: {normalized_distances}")
    
    return normalized_distances
    
        
def get_metrics(sv_explanation_predictions, sv_original_predictions, attributions, original_data_embedding, compl_dataset_embeddings):
    """
    Compute all metrics used in the paper.
    
    Args:
    - sv_explanation_predictions: list, shape (num_clusters, num_sub_trajs, num_actions)
    - sv_original_predictions: list, shape (num_sub_trajs, num_actions)
    - attributions: list, list of attributions
    - original_data_embedding: np.array, shape (128,)
    - compl_dataset_embeddings: np.array, shape (num_clusters, 128)
    
    Returns:
    - metrics: dict, dictionary containing all metrics
    """
    metrics = {}
    
    metrics["init_state_values"] = get_initial_state_values(sv_explanation_predictions, sv_original_predictions)
    metrics["expected_delta_q"] = get_action_value_difference(sv_explanation_predictions, sv_original_predictions)
    metrics["action_contrasts"] = get_action_contrast(sv_explanation_predictions, sv_original_predictions)
    metrics["normalized_distances"] = get_data_distance(original_data_embedding, compl_dataset_embeddings)
    metrics["cluster_attr_freq"] = get_attr_freq(attributions, len(sv_explanation_predictions))
        
    return metrics
    