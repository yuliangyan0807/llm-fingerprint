import matplotlib.pyplot as plt
import seaborn as sns
from model_list import MODEL_LIST, MODEL_LABEL

def distance_matrix_plot(result_dict,
                         save_dir
                         ):
    
    num_models = len(MODEL_LIST)
    
    fig, axs = plt.subplots(1, 3, figsize=(48, 16))
    
    sns.heatmap(result_dict['edit_distance_matrix'], annot=True, fmt=".2f", ax=axs[0], cmap="coolwarm", cbar=True,
                xticklabels=MODEL_LABEL, yticklabels=MODEL_LABEL)
    axs[0].set_title("Edit Distance Matrix")
    
    sns.heatmap(result_dict['jaccard_simlarity_matrix'], annot=True, fmt=".2f", ax=axs[1], cmap="coolwarm", cbar=True,
                xticklabels=MODEL_LABEL, yticklabels=MODEL_LABEL)
    axs[1].set_title("Jaccard Simlarity Matrix")
    
    sns.heatmap(result_dict['entropy_simlarity_matrix'], annot=True, fmt=".2f", ax=axs[2], cmap="coolwarm", cbar=True,
                xticklabels=MODEL_LABEL, yticklabels=MODEL_LABEL)
    axs[2].set_title("Entropy Simlarity Matrix")
    
    plt.tight_layout()
    plt.savefig(save_dir, format='png', dpi=300)
    plt.show()