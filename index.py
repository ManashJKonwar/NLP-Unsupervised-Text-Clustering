__author__ = "konwar.m"
__copyright__ = "Copyright 2023, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import pandas as pd
from tqdm import tqdm 
tqdm.pandas()
from hyperopt import hp

from src.cluster_generator import *
from src.text_preprocesser import *
from src.text_vectorizer import *
from src.cluster_optimization import *
from src.text_dimension_reduction import *
from utility.data_collection import *

if __name__ == '__main__':
    category = "reviews_Cell_Phones_and_Accessories_5"
    cluster_data = None
    text_column = "reviewText"
    vectorized_column = "reduced_vectorized_text"
    sample_frac = 0.01
    model_embedding = "sentence-transformers/all-MiniLM-L6-v2"
    
    if not os.path.exists(os.path.join("input", category.lower() + ".csv")):
        cluster_data = getDF(category=category)
        cluster_data.to_csv(os.path.join("input", category.lower() + ".csv"))
    else:
        cluster_data = pd.read_csv(os.path.join("input", category.lower() + ".csv"))   
        
    # Removing nan rows based on review text
    cluster_data = cluster_data[cluster_data[text_column].notna()].reset_index(drop=True)
    
    # Sampling
    cluster_data = cluster_data.sample(frac=sample_frac).reset_index(drop=True)
    
    # Preprocessing Pipeline
    cluster_data["preprocessed_text"] = cluster_data[text_column].progress_apply(lambda x: preprocess_text(x))
    
    # Vectorization Pipeline
    x_sbert = encoded_text(
                    data=cluster_data, 
                    col_name="preprocessed_text", 
                    model_name=model_embedding
                )
    cluster_data["vectorized_text"] = x_sbert.tolist()
    
    # Optimization Pipeline
    hspace = {
        "n_neighbors": hp.choice('n_neighbors', range(3,16)),
        "n_components": hp.choice('n_components', range(3,16)),
        "min_cluster_size": hp.choice('min_cluster_size', range(10,30)),
        "random_state": 42
    }
    label_lower = 30
    label_upper = 100
    max_evals = 50
    
    best_params_use, best_clusters_use, best_embeddings_use, trials_use = bayesian_search(
                                                                                embeddings=x_sbert, 
                                                                                space=hspace, 
                                                                                label_lower=label_lower, 
                                                                                label_upper=label_upper, 
                                                                                max_evals=max_evals
                                                                            )
    
    # Dimensionality Reduction Pipeline
    cluster_data[vectorized_column] = reduce_dimensions(
                                                    n_neighbors=best_params_use['n_neighbors'],
                                                    n_components=best_params_use['n_components'],
                                                    text_embeddings=x_sbert
                                                ).tolist()
    
    # Clustering Pipeline
    clustering_instance = Clustering(
                            cluster_algo='HDBSCAN',
                            cluster_data=cluster_data,
                            vectorized_column=vectorized_column
                        )
    cluster_output = clustering_instance._cluster_instance.run_cluster()    
    print(set(cluster_output.labels_))