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

from src.clustering import *
from src.text_preprocesser import *
from src.text_vectorizer import *
from utility.data_collection import *

if __name__ == '__main__':
    category = "reviews_Cell_Phones_and_Accessories_5"
    cluster_data = None
    text_column = "reviewText"
    sample_frac = 0.1
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
    x_sbert = encoded_text(data=cluster_data, col_name="preprocessed_text", model_name=model_embedding)
    cluster_data["vectorized_text"] = x_sbert.tolist()
    
    # Clustering Pipeline
    clustering_instance = Clustering(
                            cluster_algo='HDBSCAN',
                            cluster_data=cluster_data,
                            text_column=text_column
                        )
    clustering_instance._cluster_instance.run_cluster()