__author__ = "konwar.m"
__copyright__ = "Copyright 2023, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import pandas as pd

from src.clustering import *
from utility.data_collection import *

if __name__ == '__main__':
    category="reviews_Cell_Phones_and_Accessories_5"
    cluster_data = None
    
    if not os.path.exists(os.path.join("input", category.lower() + ".csv")):
        cluster_data = getDF(category=category)
        cluster_data.to_csv(os.path.join("input", category.lower() + ".csv"))
    else:
        cluster_data = pd.read_csv(os.path.join("input", category.lower() + ".csv"))        
    
    clustering_instance = Clustering(
                            cluster_algo='HDBSCAN',
                            cluster_data=cluster_data,
                            text_column=""
                        )