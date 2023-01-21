__author__ = "konwar.m"
__copyright__ = "Copyright 2023, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

from src.clustering import *

if __name__ == '__main__':
    clustering_instance = Clustering(
                            cluster_algo='HDBSCAN',
                            cluster_data=None
                        )