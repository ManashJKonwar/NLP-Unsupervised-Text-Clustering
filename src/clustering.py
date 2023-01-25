__author__ = "konwar.m"
__copyright__ = "Copyright 2023, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import hdbscan

class Clustering:
    def __init__(self, **kwargs):
        self._cluster_algo = kwargs.get('cluster_algo')
        self._cluster_data = kwargs.get('cluster_data')
        self._vectorized_column = kwargs.get('vectorized_column')
        self._cluster_instance = self._set_cluster()   
    
    def _set_cluster(self):
        if self._cluster_algo.__eq__('HDBSCAN'):
            return HDBSCAN(self)
            
class HDBSCAN:
    def __init__(self, master_cluster, **kwargs):
        self._master_cluster = master_cluster 
        self._min_cluster_size = kwargs.get('min_cluster_size') if 'min_cluster_size' in kwargs.keys() else 25
        self._hdbscan_cluster = hdbscan.HDBSCAN(
                                    min_cluster_size = self._min_cluster_size,
                                    metric='euclidean', 
                                    cluster_selection_method='eom')
        
    def run_cluster(self):
        return self._hdbscan_cluster.fit(self._master_cluster._cluster_data[self._master_cluster._vectorized_column].tolist())