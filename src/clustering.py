__author__ = "konwar.m"
__copyright__ = "Copyright 2023, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

class Clustering:
    def __init__(self, **kwargs):
        self._cluster_algo = kwargs.get('cluster_algo')
        self._cluster_data = kwargs.get('cluster_data')
        self._text_column = kwargs.get('text_column')
        self._cluster_instance = self._set_cluster()   
    
    def _set_cluster(self):
        if self._cluster_algo.__eq__('HDBSCAN'):
            return HDBSCAN(self)
            
class HDBSCAN:
    def __init__(self, master_cluster):
        self._master_cluster = master_cluster 
        
    def run_cluster(self):
        pass