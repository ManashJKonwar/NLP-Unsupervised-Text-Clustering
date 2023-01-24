__author__ = "konwar.m"
__copyright__ = "Copyright 2023, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

import umap

def reduce_dimensions(**kwargs):
    n_neighbors = kwargs.get('n_neighbors')
    n_components = kwargs.get('n_components')
    random_state = 42
    text_embeddings = kwargs.get('text_embeddings')
    
    umap_embeddings = umap.UMAP(
                        n_neighbors=n_neighbors, 
                        n_components=n_components, 
                        metric='cosine', 
                        random_state=random_state).fit_transform(text_embeddings)
    
    return umap_embeddings