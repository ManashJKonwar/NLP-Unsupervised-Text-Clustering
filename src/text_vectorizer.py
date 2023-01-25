__author__ = "konwar.m"
__copyright__ = "Copyright 2023, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

from sentence_transformers import SentenceTransformer

def encoded_text(**kwargs):
    data = kwargs.get('data')
    col_name = kwargs.get('col_name')
    model_name = kwargs.get('model_name')
    
    sbert_model = SentenceTransformer(model_name)
    x_sbert = sbert_model.encode(data[col_name].tolist(), show_progress_bar=True)
    
    return x_sbert