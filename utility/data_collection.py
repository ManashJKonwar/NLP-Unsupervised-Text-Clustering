__author__ = "gogineni.b"
__copyright__ = "Copyright 2023, AI R&D"
__credits__ = ["gogineni.b"]
__license__ = "Individual Ownership"
__version__ = "1.0.1"
__maintainer__ = "gogineni.b"
__email__ = "bharathchowdary.gogineni@gmail.com"
__status__ = "Development"

import os
import gzip
import requests
import pandas as pd

if not os.path.exists("input"):
    os.makedirs("input")
    
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(category="reviews_Cell_Phones_and_Accessories_5"):
    
    def write_json(category):
        URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/" + category + ".json.gz"
        response = requests.get(URL)
        open(os.path.join("input", category + ".json.gz"), "wb").write(response.content)
        
    def delete_json(category):
        os.remove(os.path.join("input", category + ".json.gz"))

    # writing json file
    write_json(category=category)
    
    # reading json file
    i = 0
    df = {}
    for d in parse(path=os.path.join("input", category + ".json.gz")):
        df[i] = d
        i += 1
        
    # deleting json file
    delete_json(category=category)
    
    return pd.DataFrame.from_dict(df, orient='index')

# df = getDF(category='reviews_Cell_Phones_and_Accessories_5.json.gz')