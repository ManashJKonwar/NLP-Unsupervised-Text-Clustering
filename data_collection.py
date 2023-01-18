import pandas as pd
#import numpy as np 
import gzip

import requests
URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz"
response = requests.get(URL)
open("reviews_Cell_Phones_and_Accessories_5.json.gz", "wb").write(response.content)




def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

df = getDF('reviews_Cell_Phones_and_Accessories_5.json.gz')


print(df.shape)