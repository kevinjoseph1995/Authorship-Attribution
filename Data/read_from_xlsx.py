import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from collections import defaultdict
import pickle
data_set='LargeTrain'
df = pd.read_excel(data_set+'.xlsx')
author_id = df['author_id'].tolist()
body=df['body'].tolist()

author_text_dict=defaultdict(list)
for i in xrange(len(author_id)):
    author_text_dict[author_id[i]].append(body[i])

pickle.dump(author_text_dict, open( data_set+".p", "wb" ) )