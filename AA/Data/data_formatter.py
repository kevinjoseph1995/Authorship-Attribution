import os
from collections import defaultdict
import pickle
DATASETPATH="C:\Users\kevin\Documents\GitHub\Authorship-Attribution\Data\C50\C50train"
root, dirs, files = os.walk(DATASETPATH).next()
author_text_dict=defaultdict(list)
for author in dirs:
    FILEPATH=DATASETPATH +"\\"+ author
    temp, temp, text_files = os.walk(FILEPATH).next()
    for text_file_name in text_files:
        COMPLETE_PATH=FILEPATH+"\\"+text_file_name
        text_file = open(COMPLETE_PATH, "r")
        author_text_dict[author].append(text_file.read())
        text_file.close()
pickle.dump(author_text_dict, open( "TrainData.p", "wb" ) )    