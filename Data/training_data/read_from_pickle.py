import pickle
from collections import defaultdict
class read_Data:
    def __init__(self,filename):
        self.author_text_dict=None
        self.filename=filename
    def read_from_pickle(self):
        self.author_text_dict = pickle.load( open( self.filename+".p", "rb" ) )
        print 'Read Success'
        return self.author_text_dict
    
                
obj=read_Data('LargeTrain')
obj.read_from_pickle()
obj.clean_data()