import pickle
class read_Data:
    def __init__(self,filename):
        self.author_text_dict=None
        self.filename=filename
    def read_from_pickle(self):
        self.author_text_dict = pickle.load( open( self.filename+".p", "rb" ) )
        print 'Read Success'
obj=read_Data('LargeTrain')
dict=obj.read_from_pickle()