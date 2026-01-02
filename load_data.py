import pickle
def load_subject(path):
    with open(path,'rb') as f:
        data=pickle.load(f,encoding='latin1')
    return data    






