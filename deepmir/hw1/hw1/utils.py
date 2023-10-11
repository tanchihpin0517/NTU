import pickle

def pickle_save(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)

def pickle_load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
