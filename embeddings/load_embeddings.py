import gensim

def load_model(model_path):
    """
    Loads word embeddings from a file into a KeyedVectors object.
    :param model_path: str, path to the embeddings file
    :return: KeyedVectors object
    """
    return gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)