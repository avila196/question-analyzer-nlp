#Imports for model
from os import path
import numpy as np
from gensim.models import fasttext
from gensim.test.utils import datapath

#Class that represents the FastText word embeddings. Uses 300-dim vectors
#and has the capability to provide word vectors for OOV words.
class FT:
    def __init__(self):
        #FIX DATAPATH!
        p = "C:/Users/p_yth/Drive Factored/factored_github/team3_textassistant/models/" + FT.path_to_fasttext_file
        self.embeddings = fasttext.load_facebook_vectors(datapath(p))

    def vector(self, word):
        return self.embeddings[word]
    
    
'''
def download(self):
    path_to_fasttext_zipfile = "wiki-news-300d-1M-subword.vec.zip"
    path_to_fasttext_file = "wiki-news-300d-1M-subword.vec"
    
    #Download and prepare the Pre-trained GloVe Word Embedding model
    if not path.exists(FT.path_to_fasttext_file):
        if not path.exists(FT.path_to_fasttext_zipfile):
            print("downloading fasttext .zip file...")
            !wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip
        print("unzipping fasttext .zip file...")
        !unzip -q wiki-news-300d-1M-subword.vec.zip
'''