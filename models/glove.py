#Imports for model
import numpy as np

#Class that represents the GloVe word embeddings. Uses 300-dim vectors
#with 42B tokens in total.
class Glove:
    def __init__(self):
        self.embeddings = {}
        try:
            with open("../processed_files/glove.42B.300d.txt", "r", encoding='utf-8') as f:
                for line in f:
                    word, coefs = line.split(maxsplit=1)
                    coefs = np.fromstring(coefs, "f", sep=" ")
                    self.embeddings[word] = coefs
            
            #create zero vector for unkown words. 
            self.zero = np.zeros_like(self.embeddings["car"])
            self.dim = len(self.embeddings["car"])
            self.voc_size = len(self.embeddings)
            print("Found %s word vectors." % len(self.embeddings))
        except:
            print("File \"glove.42B.300d.txt\" not found. Cannot load vec embeddings\n"\
                  "Please make sure the file is on the path \"/processed_files/\"")

    def vector(self, word):
        return self.embeddings.get(word, self.zero)