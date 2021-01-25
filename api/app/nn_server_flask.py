from flask import Flask, request
from flasgger import Swagger
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np
import re
from alibi.explainers import IntegratedGradients

app = Flask("Question Evaluator")
swagger = Swagger(app)

#Class that defines a Predictor object to make all predictions
class Predictor:
    def __init__(self, 
                 path_model="model_v2.h5",
                 path_embeddings="nnlm-en-dim128_2"):#-------> DOWNLOAD LINK : https://tfhub.dev/google/nnlm-en-dim128/2
        
        self.model = keras.models.load_model(path_model)
        self.embed = hub.load(path_embeddings)
        #regular expression for text pre-processing
        self.p = re.compile(r'<.*?>')
        #----------------------
        self.ig = IntegratedGradients(self.model,
                          layer=None,
                          method="gausslegendre",
                          n_steps=50,
                          internal_batch_size=100)

    #special clean, only for mathematica dataset. 
    def cleanSentence(self, sentence, html = False):
        if html: sentence = self.p.sub('', sentence) 
        sentence = ''.join([i.lower() if (i.isalpha() or i==" ") else (" "+i) for i in sentence])
        return sentence

    def predict(self, question, tags, year, max_sequence_length=20, vec_dim=128):
        #INPUTS: question: "string", tags: ["string", "string", "string"], year: integer (2010-2019) 
        #OUTPUTS: evaluation: Bool,
#                 words: ["string", "string", "string"],
#                 attributions_words: [float, float, float] 
#                 attribution year: float
        Xtext=[]
        X2=[]
        Mask=[]
 
        clean_question = self.cleanSentence(question)
        info = tags+["."]+clean_question.split() 
        
        # set x1i length equal to str of max_sequence_lenght
        xtexti =  info[0:max_sequence_length]  +  [""]*(max_sequence_length-len(info))
        maski = [1]*min(len(info), max_sequence_length) + [0]*(max_sequence_length-len(info))
        #----------------
        year = int(str(year)[3])
        x2i = [0]*10
        x2i[year] = 1
        
        #-----------------------
        Xtext += xtexti
        Mask.append(maski)
        X2.append(x2i)
        
        X1 = np.array(self.embed(Xtext))
        Mask = np.array(Mask)
        Mask = np.expand_dims(Mask, axis=-1)
        X1 = Mask * X1.reshape(1, max_sequence_length, vec_dim)
        
        X2 = np.array(X2)
        X2 = X2.reshape(1, 1, 10)
        O = np.ones((1, max_sequence_length, 10)) #for broadcasting year one-hot encoding
        X2 = X2 * O
        #-----------------------------------
        X = np.concatenate((X2, X1), axis=2)
        prediction = self.model.predict(X)
        explanation = self.ig.explain(X,
                         baselines=None,
                         target=prediction)
        
        attrs = -explanation.attributions
        #separate attributions from year information and text information 
        attrs1 = attrs[:,:,10:]
        attrs2 = attrs[:,:,0:10]
#         print("attributions 1 and 2 shapes:", attrs1.shape, attrs2.shape)
        # sum attribution by embedding vector 
        attrs1 = attrs1.sum(axis=2)
        attrs1 = attrs1.reshape(-1).tolist()
        # sum all year attributions
        attrs2 = np.sum(attrs2)
        
        #Finally, we return the predictions
        return prediction[0][0]>0.5, info, attrs1[:len(info)], attrs2

#Initialize objects to make all predictions
predictor = Predictor()

#Function that constructs the final prediction message
def construct_answer(y, words, a_words, a_year=None):
    #Given the returns of the predictor, we construct the final string to return back
    #to the user. First, we add the message if good or bad question
    if a_year:
        msg = "Good question for given year!" if y else "Bad question for given year!"
    else:
        msg = "Good question for recent years!" if y else "Bad question for recent years!"
    #Now, we add the relevance of each of the words
    msg += "\n>> Relevance of each word in question (sorted from positive to negative):"
    vals = zip(words, a_words)
    vals = sorted(vals, key = lambda x: -x[1])
    for word, attribution in vals:
        msg += "\n\t" + "{:20s}:{:.5f}".format(word, attribution)
    #Finally, we add the relevance of the year
    if a_year:
        msg += "\n\t" + "{:20s}:{:.5f}".format("Year relevance:",a_year)
    return msg
    

@app.route("/predict/default", methods=["POST"])
def predict():
    """ Endpoint to predict the score for a single typed question
    ---
    parameters:
        - name: question
          in: formData
          type: string
          required: true
          description: Question to evaluate
    responses:
        200:
            description: "Prediction on the question"
    """
    question = request.form["question"]
    #For current question, we will use no tags and default year of 2019
    y, words, a_words, a_year = predictor.predict(question, [], 2019)
    #Now, we construct the answer to return
    return construct_answer(y, words, a_words)

@app.route("/predict/year-tags", methods=["POST"])
def predict_with_year_tags():
    """ Endpoint to predict the score for a single typed question, the year
    where the question could be asked [2010-2019] and the tags for the question
    ---
    parameters:
        - name: question
          in: formData
          type: string
          required: true
          description: Question to evaluate
        - name: year
          in: formData
          type: number
          required: true
          description: Year between [2010-2019]
        - name: tags
          in: formData
          type: string
          required: true
          description: Tags separated by | char
    responses:
        200:
            description: "Prediction on the question"
    """
    question = request.form["question"]
    tags = request.form["tags"].split("|")
    year = int(request.form["year"])
    #We use the values given to predict
    y, words, a_words, a_year = predictor.predict(question, tags, year)
    #Now, we construct the answer to return
    return construct_answer(y, words, a_words, a_year)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)