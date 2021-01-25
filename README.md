# Question Evaluator and Recommender!

This ML-based project represents a question evaluator and recommender. The idea behind the project is to use NLP methods in order to evaluate a question written by any person.
In general terns, the model evaluates the question against a dataset provided by StackExchange, providing two main outputs:
* Evaluation: The model evaluates the question giving back a result as two states -> good or bad question.
* Recommender: The model recommends changes on the question in order to improve its general score to be a very good well-written question.

## Setup
It's highly recommended to clone the whole repo in order to get the latest versions of the codes.
The folders provided already consider the relative paths of files needed in order to run and load each of the steps in the model.
*Note*: The current project can use either a mid-weight or a very large dataset. The dataset's links are provided within the notebooks and they download the datasets as needed.
As an overview, there are two main datasets:
* Stack Overflow: This dataset can weigh up to 80 gb when unzipped. This is a very tech-specific dataset used for CS related questions.
* English: This dataset contains all english language related questions. Its weight is about 460 mb when unzipped.

The files are mainly Python files (.py), iPython notebook files (.ipynb) and text files with some special format (.xml or .txt)-

## Step-by-step running process
In order to run the model, clone the whole github repository and make the code download the needed dataset files. Then, run files in the next order:
* database_stack_interpolation_v2.ipynb -> Creates word embeddings and creates the needed CSV files to feed the model
* model_4v2.ipynb -> Runs the NN model with database created and predicts for some input questions

## Deployment
The deployment of the model is summarized to be deployed on a Docker container. The folder "api_server" has the *Dockerfile* needed to create the image. Once the image is running, go to localhost:5000/apidocs in order to check the API endpoints that run the model
