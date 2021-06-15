import pickle
import pandas
from .detect import Detect
from .prediction import Prediction
import logging
from sklearn.pipeline import Pipeline
class fake_news_detection:
    def __init__(self,filename):
        self.filename = filename
        self.data = self.loadModel()
        self.tr = Detect(self.data)
    def loadModel(self):
        try : 
            data = pandas.read_csv("data/file1.csv")
        except :
            logging.error("file not found")
            exit(0)           
        return  data
    def train(self, model = "LR"):      
        self.tr.train(model)
    def predict(self,text,title):
        return Prediction().predict(text , title)
    def compare(self,force = False):
        if(not force):
            try :
                m = pickle.load(open("compare_model.sav" , "rb"))
            except IOError:
                logging.warning("file not found traning all model")
                m= self.tr.compare()
        else:
               m= self.tr.compare()     
        data= {'accuracy':[self.tr.getAccuracy(model) for _ , model in m.items()],
               'cmatrix':[self.tr.getConfmatrix(model) for _ , model in m.items()],
               'creport':[self.tr.getReport(model) for _, model in m.items()]}
        return data