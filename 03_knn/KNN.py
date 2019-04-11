import numpy as np 
from math import sqrt
from collections import Counter

class KNNClassifier:
    def __init__(self,k):
        assert k>=1,"k must be valid"
        self.k=k
        self.X_train=None
        self.Y_train=None
    
    def fit(self,X_train,Y_train):
        assert X_train.shape[0]==Y_train.shape[0],\
            "the size of x and y must be same"
        assert self.k<=Y_train.shape[0],\
            "k cannot large than the size of y"
        self.X_train=X_train
        self.Y_train=Y_train
        

    def predict(self,predictX):
        assert self.X_train is not None and self.Y_train is not None,\
            "must fit before predict"
        assert self.X_train.shape[1]==predictX.shape[1],\
            "the nums of features of predict and train must be same"
        
        y_predict=[self._perdict(x) for x in predictX]
        return y_predict


    def _perdict(self,x):
        distances=[sqrt(sum((xt-x)**2)) for xt in self.X_train]
        votes=Counter([self.Y_train[y] for y in np.argsort(distances)[:self.k]])
        return votes.most_common(1)[0][0]
                
    
    def __repr__(self):
        return "KNN(k=%d)" %self.k
    