import numpy as np
from scipy.stats import mode
from sklearn.base import BaseEstimator

class knn_clf(BaseEstimator):
    """_summary_
    K Nearest Neighnors classifier
    """
    def __init__(self , n_neighbors = 5) :
        self.n_neighbors = n_neighbors
        self.X = None
        self.c = None

    def fit(self,X : np.array, y : np.array):
        """_summary_
        Args:
            X (np.array): dataset
        """
        self.X = X.copy()
        self.c = y.copy()
        
        return self
        
            
    def predict(self,X : np.array):
        
        m = X.shape[0]
        n = self.X.shape[0]
        k = self.n_neighbors
        metric = np.zeros((m,n))
        
        y_pred = np.zeros(m)
        
        for i in range(m):
            #calculate distance
            metric[i] = np.linalg.norm( (self.X - X[i]) , axis = 1 )
            
        for i in range(m):
            y_pred[i] = mode( self.c[np.argsort(metric[i])[0:k]] )[0][0]     
        
    
        return y_pred

