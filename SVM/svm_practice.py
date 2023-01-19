import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.base import BaseEstimator
import numpy as np
from numpy import linalg

from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import numpy.linalg as la

class Kernel(object):
    """Implements list of kernels from
    http://en.wikipedia.org/wiki/Support_vector_machine
    """
    @staticmethod
    def linear():
        def f(x, y):
            return np.inner(x, y)
        return f

    @staticmethod
    def gaussian(sigma):
        def f(x, y):
            exponent = -np.sqrt(la.norm(x-y) ** 2 / (2 * (sigma ** 2)))
            return np.exp(exponent)
        return f

    @staticmethod
    def _polykernel(dimension, offset):
        def f(x, y):
            return (offset + np.dot(x, y)) ** dimension
        return f



class mySVM(BaseEstimator):
    
    def __init__(self, kernel=Kernel.linear(), C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)
        
    def _gram_matrix(self,X):
        m , n = X.shape
        K = np.zeros(( m , m))
        for i,x_i in enumerate(X) : 
            for j,x_j in enumerate(X):
                K[i,j] = self.kernel(x_i,x_j)
        
        return K      

    def fit(self, X : np.array ,y : np.array ):
        
        y[y == 0] = y[y == 0] - 1
        
        m , n = X.shape
        K = self._gram_matrix(X)
        H = np.outer(y,y) * K 
        
         
        # P m by m matrix
        # q m by 1 vector
        # G m by m identity matrix   
        # A m by 1 vector
        # b m by 1 vector
        
        P = cvxopt_matrix(H)
        q = cvxopt_matrix(-np.ones((m,1)))
        A = cvxopt_matrix(y.reshape(1,-1))
        b = cvxopt_matrix(np.zeros(1))

        if self.C is None :
            G = cvxopt_matrix(np.eye(m)*-1)
            h = cvxopt_matrix(np.zeros(m))
        
        else:
            G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
            h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))
        
        
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])
        
        # sv의 alpha는 0이 아니다 
        sv = (alphas > 1e-4).flatten()
        
        w = ((y * alphas).T @ X).reshape(-1,1)
        print(w.shape)
        
        b = y[sv] - np.dot(X[sv], w)
        
        self.intercept_ = b
        self.coef_ = w.flatten()
        return self
    
    
    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(np.float64)
    
    def decision_function(self, X):
        return X.dot(self.coef_) + self.intercept_[0]
    
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # # 꽃잎 길이, 꽃잎 너비
y = (iris["target"]==2).astype(np.float64).reshape(-1, 1) # Iris virginica




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_clf = mySVM(C=4.75,kernel=Kernel.gaussian(1/0.08))
svm_clf.fit(X_train,y_train)
y_pred = svm_clf.predict(X_test_scaled)

