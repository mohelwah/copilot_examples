'''
create KNN  regression model and train it
'''
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error

class KNN:
    def __init__(self, n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=self.weights, algorithm=self.algorithm, leaf_size=self.leaf_size, p=self.p, metric=self.metric, metric_params=self.metric_params, n_jobs=self.n_jobs)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X = None
        self.y = None
        self.y_pred = None
        self.score = None
        self.mse = None
        self.mae = None
        self.mdae = None
        self.r2 = None

    def train(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.score = self.model.score(self.X_test, self.y_test)
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.mae = mean_absolute_error(self.y_test, self.y_pred)
        self.mdae = median_absolute_error(self.y_test, self.y_pred)
        self.r2 = r2_score(self.y_test, self.y_pred)

    def predict(self, X):
        self.y_pred = self.model.predict(X)
        return self.y_pred

    def get_score(self):
        return self.score

    def get_mse(self):
        return self.mse

    def get_mae(self):
        return self.mae

    def get_mdae(self):
        return self.mdae

    def get_r2(self):
        return self.r2

    def get_model(self):
        return self.model

    def get_y_pred(self):
        return self.y_pred

    def get_y_test(self):
        return self.y_test

    def get_y_train(self):
        return self.y_train

    def get_X_test(self):
        return self.X_test  
    
    def get_X_train(self):
        return self.X_train
    
    def get_X(self):
        return self.X
    
    def get_y(self):
        return self.y
    
    def set_X(self, X):
        self.X = X
    
    def set_y(self, y):
        self.y = y
    
    def set_y_pred(self, y_pred):
        self.y_pred = y_pred
    
    def set_y_test(self, y_test):
        self.y_test = y_test

    def set_y_train(self, y_train):
        self.y_train = y_train
    
    def set_X_test(self, X_test):
        self.X_test = X_test

    def set_X_train(self, X_train):
        self.X_train = X_train
    
    def set_model(self, model):
        self.model = model
    
    def set_score(self, score):
        self.score = score
    
    def set_mse(self, mse):
        self.mse = mse
    
    def set_mae(self, mae):
        self.mae = mae
    
    def set_mdae(self, mdae):
        self.mdae = mdae
    
    def set_r2(self, r2):
        self.r2 = r2
    
    def set_n_neighbors(self, n_neighbors):
        self.n_neighbors = n_neighbors
        
    def set_weights(self, weights):
        