import numpy as np
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from keras._tf_keras.keras import Sequential
from keras._tf_keras.keras.layers import Dense
from sklearn.model_selection import GridSearchCV
import joblib

class NeuronalNetwork():
    def __init__(self, input_dim):
        self.model = Sequential()
        self.model.add(Dense(128, input_dim=input_dim, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    def train(self, X_train, y_train, epochs=100, batch_size=64):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def save_model(self, path):
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        return mae, mse, rmse, r2

def optimize_hyperparameters(X_train, y_train):
    model = NeuronalNetwork(input_dim=X_train.shape[1])
    param_grid = {
        'epochs': [50, 100, 150],
        'batch_size': [10, 20, 30]
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=make_scorer(mean_squared_error), cv=3)
    grid_result = grid.fit(X_train, y_train)
    best_params = grid_result.best_params_
    best_model = NeuronalNetwork(input_dim=X_train.shape[1])
    best_model.train(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'])
    return best_model, best_params
