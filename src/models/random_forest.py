import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from skopt import BayesSearchCV

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        return mae, mse, rmse, r2

    def save_model(self, filepath):
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        self.model = joblib.load(filepath)

    def optimize_bayesians(self, X_train, Y_train):
        param_space = {
            'n_estimators': (100, 300),
            'max_depth': (3, 9),
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 5),
            'max_features': (0.1, 1.0)
        }

        opt = BayesSearchCV(
            estimator=self.model,
            search_spaces=param_space,
            n_iter=32,
            cv=3,
            n_jobs=-1,
            verbose=2,
            scoring='neg_mean_squared_error',
            random_state=42
        )
        opt.fit(X_train, Y_train)

        self.model = opt.best_estimator_
        return opt.best_params_
    
    
