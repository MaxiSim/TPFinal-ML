import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from skopt import BayesSearchCV

class XGBoostModel:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, gamma=0):
        self.model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample, colsample_bytree=colsample_bytree, gamma=gamma, objective='reg:squarederror', random_state=42)

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

    def optimize_bayesian(self, X_train, y_train):
        param_space = {
            'n_estimators': (100, 300),
            'max_depth': (3, 9),
            'learning_rate': (0.01, 0.2, 'log-uniform'),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'gamma': (0, 0.2)
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
        opt.fit(X_train, y_train)
        
        self.model = opt.best_estimator_
        return opt.best_params_
