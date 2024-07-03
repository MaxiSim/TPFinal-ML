from plotly import graph_objects as go

def get_model(model_name, optimize=False, X_train=None, y_train=None, input_dim=None):
    if model_name == 'linear_regression':
        from src.models.linear_regression import LinearRegressionModel
        return LinearRegressionModel()
    elif model_name == 'random_forest':
        from src.models.random_forest import RandomForestModel
        if optimize:
            rf = RandomForestModel()
            params = rf.optimize_bayesians(X_train, y_train)
            return rf, params
        else:
            return RandomForestModel()
    elif model_name == 'nn':
        from src.models.nn import NeuronalNetwork, optimize_hyperparameters
        if optimize:
            model, params = optimize_hyperparameters(X_train, y_train)
            return model, params
        else:
            return NeuronalNetwork(input_dim=input_dim)
    elif model_name == 'xgboost':
        from src.models.xgboost import XGBoostModel
        if optimize:
            xg = XGBoostModel()
            params = xg.optimize_bayesian(X_train, y_train)
            return xg, params
        else:
            return XGBoostModel()
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

def plot_predictions(y_test, predictions):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=predictions, mode='markers'))
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', line=dict(color='red', dash='dash')))

    fig.update_layout(
        title='Predicciones vs. Valores reales',
        xaxis_title='Valores reales',
        yaxis_title='Predicciones',
        showlegend=False
    )

    fig.show()
    