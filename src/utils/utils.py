from plotly import graph_objects as go

def get_model(model_name, optimize=False, X_train=None, y_train=None):
    if model_name == 'linear_regression':
        from src.models.linear_regression import LinearRegressionModel
        return LinearRegressionModel()
    elif model_name == 'random_forest':
        from src.models.random_forest import RandomForestModel
        return RandomForestModel()
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
    