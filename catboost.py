import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib 

class Config:
    ITERATIONS = 1000
    LR = 0.1
    DEPTH = 4
    RANDOM_STATE = 927

target_columns = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']
feature_columns = ['spectral_centroid_mean', 'spectral_bandwidth', 'zero_crossing_rate']
X = corr_df[feature_columns]

models = {}
metrics = {}
baseline_metrics = {}  
best_rmse = {target: float('inf') for target in target_columns}

def mean_absolute_percentage_error(y_true, y_pred):
    return 100 * (abs((y_true - y_pred) / y_true)).mean()

plt.figure(figsize=(12, 8))

for i, target in enumerate(target_columns):
    y = corr_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=Config.RANDOM_STATE + 1)

    model = CatBoostRegressor(
        iterations=Config.ITERATIONS,
        learning_rate=Config.LR,
        depth=Config.DEPTH,
        verbose=0,
        random_seed=Config.RANDOM_STATE
    )
    eval_set = Pool(X_test, y_test)

    model.fit(X_train, y_train, eval_set=eval_set, logging_level='Silent', plot=False)

    train_loss = model.get_evals_result()['learn']['RMSE']
    test_loss = model.get_evals_result()['validation']['RMSE']

    plt.subplot(2, 3, i + 1)
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(test_loss, label='Test Loss', color='red')
    plt.title(f'Loss Curve for {target}')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
 
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    metrics[target] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

    print(f"Metrics for {target}:")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")
    print(f"MAPE: {mape}\n")

    if rmse < best_rmse[target]:
        best_rmse[target] = rmse
        models[target] = model
        joblib.dump(model, f"{target}_best_model.joblib") 

    baseline_pred = [y_train.mean()] * len(y_test)  
    baseline_mse = mean_squared_error(y_test, baseline_pred)
    baseline_rmse = baseline_mse ** 0.5
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_r2 = r2_score(y_test, baseline_pred)
    baseline_mape = mean_absolute_percentage_error(y_test, baseline_pred)

    baseline_metrics[target] = {
        'MSE': baseline_mse,
        'RMSE': baseline_rmse,
        'MAE': baseline_mae,
        'R2': baseline_r2,
        'MAPE': baseline_mape
    }

plt.tight_layout()
plt.show()

print("All metrics for each target variable:")
for target, metric_vals in metrics.items():
    print(f"{target}: {metric_vals}")

print("\nBaseline metrics (predicting mean) for each target variable:")
for target, baseline_vals in baseline_metrics.items():
    print(f"{target}: {baseline_vals}")
