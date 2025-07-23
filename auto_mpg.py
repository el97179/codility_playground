from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_data():
    # Load the Auto MPG dataset
    data = fetch_openml("autoMPG", version=1, as_frame=True)
    return data.data, data.target

def prepare_features(X_train, X_test, feature_engineering=False):
    """Prepare features with optional feature engineering"""
    if feature_engineering:
        # Apply polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        # Apply scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_poly)
        X_test_scaled = scaler.transform(X_test_poly)
        
        return X_train_scaled, X_test_scaled
    else:
        return X_train, X_test

def linear_regression_model(X_train, X_test, y_train, feature_engineering=False):
    """Linear Regression with optional feature engineering"""
    X_train_prep, X_test_prep = prepare_features(X_train, X_test, feature_engineering)
    model = LinearRegression()
    model.fit(X_train_prep, y_train)
    return model, X_train_prep, X_test_prep

def random_forest_model(X_train, X_test, y_train, feature_engineering=False):
    """Random Forest Regressor"""
    X_train_prep, X_test_prep = prepare_features(X_train, X_test, feature_engineering)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_prep, y_train)
    return model, X_train_prep, X_test_prep

def lightgbm_model(X_train, X_test, y_train, feature_engineering=False):
    """LightGBM Regressor"""
    X_train_prep, X_test_prep = prepare_features(X_train, X_test, feature_engineering)
    model = lgb.LGBMRegressor(random_state=42, verbose=-1)
    model.fit(X_train_prep, y_train)
    return model, X_train_prep, X_test_prep

def xgboost_model(X_train, X_test, y_train, feature_engineering=False):
    """XGBoost Regressor"""
    X_train_prep, X_test_prep = prepare_features(X_train, X_test, feature_engineering)
    model = xgb.XGBRegressor(random_state=42, verbosity=0)
    model.fit(X_train_prep, y_train)
    return model, X_train_prep, X_test_prep

def svr_model(X_train, X_test, y_train, feature_engineering=False):
    """Support Vector Regression with RBF kernel"""
    X_train_prep, X_test_prep = prepare_features(X_train, X_test, feature_engineering)
    # SVR requires scaling
    if not feature_engineering:
        scaler = StandardScaler()
        X_train_prep = scaler.fit_transform(X_train_prep)
        X_test_prep = scaler.transform(X_test_prep)
    model = SVR(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_train_prep, y_train)
    return model, X_train_prep, X_test_prep

def mlp_model(X_train, X_test, y_train, feature_engineering=False):
    """Multi-layer Perceptron Regressor"""
    X_train_prep, X_test_prep = prepare_features(X_train, X_test, feature_engineering)
    # MLP requires scaling
    if not feature_engineering:
        scaler = StandardScaler()
        X_train_prep = scaler.fit_transform(X_train_prep)
        X_test_prep = scaler.transform(X_test_prep)
    model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
    model.fit(X_train_prep, y_train)
    return model, X_train_prep, X_test_prep

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate model and print metrics"""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name}:")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.2f}")
    
    return mae, r2

def cross_validate_model(model_func, X_clean, y_clean, model_name, feature_engineering=False):
    """Perform cross-validation for a model"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scores = []
    r2_scores = []
    
    for train_idx, val_idx in kf.split(X_clean):
        X_train_fold, X_val_fold = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
        y_train_fold, y_val_fold = y_clean.iloc[train_idx], y_clean.iloc[val_idx]
        
        model, X_train_prep, X_val_prep = model_func(X_train_fold, X_val_fold, y_train_fold, feature_engineering)
        y_pred = model.predict(X_val_prep)
        
        mae_scores.append(mean_absolute_error(y_val_fold, y_pred))
        r2_scores.append(r2_score(y_val_fold, y_pred))
    
    print(f"{model_name} 5-fold CV:")
    print(f"MAE: {np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f}")
    print(f"R2: {np.mean(r2_scores):.2f} ± {np.std(r2_scores):.2f}")
    
    return mae_scores, r2_scores

if __name__ == "__main__":
    X, y = load_data()
    print("Rows with null values:")
    print(X[X.isnull().any(axis=1)])
    # Convert all columns to numeric, coerce errors to NaN
    X = X.apply(pd.to_numeric, errors='coerce')
    # 1. Clean data: remove rows with NaN
    data = pd.concat([X, y], axis=1)
    data_clean = data.dropna()
    X_clean = data_clean.iloc[:, :-1]
    y_clean = data_clean.iloc[:, -1]

    # 2. Split data into train/test 80/20
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

    # Test different models
    models = [
        (linear_regression_model, "Linear Regression"),
        (linear_regression_model, "Linear Regression + Feature Engineering"),
        (random_forest_model, "Random Forest"),
        (random_forest_model, "Random Forest + Feature Engineering"),
        (lightgbm_model, "LightGBM"),
        (lightgbm_model, "LightGBM + Feature Engineering"),
        (xgboost_model, "XGBoost"),
        (xgboost_model, "XGBoost + Feature Engineering"),
        (svr_model, "SVR"),
        (svr_model, "SVR + Feature Engineering"),
        (mlp_model, "MLP"),
        (mlp_model, "MLP + Feature Engineering")
    ]
    
    results = []
    for i, (model_func, model_name) in enumerate(models):
        feature_eng = True if "Feature Engineering" in model_name else False
        try:
            model, X_train_prep, X_test_prep = model_func(X_train, X_test, y_train, feature_eng)
            mae, r2 = evaluate_model(model, X_train_prep, X_test_prep, y_train, y_test, model_name)
            results.append((model_name, mae, r2))
        except Exception as e:
            print(f"Error with {model_name}: {e}")

    # Cross-validation for all models
    print("\n" + "="*50)
    print("CROSS-VALIDATION RESULTS")
    print("="*50)
    
    cv_results = []
    for model_func, model_name in models:  # Test all models with CV
        feature_eng = True if "Feature Engineering" in model_name else False
        try:
            mae_scores, r2_scores = cross_validate_model(model_func, X_clean, y_clean, model_name, feature_eng)
            cv_results.append((model_name, np.mean(mae_scores), np.std(mae_scores), 
                              np.mean(r2_scores), np.std(r2_scores)))
        except Exception as e:
            print(f"CV Error with {model_name}: {e}")

    # Plot results with error bars
    if cv_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        model_names = [result[0] for result in cv_results]
        mae_means = [result[1] for result in cv_results]
        mae_stds = [result[2] for result in cv_results]
        r2_means = [result[3] for result in cv_results]
        r2_stds = [result[4] for result in cv_results]
        
        # MAE plot
        bars1 = ax1.bar(range(len(model_names)), mae_means, yerr=mae_stds, capsize=5, alpha=0.7)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Mean Absolute Error (MAE)')
        ax1.set_title('MAE Comparison (5-fold CV)')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on MAE bars
        for i, (mean_val, std_val) in enumerate(zip(mae_means, mae_stds)):
            ax1.text(i, mean_val + std_val + 0.1, f'{mean_val:.2f}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # R2 plot
        bars2 = ax2.bar(range(len(model_names)), r2_means, yerr=r2_stds, capsize=5, alpha=0.7, color='orange')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('R² Score')
        ax2.set_title('R² Comparison (5-fold CV)')
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on R2 bars
        for i, (mean_val, std_val) in enumerate(zip(r2_means, r2_stds)):
            ax2.text(i, mean_val + std_val + 0.02, f'{mean_val:.2f}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        plt.tight_layout()

    # Plot scatter plots for all models
    if cv_results:
        n_models = len(models)
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, (model_func, model_name) in enumerate(models):
            feature_eng = True if "Feature Engineering" in model_name else False
            try:
                # Get cross-validated predictions
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                y_pred_cv = []
                y_actual_cv = []
                
                for train_idx, val_idx in kf.split(X_clean):
                    X_train_fold, X_val_fold = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                    y_train_fold, y_val_fold = y_clean.iloc[train_idx], y_clean.iloc[val_idx]
                    
                    model, X_train_prep, X_val_prep = model_func(X_train_fold, X_val_fold, y_train_fold, feature_eng)
                    y_pred = model.predict(X_val_prep)
                    
                    y_pred_cv.extend(y_pred)
                    y_actual_cv.extend(y_val_fold)
                
                # Plot scatter
                axes[i].scatter(y_actual_cv, y_pred_cv, alpha=0.5)
                axes[i].plot([min(y_actual_cv), max(y_actual_cv)], [min(y_actual_cv), max(y_actual_cv)], 'r--')
                axes[i].set_xlabel('Actual MPG')
                axes[i].set_ylabel('Predicted MPG')
                axes[i].set_title(model_name)
                axes[i].grid(True, alpha=0.3)
                
                # Calculate and display metrics on the plot
                mae_val = mean_absolute_error(y_actual_cv, y_pred_cv)
                r2_val = r2_score(y_actual_cv, y_pred_cv)
                axes[i].text(0.05, 0.95, f'MAE: {mae_val:.2f}\nR²: {r2_val:.2f}', 
                           transform=axes[i].transAxes, va='top', ha='left',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           fontsize=10, fontweight='bold')
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                           transform=axes[i].transAxes, ha='center', va='center')
                axes[i].set_title(f"{model_name} (Error)")
        
        # Hide unused subplots if we have fewer than 12 models
        for j in range(n_models, 12):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.show()
