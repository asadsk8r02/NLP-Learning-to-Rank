import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin

classDataLoader:
    def__init__(self, file_path):
        self.file_path = file_path

    defload_data(self):
        df = pd.read_csv(self.file_path, sep=' ', header=None)
        return df

    defsample_data(self, df, sample_size=200, random_seed=42):
        query_ids = df.iloc[:, 1].unique()
        np.random.seed(random_seed)
        query_ids = np.random.choice(query_ids, size=sample_size, replace=False)
        df_sampled = df[df.iloc[:, 1].isin(query_ids)]
        return df_sampled

classDataPreprocessor:
    def__init__(self):
        self.scaler = StandardScaler()

    defsplit_data(self, df, test_size=0.3, random_state=42):
        X = df.iloc[:, 2:].values
        y = df.iloc[:, 0].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    defscale_data(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

classRegressionModels:
    def__init__(self):
        self.linear_reg = LinearRegression()
        self.svr = SVR()
        self.gbt = GradientBoostingRegressor()

    deffit_models(self, X_train, y_train):
        self.linear_reg.fit(X_train, y_train)
        self.svr.fit(X_train, y_train)
        self.gbt.fit(X_train, y_train)

    defpredict_models(self, X_test):
        y_pred_linear = self.linear_reg.predict(X_test)
        y_pred_svr = self.svr.predict(X_test)
        y_pred_gbt = self.gbt.predict(X_test)
        return y_pred_linear, y_pred_svr, y_pred_gbt

    defevaluate_models(self, y_test, y_pred_linear, y_pred_svr, y_pred_gbt):
        rmse_linear = mean_squared_error(y_test, y_pred_linear, squared=False)
        rmse_svr = mean_squared_error(y_test, y_pred_svr, squared=False)
        rmse_gbt = mean_squared_error(y_test, y_pred_gbt, squared=False)
        return rmse_linear, rmse_svr, rmse_gbt

    defplot_results(self, y_test, y_pred_linear, y_pred_svr, y_pred_gbt):
        plt.scatter(y_test, y_pred_linear, label="Linear Regression")
        plt.scatter(y_test, y_pred_svr, label="Support Vector Regression")
        plt.scatter(y_test, y_pred_gbt, label="Gradient Boosted Regression Trees")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Comparison of Regression Models")
        plt.legend()
        plt.show()

classCustomRegressor(BaseEstimator, RegressorMixin):
    def__init__(self, model='ridge', alpha=1, l1_ratio=0.1, n_estimators=10, max_depth=3, learning_rate=0.1):
        self.model = model
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    deffit(self, X, y):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        if self.model == 'ridge':
            self.reg = Ridge(alpha=self.alpha)
        elif self.model == 'lasso':
            self.reg = Lasso(alpha=self.alpha, max_iter=10000, tol=0.001)
        elif self.model == 'rf':
            self.reg = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth)
        else:
            raise ValueError("Invalid model name")
        self.reg.fit(X, y)

    defpredict(self, X):
        scaler = StandardScaler()
        X = scaler.transform(X)
        return self.reg.predict(X)

classModelEvaluator:
    def__init__(self, param_grid):
        self.param_grid = param_grid

    defevaluate_with_grid_search(self, X_train, y_train):
        regressor = CustomRegressor()
        grid_search = GridSearchCV(regressor, self.param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_, grid_search.best_score_

    defcross_validate_models(self, X, y, models):
        kfold = KFold(n_splits=5)
        for model_name, model in models.items():
            print(f"Model: {model_name}")
            fold_errors = []
            for i, (train_indices, test_indices) inenumerate(kfold.split(X, y)):
                model.fit(X[train_indices], y[train_indices])
                y_pred = model.predict(X[test_indices])
                fold_error = mean_squared_error(y[test_indices], y_pred)
                fold_errors.append(fold_error)
                print(f"Fold {i+1} MSE: {fold_error}")
            plt.plot(fold_errors, label=model_name)
        plt.xlabel('Fold')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # File path
    file_path = '/Users/asadullahkhan/Documents/SPRING2023/IR/fold1_train_sample_all_queries.csv'# Load data
    data_loader = DataLoader(file_path)
    df = data_loader.load_data()
    df_sampled = data_loader.sample_data(df, sample_size=200)

    # Preprocess data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.split_data(df_sampled)
    X_train_scaled, X_test_scaled = preprocessor.scale_data(X_train, X_test)

    # Train and evaluate models
    models = RegressionModels()
    models.fit_models(X_train_scaled, y_train)
    y_pred_linear, y_pred_svr, y_pred_gbt = models.predict_models(X_test_scaled)
    rmse_linear, rmse_svr, rmse_gbt = models.evaluate_models(y_test, y_pred_linear, y_pred_svr, y_pred_gbt)

    # Print RMSE for each modelprint("Linear Regression RMSE:", rmse_linear)
    print("Support Vector Regression RMSE:", rmse_svr)
    print("Gradient Boosted Regression Trees RMSE:", rmse_gbt)

    # Plot the results
    models.plot_results(y_test, y_pred_linear, y_pred_svr, y_pred_gbt)

    # Define parameter grid for custom regressor
    param_grid = {
        'model': ['ridge', 'lasso'],
        'alpha': [0.1, 1],
        'l1_ratio': [0.1, 0.5],
        'n_estimators': [10, 50],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.01],
    }

    # Evaluate custom regressor with grid search
    evaluator = ModelEvaluator(param_grid)
    best_params, best_score = evaluator.evaluate_with_grid_search(X_train_scaled, y_train)
    print("Best hyperparameters:", best_params)
    print("Best score:", best_score)

    # Cross-validate models
    models_dict = {
        'Linear Regression': models.linear_reg,
        'Support Vector Regression': models.svr,
        'Gradient Boosted Regression Trees': models.gbt,
        'Custom Regressor': CustomRegressor()
    }
    evaluator.cross_validate_models(X_train_scaled, y_train, models_dict)
