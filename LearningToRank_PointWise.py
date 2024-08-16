import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

# Load the dataset
file_path = '/Users/asadullahkhan/Documents/SPRING2023/IR/fold1_train_sample_all_queries.csv'

# Load MSLR dataset
def load_data(file_path):
    data = pd.read_csv(file_path, sep=' ', header=None)

df = pd.read_csv(file_path)

# Get 500 random query samples
query_ids = df.iloc[:, 1].unique()
np.random.seed(42)
query_ids = np.random.choice(query_ids, size=200, replace=False)
df = df[df.iloc[:, 1].isin(query_ids)]

# Split into features and target
X = df.iloc[:, 2:].values
y = df.iloc[:, 0].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print shapes of train and test sets
print("Shape of X_train: ", X_train.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_test: ", y_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Shape of X_train:  (28837, 136)
# Shape of y_train:  (28837,)
# Shape of X_test:  (12360, 136)
# Shape of y_test:  (12360,)

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
# Fit linear regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Fit support vector regression model
svr = SVR()
svr.fit(X_train, y_train)

# Fit gradient boosted regression trees model
gbt = GradientBoostingRegressor()
gbt.fit(X_train, y_train)

# Predict on test set
y_pred_linear = linear_reg.predict(X_test)
y_pred_svr = svr.predict(X_test)
y_pred_gbt = gbt.predict(X_test)

# Calculate RMSE for each model
rmse_linear = mean_squared_error(y_test, y_pred_linear, squared=False)
rmse_svr = mean_squared_error(y_test, y_pred_svr, squared=False)
rmse_gbt = mean_squared_error(y_test, y_pred_gbt, squared=False)

# Print RMSE for each model
print("Linear Regression RMSE:", rmse_linear)
print("Support Vector Regression RMSE:", rmse_svr)
print("Gradient Boosted Regression Trees RMSE:", rmse_gbt)

# Plot predicted vs actual values
plt.scatter(y_test, y_pred_linear, label="Linear Regression")
plt.scatter(y_test, y_pred_svr, label="Support Vector Regression")
plt.scatter(y_test, y_pred_gbt, label="Gradient Boosted Regression Trees")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Comparison of Regression Models")
plt.legend()
plt.show()
Linear Regression RMSE: 1.1232676979860061
Support Vector Regression RMSE: 1.0677999901267823
Gradient Boosted Regression Trees RMSE: 1.0497081211822867

# Define CustomRegressor class
class CustomRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model='ridge', alpha=1, l1_ratio=0.1, n_estimators=10, max_depth=3, learning_rate=0.1):
        self.model = model
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
    
    def fit(self, X, y):
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
    
    def predict(self, X):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return self.reg.predict(X)
param_grid = {
    'model': ['ridge', 'lasso'],
    'alpha': [0.1, 1],
    'l1_ratio': [0.1, 0.5],
    'n_estimators': [10, 50],
    'max_depth': [3, 5],
    'learning_rate': [0.1,0.01],
}

# Define CustomRegressor
regressor = CustomRegressor()

# Define GridSearchCV with cross-validation and hyperparameters
grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit GridSearchCV on training data
grid_search.fit(X_train, y_train)

# Print best hyperparameters and corresponding score
print("Best hyperparameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
Best hyperparameters:  {'alpha': 1, 'l1_ratio': 0.1, 'learning_rate': 0.1, 'max_depth': 3, 'model': 'ridge', 'n_estimators': 10}
Best score:  -0.5744162865874334
# Extract the features and target variable
from sklearn.model_selection import KFold, cross_val_score
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Initialize the regressor models
lr = LinearRegression()
svr = SVR()
gbr = GradientBoostingRegressor()
custom_reg = CustomRegressor()


models = {'Linear Regression': lr, 
          'Support Vector Regression': svr, 
          'Gradient Boosted Regression Trees': gbr,
          'Custom Regressor': custom_reg}

kfold = KFold(n_splits=5)

for model_name, model in models.items():
    print(f"Model: {model_name}")
    fold_errors = []
    for i, (train_indices, test_indices) in enumerate(kfold.split(X, y)):
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
# Model: Linear Regression
# Fold 1 MSE: 1.107307521098981
# Fold 2 MSE: 1.1985103688904912
# Fold 3 MSE: 1.2366609892209928
# Fold 4 MSE: 1.2741719370903943
# Fold 5 MSE: 1.2454445305625146
# Model: Support Vector Regression
# Fold 1 MSE: 1.0376375996804335
# Fold 2 MSE: 1.1441304234531553
# Fold 3 MSE: 1.1657147098476848
# Fold 4 MSE: 1.1644881082622172
# Fold 5 MSE: 1.2103181426081142
# Model: Gradient Boosted Regression Trees
# Fold 1 MSE: 1.0104336653424015
# Fold 2 MSE: 1.1025549385361006
# Fold 3 MSE: 1.1187279697231638
# Fold 4 MSE: 1.1130495093627193
# Fold 5 MSE: 1.1634068020184702
# Model: Custom Regressor
# Fold 1 MSE: 1.269880658572974
# Fold 2 MSE: 1.388342850975913
# Fold 3 MSE: 1.3473275152774369
# Fold 4 MSE: 1.2399978148554232
# Fold 5 MSE: 1.3579218587404305

 
