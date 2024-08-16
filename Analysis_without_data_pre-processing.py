# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load the MSLR-WEB10K dataset
data = pd.read_csv("data.csv")

# Split the data into training, validation, and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Extract the features and labels from the data
train_features = train_data.iloc[:, 2:].values
train_labels = train_data.iloc[:, 1].values

val_features = val_data.iloc[:, 2:].values
val_labels = val_data.iloc[:, 1].values

test_features = test_data.iloc[:, 2:].values
test_labels = test_data.iloc[:, 1].values

# Define the evaluation metric as NDCG@10
def ndcg(y_true, y_pred, k=10):
    score = ndcg_score(np.array([y_true]), np.array([y_pred]), k=k)
    return score

# Define the gradient boosting algorithm for LTR
class GradientBoostingLTR():
    def __init__(self, num_trees=100, learning_rate=0.1, max_depth=6):
        self.num_trees = num_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        
    def fit(self, X, y, qid):
        self.trees = []
        unique_qid = np.unique(qid)
        
        for q in unique_qid:
            mask = qid == q
            Xq = X[mask]
            yq = y[mask]
            n = len(yq)
            weights = np.ones(n) / n
            
            for i in range(self.num_trees):
                tree = DecisionTreeRegressor(max_depth=self.max_depth)
                tree.fit(Xq, yq, sample_weight=weights)
                
                predictions = tree.predict(Xq)
                gradient = yq - predictions
                
                weights = weights * np.exp(-self.learning_rate * gradient)
                weights = weights / np.sum(weights)
                
                self.trees.append(tree)
                
    def predict(self, X, qid):
        predictions = np.zeros(len(X))
        unique_qid = np.unique(qid)
        
        for q in unique_qid:
            mask = qid == q
            Xq = X[mask]
            n = len(Xq)
            
            if n == 0:
                continue
                
            tree_predictions = np.zeros(n)
            
            for tree in self.trees:
                tree_predictions += self.learning_rate * tree.predict(Xq)
                
            predictions[mask] = tree_predictions
            
        return predictions

# Train the gradient boosting LTR algorithm on the training set
ltr = GradientBoostingLTR(num_trees=100, learning_rate=0.1, max_depth=6)
ltr.fit(train_features, train_labels, train_data['qid'].values)

# Evaluate the LTR algorithm on the test set using NDCG@10
predictions = ltr.predict(test_features, test_data['qid'].values)
test_ndcg = ndcg(test_labels, predictions, k=10)

print("Test NDCG@10:", test_ndcg)
