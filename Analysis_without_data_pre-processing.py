import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

class DataManager:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        return pd.read_csv(self.data_path)

    def split_data(self, data, test_size=0.2, val_size=0.2, random_state=42):
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
        train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=random_state)
        return train_data, val_data, test_data

    def extract_features_labels(self, data):
        features = data.iloc[:, 2:].values
        labels = data.iloc[:, 1].values
        return features, labels

class NDCGCalculator:
    @staticmethod
    def calculate(y_true, y_pred, k=10):
        return ndcg_score(np.array([y_true]), np.array([y_pred]), k=k)

class GradientBoostingLTR:
    def __init__(self, num_trees=100, learning_rate=0.1, max_depth=6):
        self.num_trees = num_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y, qid):
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

class LTRPipeline:
    def __init__(self, data_path, num_trees=100, learning_rate=0.1, max_depth=6):
        self.data_manager = DataManager(data_path)
        self.ltr_model = GradientBoostingLTR(num_trees=num_trees, learning_rate=learning_rate, max_depth=max_depth)

    def run(self):
        # Load and split data
        data = self.data_manager.load_data()
        train_data, val_data, test_data = self.data_manager.split_data(data)

        # Extract features and labels
        train_features, train_labels = self.data_manager.extract_features_labels(train_data)
        val_features, val_labels = self.data_manager.extract_features_labels(val_data)
        test_features, test_labels = self.data_manager.extract_features_labels(test_data)

        # Train the model
        self.ltr_model.fit(train_features, train_labels, train_data['qid'].values)

        # Predict and evaluate using NDCG@10
        predictions = self.ltr_model.predict(test_features, test_data['qid'].values)
        test_ndcg = NDCGCalculator.calculate(test_labels, predictions, k=10)

        print("Test NDCG@10:", test_ndcg)

# To run the pipeline
if __name__ == "__main__":
    ltr_pipeline = LTRPipeline(data_path="data.csv")
    ltr_pipeline.run()
