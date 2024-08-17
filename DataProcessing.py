import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
from sklearn.metrics import make_scorer

class FeaturePreprocessor:
    @staticmethod
    def preprocess_features(file_path='features.csv'):
        features = pd.read_csv(file_path)
        new_header = features.iloc[0].str.replace(' ', '_')
        features = features[1:]
        features.columns = new_header
        features['feature_description'] = features['feature_description'].ffill()
        
        character_removal = [' ', '(', ')', '*']
        for char in character_removal:
            features['feature_description'] = features['feature_description'].str.replace(char, '_')
            features['stream'] = features['stream'].astype(str).str.replace(char, '_')
        
        features['feature_id'] = features['feature_id'].astype(str)
        features['cols'] = 'string'
        
        for idx in range(len(features)):
            if str(features.iloc[idx]['stream']) != 'nan':
                features.at[idx, 'cols'] = features['feature_description'].iloc[idx] + '_' + features['stream'].iloc[idx]
            else:
                features.at[idx, 'cols'] = features['feature_description'].iloc[idx]
        return features

class ColumnLabeler:
    @staticmethod
    def label_columns(df):
        for col in df.columns:
            if col == 0:
                df.rename({col: 'relevance_label'}, axis=1, inplace=True)
            elif col == 1:
                df.rename({col: 'query_id'}, axis=1, inplace=True)
            else:
                df.rename({col: f'feature_{col - 1}'}, axis=1, inplace=True)
        return df

class DataLoader:
    def __init__(self, folder_num):
        self.folder_num = folder_num

    def load_and_process_data(self):
        for folder in self.folder_num:
            df_train = pd.read_csv(f'MSLR-WEB10K/Fold{folder}/train.txt', sep=' ', header=None)
            df_test = pd.read_csv(f'MSLR-WEB10K/Fold{folder}/test.txt', sep=' ', header=None)
            df_val = pd.read_csv(f'MSLR-WEB10K/Fold{folder}/vali.txt', sep=' ', header=None)
            
            df_train = ColumnLabeler.label_columns(df_train)
            df_test = ColumnLabeler.label_columns(df_test)
            df_val = ColumnLabeler.label_columns(df_val)
            
            dataframes = {'train': df_train, 'test': df_test, 'val': df_val}
            for k, df in dataframes.items():
                for i in range(1, len(df.columns)-1):
                    df[f'feature_{i}'].replace(f'{i}:', '', regex=True, inplace=True)
                df['query_id'].replace('qid:', '', regex=True, inplace=True)

            features = FeaturePreprocessor.preprocess_features()
            for k, df in dataframes.items():
                for idx in range(len(features)):
                    id_ = features.iloc[idx]['feature_id']
                    for col in df.columns:
                        if str(id_) == col.lstrip('feature_'):
                            df.rename({col: features.iloc[idx]['cols']}, axis=1, inplace=True)
            
            df_train.to_csv(f'MSLR-WEB10K/Fold{folder}/df_train.csv', index=False)
            df_test.to_csv(f'MSLR-WEB10K/Fold{folder}/df_test.csv', index=False)
            df_val.to_csv(f'MSLR-WEB10K/Fold{folder}/df_val.csv', index=False)

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
        self.data_path = data_path
        self.num_trees = num_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def load_data(self):
        data = pd.read_csv(self.data_path)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
        return train_data, val_data, test_data

    def extract_features_labels(self, data):
        features = data.iloc[:, 2:].values
        labels = data.iloc[:, 1].values
        return features, labels

    def run(self):
        train_data, val_data, test_data = self.load_data()
        train_features, train_labels = self.extract_features_labels(train_data)
        val_features, val_labels = self.extract_features_labels(val_data)
        test_features, test_labels = self.extract_features_labels(test_data)

        ltr = GradientBoostingLTR(num_trees=self.num_trees, learning_rate=self.learning_rate, max_depth=self.max_depth)
        ltr.fit(train_features, train_labels, train_data['query_id'].values)

        predictions = ltr.predict(test_features, test_data['query_id'].values)
        test_ndcg = ndcg_score([test_labels], [predictions], k=10)
        print("Test NDCG@10:", test_ndcg)

class LightGBMModel:
    def __init__(self):
        self.params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': 10,
            'learning_rate': 0.1,
            'max_depth': 6,
            'num_leaves': 64,
            'verbose': 1
        }

    def train(self, train_features, train_labels, val_features, val_labels):
        train_dataset = lgb.Dataset(train_features, label=train_labels)
        val_dataset = lgb.Dataset(val_features, label=val_labels, reference=train_dataset)
        model = lgb.train(self.params, train_dataset, num_boost_round=200, valid_sets=[train_dataset, val_dataset],
                          early_stopping_rounds=10, verbose_eval=10)
        return model

    def evaluate(self, model, test_features, test_labels):
        predictions = model.predict(test_features)
        test_ndcg = ndcg_score(test_labels, predictions, k=10)
        print("Test NDCG@10:", test_ndcg)

if __name__ == "__main__":
    # Data preprocessing
    data_loader = DataLoader(folder_num=[1])
    data_loader.load_and_process_data()

    # Gradient Boosting LTR
    ltr_pipeline = LTRPipeline(data_path="MSLR-WEB10K/Fold1/df_train.csv")
    ltr_pipeline.run()

    # LightGBM Model
    data = pd.read_csv("MSLR-WEB10K/Fold1/df_train.csv")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    train_features = train_data.iloc[:, 2:]
    train_labels = train_data.iloc[:, 1]
    val_features = val_data.iloc[:, 2:]
    val_labels = val_data.iloc[:, 1]
    test_features = test_data.iloc[:, 2:]
    test_labels = test_data.iloc[:, 1]

    lightgbm_model = LightGBMModel()
    model = lightgbm_model.train(train_features, train_labels, val_features, val_labels)
    lightgbm_model.evaluate(model, test_features, test_labels)
