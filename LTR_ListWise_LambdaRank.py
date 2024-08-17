import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import lightgbm as lgb


classDataPreprocessor:
    def__init__(self, file_path):
        self.file_path = file_path

    defload_and_prepare_data(self):
        df = pd.read_csv(self.file_path)
        print(f'Shape of DataFrame: {df.shape}')
        df.columns = ['relevance'] + ['qid'] + [f'feature_{i}'for i inrange(df.shape[1]-2)]
        return df

    defsplit_data(self, df, test_size=0.2, random_state=42):
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        return train_df, test_df

    defget_features_labels_groups(self, df):
        qids = df["qid"].value_counts().to_numpy()
        X = df.drop(["qid", "relevance"], axis=1)
        y = df["relevance"]
        return X, y, qids

classLightGBMRankerModel:
    def__init__(self):
        self.model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg")

    deftrain(self, X_train, y_train, qids_train, X_test, y_test, qids_test):
        self.model.fit(
            X=X_train,
            y=y_train,
            group=qids_train,
            eval_set=[(X_test, y_test)],
            eval_group=[qids_test],
            eval_at=10,
            verbose=10,
        )

    defpredict(self, X_test):
        return self.model.predict(X_test)

    defget_best_ndcg_score(self):
        returnmax(self.model.evals_result_['valid_0']['ndcg@10'])

classPlotter:
    @staticmethoddefplot_predicted_vs_actual(y_test, y_pred, ndcg_score):
        plt.scatter(y_test, y_pred)
        plt.plot([0, 5], [0, 5], '--', color='red')  # Add diagonal line for perfect predictions
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predicted vs Actual Relevance Scores (NDCG: {ndcg_score:.4f})')
        plt.show()

if __name__ == "__main__":
    file_path = "C:/Users/Lenovo/Downloads/fold1_train_sample_all_queries.csv"# Data Preprocessing
    preprocessor = DataPreprocessor(file_path)
    df = preprocessor.load_and_prepare_data()
    train_df, test_df = preprocessor.split_data(df)
    
    X_train, y_train, qids_train = preprocessor.get_features_labels_groups(train_df)
    X_test, y_test, qids_test = preprocessor.get_features_labels_groups(test_df)

    # LightGBM Model Training
    model = LightGBMRankerModel()
    model.train(X_train, y_train, qids_train, X_test, y_test, qids_test)

    # Prediction and Evaluation
    y_pred = model.predict(X_test)
    ndcg_score = model.get_best_ndcg_score()

    # Plotting
    Plotter.plot_predicted_vs_actual(y_test, y_pred, ndcg_score)
