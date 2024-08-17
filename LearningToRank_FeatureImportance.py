import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance

classDataLoader:
    def__init__(self, data_file_path, column_file_path):
        self.data_file_path = data_file_path
        self.column_file_path = column_file_path

    defload_data(self):
        # Load the dataset
        df = pd.read_csv(self.data_file_path)
        column_names = pd.read_csv(self.column_file_path, header=None, squeeze=True)

        # Remove the last column name if there are 137 names in the listiflen(column_names) == 137:
            column_names = column_names[:-1]

        # Add the column names to the dataframe starting from column 2
        df.columns.values[2:138] = column_names
        return df

classDataAnalyzer:
    @staticmethoddefanalyze_data(df):
        print(f"Shape of DataFrame: {df.shape}")
        print(df.head(10))

        unique_scores = df['0'].unique()
        print("Unique Relevance Scores:", unique_scores)

        freq = df['0'].value_counts()
        DataAnalyzer.plot_distribution(freq)

    @staticmethoddefplot_distribution(freq):
        plt.bar(freq.index, freq.values)
        plt.title("Relevance Score Distribution")
        plt.xlabel('Relevance Score')
        plt.ylabel('Frequency')
        plt.show()

classModelTrainer:
    def__init__(self):
        self.model = xgb.XGBClassifier()

    deftrain_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    deffeature_importance(self, feature_names):
        self.model.get_booster().feature_names = feature_names
        plot_importance(self.model.get_booster(), max_num_features=10)
        plt.show()

classDataPreprocessor:
    @staticmethoddefsplit_data(df, test_size=0.3, random_state=42):
        X = df.iloc[:, 2:]
        y = df.iloc[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print(f"Shape of X_train: {X_train.shape}")
        print(f"Shape of y_train: {y_train.shape}")
        print(f"Shape of X_test: {X_test.shape}")
        print(f"Shape of y_test: {y_test.shape}")
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # File paths
    data_file_path = '/Users/atifsiddiqui/Documents/SPRING2023/IR/fold1_train_sample_all_queries.csv'
    column_file_path = '/Users/atifsiddiqui/Documents/SPRING2023/IR/file.csv'# Load and prepare data
    data_loader = DataLoader(data_file_path, column_file_path)
    df = data_loader.load_data()

    # Analyze data
    DataAnalyzer.analyze_data(df)

    # Preprocess data
    X_train, X_test, y_train, y_test = DataPreprocessor.split_data(df)

    # Train model and display feature importance
    model_trainer = ModelTrainer()
    model_trainer.train_model(X_train, y_train)
    model_trainer.feature_importance(list(df.columns)[2:])
