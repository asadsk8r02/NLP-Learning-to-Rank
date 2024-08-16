import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = '/Users/atifsiddiqui/Documents/SPRING2023/IR/fold1_train_sample_all_queries.csv'
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load MSLR dataset
df = pd.read_csv(file_path)
column_names = pd.read_csv('/Users/atifsiddiqui/Documents/SPRING2023/IR/file.csv', header=None, squeeze=True)

# Remove the last column name if there are 137 names in the list
if len(column_names) == 137:
    column_names = column_names[:-1]

# Add the column names to the dataframe starting from column 2
df.columns.values[2:138] = column_names

# Print the updated dataframe
print(df)

df.shape

df.head(10)

unique_scores = df['0'].unique()
print(unique_scores)

import matplotlib.pyplot as plt
# Get the frequency of unique values in column 2
freq = df['0'].value_counts()

# Create a bar graph
plt.bar(freq.index, freq.values)

# Set the x-label and y-label
plt.title("Relevance Score Distribution")
plt.xlabel('Relevance Score')
plt.ylabel('Frequency')

# Show the plot
plt.show()

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print shapes of train and test sets
print("Shape of X_train: ", X_train.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_test: ", y_test.shape)

feature_names = list(df.columns)
feature_name = feature_names[2:]

import xgboost
from xgboost import plot_importance
from matplotlib import pyplot
from xgboost import XGBClassifier

xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

xgb_model.get_booster().feature_names = feature_name
xgboost.plot_importance(xgb_model.get_booster(), max_num_features=10)
 
