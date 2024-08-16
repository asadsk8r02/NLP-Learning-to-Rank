import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import lightgbm

# Load the dataset
file_path = "C:/Users/Lenovo/Downloads/fold1_train_sample_all_queries.csv"

df = pd.read_csv(file_path)
print(f'Shape of Dataframe: {df.shape}')
Shape of Dataframe: (41197, 138)

# Assign column names
df.columns = ['relevance'] + ['qid'] + [f'feature_{i}' for i in range(df.shape[1]-2)]

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

qids_train = train_df["qid"].value_counts().to_numpy()
X_train = train_df.drop(["qid", "relevance"], axis=1)
y_train = train_df["relevance"]

qids_test = test_df["qid"].value_counts().to_numpy()
X_test = test_df.drop(["qid", "relevance"], axis=1)
y_test = test_df["relevance"]


# Lightgbm model
model = lightgbm.LGBMRanker(
    objective="rank_xendcg",
    metric="ndcg",
    ndcg_eval_at= [10],
    learning_rate=0.1,
    num_leaves=31,
    verbose= 0,
    force_col_wise=True
)
model.fit(
    X=X_train,
    y=y_train,
    group=qids_train,
    eval_set=[(X_test, y_test)],
    eval_group=[qids_test],
    eval_at=10,
    verbose=10,
)

y_pred = model.predict(X_test)
ndcg_score = max(model.evals_result_['valid_0']['ndcg@10'])
# Create scatter plot of predicted vs actual values
plt.scatter(y_test, y_pred)
plt.plot([0, 5], [0, 5], '--', color='red')  # add diagonal line for perfect predictions

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Predicted vs Actual Relevance Scores (NDGC: {ndcg_score:.4f})')

plt.show()

 
