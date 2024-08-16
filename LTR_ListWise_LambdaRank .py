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

# Shape of Dataframe: (41197, 138)
# df
# 0	1	2	3	4	5	6	7	8	9	...	128	129	130	131	132	133	134	135	136	137
# 0	0	1.0	3.0	0.0	2.0	2.0	3.0	1.0	0.0	0.666667	...	78.0	0.0	5.0	3492.0	65535.0	101.0	56.0	0.0	0.0	0.000000
# 1	0	1.0	3.0	0.0	2.0	2.0	3.0	1.0	0.0	0.666667	...	50.0	0.0	2.0	214.0	25445.0	38.0	55.0	0.0	0.0	0.000000
# 2	1	1.0	3.0	0.0	3.0	1.0	3.0	1.0	0.0	1.000000	...	136.0	0.0	1.0	944.0	48469.0	1.0	11.0	0.0	0.0	0.000000
# 3	1	1.0	3.0	0.0	3.0	0.0	3.0	1.0	0.0	1.000000	...	62.0	0.0	15.0	280.0	63568.0	1.0	3.0	0.0	0.0	0.000000
# 4	2	1.0	3.0	3.0	0.0	0.0	3.0	1.0	1.0	0.000000	...	62.0	11089534.0	2.0	116.0	64034.0	13.0	3.0	0.0	0.0	0.000000
# ...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
# 41192	2	29989.0	2.0	0.0	2.0	0.0	2.0	1.0	0.0	1.000000	...	71.0	0.0	1.0	6038.0	26752.0	1.0	70.0	0.0	0.0	0.000000
# 41193	2	29989.0	2.0	0.0	2.0	1.0	2.0	1.0	0.0	1.000000	...	61.0	10.0	0.0	541.0	19057.0	4.0	10.0	0.0	0.0	0.000000
# 41194	0	29992.0	2.0	0.0	2.0	2.0	2.0	1.0	0.0	1.000000	...	63.0	0.0	0.0	19456.0	39029.0	8.0	106.0	0.0	7.0	6.333333
# 41195	0	29992.0	2.0	0.0	1.0	0.0	2.0	1.0	0.0	0.500000	...	22.0	0.0	0.0	22383.0	20796.0	5.0	4.0	0.0	157.0	24.963929
# 41196	1	29992.0	2.0	1.0	1.0	0.0	2.0	1.0	0.5	0.500000	...	30.0	131.0	0.0	13556.0	25675.0	2.0	12.0	0.0	0.0	0.000000
# 41197 rows × 138 columns

# Assign column names
df.columns = ['relevance'] + ['qid'] + [f'feature_{i}' for i in range(df.shape[1]-2)]
df
# relevance	qid	feature_0	feature_1	feature_2	feature_3	feature_4	feature_5	feature_6	feature_7	...	feature_126	feature_127	feature_128	feature_129	feature_130	feature_131	feature_132	feature_133	feature_134	feature_135
# 0	0	1.0	3.0	0.0	2.0	2.0	3.0	1.0	0.0	0.666667	...	78.0	0.0	5.0	3492.0	65535.0	101.0	56.0	0.0	0.0	0.000000
# 1	0	1.0	3.0	0.0	2.0	2.0	3.0	1.0	0.0	0.666667	...	50.0	0.0	2.0	214.0	25445.0	38.0	55.0	0.0	0.0	0.000000
# 2	1	1.0	3.0	0.0	3.0	1.0	3.0	1.0	0.0	1.000000	...	136.0	0.0	1.0	944.0	48469.0	1.0	11.0	0.0	0.0	0.000000
# 3	1	1.0	3.0	0.0	3.0	0.0	3.0	1.0	0.0	1.000000	...	62.0	0.0	15.0	280.0	63568.0	1.0	3.0	0.0	0.0	0.000000
# 4	2	1.0	3.0	3.0	0.0	0.0	3.0	1.0	1.0	0.000000	...	62.0	11089534.0	2.0	116.0	64034.0	13.0	3.0	0.0	0.0	0.000000
# ...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
# 41192	2	29989.0	2.0	0.0	2.0	0.0	2.0	1.0	0.0	1.000000	...	71.0	0.0	1.0	6038.0	26752.0	1.0	70.0	0.0	0.0	0.000000
# 41193	2	29989.0	2.0	0.0	2.0	1.0	2.0	1.0	0.0	1.000000	...	61.0	10.0	0.0	541.0	19057.0	4.0	10.0	0.0	0.0	0.000000
# 41194	0	29992.0	2.0	0.0	2.0	2.0	2.0	1.0	0.0	1.000000	...	63.0	0.0	0.0	19456.0	39029.0	8.0	106.0	0.0	7.0	6.333333
# 41195	0	29992.0	2.0	0.0	1.0	0.0	2.0	1.0	0.0	0.500000	...	22.0	0.0	0.0	22383.0	20796.0	5.0	4.0	0.0	157.0	24.963929
# 41196	1	29992.0	2.0	1.0	1.0	0.0	2.0	1.0	0.5	0.500000	...	30.0	131.0	0.0	13556.0	25675.0	2.0	12.0	0.0	0.0	0.000000
# 41197 rows × 138 columns

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

qids_train = train_df["qid"].value_counts().to_numpy()
X_train = train_df.drop(["qid", "relevance"], axis=1)
y_train = train_df["relevance"]

qids_test = test_df["qid"].value_counts().to_numpy()
X_test = test_df.drop(["qid", "relevance"], axis=1)
y_test = test_df["relevance"]
# X_train

# Lightgbm model
model = lightgbm.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
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

 
 
 
 
 
