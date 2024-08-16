import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import ndcg_score

def load_data(file_path):
    data = pd.read_csv(file_path, sep=' ', header=None)
    data.drop(data.columns[-1], axis=1, inplace=True)  # Drop the last empty column
    data[0] = data[0].str.split(':').str.get(1).astype(int)  # Extract the relevance label
    for col in range(1, data.shape[1]):
        data[col] = data[col].str.split(':').str.get(1).astype(float)  # Extract the feature values
    return data

def generate_pairwise_data(grouped_data):
    pairwise_data_1, pairwise_data_2, binary_preferences = [], [], []

    for _, group in grouped_data:
        group = group.drop('1', axis=1)  # Remove the query column
        docs = group.drop('0', axis=1).values
        labels = group['0'].values

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] != labels[j]:
                    pairwise_data_1.append(docs[i])
                    pairwise_data_2.append(docs[j])
                    binary_preferences.append(1 if labels[i] > labels[j] else 0)

    return np.array(pairwise_data_1), np.array(pairwise_data_2), np.array(binary_preferences)
  
def ranknet_model(input_shape, hidden_units=64):
    input1 = Input(shape=(input_shape,))
    input2 = Input(shape=(input_shape,))

    hidden_layer = Dense(hidden_units, activation='relu')

    score1 = hidden_layer(input1)
    score2 = hidden_layer(input2)

    score_diff = Lambda(lambda x: x[0] - x[1])([score1, score2])
    probability = Dense(1, activation='sigmoid')(score_diff)

    model = Model(inputs=[input1, input2], outputs=probability)
    
    # Define the scoring model separately
    scoring_input = Input(shape=(input_shape,))
    scoring_output = hidden_layer(scoring_input)
    scoring_model = Model(inputs=scoring_input, outputs=scoring_output)

    return model, scoring_model
  
# Define the hyperparameters
input_shape = 136
hidden_units = 64
learning_rate = 0.001
batch_size = 32
epochs = 30

# Load the MSLR-WEB10K dataset
train_data = pd.read_csv("fold1_train_sample.csv")
test_data = pd.read_csv("fold1_test_sample.csv")

# Normalize the features
scaler = MinMaxScaler()
train_data.iloc[:, 2:] = scaler.fit_transform(train_data.iloc[:, 2:])
test_data.iloc[:, 2:] = scaler.transform(test_data.iloc[:, 2:])

# Group the data by query
train_grouped = train_data.groupby('1')
test_grouped = test_data.groupby('1')

# Generate pairwise data
train_pairwise_data_1, train_pairwise_data_2, train_binary_preferences = generate_pairwise_data(train_grouped)
test_pairwise_data_1, test_pairwise_data_2, test_binary_preferences = generate_pairwise_data(test_grouped)

# Instantiate the RankNet model and the scoring model
model, scoring_model = ranknet_model(input_shape, hidden_units)

# Compile the model with the binary cross-entropy loss and the Adam optimizer
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate))

from tensorflow.keras.utils import plot_model
from IPython.display import Image
plot_model(model, to_file='ranknet_model.png', show_shapes=True, show_layer_names=True, dpi=96)

# Image(filename='ranknet_model.png')

# Train the model using the pairwise data
model.fit([train_pairwise_data_1, train_pairwise_data_2], train_binary_preferences, batch_size=batch_size, epochs=epochs)

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max

def evaluate_ndcg(test_grouped, scoring_model, k=10):
    ndcg_scores = []

    for query, group in test_grouped:
        group = group.drop('1', axis=1)  # Remove the query column
        true_rel = group['0'].values
        features = group.drop('0', axis=1).values
        pred_rel = scoring_model.predict(features).flatten()

        # Combine true_rel and pred_rel into a list of tuples and sort by pred_rel
        combined = list(zip(true_rel, pred_rel))
        combined.sort(key=lambda x: x[1], reverse=True)

        # Extract the sorted true_rel
        true_rel_sorted = [x[0] for x in combined]

        ndcg_scores.append(ndcg_at_k(true_rel_sorted, k))

    return np.mean(ndcg_scores)

ndcg = evaluate_ndcg(test_grouped, scoring_model)
print("NDCG@10:", ndcg)
# NDCG@10: 0.7172376787429267
